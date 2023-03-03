from quadmodel.quad_model import QuadLensSystem
from quadmodel.Solvers.hierachical import HierarchicalOptimization
from quadmodel.Solvers.brute import BruteOptimization
import os
import subprocess
from time import time
from quadmodel.data.load_preset_lens import load_preset_lens
from quadmodel.inference.realization_setup import *
from quadmodel.macromodel import MacroLensModel
from quadmodel.inference.util import filenames, SimulationOutputContainer
import numpy as np
import dill
from copy import deepcopy


def forward_model(output_path, job_index, lens_data, n_keep, kwargs_sample_realization,
                  kwargs_sample_macromodel, tolerance=0.5,
                  verbose=False, readout_steps=2, kwargs_realization_other={},
                  ray_tracing_optimization='default', test_mode=False,
                  statistic_type='METRIC_DISTANCE', save_realizations=False, kwargs_preset_lens={},
                  crit_curves_in_test_mode=False, write_sampling_rate=False):

    """
    This function generates samples from a posterior distribution p(q | d) where q is a set of parameters and d
    specifies the image positions and flux ratios of a quadruply-imaged quasar

    :param output_path: a string specifying the directory where output will be generated
    :param job_index: a unique integer added to output file names
    :param lens_data: either a string specifying the name of the lens data class, see class in quadmodel.data, or an
    instance of a lens data class
    :param n_keep: the number of samples to generate from the posterior; the function will run until n_keep samples are
    generated
    :param kwargs_sample_realization: a dictionary of parameters that will be sampled in the forward model

    Format:

    kwargs_sample_realization['sigma_sub'] = ['UNIFORM', 0.0, 0.1]
    specifies a uniform prior on the parameter sigma_sub recognized by the preset model in pyHalo, with values ranging
    between 0 and 0.1

    :param tolerance: the tolerance threshold imposed on the summary statistics for accepting or rejecting
    a set of parameters. The summary statistic is

    S = sqrt( sum(df_model - df_data)^2 )

    or the metric distance between the model-predicted flux ratios and the measured flux ratios
    :param kwargs_sample_macromodel: keyword arguments for sampling macromodel parameters;
    currently only multipole amplitudes implemented
    :param verbose: determines how much output to print while running the inference
    :param readout_steps: determines how often output is printed to a file
    :param kwargs_realization_other: additional keyword arguments to be passed into a pyHalo preset model
    :param ray_tracing_optimization: sets the method used to perform ray tracing
    :param test_mode: prints output and generates plots of image positions and convergence maps
    :param save_realizations: toggles on or off saving entire accepted realizations
    :param statistic_type: the type of summary statistic selection to make
    :param crit_curves_in_test_mode: bool; whether or not to plot the critical curves over convergence maps
    when test_mode=True
    :param write_sampling_rate: bool; whether or not to create text files that contain the minutes per realization
    (useful in some cases to determine how long it will take to run a full simulation)
    :return:
    """

    # set up the filenames and folders for writing output
    filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio = \
        filenames(output_path, job_index)

    # if the required directories do not exist, create them
    if os.path.exists(output_path) is False:
        proc = subprocess.Popen(['mkdir', output_path])
        proc.wait()
    if os.path.exists(output_path + 'job_' + str(job_index)) is False:
        proc = subprocess.Popen(['mkdir', output_path + 'job_' + str(job_index)])
        proc.wait()

    if verbose:
        print('reading output to files: ')
        print(filename_parameters)
        print(filename_mags)

    # Now load the lens data, this can either be specified as a string, which is used to load a lens system with image
    # positions and magnifications specified in a lens-specific class (see quadmodel.data), or you can pass in a lens
    # data class directly. For the required structure of the lens data class, see quad_base and the preset data classes

    if isinstance(lens_data, str):
        lens_data_class = load_preset_lens(lens_data, kwargs_preset_lens)
    else:
        lens_data_class = lens_data

    magnifications, magnification_uncertainties, astrometric_uncertainty, R_ein_approx = \
        lens_data_class.m, lens_data_class.delta_m, \
        lens_data_class.delta_xy, lens_data_class.approx_einstein_radius

    # You can restart inferences from previous runs by simply running the function again. In the following lines, the
    # code looks for existing output files, and determines how many samples to add based on how much output already
    # exists.
    if os.path.exists(filename_mags):
        _m = np.loadtxt(filename_mags)
        try:
            n_kept = _m.shape[0]
        except:
            n_kept = 1
        write_param_names = False
    else:
        n_kept = 0
        _m = None
        write_param_names = True

    if n_kept >= n_keep:
        print('\nSIMULATION ALREADY FINISHED.')
        return

    # Initialize stuff for the inference
    idx_init = deepcopy(n_kept)
    parameter_array = None
    mags_out = None
    readout = False
    break_loop = False
    saved_lens_systems = []
    lens_data_class_sampling_list = []
    accepted_realizations_counter = 0
    acceptance_rate_counter = 0
    iteration_counter = 0
    acceptance_ratio = np.nan
    sampling_rate = np.nan
    t0 = time()

    if verbose:
        print('starting with '+str(n_kept)+' samples accepted, '+str(n_keep - n_kept)+' remain')
        print('existing magnifications: ', _m)
        print('samples remaining: ', n_keep - n_kept)

    magnifications = np.array(magnifications)
    _flux_ratios_data = magnifications[1:] / magnifications[0]
    zsource = lens_data_class.zsource
    # start the simulation, the while loop will execute until one has obtained n_keep samples from the posterior
    while True:

        # add astrometric uncertainties to image positions
        delta_x, delta_y = np.random.normal(0, astrometric_uncertainty), np.random.normal(0, astrometric_uncertainty)
        lens_data_class_sampling = deepcopy(lens_data_class)
        lens_data_class_sampling.x += delta_x
        lens_data_class_sampling.y += delta_y

        # get the lens redshift, for some deflectors with photometrically-estimated redshifts, we have to sample a PDF
        lens_data_class_sampling.set_zlens(reset=True)
        zlens = lens_data_class_sampling.zlens

        # Now, setup the source model, and ray trace to compute the image magnifications
        source_size_pc, kwargs_source_model, source_samples, param_names_source = \
            lens_data_class_sampling.generate_sourcemodel()

        # parse the input dictionaries into arrays with parameters drawn from their respective priors
        realization_samples, preset_model, kwargs_preset_model, param_names_realization = setup_realization(kwargs_sample_realization,
                                                                                                            kwargs_realization_other,
                                                                                                            lens_data_class_sampling.x,
                                                                                                            lens_data_class_sampling.y,
                                                                                                            source_size_pc)
        if 'log_m_host' not in kwargs_preset_model.keys():
            try:
                kwargs_preset_model['log_m_host'] = np.random.normal(lens_data_class.log10_host_halo_mass,
                                                             lens_data_class.log10_host_halo_mass_sigma)
            except:
                raise Exception('must specify the host halo mass either in the list of priors, or inside the lens data class')

        model, constrain_params_macro, optimization_routine, \
        macromodel_samples, param_names_macro = lens_data_class_sampling.generate_macromodel(**kwargs_sample_macromodel)
        macromodel = MacroLensModel(model.component_list)
        # create the realization
        # we set the cone opening angle to 6 times the Einstein radius to get all the halos near images
        cone_opening_angle = 6 * R_ein_approx
        realization = preset_model(zlens, zsource, cone_opening_angle_arcsec=cone_opening_angle,
                          **kwargs_preset_model)
        lens_model_list, _, kw, _ = realization.lensing_quantities()

        if verbose:
            print('realization contains ' + str(len(realization.halos)) + ' halos.')
            print(param_names_realization)
            print('realization hyper-parameters: ', realization_samples)
            print(param_names_source)
            print('source/lens parameters: ', source_samples)
            print(param_names_macro)
            print('macromodel samples: ', macromodel_samples)

        # This sets up a baseline lens macromodel and aligns the dark matter halos to follow the path taken by the
        # light rays. This is important if the source is significantly offset from the lens centroid
        lens_system = QuadLensSystem.shift_background_auto(lens_data_class_sampling, macromodel, zsource, realization)

        # Now we set up the optimization routine, which will solve for a set of macromodel parameters that map the
        # observed image coordinates to common source position in the presence of all the dark matter halos along the
        # line of sight and in the main lens plane.

        optimizer = HierarchicalOptimization(lens_system, settings_class=ray_tracing_optimization)
        kwargs_lens_final, lens_model_full, return_kwargs = optimizer.optimize(lens_data_class_sampling,
                                                                           constrain_params=constrain_params_macro,
                                                                           param_class_name=optimization_routine,
                                                                           verbose=verbose)



        if test_mode:
            import matplotlib.pyplot as plt
            lens_system.plot_images(lens_data_class_sampling.x, lens_data_class_sampling.y, source_size_pc,
                                    lens_model_full,
                                    kwargs_lens_final,
                                    grid_resolution_rescale=2.,
                                    **kwargs_source_model)
            plt.show()
            _r = np.linspace(-2.0 * R_ein_approx, 2.0 * R_ein_approx, 200)
            xx, yy = np.meshgrid(_r, _r)
            shape0 = xx.shape
            kappa = lens_model_full.kappa(xx.ravel(), yy.ravel(), kwargs_lens_final).reshape(shape0)
            lensmodel_macro, kwargs_macro = lens_system.get_lensmodel(include_substructure=False)
            kappa_macro = lensmodel_macro.kappa(xx.ravel(), yy.ravel(), kwargs_macro).reshape(shape0)
            extent = [-2 * R_ein_approx, 2 * R_ein_approx, -2 * R_ein_approx, 2 * R_ein_approx]
            plt.imshow(kappa - kappa_macro, origin='lower', vmin=-0.1, vmax=0.1, cmap='bwr', extent=extent)
            plt.scatter(lens_data_class_sampling.x, lens_data_class_sampling.y, color='k')
            if crit_curves_in_test_mode:
                from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
                ext = LensModelExtensions(lens_model_full)
                ra_crit_list, dec_crit_list, _, _ = ext.critical_curve_caustics(kwargs_lens_final, compute_window=4*R_ein_approx,
                                                                          grid_scale=0.05)
                for i in range(0, len(ra_crit_list)):
                    plt.plot(ra_crit_list[i], dec_crit_list[i], color='k', lw=2)

            plt.show()
            _=input('continue')

        try:
            mags = lens_system.quasar_magnification(lens_data_class_sampling.x,
                                                lens_data_class_sampling.y, source_size_pc, lens_model=lens_model_full,
                                                kwargs_lensmodel=kwargs_lens_final, grid_axis_ratio=0.5,
                                                grid_resolution_rescale=2., **kwargs_source_model)
        except:
            print('SINGULAR HESSIAN MATRIX; RETRY WITH CIRCULAR APERTURE')
            mags = lens_system.quasar_magnification(lens_data_class_sampling.x,
                                                    lens_data_class_sampling.y, source_size_pc,
                                                    lens_model=lens_model_full,
                                                    kwargs_lensmodel=kwargs_lens_final, grid_axis_ratio=1,
                                                    grid_resolution_rescale=2., **kwargs_source_model)

        # Now we account for uncertainties in the image magnifications. These uncertainties are sometimes quoted for
        # individual image fluxes, or the flux ratios.

        if statistic_type == 'METRIC_DISTANCE':

            if lens_data_class.uncertainty_in_magnifications:
                mags_with_uncertainties = []
                for j, mag in enumerate(mags):
                    if magnification_uncertainties[j] is None:
                        m = np.nan
                    else:
                        delta_m = np.random.normal(0.0, magnification_uncertainties[j] * mag)
                        m = mag + delta_m
                    mags_with_uncertainties.append(m)
                mags_with_uncertainties = np.array(mags_with_uncertainties)
                _flux_ratios = mags_with_uncertainties[1:] / mags_with_uncertainties[0]

            else:
                # If uncertainties are quoted for image flux ratios, we first compute the flux ratios, and then add
                # the uncertainties
                flux_ratios = mags[1:] / mags[0]
                fluxratios_with_uncertainties = []

                for k, fr in enumerate(flux_ratios):
                    if magnification_uncertainties[k] is None:
                        new_fr = np.nan
                    else:

                        df = np.random.normal(0, fr * magnification_uncertainties[k])
                        new_fr = fr + df

                    fluxratios_with_uncertainties.append(new_fr)
                _flux_ratios = np.array(fluxratios_with_uncertainties)

            flux_ratios_data = []
            flux_ratios = []
            for idx in lens_data_class_sampling.keep_flux_ratio_index:
                flux_ratios.append(_flux_ratios[idx])
                flux_ratios_data.append(_flux_ratios_data[idx])

            # Now we compute the summary statistic
            stat = 0
            for f_i_data, f_i_model in zip(flux_ratios_data, flux_ratios):
                stat += (f_i_data - f_i_model) ** 2
            stat = np.sqrt(stat)

        elif statistic_type == 'CHI_SQUARE':

            _flux_ratios = mags[1:]/mags[0]
            flux_ratios_data = []
            flux_ratios = []
            for idx in lens_data_class_sampling.keep_flux_ratio_index:
                flux_ratios.append(_flux_ratios[idx])
                flux_ratios_data.append(_flux_ratios_data[idx])

            if lens_data_class_sampling.uncertainty_in_magnifications:
                stat = lens_data_class_sampling.flux_chi_square(mags)
            else:
                stat = lens_data_class_sampling.flux_ratio_chi_square(flux_ratios)

        else:
            raise Exception('statistic_type must be either CHI_SQUARE or METRIC_DISTANCE')

        if verbose:
            print('flux ratios data: ', flux_ratios_data)
            print('flux ratios model: ', flux_ratios)
            print('statistic: ', stat)

        acceptance_rate_counter += 1
        # Once we have computed a couple realizations, keep a log of the time it takes to run per realization
        if acceptance_rate_counter == 10 or acceptance_rate_counter == 50:
            time_elapsed = time() - t0
            time_elapsed_minutes = time_elapsed / 60
            sampling_rate = time_elapsed_minutes / acceptance_rate_counter
            readout_sampling_rate = True
        else:
            readout_sampling_rate = False

        # this keeps track of how many realizations were analyzed, and resets after each readout (set by readout_steps)
        # The purpose of this counter is to keep track of the acceptance rate
        iteration_counter += 1
        if stat < tolerance:
            # If the statistic is less than the tolerance threshold, we keep the parameters
            accepted_realizations_counter += 1
            n_kept += 1
            params = np.append(np.append(np.append(realization_samples, source_samples), macromodel_samples), stat)
            param_names = param_names_realization + param_names_source + param_names_macro + ['summary_statistic']
            saved_lens_systems.append(lens_system)
            lens_data_class_sampling_list.append(lens_data_class_sampling)
            acceptance_ratio = accepted_realizations_counter/iteration_counter

            if parameter_array is None:
                parameter_array = params
            else:
                parameter_array = np.vstack((parameter_array, params))

            if mags_out is None:
                mags_out = mags
            else:
                mags_out = np.vstack((mags_out, mags))

            if verbose:
                print('N_kept: ', n_kept)
                print('N remaining: ', n_keep - n_kept)

        if verbose:
            print('accepeted realizations counter: ', acceptance_rate_counter)
            print('readout steps: ', readout_steps)

        # readout if either of these conditions are met
        if accepted_realizations_counter == readout_steps:
            readout = True
            accepted_realizations_counter = 0
            iteration_counter = 0
        # break loop if we have collected n_keep samples
        if n_kept == n_keep:
            readout = True
            break_loop = True
        if readout_sampling_rate and write_sampling_rate:
            with open(filename_sampling_rate, 'a') as f:
                f.write(str(np.round(sampling_rate, 2)) + ' ')
                f.write('\n')

        if readout:
            # Now write stuff to file
            readout = False

            with open(filename_acceptance_ratio, 'a') as f:
                f.write(str(np.round(acceptance_ratio, 8)) + ' ')
                f.write('\n')

            with open(filename_parameters, 'a') as f:
                if write_param_names:
                    param_name_string = ''
                    for name in param_names:
                        param_name_string += name + ' '
                    f.write(param_name_string+'\n')
                    write_param_names = False

                nrows, ncols = int(parameter_array.shape[0]), int(parameter_array.shape[1])

                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(parameter_array[row, col], 6)) + ' ')
                    f.write('\n')

            with open(filename_mags, 'a') as f:
                nrows, ncols = int(mags_out.shape[0]), int(mags_out.shape[1])
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(mags_out[row, col], 6)) + ' ')
                    f.write('\n')

            if save_realizations:
                for idx_system, system in enumerate(saved_lens_systems):

                    container = SimulationOutputContainer(lens_data_class_sampling_list[idx_system], system,
                                                          mags_out[idx_system,:],
                                                          parameter_array[idx_system,:])
                    f = open(filename_realizations + 'simulation_output_' + str(idx_system + idx_init + 1), 'wb')
                    dill.dump(container, f)
                    f.close()

            idx_init += len(saved_lens_systems)

            parameter_array = None
            mags_out = None
            lens_data_class_sampling_list = []
            saved_lens_systems = []

        if break_loop:
            print('\nSIMULATION FINISHED')
            break
