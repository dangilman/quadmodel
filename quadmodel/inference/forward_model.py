from quadmodel.inference.forward_model_util import _evaluate_model
import os
import subprocess
from time import time
from quadmodel.inference.util import filenames, SimulationOutputContainer
from quadmodel.lens_system_nopyhalo import LensSystem
import numpy as np
import dill
from copy import deepcopy


def forward_model(output_path, job_index, lens_data_class, n_keep, kwargs_sample_realization,
                  kwargs_sample_macromodel, tolerance=0.5,
                  verbose=False, readout_steps=2, kwargs_realization_other={},
                  ray_tracing_optimization='default', test_mode=False,
                  save_realizations=False, crit_curves_in_test_mode=False, write_sampling_rate=False,
                  importance_weights_function=None, readout_macromodel_samples=False, n_macro=None,
                  realization_class=None, shift_background_realization=True, readout_kappagamma_statistics=False,
                  readout_curvedarc_statistics=False, diff_scale_list=[0.0001, 0.05, 0.2],
                  subtract_exact_mass_sheets=False, log_mlow_mass_sheet=None, random_seed=None, rescale_grid_size=1.0,
                  rescale_grid_resolution=2.0, index_lens_split=None):

    """
    This function generates samples from a posterior distribution p(q | d) where q is a set of parameters and d
    specifies the image positions and flux ratios of a quadruply-imaged quasar

    :param output_path: a string specifying the directory where output will be generated
    :param job_index: a unique integer added to output file names
    :param lens_data_class: aninstance of a lens data class
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
    :param crit_curves_in_test_mode: bool; whether or not to plot the critical curves over convergence maps
    when test_mode=True
    :param write_sampling_rate: bool; whether or not to create text files that contain the minutes per realization
    (useful in some cases to determine how long it will take to run a full simulation)
    :param importance_weights_function: a function that returns an importance weight given sampled parameters;
    this function also effectively overrides the prior, as new samples will be generated with probability given
    by the function
    :param readout_macromodel_samples: bool; whether or not to readout textfiles containing all macromodel samples
    :param n_macro: integer defining how many lens models correspond to the macromodel (only if readout_macromodel_samples is True)
    For example, for an EPL+SHEAR+MULTIPOLE model n_macro = 3
    :param realization_class: a fixed instance of Realization in pyHalo to use for the simulation
    :param shift_background_realization: bool; whether or not to align halos with the center of the volume
    traversed by light
    :param readout_kappagamma_statistics: bool; reads out text files with the convegence and shear evaluated at diff_scale
    :param readout_curvedarc_statistics: bool; reads out text files with the curved arc parameters evaluated at diff_scale
    :param diff_scale_list: the angular scale(s) at which to read out the convergence, shear, and curved arc parameters
    :param subtract_exact_mass_sheets: bool; if True, then pyHalo will subtract the exact amount of mass added in
    substructure at each lens plane
    :param log_mlow_mass_sheet: the minimum halo mass used to compute the negative convergence added along the LOS.
    Note: if subtract_exact_mass_sheets is True, then this argument has no effect
    :param rescale_grid_size: rescales the size of the ray-tracing grid
    :param rescale_grid_resolution: rescales the resolution (arcsec/pixel) of the ray-tracing grid
    :param index_lens_split: for use with decoupled multi-plane model, see docs in lenstronomy
    :return:
    """

    # set up the filenames and folders for writing output
    filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio, \
    filename_macromodel_samples, filename_kappagamma_stats, filename_curvedarc_stats = filenames(output_path, job_index)
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
    # You can restart inferences from previous runs by simply running the function again. In the following lines, the
    # code looks for existing output files, and determines how many samples to add based on how much output already
    # exists.
    if os.path.exists(filename_mags):
        _m = np.loadtxt(filename_mags)
        try:
            n_kept = _m.shape[0]
        except:
            n_kept = 1
        write_param_names_kappagamma_stats = False
        write_param_names_curvedarc_stats = False
        write_param_names = False
        write_param_names_macromodel_samples = False
    else:
        n_kept = 0
        _m = None
        write_param_names_kappagamma_stats = True
        write_param_names_curvedarc_stats = True
        write_param_names = True
        write_param_names_macromodel_samples = True

    if n_kept >= n_keep:
        print('\nSIMULATION ALREADY FINISHED.')
        return
    if readout_macromodel_samples:
        assert n_macro is not None, "If readout_macromodel_samples is True, you must specify a value for n_macro, the " \
                                    "number of lens models corresponding to the macromodel"

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
        print('running simulation with a summary statistic tolerance of: ', tolerance)
    # start the simulation, the while loop will execute until one has obtained n_keep samples from the posterior
    if importance_weights_function is None:
        importance_weights_function = _flat_prior_importance_weights

    while True:

        stat, realization_samples, source_samples, macromodel_samples, param_names_realization, \
        param_names_source, param_names_macro, lens_system, lens_data_class_sampling, importance_weight, model_mags, \
        kwargs_multiplane_model = _evaluate_model(lens_data_class, kwargs_sample_realization, kwargs_realization_other,
                                                  kwargs_sample_macromodel, ray_tracing_optimization, test_mode,
                                                  verbose, crit_curves_in_test_mode,
                                                  importance_weights_function,
                                                  realization_class, shift_background_realization,
                                                  subtract_exact_mass_sheets, log_mlow_mass_sheet,
                                                  rescale_grid_size, rescale_grid_resolution, index_lens_split,
                                                  random_seed)
        acceptance_rate_counter += 1
        # Once we have computed a couple realizations, keep a log of the time it takes to run per realization
        if acceptance_rate_counter == 50:
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
            params = np.append(np.append(np.append(np.append(realization_samples, source_samples), macromodel_samples), stat), importance_weight)
            param_names = param_names_realization + param_names_source + param_names_macro + ['summary_statistic', 'importance_weight']
            saved_lens_systems.append(lens_system)
            lens_data_class_sampling_list.append(lens_data_class_sampling)
            acceptance_ratio = accepted_realizations_counter/iteration_counter

            if parameter_array is None:
                parameter_array = params
            else:
                parameter_array = np.vstack((parameter_array, params))
            if mags_out is None:
                mags_out = model_mags
            else:
                mags_out = np.vstack((mags_out, model_mags))
            if verbose:
                print('N_kept: ', n_kept)
                print('N remaining: ', n_keep - n_kept)

        if verbose:
            print('accepted realizations counter: ', acceptance_rate_counter)
            print('readout steps: ', readout_steps)

        # readout if either of these conditions are met
        if accepted_realizations_counter == readout_steps:
            readout = True
            if verbose:
                print('reading out data on this iteration.')
            accepted_realizations_counter = 0
            iteration_counter = 0
        # break loop if we have collected n_keep samples
        if n_kept == n_keep:
            readout = True
            break_loop = True
            if verbose:
                print('final data readout...')
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
            if verbose:
                print('writing parameter output to '+filename_parameters)
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
            if verbose:
                print('writing flux ratio output to '+filename_mags)
            with open(filename_mags, 'a') as f:
                nrows, ncols = int(mags_out.shape[0]), int(mags_out.shape[1])
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(mags_out[row, col], 6)) + ' ')
                    f.write('\n')

            if readout_macromodel_samples:
                if verbose:
                    print('writing macromodel samples to ' + filename_macromodel_samples)
                macromodel_samples_array = None
                for l, system in enumerate(saved_lens_systems):
                    if macromodel_samples_array is None:
                        macromodel_samples_array, param_names_macromodel_samples = system.get_model_samples(n_macro)
                    else:
                        macromodel_samples_array = np.vstack((macromodel_samples_array, system.get_model_samples(n_macro)[0]))
                nrows, ncols = int(macromodel_samples_array.shape[0]), int(macromodel_samples_array.shape[1])
                with open(filename_macromodel_samples, 'a') as f:
                    if write_param_names_macromodel_samples:
                        param_name_string = ''
                        for name in param_names_macromodel_samples:
                            param_name_string += name + ' '
                        f.write(param_name_string + '\n')
                        write_param_names_macromodel_samples = False
                    for row in range(0, nrows):
                        for col in range(0, ncols):
                            f.write(str(np.round(macromodel_samples_array[row, col], 6)) + ' ')
                        f.write('\n')

            if readout_kappagamma_statistics:
                if verbose:
                    print('writing kappa/gamma statistics to ' + filename_kappagamma_stats)
                kappagamma_stats = None
                for l, system in enumerate(saved_lens_systems):
                    xi, yi = lens_data_class_sampling_list[l].x, lens_data_class_sampling_list[l].y
                    if kappagamma_stats is None:
                        kappagamma_stats, param_names_kappagamma_statistics = system.kappa_gamma_statistics(xi, yi, diff_scale=diff_scale_list)
                    else:
                        new, _ = system.kappa_gamma_statistics(xi, yi, diff_scale=diff_scale_list)
                        kappagamma_stats = np.vstack((kappagamma_stats, new))
                nrows, ncols = int(kappagamma_stats.shape[0]), int(kappagamma_stats.shape[1])
                with open(filename_kappagamma_stats, 'a') as f:
                    if write_param_names_kappagamma_stats:
                        param_name_string = ''
                        for name in param_names_kappagamma_statistics:
                            param_name_string += name + ' '
                        f.write(param_name_string + '\n')
                        write_param_names_kappagamma_stats = False
                    for row in range(0, nrows):
                        for col in range(0, ncols):
                            f.write(str(np.round(kappagamma_stats[row, col], 6)) + ' ')
                        f.write('\n')

            if readout_curvedarc_statistics:
                if verbose:
                    print('writing curved arc statistics to ' + filename_curvedarc_stats)
                curvedarc_stats = None
                for l, system in enumerate(saved_lens_systems):
                    xi, yi = lens_data_class_sampling_list[l].x, lens_data_class_sampling_list[l].y
                    if curvedarc_stats is None:
                        curvedarc_stats, param_names_curvedarc_statistics = system.curved_arc_statistics(xi, yi, diff_scale=diff_scale_list)
                    else:
                        new, _ = system.curved_arc_statistics(xi, yi, diff_scale=diff_scale_list)
                        curvedarc_stats = np.vstack((curvedarc_stats, new))
                nrows, ncols = int(curvedarc_stats.shape[0]), int(curvedarc_stats.shape[1])
                with open(filename_curvedarc_stats, 'a') as f:
                    if write_param_names_curvedarc_stats:
                        param_name_string = ''
                        for name in param_names_curvedarc_statistics:
                            param_name_string += name + ' '
                        f.write(param_name_string + '\n')
                        write_param_names_curvedarc_stats = False
                    for row in range(0, nrows):
                        for col in range(0, ncols):
                            f.write(str(np.round(curvedarc_stats[row, col], 6)) + ' ')
                        f.write('\n')

            if save_realizations:
                if verbose:
                    print('writing curved arc statistics to ' + filename_curvedarc_stats)
                for idx_system, system_with_pyhalo in enumerate(saved_lens_systems):
                    zd, zs = system_with_pyhalo.zlens, system_with_pyhalo.zsource
                    ximg = lens_data_class_sampling_list[idx_system].x
                    yimg = lens_data_class_sampling_list[idx_system].y
                    lm, kwargs_lens_save = system_with_pyhalo.get_lensmodel()
                    astropy_class = lm.cosmo
                    num_alpha_class = None
                    if hasattr(system_with_pyhalo, '_numerical_alpha_class'):
                        num_alpha_class = system_with_pyhalo._numerical_alpha_class
                    if kwargs_multiplane_model is None:
                        decouple_multi_plane = False
                    else:
                        decouple_multi_plane = True
                    kwargs_lens_model = {'lens_model_list': lm.lens_model_list,
                                         'lens_redshift_list': lm.redshift_list,
                                         'z_source': zs, 'z_lens': zd, 'multi_plane': True,
                                         'numerical_alpha_class': num_alpha_class,
                                         'decouple_multi_plane': decouple_multi_plane,
                                         'kwargs_multi_plane_model': kwargs_multiplane_model}
                    system = LensSystem(zd, zs,
                                        ximg, yimg,
                                        kwargs_lens_model, kwargs_lens_save, astropy_class)
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

def _flat_prior_importance_weights(*args, **kwargs):
    return 1.0

