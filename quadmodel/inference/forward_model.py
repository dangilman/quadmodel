from quadmodel.quadmodel import QuadLensSystem
from quadmodel.Solvers.hierachical import HierarchicalOptimization
import os
import subprocess
from time import time
from quadmodel.data.load_preset_lens import load_preset_lens
from quadmodel.inference.realization_setup import *
from quadmodel.macromodel import MacroLensModel
import numpy as np
import dill
from copy import deepcopy

class SimulationOutputContainer(object):

    """
    This class contains the output of a forward modeling simulation for a single accepted set of parameters.
    It includes the lens data class, the accepted lens system, and the corresponding set of parameters
    """

    def __init__(self, lens_data, lens_system, magnifications, parameters):

        self.data = lens_data
        self.lens_system = lens_system
        self.parameters = parameters
        self.magnifications = magnifications

def compute_flux_ratios(lens_system, lens_data_class, constrain_params_macro, optimization_routine,
                                      source_size_pc, kwargs_source_model, verbose):

    # compute macromodel parameters that map the images to a common source coordinate
    optimizer = HierarchicalOptimization(lens_system)
    kwargs_lens_final, lens_model_full, return_kwargs = optimizer.optimize(lens_data_class,
                                                                           constrain_params=constrain_params_macro,
                                                                           param_class_name=optimization_routine,
                                                                           verbose=verbose)

    mags = lens_system.quasar_magnification(lens_data_class.x,
                                            lens_data_class.y, source_size_pc, lens_model=lens_model_full,
                                            kwargs_lensmodel=kwargs_lens_final, grid_axis_ratio=0.5,
                                            grid_resolution_rescale=2., **kwargs_source_model)


    return mags

def forward_model(output_path, job_index, lens_data, n_keep, kwargs_sample_realization, tolerance=0.5,
                  verbose=False, readout_steps=2, kwargs_realization_other={}):

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

    :param verbose: determines how much output to print while running the inference
    :param readout_steps: determines how often output is printed to a file
    :param kwargs_realization_other: additional keyword arguments to be passed into a pyHalo preset model
    :return:
    """
    filename_parameters = output_path + 'job_' + str(job_index) + '/parameters.txt'
    filename_mags = output_path + 'job_' + str(job_index) + '/fluxes.txt'
    filename_realizations = output_path + 'job_' + str(job_index) + '/'
    filename_sampling_rate = output_path + 'job_' + str(job_index) + '/sampling_rate.txt'
    filename_acceptance_ratio = output_path + 'job_' + str(job_index) + '/acceptance_ratio.txt'

    if os.path.exists(output_path + 'job_' + str(job_index)) is False:
        proc = subprocess.Popen(['mkdir', output_path + 'job_' + str(job_index)])
        proc.wait()

    if verbose:
        print('reading output to files: ')
        print(filename_parameters)
        print(filename_mags)

    if isinstance(lens_data, str):
        lens_data_class = load_preset_lens(lens_data)
    else:
        lens_data_class = lens_data

    uncertainty_in_magnifications, keep_flux_ratio_index, magnifications, magnification_uncertainties, astrometric_uncertainty, R_ein_approx = \
        lens_data_class.uncertainty_in_magnifications, lens_data_class.keep_flux_ratio_index, lens_data_class.m, lens_data_class.delta_m, \
        lens_data_class.delta_xy, lens_data_class.approx_einstein_radius

    cone_opening_angle = 6 * R_ein_approx

    magnifications = np.array(magnifications)

    flux_ratios_data = magnifications[1:] / magnifications[0]

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

    parameter_array = None
    mags_out = None
    saved_lens_systems = []
    lens_data_class_sampling_list = []
    idx_init = n_kept
    readout = False
    break_loop = False
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

    if n_kept >= n_keep:
        print('\nSIMULATION ALREADY FINISHED.')
        return

    # start the simulation
    while True:

        lens_data_class.set_zlens()
        zlens = lens_data_class.zlens
        zsource = lens_data_class.zsource

        # add astrometric uncertainties to image positions
        delta_x, delta_y = np.random.normal(0, astrometric_uncertainty), np.random.normal(0, astrometric_uncertainty)
        lens_data_class_sampling = deepcopy(lens_data_class)
        lens_data_class_sampling.x += delta_x
        lens_data_class_sampling.y += delta_y

        # parse the input dictionaries into arrays with parameters drawn from their respective priors
        realization_samples, preset_model, kwargs_preset_model, param_names_realization = setup_realization(kwargs_sample_realization, kwargs_realization_other)
        # setup the source model
        source_size_pc, kwargs_source_model, source_samples, param_names_source = lens_data_class_sampling.generate_sourcemodel()
        # load the lens macromodel defined in the data class
        model, constrain_params_macro, optimization_routine, macromodel_samples, param_names_macro = lens_data_class_sampling.generate_macromodel()
        macromodel = MacroLensModel(model.component_list)

        # create the realization
        realization = preset_model(zlens, zsource, cone_opening_angle_arcsec=cone_opening_angle,
                          **kwargs_preset_model)

        if verbose:
            print('realization contains ' + str(len(realization.halos)) + ' halos.')
            print('realization parameter array: ', realization_samples)

        lens_system = QuadLensSystem.shift_background_auto(lens_data_class, macromodel, zsource, realization)

        params = np.append(np.append(realization_samples, source_samples), macromodel_samples)
        param_names = param_names_realization + param_names_source + param_names_macro + ['summary_statistic']
        mags = compute_flux_ratios(lens_system, lens_data_class_sampling, constrain_params_macro,
                                                 optimization_routine, source_size_pc, kwargs_source_model, verbose)

        acceptance_rate_counter += 1
        if acceptance_rate_counter == 5 or acceptance_rate_counter == 10:
            time_elapsed = time() - t0
            time_elapsed_minutes = time_elapsed / 60
            sampling_rate = time_elapsed_minutes / acceptance_rate_counter
            readout_sampling_rate = True
        else:
            readout_sampling_rate = False

        if uncertainty_in_magnifications:

            mags_with_uncertainties = []

            for j, mag in enumerate(mags):
                if magnification_uncertainties[j] is None:
                    m = np.nan
                else:
                    m = abs(mag + np.random.normal(0, magnification_uncertainties[j] * mag))
                mags_with_uncertainties.append(m)

            mags_with_uncertainties = np.array(mags_with_uncertainties)
            flux_ratios = mags_with_uncertainties[1:] / mags_with_uncertainties[0]
            df = 0
            for idx in keep_flux_ratio_index:
                df = flux_ratios[idx] - flux_ratios_data[idx]

        else:
            flux_ratios = mags[1:] / mags[0]
            fluxratios_with_uncertainties = []
            for j, fr in enumerate(flux_ratios):
                fluxratios_with_uncertainties.append(abs(fr + np.random.normal(0, fr * magnification_uncertainties[j])))
            fluxratios_with_uncertainties = np.array(fluxratios_with_uncertainties)
            df = flux_ratios - fluxratios_with_uncertainties

        stat = np.sqrt(np.sum(df ** 2))

        if verbose:
            print('flux ratios data: ', flux_ratios_data)
            print('flux ratios model: ', flux_ratios)
            print('statistic: ', stat)

        iteration_counter += 1

        if stat < tolerance:

            accepted_realizations_counter += 1
            n_kept += 1
            saved_lens_systems.append(lens_system)
            lens_data_class_sampling_list.append(lens_data_class_sampling)
            acceptance_ratio = accepted_realizations_counter/iteration_counter

            if parameter_array is None:
                parameter_array = np.append(params, stat)
            else:
                parameter_array = np.vstack((parameter_array, np.append(params, stat)))

            if mags_out is None:
                mags_out = mags
            else:
                mags_out = np.vstack((mags_out, mags))

        if accepted_realizations_counter == readout_steps:
            readout = True
            accepted_realizations_counter = 0
            iteration_counter = 0
        if n_kept == n_keep:
            readout = True
            break_loop = True

        if readout_sampling_rate:
            with open(filename_sampling_rate, 'a') as f:
                f.write(str(np.round(sampling_rate, 2)) + ' ')
                f.write('\n')

        if readout:

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

            i = idx_init + 1
            for idx_system, system in enumerate(saved_lens_systems):
                container = SimulationOutputContainer(lens_data_class_sampling_list[idx_system], system,
                                                      mags_out[idx_system,:],
                                                      parameter_array[idx_system,:])
                f = open(filename_realizations + 'simulation_output_' + str(i + idx_system), 'wb')
                dill.dump(container, f)
                idx_init += 1

            parameter_array = None
            mags_out = None
            lens_data_class_sampling_list = []
            saved_lens_systems = []

        if break_loop:
            print('\nSIMULATION FINISHED')
            break
