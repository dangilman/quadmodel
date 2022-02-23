import sys

sys.path.insert(0,'/u/home/v/vdike/code/quadmodel/')
sys.path.insert(0,'/u/home/v/vdike/code/pyHalo/')
sys.path.insert(0,'/u/home/v/vdike/code/lenstronomy/')

from quadmodel.quadmodel import QuadLensSystem
from quadmodel.Solvers.hierachical import HierarchicalOptimization
import os
import subprocess
from time import time
from quadmodel.data.load_preset_lens import load_preset_lens
import quadmodel.inference.realization_setup as qm_setup # was star import, changed
from quadmodel.macromodel import MacroLensModel
from quadmodel.inference.util import filenames, SimulationOutputContainer
import numpy as np
import dill
from copy import deepcopy

from pyHalo.utilities import interpolate_ray_paths
from pyHalo.realization_extensions import RealizationExtensions
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction


# def forward_model(output_path, job_index, lens_data, n_keep, kwargs_sample_realization, tolerance=0.5,
#                   verbose=False, readout_steps=2, kwargs_realization_other={}):

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



#%% REMOVE AFTER TESTING
# output_path = os.getcwd() + '/example_inference_output/'

# import shutil 

# if os.path.exists(output_path) is True:
#     shutil.rmtree(output_path) # remove folder stored before so it starts over
#%% INPUTS FROM EXAMPLE INFERENCE SCRIPT
lens_data = 'B1422_fakeNoPBH'
# the path where we generate output samples
output_path = os.getcwd() + '/output_NoPBH_fixedFrac'
# a unique integer that is appended to the filenames
job_index = 1
# the number of realizations to accept in the posterior; the function will run until this many samples are generated
n_keep = 100
# sets the tolerance for the summary statistics (a reasonable value is 0.05, using a larger value here for speed)
summary_statistic_tolerance = 0.05

# defines the priors for parameters sampled in the forward model
realization_priors = {}
realization_priors['PRESET_MODEL'] = 'CDM'
realization_priors['sigma_sub'] = ['UNIFORM', 0.0, 0.1]
realization_priors['LOS_normalization'] = ['UNIFORM', 0.8, 1.2]
realization_priors['power_law_index'] = ['FIXED', -1.9]
realization_priors['log_m_host'] = ['GAUSSIAN', 13.3, 0.3] 
realization_priors['log_black_hole_mass'] = ['UNIFORM', 4., 6.]
realization_priors['mass_fraction'] = ['FIXED', .5]
# realization_priors['log10c0'] = ['UNIFORM', 2, 4] 
# realization_priors['beta'] = ['FIXED', 0.8] 
# realization_priors['zeta'] = ['FIXED', -0.25] 

tolerance=summary_statistic_tolerance
verbose=False
kwargs_sample_realization = realization_priors
readout_steps=2
kwargs_realization_other={}


log_mlow = 6.
log_mhigh = 10.

#%%
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
    lens_data_class = load_preset_lens(lens_data)
else:
    lens_data_class = lens_data
magnifications, magnification_uncertainties, astrometric_uncertainty, R_ein_approx = \
    lens_data_class.m, lens_data_class.delta_m, \
    lens_data_class.delta_xy, lens_data_class.approx_einstein_radius

# You can restart inferences from previous runs by simply running the function again. In the following lines, the
# code looks for existing output files, and determines how many samples to add based on how much output already
# exists.
# Why this functionality?
# This needs to be changed for testing
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
#     return

# Initialize stuff for the inference
idx_init = n_kept
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

magnifications = np.array(magnifications) # magnifications loaded in lens_data
_flux_ratios_data = magnifications[1:] / magnifications[0] # why internalize this?

log_ref_mass=6. # if you change the mass range, change the aperture scaling below


# start the simulation, the while loop will execute until one has obtained n_keep samples from the posterior
while True:

    # get the lens redshift, for some deflectors with photometrically-estimated redshifts, we have to sample a PDF
    lens_data_class.set_zlens()
    zlens = lens_data_class.zlens
    zsource = lens_data_class.zsource

    # add astrometric uncertainties to image positions
    delta_x, delta_y = np.random.normal(0, astrometric_uncertainty), np.random.normal(0, astrometric_uncertainty)
    lens_data_class_sampling = deepcopy(lens_data_class)
    lens_data_class_sampling.x += delta_x
    lens_data_class_sampling.y += delta_y

    # EDIT STARTING HERE
    # first fit lens then add pbh
    # parse the input dictionaries into arrays with parameters drawn from their respective priors
    realization_samples, preset_model, kwargs_preset_model, param_names_realization = qm_setup.setup_realization(kwargs_sample_realization, 
                                                                                                                 kwargs_realization_other)
    log_black_hole_mass = kwargs_preset_model['log_black_hole_mass'] # log10 pbh mass
    pbh_mass_fraction = kwargs_preset_model['mass_fraction']
    
    kwargs_pbh_mass_function = {'mass_function_type': 'DELTA', 'logM': log_black_hole_mass}
    aperture_smaller = np.sqrt(10**(log_black_hole_mass - log_ref_mass)) # aperture = k*sqrt(mass)
    # the Einstein radius of a point mass goes like sqrt(mass), so we can set the aperture size proportional to sqrt(mass)
    # larger aperture sizes include more black holes in the lens model and therefore take more computation time   
    r_max = 0.35 * aperture_smaller # the angular aperture inside which we render the PBH
    if r_max < 0.1 : r_max =0.1

    
    # load the lens macromodel defined in the data class
    model, constrain_params_macro, optimization_routine, macromodel_samples, param_names_macro = \
        lens_data_class_sampling.generate_macromodel()
    macromodel = MacroLensModel(model.component_list)
    
    # create the realization
    # we set the cone opening angle to 6 times the Einstein radius to get all the halos near images
    cone_opening_angle = 6 * R_ein_approx
    realization = preset_model(zlens, zsource, cone_opening_angle_arcsec=cone_opening_angle,
                      **kwargs_preset_model)
    if verbose:
        print('realization contains ' + str(len(realization.halos)) + ' halos.')
        print('realization parameter array: ', realization_samples)

    # This sets up a baseline lens macromodel and aligns the dark matter halos to follow the path taken by the
    # light rays. This is important if the source is significantly offset from the lens centroid
    lens_system = QuadLensSystem.shift_background_auto(lens_data_class, macromodel, zsource, realization)

    # Now we set up the optimization routine, which will solve for a set of macromodel parameters that map the
    # observed image coordinates to common source position in the presence of all the dark matter halos along the
    # line of sight and in the main lens plane.
    optimizer = HierarchicalOptimization(lens_system)
    kwargs_lens_final, lens_model_full, return_kwargs = optimizer.optimize(lens_data_class,
                                                                           constrain_params=constrain_params_macro,
                                                                           param_class_name=optimization_routine,
                                                                           verbose=verbose)
   
    # #Now, setup the source model, and ray trace to compute the image magnifications
    source_size_pc, kwargs_source_model, source_samples, param_names_source = \
    lens_data_class_sampling.generate_sourcemodel()
    #GET PBH MAGS START
    x_image_interp_list, y_image_interp_list = interpolate_ray_paths(lens_data_class.x, lens_data_class.y, lens_model_full,
                                                                     kwargs_lens_final, zsource)

    ext = RealizationExtensions(lens_system.realization)
    
    cosmology_class = realization.lens_cosmo.cosmo
    halo_mass_function = LensingMassFunction(cosmology_class, zlens, zsource, 
                                             mlow=10**log_mlow, mhigh=10**log_mhigh, cone_opening_angle=cone_opening_angle)
    # the mass fraction in halos is technically a function of redshift, but the redshift evolution is negligible so we can ignore it
    
    mass_fraction_in_halos = halo_mass_function.mass_fraction_in_halos(zlens, 10**log_mlow, 10**log_mhigh)
    pbh_realization = ext.add_primordial_black_holes(pbh_mass_fraction, kwargs_pbh_mass_function, mass_fraction_in_halos,
                                      x_image_interp_list, y_image_interp_list, r_max)
    lens_system_pbh = QuadLensSystem.addRealization(lens_system, pbh_realization)
    
    lensmodel, kwargs = lens_system_pbh.get_lensmodel()
    
    magnifications_with_pbh = lens_system_pbh.quasar_magnification(lens_data_class.x, 
                                       lens_data_class.y, source_size_pc, lens_model=lensmodel, 
                                       kwargs_lensmodel=kwargs, grid_axis_ratio=1., 
                                       grid_resolution_rescale=1., source_model='GAUSSIAN')
    mags = magnifications_with_pbh
    #GET PBH MAGS END

    # #EDIT to replace with getting mags with PBH
    # mags = lens_system.quasar_magnification(lens_data_class.x,
    #                                         lens_data_class.y, source_size_pc, lens_model=lens_model_full,
    #                                         kwargs_lensmodel=kwargs_lens_final, grid_axis_ratio=0.5,
    #                                         grid_resolution_rescale=2., **kwargs_source_model)

    # Now we account for uncertainties in the image magnifications. These uncertainties are sometimes quoted for
    # individual image fluxes, or the flux ratios.
    if lens_data_class.uncertainty_in_magnifications:
        mags_with_uncertainties = []
        # If uncertainties are quoted for image fluxes, we can add them to the model-predicted image magnifications
        for j, mag in enumerate(mags):
            if magnification_uncertainties[j] is None:
                m = np.nan
            else:
                m = abs(mag + np.random.normal(0, magnification_uncertainties[j] * mag))
            mags_with_uncertainties.append(m)

        mags_with_uncertainties = np.array(mags_with_uncertainties)
        _flux_ratios = mags_with_uncertainties[1:] / mags_with_uncertainties[0]

    else:
        # If uncertainties are quoted for image flux ratios, we first compute the flux ratios, and then add
        # the uncertainties
        flux_ratios = mags[1:] / mags[0]
        fluxratios_with_uncertainties = []
        for j, fr in enumerate(flux_ratios):
            df = np.random.normal(0, fr * magnification_uncertainties[j])
            new_fr = fr + df
            fluxratios_with_uncertainties.append(new_fr)
        _flux_ratios = np.array(fluxratios_with_uncertainties)

    # Next, we keep only the flux ratios for which we have good data (for most lenses with well-measured fluxes,
    # this will be all the images, so keep_flux_ratio_index would be a list [0, 1, 2]
    # are these just pre-flagged?
    flux_ratios_data = []
    flux_ratios = []
    for idx in lens_data_class_sampling.keep_flux_ratio_index:
        flux_ratios.append(_flux_ratios[idx])
        flux_ratios_data.append(_flux_ratios_data[idx])

    # Now we compute the summary statistic
    stat = 0
    for f_i_data, f_i_model in zip(flux_ratios_data, flux_ratios):
        stat += (f_i_data - f_i_model)**2
    stat = np.sqrt(stat)

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
        params = np.append(np.append(realization_samples, source_samples), macromodel_samples)
        param_names = param_names_realization + param_names_source + param_names_macro + ['summary_statistic']
        saved_lens_systems.append(lens_system_pbh)
        lens_data_class_sampling_list.append(lens_data_class_sampling)
        acceptance_ratio = accepted_realizations_counter/iteration_counter

        if parameter_array is None:
            parameter_array = np.append(params, stat)
        else:
            parameter_array = np.vstack((parameter_array, np.append(params, stat)))
        parameter_array = np.atleast_2d(parameter_array)

        if mags_out is None:
            mags_out = mags
        else:
            mags_out = np.vstack((mags_out, mags))
        mags_out = np.atleast_2d(mags_out)
        
        if verbose:
            print('N_kept: ', n_kept)
            print('N remaining: ', n_keep - n_kept)

    if verbose:
        print('accepted realizations counter: ', acceptance_rate_counter)
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
    if readout_sampling_rate:
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

        i = idx_init + 1
        for idx_system, system in enumerate(saved_lens_systems):
            container = SimulationOutputContainer(lens_data_class_sampling_list[idx_system], system,
                                                  mags_out[idx_system,:],
                                                  parameter_array[idx_system,:])
            f = open(filename_realizations + 'simulation_output_' + str(i + idx_system), 'wb')
            dill.dump(container, f)
            idx_init += 1
        print('SHAPE IS '+str(parameter_array.shape))
        parameter_array = None
        mags_out = None
        lens_data_class_sampling_list = []
        saved_lens_systems = []

    if break_loop:
        print('\nSIMULATION FINISHED')
        break

