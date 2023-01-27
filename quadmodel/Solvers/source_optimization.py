import numpy as np
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
import os
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
import sys
import pickle
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from copy import deepcopy
from lenstronomy.Plots.model_plot import ModelPlot

N_jobs = 2
job_index = int(sys.argv[1])
path_to_simulation_output = os.getenv('HOME')+'/Code/quadmodel/notebooks/fluxratio_arc_inference/inference_output/job_'+str(job_index)+'/'
path_to_data = os.getenv('HOME')+'/Code/quadmodel/notebooks/fluxratio_arc_inference/'

n_max = 4
n_threads = 8

n_pso_particles = 2
n_pso_iterations = 2
n_burn = 2
n_run = 2

for idx in range(1, N_jobs+1):

    try:
        f = open(path_to_simulation_output + 'simulation_output_'+str(idx), 'rb')
        simulation_output = pickle.load(f)
        f.close()
    except:
        print('could not find simulation output file '+path_to_simulation_output + 'simulation_output_'+str(idx))
        continue

    f = open(path_to_data+'simulated_lens_data', 'rb')
    hst_data = pickle.load(f)
    f.close()

    x_image, y_image = simulation_output.data.x, simulation_output.data.y
    fluxes = simulation_output.data.m
    lens_system = simulation_output.lens_system
    lensmodel, kwargs_lens = lens_system.get_lensmodel()
    source_x, source_y = lensmodel.ray_shooting(x_image, y_image, kwargs_lens)
    source_x = np.mean(source_x)
    source_y = np.mean(source_y)

    ###################################
    """"The rest runs lenstronomy only"""
    ###################################

    source_brightness_intrinsic = 100.0
    kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                  'point_amp': fluxes * source_brightness_intrinsic}]
    lensmodel, kwargs_lens = lens_system.get_lensmodel()
    lensmodel_macro, kwargs_macro = lens_system.get_lensmodel(include_substructure=False)

    # data specifics
    background_rms = 0.5  # background noise per pixel
    exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
    window_size = 4.0
    numPix = int(window_size / deltaPix)
    fwhm = 0.1  # full width half max of PSF

    # PSF specification
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
    data_class = ImageData(**kwargs_data)
    kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 5}
    psf_class = PSF(**kwargs_psf)

    lens_model_list_macro_data = ['EPL', 'SHEAR']
    kwargs_lens_macro_data = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.25, 'e2': -0.1, 'gamma': 2.09},
                              {'gamma1': 0.013, 'gamma2': 0.06}]

    # list of light profiles (for lens and source)
    # 'SERSIC': spherical Sersic profile
    lens_light_model_list = ['SERSIC_ELLIPSE']
    kwargs_sersic = {'amp': 800, 'R_sersic': 0.6, 'n_sersic': 4.0, 'e1': 0.1, 'e2': -0.05, 'center_x': 0.0,
                     'center_y': 0.0}
    kwargs_lens_light = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

    # 'SERSIC_ELLIPSE': elliptical Sersic profile
    source_model_list = ['SERSIC_ELLIPSE']
    kwargs_sersic_ellipse = {'amp': 3000., 'R_sersic': 0.04, 'n_sersic': 3.2, 'center_x': source_x,
                             'center_y': source_y, 'e1': 0.1, 'e2': 0.15}
    kwargs_source = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)

    # note: the relative magnification of point sources is not used as constraints in the fitting in the default settings of lenstronomy.
    # you can set this constraint with the keyword 'fixed_magnification_list' (see next block). The images are treated otherwise as separate linear amplitudes that are constraint independently of each other.
    point_source_list = ['LENSED_POSITION']
    point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    imageModel = ImageModel(data_class, psf_class, lensmodel, source_model_class,
                            lens_light_model_class,
                            point_source_class, kwargs_numerics=kwargs_numerics)
    image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
    bkg = image_util.add_background(image_sim, sigma_bkd=background_rms)
    image_sim = image_sim + bkg + poisson

    n_macro = len(kwargs_macro)
    lens_model_list_macro = lensmodel_macro.lens_model_list
    lens_model_list_halos = lensmodel.lens_model_list[n_macro:]

    kwargs_data = {'image_data': hst_data.image_data,
                   'background_rms': hst_data.background_rms,
                   'noise_map': None,
                   'exposure_time': hst_data.exposure_time,
                   'ra_at_xy_0': hst_data.ra_at_xy_0,
                   'dec_at_xy_0': hst_data.dec_at_xy_0,
                   'transform_pix2angle': np.array(hst_data.transform_pix2angle)
                   }

    likelihood_mask = hst_data.likelihood_mask
    data_class = ImageData(**kwargs_data)
    # PSF specification
    kwargs_psf = {'psf_type': 'GAUSSIAN',
                  'fwhm': 0.1,
                  'pixel_size': hst_data.deltaPix}
    psf_class = PSF(**kwargs_psf)

    lens_model_list = lens_model_list_macro + lens_model_list_halos
    source_model_list = ['SERSIC_ELLIPSE']
    if n_max>0:
        source_model_list += ['SHAPELETS']

    lens_light_model_list = ['SERSIC_ELLIPSE']
    point_source_list = ['LENSED_POSITION']

    kwargs_model = {'lens_model_list': lens_model_list,
                    'source_light_model_list': source_model_list,
                    'lens_light_model_list': lens_light_model_list,
                    'point_source_model_list': point_source_list,
                    'additional_images_list': [False],
                    'fixed_magnification_list': [False],
                    # list of bools (same length as point_source_type_list). If True, magnification ratio of point sources is fixed to the one given by the lens model
                    }

    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    num_source_model = len(source_model_list)

    kwargs_constraints = {'joint_source_with_point_source': [[0, 0]],
                          'num_point_source_list': [4],
                          'solver_type': 'PROFILE_SHEAR',
                          'fixed_lens_model': True
                          }

    prior_lens = None
    prior_lens_light = [[0, 'center_x', 0.0, 0.2], [0, 'center_y', 0.0, 0.2]]
    # have a look in the LikelihoodModule for a complete description of implemented priors

    kwargs_likelihood = {'check_bounds': True,
                         'force_no_add_image': True,
                         'source_marg': False,
                         'image_position_uncertainty': 0.004,
                         'check_matched_source_position': True,
                         'source_position_tolerance': 0.001,
                         'source_position_sigma': 0.001,
                         'prior_lens': prior_lens,
                         'prior_lens_light': prior_lens_light,
                         'image_likelihood_mask_list': [likelihood_mask]
                         }
    # 'image_likelihood_mask_list': [likelihood_mask],
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]

    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    # initial guess of non-linear parameters, we chose different starting parameters than the truth #
    shapelets_init = {'amp': 1.0, 'beta': 1e-1, 'n_max': n_max, 'center_x': 0.0, 'center_y': 0.0}
    shapelets_sigma = {'amp': 0.5, 'beta': 0.2, 'n_max': 1.0, 'center_x': 0.1, 'center_y': 0.1}
    shapelets_min = {'amp': 0.00001, 'beta': 1e-10, 'n_max': 1.0, 'center_x': -1.0, 'center_y': -1.0}
    shapelets_max = {'amp': 1000.0, 'beta': 100.0, 'n_max': 10.0, 'center_x': 1.0, 'center_y': 1.0}

    kwargs_lens_init = kwargs_lens
    kwargs_source_init = kwargs_source + [shapelets_init]
    kwargs_lens_light_init = kwargs_lens_light
    kwargs_ps_init = [{'ra_image': x_image, 'dec_image': y_image}]

    # initial spread in parameter estimation #
    kwargs_lens_sigma = deepcopy(kwargs_lens)
    kwargs_source_sigma = [{'R_sersic': 0.2, 'n_sersic': 0.5, 'e1': 0.4, 'e2': 0.4, 'center_x': .1, 'center_y': 0.1}] + [shapelets_sigma]
    kwargs_lens_light_sigma = [
        {'R_sersic': 0.1, 'n_sersic': 0.2, 'e1': 0.4, 'e2': 0.4, 'center_x': .1, 'center_y': 0.1}]
    kwargs_ps_sigma = [{'ra_image': [1e-5] * 4, 'dec_image': [1e-5] * 4}]

    # hard bound lower limit in parameter space #
    kwargs_lower_lens = deepcopy(kwargs_lens)
    kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10,
                            'center_y': -10}] + [shapelets_min]
    kwargs_lower_lens_light = [
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
    kwargs_lower_ps = [{'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}]

    # hard bound upper limit in parameter space #
    kwargs_upper_lens = deepcopy(kwargs_lens)
    kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10,
                            'center_y': 10}] + [shapelets_max]
    kwargs_upper_lens_light = [{'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
    kwargs_upper_ps = [{'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}]

    # keeping parameters fixed
    kwargs_lens_fixed = deepcopy(kwargs_lens)
    index_shear = 1
    kwargs_lens_fixed[index_shear].update({'ra_0': 0, 'dec_0': 0})
    kwargs_source_fixed = [{'center_x': source_x, 'center_y': source_y},
                           {'center_x': source_x, 'center_y': source_y, 'n_max': n_max}]
    kwargs_lens_light_fixed = [{}]
    kwargs_ps_fixed = [{}]

    lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                     kwargs_upper_source]
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed,
                         kwargs_lower_lens_light, kwargs_upper_lens_light]
    ps_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_lower_ps, kwargs_upper_ps]

    kwargs_params = {'lens_model': lens_params,
                     'source_model': source_params,
                     'lens_light_model': lens_light_params,
                     'point_source_model': ps_params}

    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model,
                                  kwargs_constraints, kwargs_likelihood, kwargs_params)

    fitting_kwargs_list = [
        ['PSO', {'sigma_scale': 1., 'n_particles': n_pso_particles, 'n_iterations': n_pso_iterations, 'threadCount': n_threads}],
        ['MCMC', {'n_burn': n_burn, 'n_run': n_run, 'walkerRatio': 4, 'sigma_scale': 0.1, 'threadCount': n_threads}]
        ]

    kwargs_result = fitting_seq.best_fit()
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat")
    log_l = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None,
                                                                 **kwargs_result)
    n_data = modelPlot._imageModel.num_data_evaluate
    chi2 = 2*log_l/n_data

    f = open(path_to_simulation_output+'chi2_'+str(idx)+'.txt','w')
    f.write(str(np.round(chi2, 6)))
    f.close()

