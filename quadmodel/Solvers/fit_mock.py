import numpy as np
from quadmodel.Solvers.light_fit_util import *
from lenstronomy.Data.coord_transforms import Coordinates
import pickle
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from copy import deepcopy


def fit_mock(hst_data, simulation_output, initialize_from_fit,
                path_to_smooth_lens_fit, add_shapelets_source, n_max_source, astrometric_uncertainty,
                delta_x_offset_init, delta_y_offset_init):

    x_image, y_image = simulation_output.data.x, simulation_output.data.y
    lens_system = simulation_output.lens_system
    lensmodel, kwargs_lens_init = lens_system.get_lensmodel()
    source_x, source_y = lensmodel.ray_shooting(x_image, y_image, kwargs_lens_init)
    source_x = np.mean(source_x)
    source_y = np.mean(source_y)

    ra_at_x0 = hst_data.ra_at_xy_0
    dec_at_x0 = hst_data.dec_at_xy_0
    pix2angle = hst_data.transform_pix2angle
    (nx, ny) = hst_data.image_data.shape
    coordinate_system = Coordinates(pix2angle, ra_at_x0, dec_at_x0)
    ra_coords, dec_coords = coordinate_system.coordinate_grid(nx, ny)
    tabulated_lens_model = FixedLensModel(ra_coords, dec_coords, lensmodel, kwargs_lens_init)
    lens_model_list_fit = ['TABULATED_DEFLECTIONS']

    if initialize_from_fit:
        f = open(path_to_smooth_lens_fit, 'rb')
        initial_smooth_lens_fit = pickle.load(f)
        f.close()
        print('USING BEST FIT LIGHT MODELS FROM LENS MODELING RESULT')
        source_model_list = initial_smooth_lens_fit.source_model_list
        kwargs_source_init = initial_smooth_lens_fit.kwargs_source_init
        kwargs_source_init[0]['center_x'] = source_x
        kwargs_source_init[0]['center_y'] = source_y
        lens_light_model_list = initial_smooth_lens_fit.lens_light_model_list
        kwargs_lens_light_init = initial_smooth_lens_fit.kwargs_lens_light_init
    else:
        print('USING RANDOM LIGHT MODELS')
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 1, 'R_sersic': 0.2,
             'n_sersic': 4.0, 'e1': 0.001, 'e2': 0.01,
             'center_x': source_x, 'center_y': source_x}]
        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 1, 'R_sersic': 0.2, 'n_sersic': 4.0,
             'e1': 0.001, 'e2': 0.001,
             'center_x': 0.0, 'center_y': 0.0}]

    kwargs_source_sigma, kwargs_lower_source, kwargs_upper_source, kwargs_fixed_source = \
        source_params_sersic_ellipse(source_x, source_y, kwargs_source_init)
    kwargs_lens_light_sigma, kwargs_lower_lens_light, kwargs_upper_lens_light, kwargs_fixed_lens_light = \
        lens_light_params_sersic_ellipse(kwargs_lens_light_init[0])
    kwargs_lens_init, kwargs_lens_sigma, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens = [{}], [{}], [{}], [
        {}], [{}]

    kwargs_fixed_source = deepcopy(kwargs_source_init)
    kwargs_fixed_lens_light = deepcopy(kwargs_lens_light_init)

    if add_shapelets_source:
        source_model_list += ['SHAPELETS']
        kwargs_source_sigma_shapelets, kwargs_lower_source_shapelets, \
        kwargs_upper_source_shapelets, kwargs_fixed_source_shapelets = source_params_shapelets(1, source_x,
                                                                                               source_y)
        kwargs_source_sigma += kwargs_source_sigma_shapelets
        kwargs_lower_source += kwargs_lower_source_shapelets
        kwargs_upper_source += kwargs_upper_source_shapelets
        kwargs_fixed_source += kwargs_fixed_source_shapelets

    point_source_list = ['UNLENSED']
    # point_source_list = None

    kwargs_ps_sigma, kwargs_ps_lower, kwargs_ps_upper, kwargs_ps_fixed = ps_params(x_image, y_image)
    kwargs_ps_init = [{'ra_image': y_image, 'dec_image': x_image}]
    ############################### SETUP THE DATA ######################################################
    kwargs_data = {'image_data': hst_data.image_data,
                   'background_rms': hst_data.background_rms,
                   'noise_map': None,
                   'exposure_time': hst_data.exposure_time,
                   'ra_at_xy_0': hst_data.ra_at_xy_0,
                   'dec_at_xy_0': hst_data.dec_at_xy_0,
                   'transform_pix2angle': np.array(hst_data.transform_pix2angle)
                   }

    ############################### SETUP THE PSF MODEL ######################################################
    if hst_data.psf_estimate is None:
        kwargs_psf = {'psf_type': 'GAUSSIAN',
                      'fwhm': 0.1,
                      'pixel_size': hst_data.deltaPix}
    else:
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': hst_data.psf_estimate,
                      'psf_error_map': hst_data.psf_error_map}
    #
    kwargs_model_fit = {'lens_model_list': lens_model_list_fit,
                        'source_light_model_list': source_model_list,
                        'lens_light_model_list': lens_light_model_list,
                        'point_source_model_list': point_source_list,
                        'additional_images_list': [False],
                        'fixed_magnification_list': [True],
                        'tabulated_deflection_angles': tabulated_lens_model}

    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    kwargs_constraints = {
        'num_point_source_list': [4],
        'point_source_offset': True
    }
    ############################### OPTIONAL PRIORS ############################
    prior_lens = None
    prior_lens_light = None

    ############################### OPTIONAL LIKELIHOOD MASK OVER IMAGES ############################
    kwargs_likelihood = {'check_bounds': True,
                         'force_no_add_image': True,
                         'source_marg': False,
                         'check_matched_source_position': False,
                         'astrometric_likelihood': True,
                         'image_position_uncertainty': astrometric_uncertainty,
                         'prior_lens': prior_lens,
                         'prior_lens_light': prior_lens_light,
                         'image_likelihood_mask_list': [hst_data.likelihood_mask]
                         }

    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]

    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_fixed_source, kwargs_lower_source,
                     kwargs_upper_source]
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_fixed_lens_light,
                         kwargs_lower_lens_light, kwargs_upper_lens_light]
    point_source_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_ps_lower, kwargs_ps_upper]

    if delta_x_offset_init is None or delta_y_offset_init is None:
        special_init = {'delta_x_image': [0.0] * 4, 'delta_y_image': [0.0] * 4}
    else:
        special_init = {'delta_x_image': delta_x_offset_init, 'delta_y_image': delta_y_offset_init}
    special_sigma = {'delta_x_image': [astrometric_uncertainty] * 4, 'delta_y_image': [astrometric_uncertainty] * 4}
    special_lower = {'delta_x_image': [-5*astrometric_uncertainty] * 4,
                     'delta_y_image': [-5*astrometric_uncertainty] * 4}
    special_upper = {'delta_x_image': [5*astrometric_uncertainty] * 4,
                     'delta_y_image': [5*astrometric_uncertainty] * 4}
    special_fixed = [{}]
    kwargs_special = [special_init, special_sigma, special_fixed, special_lower, special_upper]

    kwargs_params = {'lens_model': lens_params,
                     'source_model': source_params,
                     'lens_light_model': lens_light_params,
                     'point_source_model': point_source_params,
                     'special': kwargs_special
                     }

    source_remove_fixed = []
    for i in range(0, len(source_model_list)):
        keys_remove_source = [key for key in source_params[0][i].keys() if key not in ['center_x', 'center_y', 'n_max']]
        remove_source = [i, keys_remove_source]
        source_remove_fixed.append(remove_source)
    lens_light_remove_fixed = []
    for i in range(0, len(lens_light_model_list)):
        keys_remove_lens_light = [key for key in lens_light_params[0][i].keys()]
        remove_light = [i, keys_remove_lens_light]
        lens_light_remove_fixed.append(remove_light)

    update_settings = {'lens_light_remove_fixed': lens_light_remove_fixed,
                       'source_remove_fixed': source_remove_fixed}

    if add_shapelets_source:
        n_run = 150
        n_iterations = 150
        update_settings['source_add_fixed'] = [
            [1, ['n_max', 'center_x', 'center_y'], [int(n_max_source), source_x, source_y]]]
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 50, 'threadCount': 1}],
                               ['update_settings', update_settings],
                               ['PSO', {'sigma_scale': 1., 'n_particles': 100, 'n_iterations': n_iterations,
                                 'threadCount': 1}],
                               ['MCMC',  {'n_burn': 0, 'n_run': n_run, 'walkerRatio': 4, 'sigma_scale': .1,
                                 'threadCount': 1}]
                               ]

    else:
        n_run = 150
        n_iterations = 100
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 50, 'threadCount': 1}],
                           ['update_settings', update_settings],
                           ['PSO', {'sigma_scale': 1., 'n_particles': 100, 'n_iterations': n_iterations,
                             'threadCount': 1}],
                           ['MCMC',
                            {'n_burn': 0, 'n_run': n_run, 'walkerRatio': 4, 'sigma_scale': .1, 'threadCount': 1}]
                           ]

    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model_fit,
                                  kwargs_constraints, kwargs_likelihood, kwargs_params)
    _ = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    lens_model_list_true, lens_redshift_list_true, kwargs_lens_true, _ = lens_system._get_lenstronomy_args()
    astropy_class = lens_system.astropy
    kwargs_model_true = {'lens_model_list': lens_model_list_true,
                         'lens_redshift_list': lens_redshift_list_true,
                         'z_source': lens_system.zsource,
                         'multi_plane': True,
                         'cosmo': astropy_class,
                         'source_light_model_list': source_model_list,
                         'lens_light_model_list': lens_light_model_list,
                         'point_source_model_list': point_source_list,
                         'additional_images_list': [False],
                         'fixed_magnification_list': [True]
                         }

    kwargs_result_true = deepcopy(kwargs_result)
    kwargs_result_true['kwargs_lens'] = kwargs_lens_true
    
    fitting_kwargs_class = FittingSequenceKwargs(kwargs_data_joint, kwargs_model_true, kwargs_constraints,
                                                 kwargs_likelihood, kwargs_params, kwargs_result_true)
    return fitting_seq, fitting_kwargs_class
