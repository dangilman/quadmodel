import numpy as np
from quadmodel.Solvers.light_fit_util import FixedLensModelNew, FittingSequenceKwargs
from copy import deepcopy
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Workflow.fitting_sequence import FittingSequence


def fit_pg1115_light(hst_data, simulation_output, astrometric_uncertainty, delta_x_offset_init,
                     delta_y_offset_init, add_shapelets_source=False, n_max_source=None,
                     super_sample_factor=1):

    x_image, y_image = simulation_output.data.x, simulation_output.data.y
    lens_system = simulation_output.lens_system
    lensmodel, kwargs_lens_true = lens_system.get_lensmodel()
    source_x, source_y = lensmodel.ray_shooting(x_image, y_image, kwargs_lens_true)
    source_x = np.mean(source_x)
    source_y = np.mean(source_y)

    for i, lens_model_name in enumerate(lensmodel.lens_model_list):
        if lens_model_name in ['EPL', 'SHEAR', 'MULTIPOLE', 'SIS']:
            print('kwargs ' + lens_model_name + ': ', kwargs_lens_true[i])
        else:
            break

    print('super sampling: ', super_sample_factor)
    ra_at_x0 = hst_data.ra_at_xy_0
    dec_at_x0 = hst_data.dec_at_xy_0
    pix2angle = hst_data.transform_pix2angle
    (nx, ny) = hst_data.image_data.shape
    # coordinate_system = Coordinates(pix2angle, ra_at_x0, dec_at_x0)
    # ra_coords, dec_coords = coordinate_system.coordinate_grid(nx, ny)
    # tabulated_lens_model = FixedLensModel(ra_coords, dec_coords, lensmodel, kwargs_lens_true)
    tabulated_lens_model = FixedLensModelNew(nx, ny, pix2angle, ra_at_x0, dec_at_x0,
                                             lensmodel, kwargs_lens_true, super_sample_factor)
    lens_model_list_fit = ['TABULATED_DEFLECTIONS']

    source_model_list = ['SERSIC_ELLIPSE']

    kwargs_source_init_sersic_ellipse = {'amp': 17.594226315853195, 'R_sersic': 0.12136390483907997,
                                         'n_sersic': 4.7042467732006275, 'e1': -0.05771360473680025,
                                         'e2': -0.06021057359113569, 'center_x': source_x,
                                         'center_y': source_y}
    # kwargs_source_init_sersic = {'amp': 10000, 'R_sersic': 0.1,
    #      'n_sersic': 5.0, 'center_x': source_x, 'center_y': source_y}
    kwargs_source_init = [kwargs_source_init_sersic_ellipse,
                          # kwargs_source_init_sersic
                          ]
    kwargs_sigma_source = [{'amp': 100, 'R_sersic': 0.1, 'n_sersic': 1.0, 'e1': 0.25,
                            'e2': 0.25, 'center_x': 0.1, 'center_y': 0.1},
                           # {'amp': 10000, 'R_sersic': 0.1, 'n_sersic': 2.0, 'center_x': 0.1, 'center_y': 0.1}
                           ]
    kwargs_lower_source = [
        {'amp': 1e-9, 'R_sersic': 0.001, 'n_sersic': 1.0, 'e1': -0.4, 'e2': -0.4, 'center_x': -10, 'center_y': -10},
    ]
    kwargs_upper_source = [
        {'amp': 1e9, 'R_sersic': 100.0, 'n_sersic': 10.0, 'e1': 0.4, 'e2': 0.4, 'center_x': 10, 'center_y': 10},
    ]

    lens_light_model_list = ['SERSIC_ELLIPSE']
    kwargs_lens_light_init = [
        {'amp': 24.4, 'R_sersic': 0.52, 'n_sersic': 4.24, 'e1': -0.0, 'e2': -0.0088, 'center_x': 0.0, 'center_y': 0.0}]
    kwargs_lens_light_sigma = [{'amp': 200, 'R_sersic': 0.2, 'n_sersic': 1.0, 'e1': 0.25,
                                'e2': 0.25, 'center_x': 0.1, 'center_y': 0.1}]
    kwargs_lower_lens_light = [
        {'amp': 1e-9, 'R_sersic': 0.001, 'n_sersic': 1.0, 'e1': -0.4, 'e2': -0.4, 'center_x': -10, 'center_y': -10}]
    kwargs_upper_lens_light = [
        {'amp': 1e9, 'R_sersic': 2.5, 'n_sersic': 10.0, 'e1': 0.4, 'e2': 0.4, 'center_x': 10, 'center_y': 10}]
    kwargs_lens_init, kwargs_lens_sigma, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens = [{}], [{}], [{}], [
        {}], [{}]

    if add_shapelets_source:
        source_model_list += ['SHAPELETS']
        kwargs_source_init += [{'amp': 0.0, 'beta': 1e-2, 'n_max': n_max_source, 'center_x': source_x, 'center_y': source_y}]
        kwargs_sigma_source += [{'amp': 1.0, 'beta': 0.1, 'n_max': 1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_source += [{'amp': 1e-9, 'beta': 1e-9, 'n_max': 1, 'center_x': -0.5, 'center_y': -0.5}]
        kwargs_upper_source += [{'amp': 1e9, 'beta': 1e9, 'n_max': 30, 'center_x': 0.5, 'center_y': 0.5}]

    kwargs_fixed_source = deepcopy(kwargs_source_init)
    kwargs_fixed_lens_light = deepcopy(kwargs_lens_light_init)

    point_source_list = ['UNLENSED']
    kwargs_ps_sigma = [{'ra_image': [0.01] * len(x_image), 'dec_image': [0.01] * len(y_image)}]
    kwargs_ps_lower = [{'ra_image': x_image - 0.1, 'dec_image': y_image - 0.1}]
    kwargs_ps_upper = [{'ra_image': x_image + 0.1, 'dec_image': y_image + 0.1}]
    kwargs_ps_fixed = [{'ra_image': x_image, 'dec_image': y_image}]
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
    kwargs_psf = {'psf_type': 'PIXEL',
                  'kernel_point_source': hst_data.psf_estimate,
                  'psf_error_map': hst_data.psf_error_map}

    kwargs_model_fit = {'lens_model_list': lens_model_list_fit,
                        'source_light_model_list': source_model_list,
                        'lens_light_model_list': lens_light_model_list,
                        'point_source_model_list': point_source_list,
                        'additional_images_list': [False],
                        'fixed_magnification_list': [True],
                        'tabulated_deflection_angles': tabulated_lens_model}

    kwargs_numerics = {'supersampling_factor': super_sample_factor, 'supersampling_convolution': False}
    kwargs_constraints = {
        'num_point_source_list': [4],
        'point_source_offset': True
    }
    ############################### OPTIONAL PRIORS ############################
    prior_lens = None
    prior_lens_light = [[0, 'e1', 0.0, 0.25], [0, 'e2', 0.0, 0.25]]

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
    kwargs_likelihood_compute_statistic = deepcopy(kwargs_likelihood)
    kwargs_likelihood_compute_statistic['image_likelihood_mask_list'] = [hst_data.likelihood_mask]

    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_sigma_source, kwargs_fixed_source, kwargs_lower_source,
                     kwargs_upper_source]
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_fixed_lens_light,
                         kwargs_lower_lens_light, kwargs_upper_lens_light]
    point_source_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_ps_lower, kwargs_ps_upper]

    if delta_x_offset_init is None or delta_y_offset_init is None:
        special_init = {'delta_x_image': [0.0] * 4, 'delta_y_image': [0.0] * 4}
    else:
        special_init = {'delta_x_image': delta_x_offset_init, 'delta_y_image': delta_y_offset_init}
    special_sigma = {'delta_x_image': [astrometric_uncertainty] * 4, 'delta_y_image': [astrometric_uncertainty] * 4}
    special_lower = {'delta_x_image': [-5 * astrometric_uncertainty] * 4,
                     'delta_y_image': [-5 * astrometric_uncertainty] * 4}
    special_upper = {'delta_x_image': [5 * astrometric_uncertainty] * 4,
                     'delta_y_image': [5 * astrometric_uncertainty] * 4}
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
    n_run = 100
    n_iterations = 50
    if add_shapelets_source:
        n_run = 150
        n_iterations = 100

    fitting_kwargs_list = [
            ['PSO', {'sigma_scale': 1.0, 'n_particles': 50, 'n_iterations': n_iterations}],
            ['update_settings', update_settings],
            ['PSO', {'sigma_scale': 1.0, 'n_particles': 100, 'n_iterations': n_iterations}],
            ['psf_iteration', {'psf_symmetry': hst_data.psf_symmetry, 'keep_psf_error_map': True}],
            ['MCMC', {'n_burn': 0, 'n_run': n_run, 'walkerRatio': 4, 'sigma_scale': 0.1, 'threadCount': 1}]
        ]

    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model_fit,
                                  kwargs_constraints, kwargs_likelihood, kwargs_params)
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    astropy_class = lens_system.astropy
    kwargs_model_true = {'lens_model_list': lensmodel.lens_model_list,
                         'lens_redshift_list': lensmodel.redshift_list,
                         'z_source': lensmodel.z_source,
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
    #     # update the likelihood mask with the one tht cuts out images and parts far from the arc
    #     print('log_L before new mask: ', fitting_seq.best_fit_likelihood)
    #     kwargs_likelihood['image_likelihood_mask_list'] = [hst_data.custom_mask]
    #     fitting_seq.kwargs_likelhood = kwargs_likelihood
    #     print('log_L after new mask: ', fitting_seq.best_fit_likelihood)
    #     a=input('continue')

    fitting_kwargs_class = FittingSequenceKwargs(kwargs_data_joint, kwargs_model_true, kwargs_constraints,
                                                 kwargs_likelihood_compute_statistic, kwargs_params, kwargs_result_true)

    return fitting_seq, fitting_kwargs_class, chain_list
