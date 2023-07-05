import numpy as np
from quadmodel.Solvers.light_fit_util import FixedLensModel, FixedLensModelNew, FittingSequenceKwargs
from copy import deepcopy
from lenstronomy.Workflow.fitting_sequence import FittingSequence


def fit_wfi2033_light(hst_data, simulation_output, astrometric_uncertainty,
                           delta_x_offset_init, delta_y_offset_init, add_shapelets_source, n_max=6,
                      super_sample_factor=1):

    x_image, y_image = simulation_output.data.x, simulation_output.data.y
    lens_system = simulation_output.lens_system
    lensmodel, kwargs_lens_true = lens_system.get_lensmodel()
    source_x, source_y = lensmodel.ray_shooting(x_image, y_image, kwargs_lens_true)
    source_x = np.mean(source_x)
    source_y = np.mean(source_y)

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
    kwargs_source_init = [{'amp': 2.0, 'center_x': source_x, 'center_y': source_y, 'R_sersic': 0.6,
                              'n_sersic': 5.5, 'e1': 0.25, 'e2': -0.2}]
    kwargs_sigma_source = [{'amp': 100, 'R_sersic': 0.1, 'n_sersic': 1.0, 'e1': 0.25,
                            'e2': 0.25, 'center_x': 0.1, 'center_y': 0.1}]
    kwargs_lower_source = [
        {'amp': 1e-9, 'R_sersic': 0.0001, 'n_sersic': 1.0, 'e1': -0.4, 'e2': -0.4, 'center_x': -10, 'center_y': -10},
        ]
    kwargs_upper_source = [
        {'amp': 1e9, 'R_sersic': 100.0, 'n_sersic': 10.0, 'e1': 0.4, 'e2': 0.4, 'center_x': 10, 'center_y': 10},
        ]

    lens_light_model_list = ['SERSIC_ELLIPSE',
                             'SERSIC',
                             'SERSIC'
                             ]
    sat_x1, sat_y1 = 0.245, 2.037
    sat_x2, sat_y2 = -4.0, -0.08
    kwargs_lens_light_init = [{'amp': 5.834852894299137, 'R_sersic': 1.620882895620721, 'n_sersic': 4.256740262678284, 'e1': -0.07154631229864938, 'e2': 0.11002873485221977, 'center_x': -0.04917802051950834, 'center_y': -0.050025007699314866},
                              {'amp': 5942.936336278121, 'R_sersic': 0.010776911321736414, 'n_sersic': 3.49354122654554, 'center_x': 0.20594761858575142, 'center_y': 1.9817344304057074},
                              {'amp': 4.101586074661848, 'R_sersic': 1.7614442529859455, 'n_sersic': 4.045773694914706, 'center_x': -4.139544129164365, 'center_y': 0.0910019935500852}]
    kwargs_lens_light_sigma = [{'amp': 20, 'R_sersic': 0.5, 'n_sersic': 1.0, 'e1': 0.25,
                                'e2': 0.25, 'center_x': 0.2, 'center_y': 0.2},
                               {'amp': 2000, 'R_sersic': 0.05, 'n_sersic': 1.0, 'center_x': 0.05, 'center_y': 0.05},
                               {'amp': 20, 'R_sersic': 0.2, 'n_sersic': 1.0, 'center_x': 0.05, 'center_y': 0.05}]
    kwargs_lower_lens_light = [
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10},
        {'R_sersic': 0.0001, 'n_sersic': 0.5, 'center_x': sat_x1 - 0.5, 'center_y': sat_y1 - 0.5},
        {'R_sersic': 0.0001, 'n_sersic': 0.5, 'center_x': sat_x2 - 0.5, 'center_y': sat_y2 - 0.5}]
    kwargs_upper_lens_light = [{'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
                               {'R_sersic': 10.0, 'n_sersic': 10.0, 'center_x': sat_x1 + 0.5, 'center_y': sat_y1 + 0.5},
                               {'R_sersic': 10.0, 'n_sersic': 10.0, 'center_x': sat_x2 + 0.5, 'center_y': sat_y2 + 0.5}]
    kwargs_lens_init, kwargs_lens_sigma, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens = [{}], [{}], [{}], [
        {}], [{}]

    if add_shapelets_source:
        source_model_list += ['SHAPELETS']
        shapelets_init = {'amp': 1.0, 'beta': 1e-7, 'n_max': 1, 'center_x': source_x, 'center_y': source_y}
        shapelets_sigma = {'amp': 0.5, 'beta': 0.2, 'n_max': 1.0, 'center_x': 0.1, 'center_y': 0.1}
        shapelets_min = {'amp': 0.00001, 'beta': 1e-16, 'n_max': 1.0, 'center_x': -1.0, 'center_y': -1.0}
        shapelets_max = {'amp': 1000.0, 'beta': 100.0, 'n_max': 10.0, 'center_x': 1.0, 'center_y': 1.0}
        kwargs_source_init += [shapelets_init]
        kwargs_sigma_source += [shapelets_sigma]
        kwargs_lower_source += [shapelets_min]
        kwargs_upper_source += [shapelets_max]

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
                  'kernel_point_source': np.absolute(hst_data.psf_estimate),
                  'psf_error_map': np.absolute(hst_data.psf_error_map)}
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
    prior_lens_light = [[1, 'center_x', sat_x1, 0.05], [1, 'center_y', sat_y1, 0.05],
                        [2, 'center_x', sat_x2, 0.2], [2, 'center_y', sat_y2, 0.2]]

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
    n_run = 10
    n_iterations = 10
    if add_shapelets_source:
        n_run = 250
        n_iterations = 150
        update_settings['source_add_fixed'] = [[1, ['n_max', 'center_x', 'center_y'], [int(n_max), source_x, source_y]]]

    nthreads = 1
    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 20, 'n_iterations': 20, 'threadCount': nthreads}],
                           ['update_settings', update_settings],
                           ['PSO',
                            {'sigma_scale': 1., 'n_particles': 100, 'n_iterations': n_iterations, 'threadCount': nthreads}],
                           ['psf_iteration', {'psf_symmetry': hst_data.psf_symmetry, 'keep_psf_error_map': True}],
                           ['MCMC',
                            {'n_burn': 0, 'n_run': n_run, 'walkerRatio': 4, 'sigma_scale': .1, 'threadCount': nthreads}]
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
    fitting_kwargs_class = FittingSequenceKwargs(kwargs_data_joint, kwargs_model_true, kwargs_constraints,
                                                 kwargs_likelihood_compute_statistic, kwargs_params, kwargs_result_true)

    return fitting_seq, fitting_kwargs_class, chain_list
