import numpy as np
import os
import sys
import dill
import pickle
from quadmodel.Solvers.light_fit_util import *
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from copy import deepcopy
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot
import matplotlib.pyplot as plt
from lenstronomy.Data.coord_transforms import Coordinates
from quadmodel.data.hst import HSTData, HSTDataModel
from scipy.interpolate import RegularGridInterpolator

def run_optimization(N_jobs, lens_data_name, filename_suffix, path_to_simulation_output, path_to_data, fitting_kwargs_list,
                     initialize_from_fit=False, path_to_smooth_lens_fit=None, add_shapelets_source=False,
                     n_max_source=None,plot_results=False, overwrite=False, random_seed=None,
                     npix_mask_images=0, run_index_list=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    if run_index_list is None:
        run_index_list = range(1, N_jobs+1)

    f = open(path_to_data + lens_data_name, 'rb')
    hst_data = pickle.load(f)
    f.close()

    for idx in run_index_list:

        print('fitting light to realization ' + str(idx) + ' ...')
        fname_chi2 = path_to_simulation_output + 'chi2_image_data' + filename_suffix + '_' + str(idx) + '.txt'

        try:
            f = open(path_to_simulation_output + 'simulation_output_' + str(idx), 'rb')
            simulation_output = dill.load(f)
            f.close()
        except:
            print(
                'could not find simulation output file ' + path_to_simulation_output + 'simulation_output_' + str(idx))
            continue

        if os.path.exists(fname_chi2):
            if overwrite is False:
                print('logL computation already performed for file ' + str(idx))
                continue

        fitting_seq, fitting_kwargs_class = _run_single(fitting_kwargs_list, hst_data, simulation_output, initialize_from_fit,
                path_to_smooth_lens_fit, add_shapelets_source, n_max_source, npix_mask_images)
        # modelPlot = ModelPlot(multi_band_list, kwargs_model,
        #                       kwargs_result, arrow_size=0.02, cmap_string="gist_heat")
        kwargs_best = fitting_seq.best_fit()
        neff = fitting_seq.likelihoodModule.effective_num_data_points(**kwargs_best)
        log_l = fitting_seq.best_fit_likelihood
        print('CHI2 FROM FIT: ', 2 * log_l / neff)

        f = open(path_to_simulation_output + 'kwargs_fitting_sequence_' + str(idx) + filename_suffix, 'wb')
        dill.dump(fitting_kwargs_class, f)
        f.close()
        np.savetxt(fname_chi2, X=np.atleast_1d(log_l))

        if plot_results:

            modelPlot = ModelPlot(**fitting_kwargs_class.kwargs_model_plot)
            chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
            for i in range(len(chain_list)):
                chain_plot.plot_chain_list(chain_list, i)

            f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

            modelPlot.data_plot(ax=axes[0, 0])
            modelPlot.model_plot(ax=axes[0, 1])
            modelPlot.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)
            modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100)
            modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
            modelPlot.magnification_plot(ax=axes[1, 2])
            f.tight_layout()
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
            plt.show()

            f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

            modelPlot.decomposition_plot(ax=axes[0, 0], text='Lens light', lens_light_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 0], text='Lens light convolved', lens_light_add=True)
            modelPlot.decomposition_plot(ax=axes[0, 1], text='Source light', source_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 1], text='Source light convolved', source_add=True)
            modelPlot.decomposition_plot(ax=axes[0, 2], text='All components', source_add=True, lens_light_add=True,
                                         unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 2], text='All components convolved', source_add=True,
                                         lens_light_add=True, point_source_add=True)
            f.tight_layout()
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
            plt.show()
            print('OKOKOKKOKOK')
            a=input('continue')


def _run_single(fitting_kwargs_list, hst_data, simulation_output, initialize_from_fit,
                path_to_smooth_lens_fit, add_shapelets_source, n_max_source, npix_mask_images):

    x_image, y_image = simulation_output.data.x, simulation_output.data.y
    # x_image_data, y_image_data = hst_data.arcsec_coordinates
    # x_image_data, y_image_data = x_image, y_image
    fluxes = simulation_output.data.m
    lens_system = simulation_output.lens_system
    lensmodel, kwargs_lens_init = lens_system.get_lensmodel()
    source_x, source_y = lensmodel.ray_shooting(x_image, y_image, kwargs_lens_init)
    source_x = np.mean(source_x)
    source_y = np.mean(source_y)
    print(source_x, source_y)

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
            {'amp': 1, 'R_sersic': 2.7640786091513947,
             'n_sersic': 9.997486183214777, 'e1': -0.09586436369120921, 'e2': 0.08652509597040224,
             'center_x': source_x, 'center_y': source_x}]
        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 1, 'R_sersic': 0.18033276090370226, 'n_sersic': 3.007623615897361,
             'e1': 0.02475164426715629, 'e2': 0.115023337548889,
             'center_x': 0.0, 'center_y': 0.0}]

    kwargs_source_sigma, kwargs_lower_source, kwargs_upper_source, kwargs_fixed_source = \
        source_params_sersic_ellipse(source_x, source_y, kwargs_source_init)
    kwargs_lens_light_sigma, kwargs_lower_lens_light, kwargs_upper_lens_light, kwargs_fixed_lens_light = \
        lens_light_params_sersic_ellipse(kwargs_lens_light_init[0])
    kwargs_fixed_lens_light[0]['e1'] = 0.0001
    kwargs_fixed_lens_light[0]['e2'] = 0.0001
    kwargs_lens_sigma, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens = [{}], [{}], [{}], [{}]

    if add_shapelets_source:
        source_model_list += ['SHAPELETS']
        kwargs_source_sigma_shapelets, kwargs_lower_source_shapelets, \
        kwargs_upper_source_shapelets, kwargs_fixed_source_shapelets = source_params_shapelets(n_max_source, source_x,
                                                                                               source_y)
        kwargs_source_sigma += kwargs_source_sigma_shapelets
        kwargs_lower_source += kwargs_lower_source_shapelets
        kwargs_upper_source += kwargs_upper_source_shapelets
        kwargs_fixed_source += kwargs_fixed_source_shapelets

    point_source_list = ['UNLENSED']
    # point_source_list = None

    kwargs_ps_sigma, kwargs_ps_lower, kwargs_ps_upper, kwargs_ps_fixed = ps_params(x_image, y_image)
    kwargs_ps_init = [{'ra_image': x_image, 'dec_image': y_image}]
    # kwargs_ps_fixed = [{}]
    # kwargs_fixed_source = deepcopy(kwargs_source_init)
    # del kwargs_fixed_source[0]['amp']
    # kwargs_fixed_source = [{}]

    #     kwargs_fixed_source[0]['amp'] = 20000
    # del kwargs_fixed_source[0]['R_sersic']
    # del kwargs_fixed_source[0]['e1']
    # del kwargs_fixed_source[0]['e2']
    # print(kwargs_fixed_source)
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
                         'image_position_uncertainty': 0.075,
                         'prior_lens': prior_lens,
                         'prior_lens_light': prior_lens_light,
                         'image_likelihood_mask_list': [hst_data.likelihood_mask]
                         }
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]

    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    # print(source_x, source_y)
    # print(kwargs_fixed_source)
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_fixed_source, kwargs_lower_source,
                     kwargs_upper_source]
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_fixed_lens_light,
                         kwargs_lower_lens_light, kwargs_upper_lens_light]
    point_source_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_ps_lower, kwargs_ps_upper]

    special_init = {'delta_x_image': [0.0]*4, 'delta_y_image': [0.0]*4}
    special_sigma = {'delta_x_image': [0.05] * 4, 'delta_y_image': [0.05] * 4}
    special_lower = {'delta_x_image': [-0.25] * 4, 'delta_y_image': [-0.25] * 4}
    special_upper = {'delta_x_image': [0.25] * 4, 'delta_y_image': [0.25] * 4}
    special_fixed = [{}]
    kwargs_special = [special_init, special_sigma, special_fixed, special_lower, special_upper]
    kwargs_params = {'lens_model': lens_params,
                     'source_model': source_params,
                     'lens_light_model': lens_light_params,
                     'point_source_model': point_source_params,
                     'special': kwargs_special
                     }

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
    # update the likelihood mask with the one tht cuts out images and parts far from the arc
    # print('log_L before new mask: ', fitting_seq.best_fit_likelihood)
    # kwargs_likelihood['image_likelihood_mask_list'] = [hst_data.custom_mask]
    # fitting_seq.kwargs_likelhood = kwargs_likelihood
    # print('log_L after new mask: ', fitting_seq.best_fit_likelihood)
    # a=input('continue')
    fitting_kwargs_class = FittingSequenceKwargs(kwargs_data_joint, kwargs_model_true, kwargs_constraints,
                                                 kwargs_likelihood, kwargs_params, kwargs_result_true)
    return fitting_seq, fitting_kwargs_class


class FixedLensModel(object):

    def __init__(self, ra_coords, dec_coords, lens_model, kwargs_lens, super_sample_factor=10.0):

        nx_0 = int(np.sqrt(len(ra_coords.ravel())))
        ny_0 = int(np.sqrt(len(dec_coords.ravel())))
        nx = int(nx_0 * super_sample_factor)
        ny = int(ny_0 * super_sample_factor)
        _ra_coords = np.linspace(np.min(ra_coords), np.max(ra_coords), nx)
        _dec_coords = np.linspace(np.min(dec_coords), np.max(dec_coords), ny)
        ra_coords, dec_coords = np.meshgrid(_ra_coords, _dec_coords)

        alpha_x, alpha_y = lens_model.alpha(ra_coords.ravel(), dec_coords.ravel(), kwargs_lens)
        points = (ra_coords[0, :], dec_coords[:, 0])
        self._interp_x = RegularGridInterpolator(points, alpha_x.reshape(nx, ny), bounds_error=False, fill_value=None)
        self._interp_y = RegularGridInterpolator(points, alpha_y.reshape(nx, ny), bounds_error=False, fill_value=None)

    def __call__(self, x, y, *args, **kwargs):

        point = (y, x)
        alpha_x = self._interp_x(point)
        alpha_y = self._interp_y(point)

        if isinstance(x, float) or isinstance(x, int) and isinstance(y, float) or isinstance(y, int):
            alpha_x = float(alpha_x)
            alpha_y = float(alpha_y)
        else:
            alpha_x = np.squeeze(alpha_x)
            alpha_y = np.squeeze(alpha_y)

        return alpha_x, alpha_y

