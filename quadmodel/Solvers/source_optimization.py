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
                print('chi2 computation already performed for file ' + str(idx))
                continue

        fitting_seq, fitting_kwargs_class, modelPlot = _run_single(fitting_kwargs_list, hst_data, simulation_output, initialize_from_fit,
                path_to_smooth_lens_fit, add_shapelets_source, n_max_source, npix_mask_images)
        # modelPlot = ModelPlot(multi_band_list, kwargs_model,
        #                       kwargs_result, arrow_size=0.02, cmap_string="gist_heat")

        num_data = fitting_seq.likelihoodModule.num_data
        num_param_nonlinear = fitting_seq.param_class.num_param()[0]
        num_param_linear = fitting_seq.param_class.num_param_linear()
        num_param = num_param_nonlinear + num_param_linear
        print(num_param, num_param_linear, num_param_nonlinear, num_data)
        log_l = fitting_seq.best_fit_likelihood

        chi2_array = np.array([2 * log_l / num_data, num_data])
        print('CHI2 FROM FIT: ', 2 * log_l / num_data)

        f = open(path_to_simulation_output + 'kwargs_fitting_sequence_' + str(idx) + filename_suffix, 'wb')
        dill.dump(fitting_kwargs_class, f)
        f.close()
        np.savetxt(fname_chi2, X=chi2_array)

        if plot_results:
            print('chi2: ', chi2_array[0])
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
            # f.tight_layout()
            # f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
            # plt.show()

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


def _run_single(fitting_kwargs_list, hst_data, simulation_output, initialize_from_fit,
                path_to_smooth_lens_fit, add_shapelets_source, n_max_source, npix_mask_images):

    x_image, y_image = simulation_output.data.x, simulation_output.data.y
    fluxes = simulation_output.data.m
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
    lens_model_list = ['TABULATED_DEFLECTIONS']

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
            {'amp': 1.0, 'center_x': source_x, 'center_y': source_y, 'e1': 0.1, 'e2': 0.1, 'R_sersic': 0.1,
             'n_sersic': 4}]
        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 1.0, 'center_x': source_x, 'center_y': source_y, 'e1': 0.1, 'e2': 0.1, 'R_sersic': 0.5,
             'n_sersic': 4}]

    kwargs_source_sigma, kwargs_lower_source, kwargs_upper_source, kwargs_fixed_source = \
        source_params_sersic_ellipse(source_x, source_y, kwargs_source_init)
    kwargs_lens_light_sigma, kwargs_lower_lens_light, kwargs_upper_lens_light, kwargs_fixed_lens_light = \
        lens_light_params_sersic_ellipse(kwargs_lens_light_init[0])

    kwargs_lens_sigma, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens = [{}], [{}], [{}], [{}]
    # kwargs_lens_sigma, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens = lensmodel_params(lens_model_list,
    #                                                                                               kwargs_lens_init)

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
    kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                  'point_amp': np.array(fluxes) * 100}]
    # kwargs_ps = {}
    kwargs_ps_sigma, kwargs_ps_lower, kwargs_ps_upper, kwargs_ps_fixed = ps_params(x_image, y_image)

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
        raise Exception('REALISTIC PSF NOT YET IMPLEMENTED')

    kwargs_model = {'lens_model_list': lens_model_list,
                    'source_light_model_list': source_model_list,
                    'lens_light_model_list': lens_light_model_list,
                    'point_source_model_list': point_source_list,
                    'additional_images_list': [False],
                    'fixed_magnification_list': [True],
                    'tabulated_deflection_angles': tabulated_lens_model
                    # list of bools (same length as point_source_type_list). If True, magnification ratio of point sources is fixed to the one given by the lens model
                   # 'fixed_lens_model': True
                    }
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    kwargs_constraints = {
        'joint_source_with_point_source': [[0, 0]],
        'num_point_source_list': [4],
        #'solver_type': 'PROFILE_SHEAR'
    }
    ############################### OPTIONAL PRIORS ############################
    prior_lens = None
    prior_lens_light = None

    ############################### OPTIONAL LIKELIHOOD MASK OVER IMAGES ############################
    coordinate_system = Coordinates(hst_data.transform_pix2angle, hst_data.ra_at_xy_0, hst_data.dec_at_xy_0)
    likelihood_mask = mask_images(x_image, y_image, npix_mask_images, hst_data.likelihood_mask, coordinate_system)
    # plt.imshow(likelihood_mask, origin='lower')
    # plt.show()
    kwargs_likelihood = {'check_bounds': True,
                         'force_no_add_image': True,
                         'source_marg': False,
                         'image_position_uncertainty': 0.005,
                         'check_matched_source_position': False,
                         # 'source_position_tolerance': 0.001,
                         # 'source_position_sigma': 0.001,
                         'prior_lens': prior_lens,
                         'prior_lens_light': prior_lens_light,
                         'image_likelihood_mask_list': [likelihood_mask]
                         }
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]

    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_fixed_source, kwargs_lower_source,
                     kwargs_upper_source]
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_fixed_lens_light,
                         kwargs_lower_lens_light, kwargs_upper_lens_light]
    point_source_params = [kwargs_ps, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_ps_lower, kwargs_ps_upper]

    kwargs_params = {'lens_model': lens_params,
                     'source_model': source_params,
                     'lens_light_model': lens_light_params,
                     'point_source_model': point_source_params
                     }

    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model,
                                  kwargs_constraints, kwargs_likelihood, kwargs_params)
    _ = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    fitting_kwargs_class = FittingSequenceKwargs(kwargs_data_joint, kwargs_model, kwargs_constraints,
                                           kwargs_likelihood, kwargs_params, kwargs_result)
    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02,
                          cmap_string="gist_heat")
    return fitting_seq, fitting_kwargs_class, modelPlot

class FixedLensModel(object):

    def __init__(self, ra_coords, dec_coords, lens_model, kwargs_lens):

        nx = int(np.sqrt(len(ra_coords.ravel())))
        ny = int(np.sqrt(len(dec_coords.ravel())))
        alpha_x, alpha_y = lens_model.alpha(ra_coords.ravel(), dec_coords.ravel(), kwargs_lens)
        points = (ra_coords[0,:], dec_coords[:,0])
        self._interp_x = RegularGridInterpolator(points, alpha_x.reshape(nx, ny))
        self._interp_y = RegularGridInterpolator(points, alpha_y.reshape(nx, ny))

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

