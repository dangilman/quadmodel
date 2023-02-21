import numpy as np
import os
import sys
import pickle
from quadmodel.Solvers.light_fit_util import *
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from copy import deepcopy
from lenstronomy.Plots.model_plot import ModelPlot
import matplotlib.pyplot as plt
from lenstronomy.Data.coord_transforms import Coordinates
from quadmodel.data.hst import HSTData, HSTDataModel


def run_optimization(N_jobs, lens_data_name, filename_suffix, path_to_simulation_output, path_to_data, fitting_kwargs_list,
                     initialize_from_fit=False, path_to_smooth_lens_fit=None, add_shapelets_source=False,
                     n_max_source=None,plot_results=False, save_model_plot=True, overwrite=False):

    chi2_array = None
    fname_chi2 = path_to_simulation_output + 'chi2_image_data' + filename_suffix + '.txt'
    for idx in range(1, N_jobs+1):

        try:
            f = open(path_to_simulation_output + 'simulation_output_'+str(idx), 'rb')
            simulation_output = pickle.load(f)
            f.close()
        except:
            print('could not find simulation output file '+path_to_simulation_output + 'simulation_output_'+str(idx))
            continue

        if os.path.exists(fname_chi2):
            if overwrite is False:
                print('chi2 computation already performed for file '+str(idx))
                continue

        f = open(path_to_data+lens_data_name, 'rb')
        hst_data = pickle.load(f)
        f.close()

        x_image, y_image = simulation_output.data.x, simulation_output.data.y
        fluxes = simulation_output.data.m
        lens_system = simulation_output.lens_system
        lensmodel, kwargs_lens_init = lens_system.get_lensmodel()
        source_x, source_y = lensmodel.ray_shooting(x_image, y_image, kwargs_lens_init)
        source_x = np.mean(source_x)
        source_y = np.mean(source_y)
        lensmodel_macro, kwargs_macro = lens_system.get_lensmodel(include_substructure=False)
        n_macro = len(kwargs_macro)
        lens_model_list_macro = lensmodel_macro.lens_model_list
        lens_model_list_halos = lensmodel.lens_model_list[n_macro:]

        if initialize_from_fit:
            f = open(path_to_smooth_lens_fit,'rb')
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
            kwargs_source_init = [{'amp': 1.0, 'center_x': source_x, 'center_y': source_y, 'e1': 0.1, 'e2': 0.1, 'R_sersic': 0.5, 'n_sersic': 4}]
            lens_light_model_list = ['SERSIC_ELLIPSE']
            kwargs_lens_light_init = [
                {'amp': 1.0, 'center_x': source_x, 'center_y': source_y, 'e1': 0.1, 'e2': 0.1, 'R_sersic': 0.5,
                 'n_sersic': 4}]

        kwargs_source_sigma, kwargs_lower_source, kwargs_upper_source, kwargs_fixed_source = \
            source_params_sersic_ellipse(source_x, source_y)
        kwargs_lens_light_sigma, kwargs_lower_lens_light, kwargs_upper_lens_light, kwargs_fixed_lens_light = \
            lens_light_params_sersic_ellipse(kwargs_lens_light_init[0])
        lens_model_list = lens_model_list_macro + lens_model_list_halos
        kwargs_lens_sigma, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens = lensmodel_params(lens_model_list,
                                                                                                      kwargs_lens_init)
        if add_shapelets_source:
            source_model_list += ['SHAPELETS']
            kwargs_source_sigma_shapelets, kwargs_lower_source_shapelets, \
            kwargs_upper_source_shapelets, kwargs_fixed_source_shapelets = source_params_shapelets(n_max_source, source_x, source_y)
            kwargs_source_sigma += kwargs_source_sigma_shapelets
            kwargs_lower_source += kwargs_lower_source_shapelets
            kwargs_upper_source += kwargs_upper_source_shapelets
            kwargs_fixed_source += kwargs_fixed_source_shapelets

        point_source_list = ['LENSED_POSITION']
        kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                      'point_amp': np.array(fluxes) * 100}]
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
                        # list of bools (same length as point_source_type_list). If True, magnification ratio of point sources is fixed to the one given by the lens model
                        'fixed_lens_model': True
                        }
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

        kwargs_constraints = {'joint_source_with_point_source': [[0, 0]],
                              'num_point_source_list': [4],
                              'solver_type': 'PROFILE_SHEAR'
                              }
        ############################### OPTIONAL PRIORS ############################
        prior_lens = None
        prior_lens_light = [[0, 'center_x', 0.0, 0.2], [0, 'center_y', 0.0, 0.2]]

        ############################### OPTIONAL LIKELIHOOD MASK OVER IMAGES ############################
        npix_image_mask = 2
        coordinate_system = Coordinates(hst_data.transform_pix2angle, hst_data.ra_at_xy_0, hst_data.dec_at_xy_0)
        likelihood_mask = mask_images(x_image, y_image, npix_image_mask, hst_data.likelihood_mask, coordinate_system)
        # plt.imshow(likelihood_mask, origin='lower')
        # plt.show()
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': True,
                             'source_marg': False,
                             'image_position_uncertainty': 0.005,
                             'check_matched_source_position': False,
                             #'source_position_tolerance': 0.001,
                             #'source_position_sigma': 0.001,
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
                         'point_source_model': point_source_params}

        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model,
                                      kwargs_constraints, kwargs_likelihood, kwargs_params)

        _ = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()

        modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat")

        log_l = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None,
                                                                  **kwargs_result)
        n_data = modelPlot._imageModel.num_data_evaluate
        out = np.array([2*log_l/n_data, n_data])

        if save_model_plot:
            f = open(path_to_simulation_output + 'modelplot_'+ str(idx) + filename_suffix, 'wb')
            pickle.dump(modelPlot, f)
            f.close()

        if chi2_array is None:
            chi2_array = out
        else:
            chi2_array = np.vstack((chi2_array, out))

        if plot_results:
            modelPlot.plot_main()
            plt.show()
            print(kwargs_result)
            a=input('continue')

    if chi2_array is not None:
        np.savetxt(fname_chi2, X=chi2_array)

