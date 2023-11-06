from copy import deepcopy

import numpy as np
from quadmodel.Solvers.hierachical import HierarchicalOptimization
from quadmodel.Solvers.brute import BruteOptimization
from quadmodel.Solvers.multiplane_decoupled import DecoupledMultiPlane
from quadmodel.inference.realization_setup import setup_realization
from quadmodel.macromodel import MacroLensModel
from quadmodel.quad_model import QuadLensSystem
from lenstronomy.LensModel.Util.decouple_multi_plane_util import setup_raytracing_lensmodels, setup_grids,\
    coordinates_and_deflections, class_setup, setup_lens_model
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution
from lenstronomy.LensModel.lens_model import LensModel
from time import time

def _evaluate_model(lens_data_class, kwargs_sample_realization, kwargs_realization_other,
                    kwargs_sample_macromodel, ray_tracing_optimization, test_mode, verbose, crit_curves_in_test_mode,
                    importance_weights_function, realization_class,
                    shift_background_realization, subtract_exact_mass_sheets, log_mlow_mass_sheet,
                    rescale_grid_size, rescale_grid_resolution, index_lens_split, seed):

    if seed is not None:
        np.random.seed(seed)

    # add astrometric uncertainties to image positions
    magnifications, magnification_uncertainties, astrometric_uncertainty = \
        lens_data_class.m, lens_data_class.delta_m, \
        lens_data_class.delta_xy
    magnifications = np.array(magnifications)
    _flux_ratios_data = magnifications[1:] / magnifications[0]

    delta_x, delta_y = np.random.normal(0.0, astrometric_uncertainty, 4), np.random.normal(0.0, astrometric_uncertainty, 4)
    lens_data_class_sampling = deepcopy(lens_data_class)
    lens_data_class_sampling.x += delta_x
    lens_data_class_sampling.y += delta_y

    # get the lens redshift, for some deflectors with photometrically-estimated redshifts, we have to sample a PDF
    lens_data_class_sampling.set_zlens(reset=True)
    zlens = lens_data_class_sampling.zlens
    zsource = lens_data_class.zsource

    source_size_pc, kwargs_source_model, source_samples, param_names_source, realization_samples, \
    preset_model, kwargs_preset_model, param_names_realization, model, constrain_params_macro, \
    optimization_routine, macromodel_samples, param_names_macro, importance_weight = _parameters_from_priors(lens_data_class_sampling,
                                                                                          kwargs_sample_realization,
                                                                                          kwargs_realization_other,
                                                                                          kwargs_sample_macromodel,
                                                                                          importance_weights_function,
                                                                                          verbose)
    macromodel = MacroLensModel(model.component_list)
    R_ein_approx = lens_data_class.approx_einstein_radius
    if realization_class is None:
        # create the realization
        if 'cone_opening_angle_arcsec' not in kwargs_preset_model.keys():
            # we set the cone opening angle to 6 times the Einstein radius to get all the halos near images
            kwargs_preset_model['cone_opening_angle_arcsec'] = 6 * R_ein_approx
        realization = preset_model(zlens, zsource, **kwargs_preset_model)
    else:
        realization_samples = np.array([])
        param_names_realization = []
        realization = deepcopy(realization_class)
        if verbose: print('using fixed realization instance')

    if shift_background_realization is False:
        realization._has_been_shifted = True
    if verbose:
        print('realization contains ' + str(len(realization.halos)) + ' halos.')
        print(param_names_realization)
        print('realization hyper-parameters: ', realization_samples)
        print(param_names_source)
        print('source/lens parameters: ', source_samples)
        print(param_names_macro)
        print('macromodel samples: ', macromodel_samples)
        print('\n')
        print('keyword arguments for realization: ')
        print('preset model function: ', preset_model)
        print('kwargs preset model: ', kwargs_preset_model)

    # This sets up a baseline lens macromodel and aligns the dark matter halos to follow the path taken by the
    # light rays. This is important if the source is significantly offset from the lens centroid
    lens_system = QuadLensSystem.shift_background_auto(lens_data_class_sampling,
                                                       macromodel, zsource,
                                                       realization)

    # Now we set up the optimization routine, which will solve for a set of macromodel parameters that map the
    # observed image coordinates to common source position in the presence of all the dark matter halos along the
    # line of sight and in the main lens plane.
    kwargs_multiplane_model = None
    if ray_tracing_optimization == 'IMAGE_DATA_FIT':
        raise Exception('not yet implemented')

    elif ray_tracing_optimization == 'DECOUPLED_MULTI_PLANE':
        grid_rmax = 2*auto_raytracing_grid_size(source_size_pc) * rescale_grid_size
        grid_resolution = rescale_grid_resolution * auto_raytracing_grid_resolution(source_size_pc)
        lens_model_init, kwargs_lens_init = lens_system.get_lensmodel(
            log_mlow_mass_sheet=log_mlow_mass_sheet, subtract_exact_mass_sheets=subtract_exact_mass_sheets)
        optimizer = DecoupledMultiPlane(lens_system, lens_model_init, kwargs_lens_init, index_lens_split)
        kwargs_lens_final, lens_model_final, _ = optimizer.optimize(lens_data_class_sampling, constrain_params=constrain_params_macro,
                                                                    param_class_name=optimization_routine,
                                                                    verbose=verbose)
        kwargs_multiplane_model = lens_model_final.lens_model.kwargs_multiplane_model
        mags = lens_system.quasar_magnification(lens_data_class_sampling.x, lens_data_class_sampling.y,
                                                source_size_pc, lens_model_final, kwargs_lens_final,
                                                grid_rmax=grid_rmax, grid_resolution=grid_resolution,
                                                decoupled_multi_plane=True, index_lens_split=index_lens_split,
                                                lens_model_init=lens_model_init, kwargs_lens_init=kwargs_lens_init)

    elif ray_tracing_optimization == 'default' or ray_tracing_optimization == 'brute':
        grid_rmax = auto_raytracing_grid_size(source_size_pc) * rescale_grid_size
        grid_resolution = auto_raytracing_grid_resolution(source_size_pc) * rescale_grid_resolution
        if ray_tracing_optimization == 'default':
            optimizer = HierarchicalOptimization(lens_system)
        else:
            optimizer = BruteOptimization(lens_system)
        if log_mlow_mass_sheet is None:
            # set this to the value specified in the settings class unless it is explicitely set by the user
            log_mlow_mass_sheet = optimizer.settings.log_mlow_mass_sheet
        kwargs_lens_final, lens_model_full, _ = optimizer.optimize(lens_data_class_sampling,
                                                                   optimization_routine,
                                                                           constrain_params=constrain_params_macro,
                                                                           log_mlow_mass_sheet=log_mlow_mass_sheet,
                                                                           subtract_exact_mass_sheets=subtract_exact_mass_sheets,
                                                                           verbose=verbose)
        mags = lens_system.quasar_magnification(lens_data_class_sampling.x,
                                                    lens_data_class_sampling.y,
                                                source_size_pc, lens_model=lens_model_full,
                                                    kwargs_lensmodel=kwargs_lens_final, grid_axis_ratio=0.5,
                                                    grid_rmax=grid_rmax, normed=False,
                                                    grid_resolution=grid_resolution, **kwargs_source_model)
    else:
        raise Exception('ray tracing optimization '+str(ray_tracing_optimization)+' not recognized.')

    if verbose:
        print('magnifications: ', mags)

    if test_mode:
        import matplotlib.pyplot as plt

        _r = np.linspace(-2.0 * R_ein_approx, 2.0 * R_ein_approx, 200)
        xx, yy = np.meshgrid(_r, _r)
        shape0 = xx.shape
        if ray_tracing_optimization == 'DECOUPLED_MULTI_PLANE':
            lens_system.plot_images(lens_data_class_sampling.x, lens_data_class_sampling.y,
                                    source_size_pc, None, kwargs_lens_final, grid_rmax=grid_rmax,
                                    grid_resolution=grid_resolution, decoupled_multi_plane=True,
                                    index_lens_split=index_lens_split, lens_model_init=lens_model_init,
                                    kwargs_lens_init=kwargs_lens_init)
            plt.show()
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free, z_source, z_split, cosmo_bkg = \
                setup_lens_model(
                lens_model_init, kwargs_lens_init, index_lens_split)
            grid_x, grid_y, interp_points, npix = setup_grids(4 * R_ein_approx, 0.025,
                                                              0.0, 0.0)
            xD, yD, alpha_x_foreground, alpha_y_foreground, alpha_x_background, alpha_y_background = \
                coordinates_and_deflections(lens_model_fixed,
                                            lens_model_free,
                                            kwargs_lens_fixed,
                                            kwargs_lens_final,
                                            grid_x,
                                            grid_y,
                                            z_split,
                                            z_source,
                                            cosmo_bkg)
            kwargs_multiplane_lens_model_full_grid = class_setup(lens_model_free, xD, yD, alpha_x_foreground, alpha_y_foreground,
                                                       alpha_x_background, alpha_y_background, z_split,
                                                       coordinate_type='GRID', interp_points=interp_points)
            lens_model_hessian = LensModel(**kwargs_multiplane_lens_model_full_grid)
        else:
            lens_system.plot_images(lens_data_class_sampling.x, lens_data_class_sampling.y, source_size_pc,
                                    lens_model_full,
                                    kwargs_lens_final,
                                    grid_rmax=grid_rmax,
                                    grid_resolution=grid_resolution,
                                    **kwargs_source_model)
            plt.show()
            lens_model_hessian = lens_model_full
        fxx, fxy, fyx, fyy = lens_model_hessian.hessian(xx.ravel(), yy.ravel(), kwargs_lens_final)
        kappa = 0.5 * (fxx + fyy)
        det_A = (1 - fxx) * (1 - fyy) - fxy * fyx
        magnification_surface = 1. / det_A

        lensmodel_macro, kwargs_macro = lens_system.get_lensmodel(include_substructure=False)
        kappa_macro = lensmodel_macro.kappa(xx.ravel(), yy.ravel(), kwargs_macro).reshape(shape0)
        extent = [-2 * R_ein_approx, 2 * R_ein_approx, -2 * R_ein_approx, 2 * R_ein_approx]
        plt.imshow(kappa.reshape(shape0) - kappa_macro, origin='lower', vmin=-0.05, vmax=0.05, cmap='bwr', extent=extent)
        plt.scatter(lens_data_class_sampling.x, lens_data_class_sampling.y, color='k')
        if crit_curves_in_test_mode:
            from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
            ext = LensModelExtensions(lens_model_full)
            ra_crit_list, dec_crit_list, _, _ = ext.critical_curve_caustics(kwargs_lens_final,
                                                                            compute_window=4 * R_ein_approx,
                                                                            grid_scale=0.05)
            for i in range(0, len(ra_crit_list)):
                plt.plot(ra_crit_list[i], dec_crit_list[i], color='k', lw=2)

        plt.show()
        plt.imshow(magnification_surface.reshape(shape0), vmin=-10, vmax=10,
        origin='lower', extent=extent, cmap='gist_heat')
        plt.show()

    # Now we account for uncertainties in the image magnifications. These uncertainties are sometimes quoted for
    # individual image fluxes, or the flux ratios.
    if lens_data_class.uncertainty_in_magnifications:
        mags_with_uncertainties = []
        for j, mag in enumerate(mags):
            if magnification_uncertainties[j] is None:
                m = np.nan
            else:
                delta_m = np.random.normal(0.0, magnification_uncertainties[j] * mag)
                m = mag + delta_m
            mags_with_uncertainties.append(m)
        mags_with_uncertainties = np.array(mags_with_uncertainties)
        _flux_ratios = mags_with_uncertainties[1:] / mags_with_uncertainties[0]

    else:
        # If uncertainties are quoted for image flux ratios, we first compute the flux ratios, and then add
        # the uncertainties
        flux_ratios = mags[1:] / mags[0]
        fluxratios_with_uncertainties = []
        for k, fr in enumerate(flux_ratios):
            if magnification_uncertainties[k] is None:
                new_fr = np.nan
            else:
                df = np.random.normal(0, fr * magnification_uncertainties[k])
                new_fr = fr + df
            fluxratios_with_uncertainties.append(new_fr)
        _flux_ratios = np.array(fluxratios_with_uncertainties)

    flux_ratios_data = []
    flux_ratios = []
    for idx in lens_data_class_sampling.keep_flux_ratio_index:
        flux_ratios.append(_flux_ratios[idx])
        flux_ratios_data.append(_flux_ratios_data[idx])

    # Now we compute the summary statistic
    stat = 0
    for f_i_data, f_i_model in zip(flux_ratios_data, flux_ratios):
        stat += (f_i_data - f_i_model) ** 2
    stat = np.sqrt(stat)

    if verbose:
        print('flux ratios data: ', flux_ratios_data)
        print('flux ratios model: ', flux_ratios)
        print('statistic: ', stat)

    return stat, realization_samples, source_samples, macromodel_samples, param_names_realization, \
        param_names_source, param_names_macro, lens_system, lens_data_class_sampling, importance_weight, mags, \
           kwargs_multiplane_model


def _parameters_from_priors(lens_data_class_sampling, kwargs_sample_realization,
                            kwargs_realization_other, kwargs_sample_macromodel, importance_weights_function,
                            verbose):

    while True:
        u = np.random.rand()
        # Now, set up the source model
        source_size_pc, kwargs_source_model, source_samples, param_names_source = \
            lens_data_class_sampling.generate_sourcemodel()

        # parse the input dictionaries into arrays with parameters drawn from their respective priors
        realization_samples, preset_model, kwargs_preset_model, param_names_realization = setup_realization(
            kwargs_sample_realization,
            kwargs_realization_other,
            lens_data_class_sampling.x,
            lens_data_class_sampling.y,
            source_size_pc)

        model, constrain_params_macro, optimization_routine, \
        macromodel_samples, param_names_macro = lens_data_class_sampling.generate_macromodel(**kwargs_sample_macromodel)
        model_probability = importance_weights_function(realization_samples, param_names_realization,
                                                        macromodel_samples, param_names_macro,
                                                        source_samples, param_names_source,
                                                        verbose)

        if model_probability >= u:
            break

    importance_weight = 1.0 / model_probability
    if verbose:
        print('importance weight for sample: ', importance_weight)
        print('sample (from realization samples): ', realization_samples)

    return source_size_pc, kwargs_source_model, source_samples, param_names_source, realization_samples, \
           preset_model, kwargs_preset_model, param_names_realization, model, constrain_params_macro, \
           optimization_routine, macromodel_samples, param_names_macro, importance_weight
