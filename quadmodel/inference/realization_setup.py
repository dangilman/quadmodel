from pyHalo.preset_models import preset_model_from_name
import numpy as np
from copy import deepcopy
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size
from pyHalo.realization_extensions import RealizationExtensions

def _draw(prior, prior_type):
    if prior_type == 'UNIFORM':
        value = np.random.uniform(prior[1], prior[2])
    elif prior_type == 'GAUSSIAN':
        value = np.random.normal(prior[1], prior[2])
    elif prior_type == 'FIXED':
        value = prior[1]
    elif prior_type == 'DISCRETE':
        value = np.random.choice(prior[1])
    else:
        raise Exception('prior type ' + str(prior_type) + ' not recognized.')
    return value

def setup_macromodel(priors):

    macromodel_priors = deepcopy(priors)
    macro_params = np.array([])
    kwargs_hyper_macro = {}
    param_names_macro = []
    init = True
    for parameter_name in macromodel_priors.keys():

        prior_type = macromodel_priors[parameter_name][0]
        prior = macromodel_priors[parameter_name]
        value = _draw(prior, prior_type)
        kwargs_hyper_macro[parameter_name] = value

        if prior_type == 'FIXED':
            continue
        else:
            param_names_macro.append(parameter_name)
        if init:
            macro_params = value
            init = False
        else:
            macro_params = np.append(macro_params, value)

    return macro_params, kwargs_hyper_macro, param_names_macro

def setup_realization(priors, kwargs_other, x_image, y_image, source_size_pc):

    realization_priors = deepcopy(priors)
    realization_params = np.array([])
    kwargs_realization = {}
    preset_model_name = realization_priors['PRESET_MODEL']

    del realization_priors['PRESET_MODEL']
    param_names_realization = []

    start = True
    for parameter_name in realization_priors.keys():

        prior_type = realization_priors[parameter_name][0]
        prior = realization_priors[parameter_name]
        value = _draw(prior, prior_type)
        if parameter_name == 'log10_sigma_sub':
            kwargs_realization['sigma_sub'] = 10 ** value
        elif parameter_name == 'log10_LOS_normalization':
            kwargs_realization['LOS_normalization'] = 10 ** value
        else:
            kwargs_realization[parameter_name] = value

        if prior_type == 'FIXED':
            continue
        else:
            param_names_realization.append(parameter_name)

        if start:
            realization_params = value
            start = False
        else:
            realization_params = np.append(realization_params, value)

    for arg in kwargs_other.keys():
        kwargs_realization[arg] = kwargs_other[arg]

    if preset_model_name == 'WDM_x':
        preset_model = CUSTOM_WDM
    elif preset_model_name == 'SIDM_CORE_COLLAPSE':
        preset_model = SIDM_CORE_COLLAPSE
    elif preset_model_name == 'ULDM':

        if kwargs_realization['flucs'] is True:
            aperture_radius = auto_raytracing_grid_size(source_size_pc) * 1.25
            kwargs_realization['flucs_args'] = {'x_images': x_image,
                                                'y_images': y_image,
                                                'aperture': aperture_radius}
        else:
            pass

        preset_model = preset_model_from_name(preset_model_name)

    else:
        try:
            preset_model = preset_model_from_name(preset_model_name)
        except:
            raise Exception('preset model '+str(preset_model_name)+' not recognized.')

    return realization_params, preset_model, kwargs_realization, param_names_realization

def SIDM_CORE_COLLAPSE(zlens, zsource, **kwargs_rendering):

    CDM = preset_model_from_name('CDM')
    realization_cdm = CDM(zlens, zsource, **kwargs_rendering)
    lens_cosmo = realization_cdm.lens_cosmo
    ext = RealizationExtensions(realization_cdm)
    mass_range = [[6.0, 7.5], [7.5, 8.5], [8.5, 10.0]]
    p675_sub = kwargs_rendering['f_675_sub']
    p7585_sub = kwargs_rendering['f_7585_sub']
    p910_sub = kwargs_rendering['f_8510_sub']

    def _collape_probability_field(z, prob, zlens, lens_cosmo):
        rescale = lens_cosmo.cosmo.halo_age(zlens)/lens_cosmo.cosmo.halo_age(z)
        return prob * rescale

    p1 = min(1.0, kwargs_rendering['r_1_field'] * p675_sub)
    p2 = min(1.0, kwargs_rendering['r_2_field'] * p7585_sub)
    p3 = min(1.0, kwargs_rendering['r_3_field'] * p910_sub)

    kwargs_field_1 = {'prob': p1, 'zlens': zlens, 'lens_cosmo': lens_cosmo}
    kwargs_field_2 = {'prob': p2, 'zlens': zlens, 'lens_cosmo': lens_cosmo}
    kwargs_field_3 = {'prob': p3, 'zlens': zlens, 'lens_cosmo': lens_cosmo}
    p675_field = _collape_probability_field
    p759_field = _collape_probability_field
    p910_field = _collape_probability_field
    kwargs_field = [kwargs_field_1, kwargs_field_2, kwargs_field_3]

    probabilities_subhalos = [p675_sub, p7585_sub, p910_sub]
    probabilities_field_halos = [p675_field, p759_field, p910_field]

    indexes = ext.core_collapse_by_mass(mass_range, mass_range,
                              probabilities_subhalos, probabilities_field_halos, kwargs_field=kwargs_field)

    if kwargs_rendering['halo_profile'] == 'SPL_CORE':
        kwargs_core_collapse_profile = {'x_match': kwargs_rendering['x_match'],
                                        'x_core_halo': kwargs_rendering['x_core_halo'],
                                        'log_slope_halo': kwargs_rendering['log_slope_halo']}
    elif kwargs_rendering['halo_profile'] == 'GNFW':
        kwargs_core_collapse_profile = {'x_match': kwargs_rendering['x_match'],
                                        'gamma_inner': kwargs_rendering['gamma_inner'],
                                        'gamma_outer': kwargs_rendering['gamma_outer']}
    else:
        raise Exception('halo profile must be specified')

    realization_sidm = ext.add_core_collapsed_halos(indexes, kwargs_rendering['halo_profile'],
                                                    **kwargs_core_collapse_profile)
    return realization_sidm

def CUSTOM_WDM(zlens, zsource, **kwargs_rendering):

    def a_func_mfunc(x, norm=0.33, slope=-0.072):
        y = norm + slope * (x - 0.5)
        return y

    def b_func_mfunc(x, norm=0.28, slope=0.6):
        y = norm + slope * (x - 0.5)
        return y

    def a_func_mcrel(x, norm=0.75, slope=0.4):
        y = norm * (0.7 / (x)) ** slope
        return y

    def b_func_mcrel(x, norm=0.94, slope=0.7, shift=0.6):
        y = norm * abs(np.log(1.76 + shift) / np.log(x + shift)) ** slope
        return y

    x = kwargs_rendering['x_wdm']
    a_wdm = a_func_mfunc(x)
    b_wdm = b_func_mfunc(x)
    c_wdm = -3.
    a_mc = a_func_mcrel(x)
    b_mc = b_func_mcrel(x)

    kwargs_mass_function_subhalos = {}
    kwargs_mass_function_fieldhalos = {}
    kwargs_mass_function_fieldhalos['a_wdm'] = a_wdm
    kwargs_mass_function_fieldhalos['b_wdm'] = b_wdm
    kwargs_mass_function_fieldhalos['c_wdm'] = c_wdm
    kwargs_mass_function_subhalos['a_wdm'] = a_wdm
    kwargs_mass_function_subhalos['b_wdm'] = b_wdm
    kwargs_mass_function_subhalos['c_wdm'] = c_wdm

    kwargs_mc_relation_subhalos = {'a': a_mc, 'b': b_mc}
    kwargs_mc_relation_fieldhalos = {'a': a_mc, 'b': b_mc}

    kwargs_rendering['mass_function_model_subhalos'] = 'LOVELL2020'
    kwargs_rendering['mass_function_model_fieldhalos'] = 'LOVELL2020'
    kwargs_rendering['concentration_model_subhalos'] = 'WDM_HYPERBOLIC'
    kwargs_rendering['concentration_model_fieldhalos'] = 'WDM_HYPERBOLIC'

    kwargs_rendering['kwargs_mass_function_subhalos'] = kwargs_mass_function_subhalos
    kwargs_rendering['kwargs_mass_function_fieldhalos'] = kwargs_mass_function_fieldhalos
    kwargs_rendering['kwargs_concentration_model_subhalos'] = kwargs_mc_relation_subhalos
    kwargs_rendering['kwargs_concentration_model_fieldhalos'] = kwargs_mc_relation_fieldhalos
    del kwargs_rendering['x_wdm']
    WDM = preset_model_from_name('WDM')

    return WDM(zlens, zsource, **kwargs_rendering)
