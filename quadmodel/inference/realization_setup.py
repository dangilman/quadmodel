from pyHalo.preset_models import preset_model_from_name
import numpy as np
from copy import deepcopy
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size
from pyHalo.realization_extensions import RealizationExtensions

def setup_realization(priors, kwargs_other, x_image, y_image, source_size_pc):

    realization_priors = deepcopy(priors)
    realization_params = None
    kwargs_realization = {}
    preset_model_name = realization_priors['PRESET_MODEL']

    del realization_priors['PRESET_MODEL']
    param_names_realization = []
    for parameter_name in realization_priors.keys():

        prior_type = realization_priors[parameter_name][0]
        prior = realization_priors[parameter_name]

        if prior_type == 'UNIFORM':
            value = np.random.uniform(prior[1], prior[2])
        elif prior_type == 'GAUSSIAN':
            value = np.random.normal(prior[1], prior[2])
        elif prior_type == 'FIXED':
            value = prior[1]
        elif prior_type == 'DISCRETE':
            value = np.random.choice(prior[1])
        else:
            raise Exception('prior type '+str(prior_type)+' not recognized.')

        kwargs_realization[parameter_name] = value

        if prior_type == 'FIXED':
            continue
        else:
            param_names_realization.append(parameter_name)

        if realization_params is None:
            realization_params = value
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
    
    realization_sidm = ext.add_core_collapsed_halos(indexes, **kwargs_core_collapse_profile)
    return realization_sidm

def CUSTOM_WDM(zlens, zsource, **kwargs_rendering):

    x = kwargs_rendering['x_wdm']
    a_wdm = 0.27 - 0.05 * (x - 0.85)
    b_wdm = 0.5 + 0.95 * (x - 0.6)
    c_wdm = -3.

    a_mc = 0.64 - 0.35 * (x - 0.7)
    b_mc = 0.49 + 1.20 * (x - 1.15) ** 2

    kwargs_suppresion_field = {'a_mc': a_mc, 'b_mc': b_mc}
    kwargs_rendering['a_wdm_los'] = a_wdm
    kwargs_rendering['b_wdm_los'] = b_wdm
    kwargs_rendering['c_wdm_los'] = c_wdm
    kwargs_rendering['a_wdm_sub'] = a_wdm
    kwargs_rendering['b_wdm_sub'] = b_wdm
    kwargs_rendering['c_wdm_sub'] = c_wdm
    kwargs_rendering['kwargs_suppression_field'] = kwargs_suppresion_field
    kwargs_rendering['kwargs_suppression_sub'] = kwargs_suppresion_field
    kwargs_rendering['suppression_model_field'], kwargs_rendering['suppression_model_sub'] = \
        'hyperbolic', 'hyperbolic'
    WDM = preset_model_from_name('WDM')
    return WDM(zlens, zsource, **kwargs_rendering)
