from quadmodel.data.quad_base import Quad
from quadmodel.deflector_models.sis import SIS
import numpy as np


class WFI2033(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian'):

        zlens = 0.66
        zsource = 1.66
        x = [-0.751, -0.039, 1.445, -0.668]
        y = [0.953, 1.068, -0.307, -0.585]
        m = [1.0, 0.65, 0.5, 0.53]
        delta_m = [0.03, 0.03/0.64, 0.02/0.5, 0.02/0.53]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        macromodel_type = 'EPL_FIXED_SHEAR_MULTIPOLE'
        kwargs_macromodel = {'shear_amplitude_min': 0.07, 'shear_amplitude_max': 0.26}

        super(WFI2033, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

    def generate_macromodel(self):
        """
        Used only if lens-specific data class has no satellite galaxies; for systems with satellites, add them in the
        lens-specific data class and override this method
        :return:
        """

        model, constrain_params, optimization_routine, params_sampled, param_names_macro = self._generate_macromodel()
        model_satellite, params_satellite, param_names_satellite = self.satellite_galaxy()
        model.add_satellite(model_satellite)
        params_sampled = np.append(params_sampled, params_satellite)
        param_names_macro += param_names_satellite
        return model, constrain_params, optimization_routine, params_sampled, param_names_macro

    def satellite_galaxy(self, sample=True):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """

        theta_E = 0.03
        center_x = 0.245
        center_y = 2.037
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.03))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init_1 = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite_1 = SIS(0.66, kwargs_init_1)
        params_1 = np.array([theta_E, center_x, center_y])
        param_names_1 = ['theta_E_1', 'center_x_1', 'center_y_1']

        theta_E = 0.93
        center_x = -3.36
        center_y = -0.08
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init_2 = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite_2 = SIS(0.745, kwargs_init_2)
        params_2 = np.array([theta_E, center_x, center_y])
        param_names_2 = ['theta_E_2', 'center_x_2', 'center_y_2']
        return [satellite_1, satellite_2], np.append(params_1, params_2), param_names_1 + param_names_2
