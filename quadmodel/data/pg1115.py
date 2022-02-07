from quadmodel.data.quad_base import Quad
import numpy as np
from quadmodel.deflector_models.sis import SIS


class PG1115(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.31
        self.zlens = zlens
        zsource = 1.72
        x = [0.947, 1.096, -0.722, -0.381]
        y = [-0.69, -0.232, -0.617, 1.344]
        m = [1.0, 0.93, 0.16, 0.21]
        delta_m = [0.06/0.93, 0.07/0.16, 0.04/0.21]
        delta_xy = [0.003] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.0025, 'shear_amplitude_max': 0.1}

        super(PG1115, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index, uncertainty_in_magnifications=False)

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
        theta_E = 2.0
        center_x = -9.205
        center_y = -3.907
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(self.zlens, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names
