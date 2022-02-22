from quadmodel.data.quad_base import Quad
import numpy as np
from quadmodel.deflector_models.sis import SIS


class MG0414(Quad):

    def __init__(self, sourcemodel_type='CO11-10_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.96
        self.zlens = zlens
        zsource = 2.64
        x = [1.078, 1.191, 0.482, -0.863]
        y = [-0.584, -0.341, 1.279, -0.369]
        m = [1.0, 0.86, 0.36, 0.16]
        delta_m = [0.05/0.83, 0.04/0.36, 0.04/0.34]
        delta_xy = [0.01] * 4 # increase the uncertainties due to the difference between redsshifted a
        # nd blue-shifted emission regions
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.0025, 'shear_amplitude_max': 0.1}

        super(MG0414, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
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
        theta_E = 0.2
        center_x = -0.403
        center_y = 1.471
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(0.96, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names
