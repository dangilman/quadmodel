from quadmodel.data.quad_base import Quad
from quadmodel.deflector_models.sis import SIS
import numpy as np


class HE0435_version2020(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.45
        zsource = 1.69
        x = [1.272, 0.306, -1.152, -0.384]
        y = [0.156, -1.092, -0.636, 1.026]
        m = [0.96, 0.976, 1.0, 0.65]
        delta_m = [0.05, 0.049, 0.048, 0.056]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.2
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.015, 'shear_amplitude_max': 0.15}

        super(HE0435, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

    def satellite_galaxy(self, sample=True):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """
        theta_E = 0.37
        center_x = -2.27
        center_y = 1.98
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(0.78, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names

class HE0435(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.45
        zsource = 1.69
        x = [-1.272, -0.306,  1.152,  0.384]
        y = [-0.156,  1.092,  0.636, -1.026]
        m = [0.96, 0.976, 1.0, 0.65]
        delta_m = [0.05, 0.049, 0.048, 0.056]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.2
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.015, 'shear_amplitude_max': 0.15}

        super(HE0435, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

    def satellite_galaxy(self, sample=True):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """
        theta_E = 0.37
        center_x_observed = -2.6
        center_y_observed = -3.65
        center_x = -1.9663
        center_y = -3.079
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(0.78, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names
