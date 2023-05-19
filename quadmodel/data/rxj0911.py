from quadmodel.data.quad_base import Quad
from quadmodel.deflector_models.sis import SIS
import numpy as np


class RXJ0911(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type = 'EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.77
        zsource = 2.76
        x = [0.688, 0.946, 0.672, -2.283]
        y = [-0.517, -0.112, 0.442, 0.274]
        m = [0.56, 1.0, 0.53, 0.24]
        delta_m = [0.04/0.56, 0.05, 0.04/0.53, 0.04/0.24]
        delta_xy = [0.003] * 4
        keep_flux_ratio_index = [0, 1, 2]
        self.log10_host_halo_mass = 13.1
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {}

        super(RXJ0911, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

    def satellite_galaxy(self, sample=True):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """
        theta_E = 0.25
        center_x = -0.767
        center_y = 0.657
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(self.zlens, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names
