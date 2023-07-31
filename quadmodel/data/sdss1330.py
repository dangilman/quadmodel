from quadmodel.data.quad_base import Quad
import numpy as np


class SDSS1330(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian'):

        zlens = 0.37
        zsource = 1.38

        x = np.array([0.226, -0.188, -1.023, 0.463])
        y = np.array([-0.978, -0.99, 0.189, 0.604])
        m = np.array([1., 0.79, 0.41, 0.25])
        delta_m = 0.05 * m
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        macromodel_type = 'EPL_FIXED_SHEAR_MULTIPOLE'
        kwargs_macromodel = {'shear_amplitude_min': 0.005, 'shear_amplitude_max': 0.3}

        super(SDSS1330, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)
