from quadmodel.data.quad_base import Quad
import numpy as np

class HS0810(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR'):
        #raise Exception('this lens not yet implemented')
        zlens = 0.5
        zsource = 1.5

        x = np.array([-0.46, -0.373, 0.315, 0.153])
        y = np.array([-0.15, -0.317, -0.408, 0.439])
        m = np.array([1., 0.93, 0.48, 0.19])

        delta_m = [0.1] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.001, 'shear_amplitude_max': 0.3}

        super(HS0810, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)
