from quadmodel.data.quad_base import Quad

class Mock7(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.4
        zsource = 2.4
        x = [0.20282109,  0.74552311,  0.96868232, -0.78415804]
        y = [-1.11403663,  0.73341575,  0.30516346,  0.23571821]
        m = [0.32197401, 0.9649838,  1.,         0.19538743]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.14}

        super(Mock7, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

