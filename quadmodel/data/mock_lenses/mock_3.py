from quadmodel.data.quad_base import Quad

class Mock3(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.4
        zsource = 1.5
        x = [0.30819235,  0.28458679,  0.9593208 , -0.90162512]
        y = [-0.97055425,  0.95712065,  0.26731158, -0.01614304]
        m = [0.83991424, 1., 0.87028334, 0.50401647]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.1}

        super(Mock3, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

