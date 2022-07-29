from quadmodel.data.quad_base import Quad

class Mock3(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.4
        zsource = 1.5
        x = [0.67345069, -0.95865037, -0.87226427,  0.45683932]
        y = [0.89975568, -0.38103759,  0.43455785, -0.70718432]
        m = [0.56688695, 1.        , 0.86294633, 0.41020866]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.02, 'shear_amplitude_max': 0.15}

        super(Mock3, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

