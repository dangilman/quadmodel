from quadmodel.data.quad_base import Quad

class Mock4(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.5
        zsource = 1.6
        x = [0.83404141, -0.10384225,  0.83345926, -0.74384427]
        y = [0.73683237, -1.03329106, -0.51659549,  0.32366071]
        m = [0.75240312, 1.        , 0.86573874, 0.37159584]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.15}

        super(Mock4, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

