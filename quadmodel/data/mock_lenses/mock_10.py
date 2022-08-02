from quadmodel.data.quad_base import Quad

class Mock10(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.4
        zsource = 2.0
        x = [-0.0579367 ,  0.51446413,  0.90939812, -0.72091025]
        y = [1.19672824, -0.82043596, -0.08068952, -0.43813965]
        m = [0.43965132, 1., 0.83196568, 0.31992258]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.025, 'shear_amplitude_max': 0.24}

        super(Mock10, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

