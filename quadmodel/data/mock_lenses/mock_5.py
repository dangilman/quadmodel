from quadmodel.data.quad_base import Quad

class Mock5(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.55
        zsource = 2.2
        x = [1.07567439,  0.61875215,  1.04178484, -0.63560938]
        y = [0.55103722, -0.9327763 , -0.44885363,  0.18652581]
        m = [0.48690356, 1.        , 0.8831887 , 0.10029187]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.15}

        super(Mock5, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

