from quadmodel.data.quad_base import Quad

class Mock8(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.6
        zsource = 1.7
        x = [0.29910032,  0.11631348,  0.84787401, -0.87390181]
        y = [-1.04283586,  0.89627613,  0.40131327,  0.08231104]
        m = [0.4468366,  1., 0.57099186, 0.31923909]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.14}

        super(Mock8, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

