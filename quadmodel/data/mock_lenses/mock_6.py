from quadmodel.data.quad_base import Quad

class Mock6(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.4
        zsource = 2.4
        x = [1.04331399, -0.09841349,  0.2785511 , -0.6790812]
        y = [0.57683118, -1.00508681, -0.94015575,  0.40684181]
        m = [0.25970027, 0.95092259, 1., 0.14074434]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.025, 'shear_amplitude_max': 0.2}

        super(Mock6, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

