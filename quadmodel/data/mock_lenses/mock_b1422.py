from quadmodel.data.quad_base import Quad

class B1422SmoothModel(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        """
        Uses flux ratios computed from a smooth lens model according to Table 3 in Nierenberg et al. (2014)
        """
        zlens = 0.34
        zsource = 3.62
        x = [-0.347, -0.734, -1.096, 0.207]
        y = [0.964, 0.649, -0.079, -0.148]
        m = [0.83, 1., 0.484, 0.04]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.12, 'shear_amplitude_max': 0.35}

        super(B1422SmoothModel, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)
