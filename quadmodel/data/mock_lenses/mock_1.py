from quadmodel.data.quad_base import Quad

class Mock1(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        """

        """
        zlens = 0.5
        zsource = 2.0
        x = [1.10502282,-0.08588209, 0.57245819, -0.58034621]
        y = [-0.3520062, 1.05155373, 0.85025812, -0.46824808]
        m = [0.52862854, 0.98170444, 1., 0.20642608]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.02, 'shear_amplitude_max': 1.6}

        super(Mock1, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

