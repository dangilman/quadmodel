from quadmodel.data.quad_base import Quad


class Mock_6_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.6
        self.zlens = zlens
        zsource = 1.9
        x = [ 0.6647,  0.4453,  1.0898, -1.022 ]
        y = [ 0.9849, -0.9605, -0.2931,  0.01  ]
        m = [0.4741, 0.9908, 0.8639, 0.2644]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.03, 'shear_amplitude_max': 0.15}

        super(Mock_6_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_6_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.6
        self.zlens = zlens
        zsource = 1.9
        x = [0.6647, 0.4453, 1.0898, -1.022]
        y = [0.9849, -0.9605, -0.2931, 0.01]
        m = [0.4834, 0.9492, 0.7845, 0.2766]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.03, 'shear_amplitude_max': 0.15}

        super(Mock_6_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)
