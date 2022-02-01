from quadmodel.data.quad_base import Quad


class Mock_4_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.5
        self.zlens = zlens
        zsource = 2.5
        x = [ 0.1828,  1.0052,  0.9566, -0.5731]
        y = [-1.0486,  0.2075, -0.3743,  0.36  ]
        m = [0.5671, 0.8864, 1.0341, 0.1544]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.03, 'shear_amplitude_max': 0.2}

        super(Mock_4_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_4_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.5
        self.zlens = zlens
        zsource = 2.5
        x = [0.1828, 1.0052, 0.9566, -0.5731]
        y = [-1.0486, 0.2075, -0.3743, 0.36]
        m = [0.5592, 0.8528, 0.9531, 0.1483]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.03, 'shear_amplitude_max': 0.2}

        super(Mock_4_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)
