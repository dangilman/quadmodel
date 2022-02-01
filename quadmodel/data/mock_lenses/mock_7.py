from quadmodel.data.quad_base import Quad


class Mock_7_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.3
        self.zlens = zlens
        zsource = 1.4
        x = [ 1.0696,  0.3761,  0.8713, -0.6135]
        y = [-0.3888,  1.0895,  0.6123, -0.328 ]
        m = [0.6758, 0.9638, 1.0335, 0.3474]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.05, 'shear_amplitude_max': 0.2}

        super(Mock_7_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_7_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.3
        self.zlens = zlens
        zsource = 1.4
        x = [1.0696, 0.3761, 0.8713, -0.6135]
        y = [-0.3888, 1.0895, 0.6123, -0.328]
        m = [0.6683, 0.963 , 0.9546, 0.342 ]

        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.05, 'shear_amplitude_max': 0.2}

        super(Mock_7_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)
