from quadmodel.data.quad_base import Quad


class Mock_9_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.3
        self.zlens = zlens
        zsource = 2.2
        x = [-0.0647,  0.8595,  0.9379, -0.649 ]
        y = [ 1.0945, -0.5758,  0.287 , -0.387 ]
        m = [0.6681, 0.9813, 0.9907, 0.3064]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.005, 'shear_amplitude_max': 0.14}

        super(Mock_9_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_9_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.3
        self.zlens = zlens
        zsource = 2.2
        x = [-0.0647, 0.8595, 0.9379, -0.649]
        y = [1.0945, -0.5758, 0.287, -0.387]
        m = [0.7075, 1.0022, 0.9694, 0.3223]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.005, 'shear_amplitude_max': 0.14}

        super(Mock_9_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                        macromodel_type,
                                        kwargs_macromodel, keep_flux_ratio_index)
