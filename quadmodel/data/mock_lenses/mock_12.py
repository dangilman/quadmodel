from quadmodel.data.quad_base import Quad


class Mock_12_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.35
        self.zlens = zlens
        zsource = 2.8
        x = [ 1.0599, -0.4645,  0.411 , -0.4873]
        y = [ 0.1375, -0.8889, -0.8213,  0.6456]
        m = [0.5903, 0.8896, 0.9748, 0.2501]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.05, 'shear_amplitude_max': 0.25}

        super(Mock_12_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_12_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.35
        self.zlens = zlens
        zsource = 2.8
        x = [1.0599, -0.4645, 0.411, -0.4873]
        y = [0.1375, -0.8889, -0.8213, 0.6456]
        m = [0.6241, 0.82  , 0.9662, 0.2556]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.05, 'shear_amplitude_max': 0.25}

        super(Mock_12_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                        macromodel_type,
                                        kwargs_macromodel, keep_flux_ratio_index)
