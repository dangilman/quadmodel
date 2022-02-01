from quadmodel.data.quad_base import Quad


class Mock_2_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.6
        self.zlens = zlens
        zsource = 1.6
        x = [ 0.6107,  0.0368,  0.4096, -0.6426]
        y = [ 0.7455, -0.7335, -0.6147,  0.0999]
        m = [0.3152, 1.0411, 0.9251, 0.2452]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.02, 'shear_amplitude_max': 0.18}

        super(Mock_2_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_2_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):
        zlens = 0.6
        self.zlens = zlens
        zsource = 1.6
        x = [0.6107, 0.0368, 0.4096, -0.6426]
        y = [0.7455, -0.7335, -0.6147, 0.0999]
        m = [0.3033, 0.9527, 0.9059, 0.2507]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.02, 'shear_amplitude_max': 0.18}

        super(Mock_2_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)
