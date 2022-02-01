from quadmodel.data.quad_base import Quad


class Mock_2_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):
        zlens = 0.6
        self.zlens = zlens
        zsource = 1.6
        x = [ 1.2223, -1.1035, -0.9108, -0.4909]
        y = [ 0.3561, -0.0555,  0.5006, -0.9336]
        m = [0.2148, 0.9716, 0.596, 0.4144]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.03, 'shear_amplitude_max': 0.16}

        super(Mock_2_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_2_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):
        zlens = 0.6
        self.zlens = zlens
        zsource = 1.6
        x = [1.2223, -1.1035, -0.9108, -0.4909]
        y = [0.3561, -0.0555, 0.5006, -0.9336]
        m = [0.2048, 1.0112, 0.6556, 0.3939]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.03, 'shear_amplitude_max': 0.16}

        super(Mock_2_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)
