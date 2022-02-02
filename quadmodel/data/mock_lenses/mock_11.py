from quadmodel.data.quad_base import Quad


class Mock_11_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.25
        self.zlens = zlens
        zsource = 0.6
        x = [ 0.6782, -0.08  ,  0.8954, -1.009 ]
        y = [-1.2316,  1.2167,  0.6966, -0.1332]
        m = [0.5945, 0.9949, 0.8519, 0.4622]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.125}

        super(Mock_11_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_11_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.25
        self.zlens = zlens
        zsource = 0.6
        x = [0.6782, -0.08, 0.8954, -1.009]
        y = [-1.2316, 1.2167, 0.6966, -0.1332]
        m = [0.5712, 1.0677, 0.8357, 0.4508]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.125}

        super(Mock_11_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                        macromodel_type,
                                        kwargs_macromodel, keep_flux_ratio_index)
