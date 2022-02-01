from quadmodel.data.quad_base import Quad


class Mock_8_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.4
        self.zlens = zlens
        zsource = 1.2
        x = [-0.4307, -0.1201, -0.6941,  0.7474]
        y = [ 1.0674, -0.951 , -0.4659, -0.0786]
        m = [0.4946, 0.9922, 0.7714, 0.3746]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.03, 'shear_amplitude_max': 0.2}

        super(Mock_8_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_8_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.4
        self.zlens = zlens
        zsource = 1.2
        x = [-0.4307, -0.1201, -0.6941, 0.7474]
        y = [1.0674, -0.951, -0.4659, -0.0786]
        m = [0.5145, 1.057 , 0.8123, 0.3666]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.03, 'shear_amplitude_max': 0.2}

        super(Mock_8_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                        macromodel_type,
                                        kwargs_macromodel, keep_flux_ratio_index)
