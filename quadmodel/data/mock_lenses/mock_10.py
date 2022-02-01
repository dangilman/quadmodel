from quadmodel.data.quad_base import Quad


class Mock_10_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.5
        self.zlens = zlens
        zsource = 1.6
        x = [ 0.5371,  1.3934,  1.2266, -0.6797]
        y = [-1.3021,  0.1234, -0.6517,  0.4098]
        m = [0.406 , 0.6299, 0.995 , 0.0931]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.1, 'shear_amplitude_max': 0.35}

        super(Mock_10_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_10_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.5
        self.zlens = zlens
        zsource = 1.6
        x = [0.5371, 1.3934, 1.2266, -0.6797]
        y = [-1.3021, 0.1234, -0.6517, 0.4098]
        m = [0.3995, 0.6681, 1.0086, 0.0879]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.1, 'shear_amplitude_max': 0.35}

        super(Mock_10_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                        macromodel_type,
                                        kwargs_macromodel, keep_flux_ratio_index)
