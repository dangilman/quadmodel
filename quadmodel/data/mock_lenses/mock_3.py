from quadmodel.data.quad_base import Quad


class Mock_3_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.3
        self.zlens = zlens
        zsource = 1.7
        x = [ 0.1037,  0.4203,  0.8937, -0.7817]
        y = [-0.9639,  0.8094, -0.063 ,  0.2349]
        m = [0.6071, 0.7904, 1.0032, 0.4773]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.005, 'shear_amplitude_max': 0.08}

        super(Mock_3_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_3_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.3
        self.zlens = zlens
        zsource = 1.7
        x = [0.1037, 0.4203, 0.8937, -0.7817]
        y = [-0.9639, 0.8094, -0.063, 0.2349]
        m = [0.5758, 0.7306, 0.9997, 0.464 ]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.005, 'shear_amplitude_max': 0.08}

        super(Mock_3_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)
