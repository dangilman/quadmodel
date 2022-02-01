from quadmodel.data.quad_base import Quad

class Mock_1_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.4
        self.zlens = zlens
        zsource = 1.2
        x = [ 0.9109, -0.042 ,  0.7291, -0.7783]
        y = [-0.5205,  0.9844,  0.738 , -0.4425]
        m = [0.6413, 1.0071, 0.9886, 0.343 ]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.1}

        super(Mock_1_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

class Mock_1_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.4
        self.zlens = zlens
        zsource = 1.2
        x = [0.9109, -0.042, 0.7291, -0.7783]
        y = [-0.5205, 0.9844, 0.738, -0.4425]
        m = [0.6289, 0.9843, 0.8869, 0.3381]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.1}

        super(Mock_1_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)
