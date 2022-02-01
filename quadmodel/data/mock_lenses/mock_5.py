from quadmodel.data.quad_base import Quad


class Mock_5_MIDIR(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.45
        self.zlens = zlens
        zsource = 2.2
        x = [ 0.1501, -0.7479, -0.9553,  0.732 ]
        y = [-0.9611,  0.53  , -0.0109,  0.5036]
        m = [0.389 , 0.9929, 0.6426, 0.2635]
        delta_m = [0.02] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.02, 'shear_amplitude_max': 0.16}

        super(Mock_5_MIDIR, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)


class Mock_5_NL(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR_MULTIPOLE'):

        zlens = 0.45
        self.zlens = zlens
        zsource = 2.2
        x = [0.1501, -0.7479, -0.9553, 0.732]
        y = [-0.9611, 0.53, -0.0109, 0.5036]
        m = [0.373 , 1.0265, 0.6408, 0.2559]
        delta_m = [0.04] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        kwargs_macromodel = {'shear_amplitude_min': 0.02, 'shear_amplitude_max': 0.16}

        super(Mock_5_NL, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {},
                                           macromodel_type,
                                           kwargs_macromodel, keep_flux_ratio_index)
