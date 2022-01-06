from quadmodel.data.quad_base import Quad

class WGD2038(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian'):

        zlens = 0.23
        zsource = 0.78
        x = [-1.474, 0.832, -0.686, 0.706]
        y = [0.488, -1.22, -1.191, 0.869]
        m = [0.862069, 1., 0.793103, 0.396552]
        delta_m = [0.01, 0.02/1.16, 0.02/0.92, 0.01/0.46]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        macromodel_type = 'EPL_FIXED_SHEAR_MULTIPOLE'
        kwargs_macromodel = {'shear_amplitude_min': 0.02, 'shear_amplitude_max': 0.12}

        super(WGD2038, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)
