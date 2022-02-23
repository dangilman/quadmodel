from quadmodel.data.quad_base import Quad

class B1422_fakeYesPBH(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian'):

        zlens = 0.34
        zsource = 3.62
        x = [-0.347, -0.734, -1.096, 0.207]
        y = [0.964, 0.649, -0.079, -0.148]
        m = [0.896901, 1., 0.453252, 0.015182]
        #m = [0.88, 1., 0.474, 0.025]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        #delta_m = [0.01/0.88, 0.01, 0.006/0.47, None]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1]

        macromodel_type = 'EPL_FIXED_SHEAR_MULTIPOLE'
        kwargs_macromodel = {'shear_amplitude_min': 0.12, 'shear_amplitude_max': 0.35}

        super(B1422_fakeYesPBH, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)
