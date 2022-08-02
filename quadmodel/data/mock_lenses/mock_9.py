from quadmodel.data.quad_base import Quad

class Mock9(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR'):

        zlens = 0.3
        zsource = 1.2
        x = [0.23662011, -0.25905741, -0.8688029, 0.68669581]
        y = [1.53191442, -1.01250613, -0.4642428, -0.76093979]
        m = [0.27680572, 1., 0.63627885, 0.39502489]
        delta_m = [0.0001, 0.0001, 0.0001, 0.0001]
        delta_xy = [0.001] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.18}

        super(Mock9, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)

