from quadmodel.data.quad_base import Quad

class J1042(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.6
        zsource = 2.5
        x_main_deflector = -0.052
        y_main_deflector = 0.034
        image_A = [-0.858, 0.687]
        image_B = [0.694, 0.111]
        image_C = [0.526, -0.468]
        image_D = [-0.087, -0.803]
        x = [image_A[0] - x_main_deflector,
             image_B[0] - x_main_deflector,
             image_C[0] - x_main_deflector,
             image_D[0] - x_main_deflector]
        y = [image_A[1] - y_main_deflector,
             image_B[1] - y_main_deflector,
             image_C[1] - y_main_deflector,
             image_D[1] - y_main_deflector]
        m = [1.0] * 4
        delta_m = [0.01] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 12.9
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.005, 'shear_amplitude_max': 0.15}

        super(J1042, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)
