from quadmodel.data.quad_base import Quad

class M1134(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.66 # Anguita et al. (in prep)
        zsource = 2.77
        x_main_deflector = -0.154
        y_main_deflector = 0.174
        image_A = [1.326, 1.150]
        image_B = [0.593, -0.609]
        image_C = [-1.356, -1.384]
        image_D = [-0.660, 0.765]
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

        self.log10_host_halo_mass = 13.4
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.2, 'shear_amplitude_max': 0.45}

        super(M1134, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)


class 2M1134(M1134):
    def __init__(self):





print('test')

class 2M1134m2103_JWST(M1134):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        super(2M1134m2103_JWST, self).__init__(sourcemodel_type, macromodel_type)
        
        # now replace the data with the JWST measurements
        x = [-1.7102934,   0.52474735,  1.53467056, -0.3491245 ] 
        y = [-0.80380298, -0.8440081,   0.87923354,  0.76857754]
        self.x = x
        self.y = y
        normalized_fluxes = [1.00, 0.70, 1.07, 1.28]
        self.m = np.array(normalized_fluxes)
        flux_uncertainties = [0.01] * 4  # percent uncertainty
        self.delta_m = np.array(flux_uncertainties)    