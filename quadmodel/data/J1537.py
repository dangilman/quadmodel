from quadmodel.data.quad_base import Quad
import numpy as np

class J1537(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.59
        zsource = 1.72
        x_main_deflector = 0.095
        y_main_deflector = -0.03
        image_A = [1.491, -0.815]
        image_B = [-0.471, -1.126]
        image_C = [-1.352, 0.822]
        image_D = [0.772, 0.970]
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

        self.log10_host_halo_mass = 13.6
        self.log10_host_halo_mass_sigma = 0.35

        kwargs_macromodel = {'shear_amplitude_min': 0.05, 'shear_amplitude_max': 0.25}

        super(J1537, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)
        
class J1537_JWST(J1537):
    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):
        super(J1537_JWST, self).__init__(sourcemodel_type, macromodel_type)
        
        # now replace the data with the JWST measurements
        x = [ 0.20448753,  1.22923212, -0.27489248, -1.15882718] 
        y = [-1.56949733,  0.15095658,  1.65685658, -0.23831582]
        self.x = x
        self.y = y
        normalized_fluxes = [1.00, 0.70, 1.07, 1.28]
        self.m = np.array(normalized_fluxes)
        flux_uncertainties = [0.01] * 4  # percent uncertainty
        self.delta_m = np.array(flux_uncertainties)    
        
        

