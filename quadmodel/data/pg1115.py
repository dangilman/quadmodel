from quadmodel.data.quad_base import Quad
import numpy as np
from quadmodel.deflector_models.sis import SIS


class PG1115(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.31
        zsource = 1.72
        x = [0.947, 1.096, -0.722, -0.381]
        y = [-0.69, -0.232, -0.617, 1.344]
        m = [1.0, 0.93, 0.16, 0.21]
        delta_m = [0.06/0.93, 0.07/0.16, 0.04/0.21]
        delta_xy = [0.003] * 4
        keep_flux_ratio_index = [0, 1, 2]
        self.log10_host_halo_mass = 13.0
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.002, 'shear_amplitude_max': 0.12}

        super(PG1115, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index, uncertainty_in_magnifications=False)

    # Omit the group as the imaging data seems to do better without it
    # def satellite_galaxy(self, sample=True):
    #     """
    #     If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
    #     """
    #     theta_E = 2.0
    #     center_x = -9.205
    #     center_y = -3.907
    #     if sample:
    #         theta_E = abs(np.random.normal(theta_E, 0.05))
    #         center_x = np.random.normal(center_x, 0.05)
    #         center_y = np.random.normal(center_y, 0.05)
    #
    #     kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
    #     satellite = SIS(self.zlens, kwargs_init)
    #     params = np.array([theta_E, center_x, center_y])
    #     param_names = ['theta_E', 'center_x', 'center_y']
    #     return [satellite], params, param_names
    
class PG1115p080_JWST(PG1115):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        super(RXJ1131_JWST, self).__init__(sourcemodel_type, macromodel_type, sample_zlens_pdf)
        
        # now replace the data with the JWST measurements
        x = [-0.95171665,  0.92010584,  0.24710189, -0.21549107] 
        y = [-0.56301239, -1.20018202,  0.83951884,  0.92367556]
        self.x = x
        self.y = y
        normalized_fluxes = [1.00, 0.70, 1.07, 1.28]
        self.m = np.array(normalized_fluxes)
        flux_uncertainties = [0.01] * 4  # percent uncertainty
        self.delta_m = np.array(flux_uncertainties)    
