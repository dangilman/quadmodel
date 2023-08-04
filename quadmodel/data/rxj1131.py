from quadmodel.data.quad_base import Quad
from quadmodel.deflector_models.sis import SIS
import numpy as np


class RXJ1131(Quad):

    def __init__(self, sourcemodel_type='DOUBLE_NL_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.3
        zsource = 0.66
        m = [1.63, 1.0, 1.19, 0.2]
        x = [-2.137, -2.076, -1.68, 1.074]
        y = [-0.27, 1.012, -1.352, 0.236]
        delta_m = [0.04/1.63, 0.12/1.19, None]
        delta_xy = [0.025] * 4
        keep_flux_ratio_index = [0, 1]
        self.log10_host_halo_mass = 13.9
        self.log10_host_halo_mass_sigma = 0.3
        kwargs_source_model = {}

        kwargs_macromodel = {'shear_amplitude_min': 0.06, 'shear_amplitude_max': 0.28}

        super(RXJ1131, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, kwargs_source_model,
                                      macromodel_type, kwargs_macromodel, keep_flux_ratio_index,
                                      uncertainty_in_magnifications=False)

    def satellite_galaxy(self, sample=True):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """
        theta_E = 0.28
        center_x = 0.097
        center_y = -0.614
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(self.zlens, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names


class RXJ1131_Sugai(Quad):
    """
    After inspecting imaging data I think the astrometry from Sugai is wrong
    """
    def __init__(self, sourcemodel_type='DOUBLE_NL_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.3
        zsource = 0.66
        # x = [-2.076, -2.037, -1.46, 1.0739999999999998]
        # y = [0.6620000000000004, -0.52, -1.6320000000000001, 0.3560000000000003]
        # m = [1.0, 1.63, 1.19, 0.2]
        m = [1.63, 1.0, 1.19, 0.2]
        x = [-2.037, -2.076, -1.46, 1.074]
        y = [-0.52, 0.662, -1.632, 0.356]
        delta_m = [0.04/1.63, 0.12/1.19, None]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1]
        self.log10_host_halo_mass = 13.9
        self.log10_host_halo_mass_sigma = 0.3
        kwargs_source_model = {}

        kwargs_macromodel = {'shear_amplitude_min': 0.06, 'shear_amplitude_max': 0.28}

        super(RXJ1131_Sugai, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, kwargs_source_model,
                                      macromodel_type, kwargs_macromodel, keep_flux_ratio_index,
                                      uncertainty_in_magnifications=False)

    def satellite_galaxy(self, sample=True):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """
        theta_E = 0.28
        center_x = -0.097
        center_y = 0.614
        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(self.zlens, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names



class RXJ1131_JWST(RXJ1131):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        super(RXJ1131_JWST, self).__init__(sourcemodel_type, macromodel_type)
        
        # now replace the data with the JWST measurements
        x = [-0.54388937, -0.99753444,  0.25821667,  1.28320714] 
        y = [-2.21168049,  0.95854212,  0.90545154,  0.34768682]
        self.x = x
        self.y = y
        normalized_fluxes = [1.00, 0.70, 1.07, 1.28]
        self.m = np.array(normalized_fluxes)
        flux_uncertainties = [0.01] * 4  # percent uncertainty
        self.delta_m = np.array(flux_uncertainties)