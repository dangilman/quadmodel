import numpy as np
from quadmodel.util import approx_theta_E
from quadmodel.inference.sample_prior import sample_from_prior

def default_priors(param):

    if param == 'gamma_macro':
        gamma_min, gamma_max = 1.9, 2.2
        return np.random.uniform(gamma_min, gamma_max)
    elif param == 'multipole_amplitude':
        am_mean, am_sigma = 0.0, 0.01
        return np.random.normal(am_mean, am_sigma)
    elif param == 'NARROW_LINE_Gaussian': # short for "Narrow-line Gaussian"
        source_fwhm_pc = np.random.uniform(25, 60)
        return source_fwhm_pc
    elif param == 'midIR_Gaussian': # short for "MidIR Gaussian"
        source_fwhm_pc = np.random.uniform(1, 20)
        return source_fwhm_pc
    elif param == 'DOUBLE_NL_Gaussian': # short for "MidIR Gaussian"
        source_fwhm_pc = np.random.uniform(25, 80)
        dx = np.random.uniform(1e-5, 1e5)
        dy = np.random.uniform(0, 0.12)
        amp_scale = np.random.uniform(0.25, 1.0)
        size_scale = np.random.uniform(0.25, 1.0)
        return source_fwhm_pc, dx, dy, amp_scale, size_scale

    else:
        raise Exception('parameter '+str(param)+' not recognized.')

class Quad(object):

    def __init__(self, zlens, zsource, x_image, y_image, magnifications, magnification_uncertainties, astrometric_uncertainties,
                 sourcemodel_type, kwargs_source_model, macromodel_type, kwargs_macromodel, keep_flux_ratio_index):

        self._zlens = zlens
        self.zsource = zsource
        self.x = np.array(x_image)
        self.y = np.array(y_image)
        self.m = np.array(magnifications)
        self.delta_m = np.array(magnification_uncertainties)
        self.delta_xy = np.array(astrometric_uncertainties)
        self.approx_einstein_radius = approx_theta_E(self.x, self.y)

        self._macromodel_type = macromodel_type
        self._kwargs_macromodel = kwargs_macromodel
        self._kwargs_source_model = kwargs_source_model

        self.uncertainty_in_magnifications = True
        self._sourcemodel_type = sourcemodel_type
        self.zlens = self.set_zlens()
        self.keep_flux_ratio_index = keep_flux_ratio_index

    def generate_macromodel(self):

        if self._macromodel_type == 'EPL_FIXED_SHEAR_MULTIPOLE':

            shear_min, shear_max = self._kwargs_macromodel['shear_amplitude_min'], \
                                   self._kwargs_macromodel['shear_amplitude_max']
            gamma_macro = default_priors('gamma_macro')
            shear_amplitude = sample_from_prior(['UNIFORM', shear_min, shear_max])
            optimization_routine = 'fixed_shear_powerlaw_multipole'
            constrain_params = {'shear': shear_amplitude}
            multipole_amplitude = default_priors('multipole_amplitude')
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole
            model = EPLShearMultipole(self.zlens, gamma_macro, shear_amplitude, multipole_amplitude, self.approx_einstein_radius,
                                      0.0, 0.0, 0.2, 0.1)
            params_sampled = np.array([gamma_macro, shear_amplitude, multipole_amplitude])
            return model, constrain_params, optimization_routine, params_sampled

        elif self._macromodel_type == 'EPL_FREE_SHEAR_MULTIPOLE':

            shear_min, shear_max = 0.001, 0.25
            gamma_macro = default_priors('gamma_macro')
            shear_amplitude = sample_from_prior(['UNIFORM', shear_min, shear_max])
            optimization_routine = 'free_shear_powerlaw_multipole'
            constrain_params = None
            multipole_amplitude = default_priors('multipole_amplitude')
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole
            model = EPLShearMultipole(self.zlens, gamma_macro, shear_amplitude, multipole_amplitude,
                                      self.approx_einstein_radius,
                                      0.0, 0.0, 0.2, 0.1)
            params_sampled = np.array([gamma_macro, shear_amplitude, multipole_amplitude])
            return model, constrain_params, optimization_routine, params_sampled

        else:
            raise Exception('other macromodels not yet implemented.')

    def generate_sourcemodel(self):

        if self._sourcemodel_type == 'NARROW_LINE_Gaussian':

            source_size_pc = default_priors(self._sourcemodel_type)
            kwargs_source_model = {'source_model': 'GAUSSIAN'}
            source_samples = np.array(source_size_pc)
            return source_size_pc, kwargs_source_model, source_samples

        elif self._sourcemodel_type == 'midIR_Gaussian':

            source_size_pc = default_priors(self._sourcemodel_type)
            kwargs_source_model = {'source_model': 'GAUSSIAN'}
            source_samples = np.array(source_size_pc)
            return source_size_pc, kwargs_source_model, source_samples

        elif self._sourcemodel_type == 'DOUBLE_NL_Gaussian':

            source_size_pc, dx, dy, amp_scale, size_scale = default_priors(self._sourcemodel_type)
            kwargs_source_model = {'source_model': 'DOUBLE_GAUSSIAN', 'dx': dx, 'dy': dy, 'amp_scale': amp_scale, 'size_scale': size_scale}
            kwargs_source_model = kwargs_source_model.update(self._kwargs_source_model)
            source_samples = np.array([source_size_pc, dy, amp_scale, size_scale])
            return source_size_pc, kwargs_source_model, source_samples

        else:
            raise Exception('other macromodels not yet implemented.')

    def satellite_galaxies(self):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """
        return []

    def set_zlens(self):

        if isinstance(self._zlens, float) or isinstance(self._zlens, int):
            self.zlens = self._zlens
        else:
            args = ['CUSTOM_PDF', self._zlens[0], self._zlens[1]]
            self.zlens = sample_from_prior(args)
