import numpy as np
from quadmodel.util import approx_theta_E
from quadmodel.inference.sample_prior import sample_from_prior
from scipy.optimize import minimize

def default_priors(param):

    if param == 'gamma_macro':
        gamma_min, gamma_max = 1.9, 2.2
        return np.random.uniform(gamma_min, gamma_max)
    elif param == 'multipole_amplitude':
        am_mean, am_sigma = 0.0, 0.01
        return np.random.normal(am_mean, am_sigma)
    elif param == 'NARROW_LINE_Gaussian':
        source_fwhm_pc = np.random.uniform(25, 60)
        return source_fwhm_pc
    elif param == 'midIR_Gaussian':
        source_fwhm_pc = np.random.uniform(0.5, 10.0)
        return source_fwhm_pc
    elif param == 'CO11-10_Gaussian':
        source_fwhm_pc = np.random.uniform(5.0, 15.0)
        return source_fwhm_pc
    elif param == 'DOUBLE_NL_Gaussian':
        source_fwhm_pc = np.random.uniform(25, 80)
        dx = np.random.uniform(1e-5, 1e-5)
        dy = np.random.uniform(0, 0.01)
        amp_scale = np.random.uniform(0.25, 1.0)
        size_scale = np.random.uniform(0.25, 1.0)
        return source_fwhm_pc, dx, dy, amp_scale, size_scale

    else:
        raise Exception('parameter '+str(param)+' not recognized.')

class Quad(object):

    def __init__(self, zlens, zsource, x_image, y_image, magnifications, magnification_uncertainties, astrometric_uncertainties,
                 sourcemodel_type, kwargs_source_model, macromodel_type, kwargs_macromodel, keep_flux_ratio_index,
                 uncertainty_in_magnifications=True):

        self._zlens = zlens
        self.zlens = self.set_zlens()
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

        self.uncertainty_in_magnifications = uncertainty_in_magnifications
        self._sourcemodel_type = sourcemodel_type
        self.keep_flux_ratio_index = keep_flux_ratio_index


    @staticmethod
    def _flux_chi_square(scale, fluxes_measured, fluxes_modeled, measurement_uncertainties, dof_increment=0):

        df = 0
        dof = 0

        for i in range(0, len(fluxes_measured)):
            if measurement_uncertainties[i] is None or fluxes_measured[i] is None:
                continue
            else:
                dof += 1
                df += (fluxes_measured[i] - scale * fluxes_modeled[i])**2/measurement_uncertainties[i]**2
        dof += dof_increment
        return df/dof

    def flux_chi_square(self, fluxes_modeled):

        solution = float(minimize(self._flux_chi_square, x0=1.0, args=(self.m, fluxes_modeled, self.delta_m))['x'])

        return self._flux_chi_square(solution, self.m, fluxes_modeled, self.delta_m, dof_increment=1)

    def flux_ratio_chi_square(self, flux_ratios_modeled):

        flux_ratios_measured = self.m[1:]/self.m[0]
        df = 0
        dof = len(self.keep_flux_ratio_index)
        for i in self.keep_flux_ratio_index:
            df += (flux_ratios_measured[i] - flux_ratios_modeled[i])**2/self.delta_m[i]**2
        return df/dof

    def generate_macromodel(self):
        """
        Used only if lens-specific data class has no satellite galaxies; for systems with satellites, add them in the
        lens-specific data class and override this method
        :return:
        """
        return self._generate_macromodel()

    def _generate_macromodel(self):

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
            param_names_macro = ['gamma', 'gamma_ext', 'a4']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self._macromodel_type == 'EPL_FIXED_SHEAR':

            shear_min, shear_max = self._kwargs_macromodel['shear_amplitude_min'], \
                                   self._kwargs_macromodel['shear_amplitude_max']
            gamma_macro = default_priors('gamma_macro')
            shear_amplitude = sample_from_prior(['UNIFORM', shear_min, shear_max])
            optimization_routine = 'fixed_shear_powerlaw'
            constrain_params = {'shear': shear_amplitude}
            from quadmodel.deflector_models.preset_macromodels import EPLShear
            model = EPLShear(self.zlens, gamma_macro, shear_amplitude, self.approx_einstein_radius,
                                      0.0, 0.0, 0.2, 0.1)
            params_sampled = np.array([gamma_macro])
            param_names_macro = ['gamma', 'gamma_ext']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro


        elif self._macromodel_type == 'EPL_FREE_SHEAR_MULTIPOLE':

            random_shear_init = np.random.uniform(0.05, 0.25)
            gamma_macro = default_priors('gamma_macro')
            optimization_routine = 'free_shear_powerlaw_multipole'
            constrain_params = None
            multipole_amplitude = default_priors('multipole_amplitude')
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole
            model = EPLShearMultipole(self.zlens, gamma_macro, random_shear_init, multipole_amplitude,
                                      self.approx_einstein_radius,
                                      0.0, 0.0, 0.2, 0.1)
            params_sampled = np.array([gamma_macro, multipole_amplitude])
            param_names_macro = ['gamma', 'a4']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self._macromodel_type == 'EPL_FREE_SHEAR':

            random_shear_init = np.random.uniform(0.05, 0.25)
            gamma_macro = default_priors('gamma_macro')
            optimization_routine = 'free_shear_powerlaw'
            constrain_params = None
            from quadmodel.deflector_models.preset_macromodels import EPLShear
            model = EPLShear(self.zlens, gamma_macro, random_shear_init,
                                      self.approx_einstein_radius,
                                      0.0, 0.0, 0.2, 0.1)
            params_sampled = np.array([gamma_macro])
            param_names_macro = ['gamma']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro


        else:
            raise Exception('other macromodels not yet implemented.')

    def generate_sourcemodel(self):

        if self._sourcemodel_type in ['NARROW_LINE_Gaussian', 'midIR_Gaussian', 'CO11-10_Gaussian']:

            source_size_pc = default_priors(self._sourcemodel_type)
            kwargs_source_model = {'source_model': 'GAUSSIAN'}
            source_samples = np.array(source_size_pc)
            param_names_source = ['source_size_pc']
            return source_size_pc, kwargs_source_model, source_samples, param_names_source

        elif self._sourcemodel_type == 'DOUBLE_NL_Gaussian':

            source_size_pc, dx, dy, amp_scale, size_scale = default_priors(self._sourcemodel_type)
            kwargs_source_model = {'source_model': 'DOUBLE_GAUSSIAN', 'dx': dx, 'dy': dy, 'amp_scale': amp_scale, 'size_scale': size_scale}
            kwargs_source_model.update(self._kwargs_source_model)
            source_samples = np.array([source_size_pc, dx, dy, amp_scale, size_scale])
            param_names_source = ['source_size_pc', 'dx', 'dy', 'amp_scale', 'size_scale']
            return source_size_pc, kwargs_source_model, source_samples, param_names_source

        else:
            raise Exception('other macromodels not yet implemented.')

    def set_zlens(self):

        if not hasattr(self, '_zlens_sampled'):
            if isinstance(self._zlens, float) or isinstance(self._zlens, int):
                self._zlens_sampled = self._zlens
            else:
                args = ['CUSTOM_PDF', self._zlens[0], self._zlens[1]]
                self._zlens_sampled = sample_from_prior(args)
            return np.round(self._zlens_sampled, 2)

        else:
            return np.round(self._zlens_sampled, 2)
