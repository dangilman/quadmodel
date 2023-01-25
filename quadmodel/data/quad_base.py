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
    elif param == 'multipole_amplitude_m3':
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

        self._zlens_init = zlens
        self.set_zlens(reset=True)
        self.zlens = self._zlens
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

    @classmethod
    def from_hst_data(cls, hst_data_class,
                      sourcemodel_type='midIR_Gaussian',
                      macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE',
                      keep_flux_ratio_index=[0, 1, 2],
                      uncertainty_in_magnifications=True,
                      kwargs_source_model={},
                      kwargs_macromodel={}):

        """

        :param hst_data_class: an instance of the HSTData class
        :param sourcemodel_type: possibilities include
        1) midIR_Gaussian (0.5-10 pc fwhm)
        2) NARROW_LINE_Gaussian (25-60 pc fwhm)
        3) CO11-10_Gaussian (5-15 pc fwhm)
        4) DOUBLE_NL_Gaussian
        two components, one with 25-80 pc fwhm and the other
        rescaled by a random factor 0.25 - 1 and shifted vertically by 0-0.01 arcsec (source plane)
        this is implemented specifically for RXJ-1131
        :param macromodel_type: the type of macromodel profile, possibilities include
        1)
        :param keep_flux_ratio_index:
        :param uncertainty_in_magnifications:
        :return:
        """
        zlens = hst_data_class.zlens
        zsource = hst_data_class.zsource
        x_image = hst_data_class.x
        y_image = hst_data_class.y
        magnifications = hst_data_class.m
        magnification_uncertainties = hst_data_class.delta_m
        astrometric_uncertainties = hst_data_class.delta_xy
        return Quad(zlens, zsource, x_image, y_image, magnifications, magnification_uncertainties, astrometric_uncertainties,
                 sourcemodel_type, kwargs_source_model, macromodel_type, kwargs_macromodel, keep_flux_ratio_index,
                 uncertainty_in_magnifications)


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

    def generate_macromodel(self, m3_amplitude=None, m4_amplitude=None):
        """
        Used only if lens-specific data class has no satellite galaxies; for systems with satellites, add them in the
        lens-specific data class and override this method
        :return: An instance of MacroModel class, a dict of parameters relevant for the class (e.g. external shear strength),
        a default optimization routine to fit the lens (can be changed), random parameters sampled to
        create the macromodel class, the names of sampled parameters
        """
        if self._macromodel_type == 'EPL_FIXED_SHEAR_MULTIPOLE':

            shear_min, shear_max = self._kwargs_macromodel['shear_amplitude_min'], \
                                   self._kwargs_macromodel['shear_amplitude_max']
            gamma_macro = default_priors('gamma_macro')
            shear_amplitude = sample_from_prior(['UNIFORM', shear_min, shear_max])
            optimization_routine = 'fixed_shear_powerlaw_multipole'
            constrain_params = {'shear': shear_amplitude}
            if m4_amplitude is None:
                multipole_amplitude = default_priors('multipole_amplitude')
            else:
                multipole_amplitude = np.random.normal(0, m4_amplitude)
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole
            model = EPLShearMultipole(self.zlens, gamma_macro, shear_amplitude, multipole_amplitude, self.approx_einstein_radius,
                                      0.0, 0.0, 0.2, 0.1)
            params_sampled = np.array([multipole_amplitude, gamma_macro, shear_amplitude])
            param_names_macro = ['a4', 'gamma', 'gamma_ext']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self._macromodel_type == 'EPL_FIXED_SHEAR_MULTIPOLE_34':

            shear_min, shear_max = self._kwargs_macromodel['shear_amplitude_min'], \
                                   self._kwargs_macromodel['shear_amplitude_max']
            gamma_macro = default_priors('gamma_macro')
            shear_amplitude = sample_from_prior(['UNIFORM', shear_min, shear_max])
            optimization_routine = 'fixed_shear_powerlaw_multipole_34'
            m3_orientation = np.random.uniform(0, 2*np.pi)
            constrain_params = {'shear': shear_amplitude, 'delta_phi_m3': m3_orientation}

            if m4_amplitude is None:
                multipole_amplitude_m4 = default_priors('multipole_amplitude')
            else:
                multipole_amplitude_m4 = np.random.normal(0, m4_amplitude)
            if m3_amplitude is None:
                multipole_amplitude_m3 = default_priors('multipole_amplitude_m3')
            else:
                multipole_amplitude_m3 = np.random.normal(0, m3_amplitude)

            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole_34
            model = EPLShearMultipole_34(self.zlens, gamma_macro, shear_amplitude, multipole_amplitude_m4,
                                      multipole_amplitude_m3,
                                      self.approx_einstein_radius,
                                      0.0, 0.0, 0.2, 0.1)

            params_sampled = np.array([multipole_amplitude_m3, multipole_amplitude_m4, m3_orientation, gamma_macro, shear_amplitude])
            param_names_macro = ['a3', 'a4', 'd_phi_a3', 'gamma', 'gamma_ext']
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
            params_sampled = np.array([gamma_macro, shear_amplitude])
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
            params_sampled = np.array([multipole_amplitude, gamma_macro])
            param_names_macro = ['a4', 'gamma']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self._macromodel_type == 'EPL_FREE_SHEAR_MULTIPOLE_34':

            # doesn't matter how this is initialized
            random_shear_init = np.random.uniform(0.05, 0.25)
            gamma_macro = default_priors('gamma_macro')
            optimization_routine = 'free_shear_powerlaw_multipole_34'
            m3_orientation = np.random.uniform(0, 2*np.pi)
            constrain_params = {'delta_phi_m3': m3_orientation}
            if m4_amplitude is None:
                multipole_amplitude_m4 = default_priors('multipole_amplitude')
            else:
                multipole_amplitude_m4 = np.random.normal(0, m4_amplitude)
            if m3_amplitude is None:
                multipole_amplitude_m3 = default_priors('multipole_amplitude_m3')
            else:
                multipole_amplitude_m3 = np.random.normal(0, m3_amplitude)
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole_34
            model = EPLShearMultipole_34(self.zlens, gamma_macro, random_shear_init, multipole_amplitude_m4,
                                      multipole_amplitude_m3,
                                      self.approx_einstein_radius,
                                      0.0, 0.0, 0.2, 0.1)
            params_sampled = np.array([multipole_amplitude_m3, multipole_amplitude_m4, m3_orientation, gamma_macro])
            param_names_macro = ['a3', 'a4', 'd_phi_a3', 'gamma']
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

    def set_zlens(self, reset=False):

        if reset:
            self._zlens = None

        if self._zlens is None:

            if isinstance(self._zlens_init, float) or isinstance(self._zlens_init, int):
                self._zlens = self._zlens_init

            else:
                args = ['CUSTOM_PDF', self._zlens_init[0], self._zlens_init[1]]
                zlens_sampled = sample_from_prior(args)
                self._zlens = np.round(zlens_sampled, 2)
