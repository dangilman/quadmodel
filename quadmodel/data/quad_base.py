import numpy as np
from quadmodel.util import approx_theta_E
from quadmodel.inference.sample_prior import sample_from_prior
from scipy.optimize import minimize
from copy import deepcopy


def default_priors(param):

    if param == 'gamma_macro':
        gamma_min, gamma_max = 1.9, 2.2
        return np.random.uniform(gamma_min, gamma_max)
    elif param == 'multipole_amplitude_m4':
        am_mean, am_sigma = 0.0, 0.01
        return np.random.normal(am_mean, am_sigma)
    elif param == 'multipole_amplitude_m3':
        am_mean, am_sigma = 0.0, 0.005
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
    elif param == 'EFFECTIVE_POINT_SOURCE':
        source_fwhm_pc = np.random.uniform(0.01, 0.02)
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
                 uncertainty_in_magnifications=True, sample_zlens_pdf=False):

        self._sample_zlens_pdf = sample_zlens_pdf
        self._zlens_init = zlens
        self.set_zlens(reset=True)
        self.zsource = zsource
        self.x = np.array(x_image)
        self.y = np.array(y_image)
        self.m = np.array(magnifications)
        self.delta_m = np.array(magnification_uncertainties)
        self.delta_xy = np.array(astrometric_uncertainties)
        self.approx_einstein_radius = approx_theta_E(self.x, self.y)

        self.macromodel_type = macromodel_type
        self.kwargs_macromodel = kwargs_macromodel
        self.kwargs_source_model = kwargs_source_model

        self.uncertainty_in_magnifications = uncertainty_in_magnifications
        self.sourcemodel_type = sourcemodel_type
        self.keep_flux_ratio_index = keep_flux_ratio_index

    @property
    def zlens(self):
        """

        :return: main defelector redshift
        """
        return self._zlens

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

    def satellite_galaxy(self, *args, **kwargs):
        """
        The default is no satellites or individual components
        :return:
        """
        satellite_list = []
        satellite_params = np.array([])
        satellite_param_names = []
        return satellite_list, satellite_params, satellite_param_names

    def generate_macromodel(self, **kwargs_main):
        """
        Used only if lens-specific data class has no satellite galaxies; for systems with satellites, add them in the
        lens-specific data class and override this method
        :return:
        """

        model, constrain_params, optimization_routine, params_sampled, param_names_macro = self._generate_macromodel_main(
            **kwargs_main)
        model_satellite, params_satellite, param_names_satellite = self.satellite_galaxy()
        model.add_satellite(model_satellite)
        params_sampled = np.append(params_sampled, params_satellite)
        param_names_macro += param_names_satellite
        return model, constrain_params, optimization_routine, params_sampled, param_names_macro

    def _generate_macromodel_main(self, m3_amplitude_prior=None, m4_amplitude_prior=None, gamma_macro_prior=None,
                                  shear_strength_prior=None, kwargs_lens_macro_init=None, center_x_prior=None,
                                  center_y_prior=None, e1_prior=None, e2_prior=None):
        """
        Used only if lens-specific data class has no satellite galaxies; for systems with satellites, add them in the
        lens-specific data class and override this method
        :return: An instance of MacroModel class, a dict of parameters relevant for the class (e.g. external shear strength),
        a default optimization routine to fit the lens (can be changed), random parameters sampled to
        create the macromodel class, the names of sampled parameters
        """

        shear_amplitude = None
        gamma_macro = None
        multipole_amplitude_m4 = None
        multipole_amplitude_m3 = None
        random_thetaE = np.random.uniform(-0.2, 0.2) + self.approx_einstein_radius

        if center_x_prior is not None:
            random_center_x = center_x_prior[0](center_x_prior[1], center_x_prior[2])
        else:
            random_center_x = np.random.normal(0.0, 0.1)
        if center_y_prior is not None:
            random_center_y = center_y_prior[0](center_y_prior[1], center_y_prior[2])
        else:
            random_center_y = np.random.normal(0.0, 0.1)
        if e1_prior is not None:
            random_e1 = e1_prior[0](e1_prior[1], e1_prior[2])
        else:
            random_e1 = np.random.uniform(-0.25, 0.25)
        if e2_prior is not None:
            random_e2 = e2_prior[0](e2_prior[1], e2_prior[2])
        else:
            random_e2 = np.random.uniform(-0.25, 0.25)

        if kwargs_lens_macro_init is None:
            kwargs_lens_macro_init = {'theta_E': random_thetaE,
                                      'center_x': random_center_x,
                                      'center_y': random_center_y,
                                      'e1': random_e1,
                                      'e2': random_e2}
        elif callable(kwargs_lens_macro_init):
            kwargs_lens_macro_init = kwargs_lens_macro_init()

            if 'gamma1' in kwargs_lens_macro_init.keys():
                assert 'gamma2' in kwargs_lens_macro_init.keys()
                shear_amplitude = np.sqrt(kwargs_lens_macro_init['gamma1'] ** 2 +
                                          kwargs_lens_macro_init['gamma2'] ** 2)
                del kwargs_lens_macro_init['gamma1']
                del kwargs_lens_macro_init['gamma2']
            if 'a_m_4' in kwargs_lens_macro_init.keys():
                multipole_amplitude_m4 = kwargs_lens_macro_init['a_m_4']
                del kwargs_lens_macro_init['a_m_4']
            if 'a_m_3' in kwargs_lens_macro_init.keys():
                multipole_amplitude_m3 = kwargs_lens_macro_init['a_m_3']
                del kwargs_lens_macro_init['a_m_3']
            if 'gamma_macro' in kwargs_lens_macro_init.keys():
                gamma_macro = kwargs_lens_macro_init['gamma_macro']
                del kwargs_lens_macro_init['gamma_macro']

        if gamma_macro is None:
            if gamma_macro_prior is None:
                gamma_macro = default_priors('gamma_macro')
            else:
                gamma_macro = gamma_macro_prior[0](gamma_macro_prior[1], gamma_macro_prior[2])

        if multipole_amplitude_m4 is None:
            if m4_amplitude_prior is None:
                multipole_amplitude_m4 = default_priors('multipole_amplitude_m4')
            else:
                multipole_amplitude_m4 = m4_amplitude_prior[0](m4_amplitude_prior[1], m4_amplitude_prior[2])

        if multipole_amplitude_m3 is None:
            if m3_amplitude_prior is None:
                multipole_amplitude_m3 = default_priors('multipole_amplitude_m3')
            else:
                multipole_amplitude_m3 = m3_amplitude_prior[0](m3_amplitude_prior[1], m3_amplitude_prior[2])

        if shear_amplitude is None:
            if shear_strength_prior is None:
                shear_amplitude = 10**np.random.uniform(-3, np.log10(0.3))
            else:
                shear_amplitude = shear_strength_prior[0](shear_strength_prior[1], shear_strength_prior[2])
        for required_param, param in zip(['theta_E', 'center_x', 'center_y', 'e1', 'e2'], [random_thetaE, random_center_x, random_center_y, random_e1, random_e2]):
            if required_param not in kwargs_lens_macro_init.keys():
                kwargs_lens_macro_init[required_param] = param

        if self.macromodel_type == 'EPL_FIXED_SHEAR_MULTIPOLE':
            optimization_routine = 'fixed_shear_powerlaw_multipole'
            constrain_params = {'shear': shear_amplitude}
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole
            model = EPLShearMultipole(self.zlens, gamma_macro, shear_amplitude, multipole_amplitude_m4,
                                          **kwargs_lens_macro_init)
            params_sampled = np.array([multipole_amplitude_m4, gamma_macro, shear_amplitude])
            param_names_macro = ['a_m_4', 'gamma_macro', 'gamma_ext']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self.macromodel_type == 'EPL_FIXED_SHEAR_MULTIPOLE_34':
            optimization_routine = 'fixed_shear_powerlaw_multipole_34'
            m3_orientation = np.random.uniform(0, 2*np.pi)
            constrain_params = {'shear': shear_amplitude, 'delta_phi_m3': m3_orientation}
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole_34
            model = EPLShearMultipole_34(self.zlens, gamma_macro, shear_amplitude, multipole_amplitude_m4,
                                      multipole_amplitude_m3, **kwargs_lens_macro_init)
            params_sampled = np.array([multipole_amplitude_m3, multipole_amplitude_m4, m3_orientation, gamma_macro, shear_amplitude])
            param_names_macro = ['a_m_3', 'a_m_4', 'd_phi_a3', 'gamma_macro', 'gamma_ext']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self.macromodel_type == 'EPL_FIXED_SHEAR':
            optimization_routine = 'fixed_shear_powerlaw'
            constrain_params = {'shear': shear_amplitude}
            from quadmodel.deflector_models.preset_macromodels import EPLShear
            model = EPLShear(self.zlens, gamma_macro, shear_amplitude, **kwargs_lens_macro_init)
            params_sampled = np.array([gamma_macro, shear_amplitude])
            param_names_macro = ['gamma_macro', 'gamma_ext']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self.macromodel_type == 'EPL_FREE_SHEAR_MULTIPOLE':
            optimization_routine = 'free_shear_powerlaw_multipole'
            constrain_params = None
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole
            model = EPLShearMultipole(self.zlens, gamma_macro, shear_amplitude, multipole_amplitude_m4,
                                      **kwargs_lens_macro_init)
            params_sampled = np.array([multipole_amplitude_m4, gamma_macro])
            param_names_macro = ['a_m_4', 'gamma_macro']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self.macromodel_type == 'EPL_FREE_SHEAR_MULTIPOLE_34':
            optimization_routine = 'free_shear_powerlaw_multipole_34'
            m3_orientation = np.random.uniform(0, np.pi/3)
            constrain_params = {'delta_phi_m3': m3_orientation}
            from quadmodel.deflector_models.preset_macromodels import EPLShearMultipole_34
            model = EPLShearMultipole_34(self.zlens, gamma_macro, shear_amplitude, multipole_amplitude_m4,
                                      multipole_amplitude_m3, **kwargs_lens_macro_init)
            params_sampled = np.array([multipole_amplitude_m3, multipole_amplitude_m4, m3_orientation, gamma_macro])
            param_names_macro = ['a_m_3', 'a_m_4', 'd_phi_a3', 'gamma_macro']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        elif self.macromodel_type == 'EPL_FREE_SHEAR':
            optimization_routine = 'free_shear_powerlaw'
            constrain_params = None
            from quadmodel.deflector_models.preset_macromodels import EPLShear
            model = EPLShear(self.zlens, gamma_macro, shear_amplitude, **kwargs_lens_macro_init)
            params_sampled = np.array([gamma_macro])
            param_names_macro = ['gamma_macro']
            return model, constrain_params, optimization_routine, params_sampled, param_names_macro

        else:
            raise Exception('other macromodels not yet implemented.')

    def generate_sourcemodel(self):

        if self.sourcemodel_type in ['NARROW_LINE_Gaussian', 'midIR_Gaussian', 'CO11-10_Gaussian',
                                     'EFFECTIVE_POINT_SOURCE']:

            source_size_pc = default_priors(self.sourcemodel_type)
            kwargs_source_model = {'source_model': 'GAUSSIAN'}
            source_samples = np.array(source_size_pc)
            param_names_source = ['source_size_pc']
            return source_size_pc, kwargs_source_model, source_samples, param_names_source

        elif self.sourcemodel_type == 'DOUBLE_NL_Gaussian':

            source_size_pc, dx, dy, amp_scale, size_scale = default_priors(self.sourcemodel_type)
            kwargs_source_model = {'source_model': 'DOUBLE_GAUSSIAN', 'dx': dx, 'dy': dy, 'amp_scale': amp_scale, 'size_scale': size_scale}
            kwargs_source_model.update(self.kwargs_source_model)
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
                self._zlens = deepcopy(self._zlens_init)

            else:
                if self._sample_zlens_pdf:
                    args = ['CUSTOM_PDF', self._zlens_init[0], self._zlens_init[1]]
                    zlens_sampled = sample_from_prior(args)
                    self._zlens = np.round(zlens_sampled, 2)
                else:
                    values, weights = self._zlens_init[0], self._zlens_init[1]
                    weighted_values = np.array(values) * np.array(weights) / np.sum(weights)
                    median_zlens = np.round(np.sum(weighted_values), 2)
                    self._zlens = median_zlens
