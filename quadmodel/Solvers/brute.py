from quadmodel.Solvers.optimization_param_managers import PowerLawFixedShear, \
    PowerLawFixedShearMultipole, PowerLawFreeShear, PowerLawFreeShearMultipole, PowerLawFixedShearMultipole_34, \
    PowerLawFreeShearMultipole_34
from lenstronomy.LensModel.QuadOptimizer.optimizer import Optimizer
from quadmodel.Solvers.base import OptimizationBase

class BruteOptimization(OptimizationBase):

    def __init__(self, lens_system, n_particles=None, simplex_n_iter=None):

        """
        This class executes a lens model fit to the data using the Optimizer class in lenstronomy

        More sophisticated optimization routines are wrappers around this main class

        :param lens_system: the lens system class to optimize (instance of lenstronomywrapper.LensSystem.quad_lens
        :param n_particles: the number of particle swarm particles to use
        :param simplex_n_iter: the number of iterations for the downhill simplex routine
        :param reoptimize: whether to start the particle swarm particles close together if the initial
        guess for the lens model is close to the `true model'
        :param log_mass_sheet_front: the log(mass) used when subtracting foreground convergence sheets from the lens mdoel
        :param log_mass_sheet_back: same as ^ but for the background lens planes
        """

        settings = BruteSettingsDefault()

        if n_particles is None:
            n_particles = settings.n_particles
        if simplex_n_iter is None:
            simplex_n_iter = settings.n_iterations

        self.n_particles = n_particles
        self.n_iterations = simplex_n_iter

        self.realization_initial = lens_system.realization

        super(BruteOptimization, self).__init__(lens_system)

    def optimize(self, data_to_fit, param_class, constrain_params, verbose=False,
                 include_substructure=True, kwargs_optimizer={}):

        kwargs_lens_final, lens_model_full, [source_x, source_y] = self.fit(data_to_fit, param_class,
                                      constrain_params, verbose, include_substructure, **kwargs_optimizer)

        self.lens_system.clear_static_lensmodel()
        self.lens_system.set_lensmodel_static(lens_model_full, kwargs_lens_final)
        self.lens_system.update_kwargs_macro(kwargs_lens_final)

        return self.return_results(
            [source_x, source_y], kwargs_lens_final, lens_model_full,
            self.realization_initial, None
        )

    @staticmethod
    def set_param_class(param_class_name, constrain_params):

        if param_class_name == 'free_shear_powerlaw':
            return PowerLawFreeShear, None
        elif param_class_name == 'fixed_shear_powerlaw':
            return PowerLawFixedShear, [constrain_params['shear']]
        elif param_class_name == 'free_shear_powerlaw_multipole':
            return PowerLawFreeShearMultipole, None
        elif param_class_name == 'fixed_shear_powerlaw_multipole':
            return PowerLawFixedShearMultipole, [constrain_params['shear']]
        elif param_class_name == 'fixed_shear_powerlaw_multipole_34':
            return PowerLawFixedShearMultipole_34, [constrain_params['shear'], constrain_params['delta_phi_m3']]
        elif param_class_name == 'free_shear_powerlaw_multipole_34':
            return PowerLawFreeShearMultipole_34, [constrain_params['delta_phi_m3']]

        else:
            raise Exception('did not recognize param_class_name = '+param_class_name)

    def fit(self, data_to_fit, param_class, constrain_params, verbose=False,
                 include_substructure=True, realization=None, re_optimize=False,
            re_optimize_scale=1., particle_swarm=True, n_particles=None, pso_convergence_mean=80000,
            threadCount=1, z_mass_sheet_max=None):

        if n_particles is None:
            n_particles = self.n_particles

        param_class, args_param_class = self.set_param_class(param_class, constrain_params)

        run_kwargs = {'x_image': data_to_fit.x, 'y_image': data_to_fit.y, 'z_lens': self.lens_system.zlens,
                      'z_source': self.lens_system.zsource, 'astropy_instance': self.lens_system.astropy,
                 'particle_swarm': particle_swarm, 're_optimize':re_optimize, 're_optimize_scale': re_optimize_scale,
                      'pso_convergence_mean': pso_convergence_mean,'tol_source':1e-6,
                      'foreground_rays': None, 'simplex_n_iterations': self.n_iterations}

        kwargs_lens_final, ray_shooting_class, source = self._fit(run_kwargs, param_class, args_param_class,
                                    include_substructure, n_particles, realization, verbose, threadCount,
                                                                  z_mass_sheet_max)

        return kwargs_lens_final, ray_shooting_class, source

    def _fit(self, run_kwargs, param_class, args_param_class, include_substructure, nparticles,
            realization, verbose, threadCount=1, z_mass_sheet_max=None):

        """
        run_kwargs: {'optimizer_routine', 'constrain_params', 'simplex_n_iter'}
        filter_kwargs: {'re_optimize', 'particle_swarm'}
        """

        lens_model_list, redshift_list, kwargs_lens, numerical_alpha_class = \
            self.lens_system._get_lenstronomy_args(include_substructure, realization=realization,
                                                   z_mass_sheet_max=z_mass_sheet_max)

        if args_param_class is None:
            param_class = param_class(kwargs_lens)
        else:
            param_class = param_class(kwargs_lens, *args_param_class)

        run_kwargs['lens_model_list'] = lens_model_list
        run_kwargs['redshift_list'] = redshift_list
        run_kwargs['numerical_alpha_class'] = numerical_alpha_class
        run_kwargs['parameter_class'] = param_class

        opt = Optimizer(**run_kwargs)

        kwargs_lens_final, [source_x, source_y] = opt.optimize(nparticles, self.n_iterations,
                                                               verbose=verbose, threadCount=threadCount)

        ray_shooting_class = opt.fast_rayshooting.lensModel

        return kwargs_lens_final, ray_shooting_class, [source_x, source_y]

class BruteSettingsDefault(object):

    @property
    def reoptimize(self):
        return False

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 400
