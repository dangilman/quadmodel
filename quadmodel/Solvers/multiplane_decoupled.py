from lenstronomy.LensModel.QuadOptimizer.optimizer import Optimizer
from quadmodel.Solvers.base import OptimizationBase
from lenstronomy.LensModel.Util.decouple_multi_plane_util import *

class DecoupledMultiPlane(OptimizationBase):

    def __init__(self, lens_system, lens_model_init, kwargs_lens_init, index_lens_split,
                 n_particles=None, simplex_n_iter=None):

        """
        This class executes a lens model fit using the decoupled multi-plane approximation

        :param lens_system: the lens system class to optimize (instance of lenstronomywrapper.LensSystem.quad_lens
        :param n_particles: the number of particle swarm particles to use
        :param simplex_n_iter: the number of iterations for the downhill simplex routine
        """

        if n_particles is None:
            n_particles = 30
        if simplex_n_iter is None:
            simplex_n_iter = 400

        self.n_particles = n_particles
        self.n_iterations = simplex_n_iter
        self.realization_initial = None
        self._index_lens_split = None
        self._lens_model_init = lens_model_init
        self._kwargs_lens_init = kwargs_lens_init
        self.index_lens_split = index_lens_split
        super(DecoupledMultiPlane, self).__init__(lens_system)

    def optimize(self, data_to_fit, param_class_name, constrain_params, particle_swarm=False, verbose=False):

        param_class, args_param_class = self.set_param_class(param_class_name, constrain_params)
        if args_param_class is None:
            param_class = param_class(self._kwargs_lens_init)
        else:
            param_class = param_class(self._kwargs_lens_init, *args_param_class)
        opt = Optimizer.decoupled_multiplane(data_to_fit.x,
                                                   data_to_fit.y,
                                                   self._lens_model_init,
                                                   self._kwargs_lens_init,
                                                   self.index_lens_split,
                                                   param_class,
                                                   particle_swarm=particle_swarm)
        kwargs_lens_final, [source_x, source_y] = opt.optimize(self.n_particles,
                                                               self.n_iterations,
                                                               verbose=verbose)

        lens_model_full = opt.ray_shooting_class
        new_macromodel = self.lens_system.macromodel.split(self.index_lens_split)
        self.lens_system.macromodel = new_macromodel
        self.lens_system.clear_static_lensmodel()
        self.lens_system.set_lensmodel_static(lens_model_full, kwargs_lens_final)
        self.lens_system.update_kwargs_macro(kwargs_lens_final)
        return self.return_results(
            [source_x, source_y], kwargs_lens_final, lens_model_full,
            None, None
        )
