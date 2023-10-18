from quadmodel.Solvers.settings import *
from quadmodel.Solvers.brute import BruteOptimization
import numpy as np
from quadmodel.util import interpolate_ray_paths_system

class HierarchicalOptimization(BruteOptimization):

    def __init__(self, lens_system, n_particles=None, simplex_n_iter=None, settings_class='default',
                 kwargs_settings_class={}):

        if settings_class == 'default':
            settings_class = HierarchicalSettings(**kwargs_settings_class)
        elif settings_class == 'no_substructure':
            settings_class = HierarchicalSettingsNoSubstructure(**kwargs_settings_class)
        else:
            raise Exception('settings class not recognized')

        if n_particles is None:
            n_particles = settings_class.n_particles
        if simplex_n_iter is None:
            simplex_n_iter = settings_class.n_iterations

        self.settings = settings_class

        super(HierarchicalOptimization, self).__init__(lens_system, n_particles, simplex_n_iter)

    def optimize(self, data_to_fit, param_class_name, constrain_params, log_mlow_mass_sheet=7.0,
                 subtract_exact_mass_sheets=False, verbose=False):

        _realization_iteration = None
        lens_model_full, kwargs_lens_final = self.lens_system.get_lensmodel(include_substructure=False)
        source_x, source_y = None, None

        mass_global_front = self.settings.mass_global_front
        mass_global_back = self.settings.mass_global_back
        aperture_mass_list_front = self.settings.aperture_mass_list_front
        aperture_mass_list_back = self.settings.aperture_mass_list_back
        aperture_sizes_front = self.settings.aperture_sizes_front
        aperture_sizes_back = self.settings.aperture_sizes_back
        re_optimize_list = self.settings.re_optimize_list
        reoptimized_realizations = []

        for run in range(0, len(aperture_mass_list_back)):

            re_optimize = re_optimize_list[run]
            if run == 0:
                use_pso = True
                ray_x_interp, ray_y_interp = interpolate_ray_paths_system(data_to_fit.x, data_to_fit.y,
                                                                          self.lens_system,
                                                                          include_substructure=False)

            else:
                use_pso = False
                ray_x_interp, ray_y_interp = interpolate_ray_paths_system(data_to_fit.x, data_to_fit.y,
                                                                          self.lens_system,
                                                                          realization=_realization_iteration)

            filter_kwargs = {'aperture_radius_front': aperture_sizes_front[run],
                             'aperture_radius_back': aperture_sizes_back[run],
                             'log_mass_allowed_in_aperture_front': aperture_mass_list_front[run],
                             'log_mass_allowed_in_aperture_back': aperture_mass_list_back[run],
                             'log_mass_allowed_global_front': mass_global_front[run],
                             'log_mass_allowed_global_back': mass_global_back[run],
                             'interpolated_x_angle': ray_x_interp,
                             'interpolated_y_angle': ray_y_interp,
                             'zmax': self.lens_system.zsource,
                             'aperture_units': 'ANGLES'
                             }

            if run==0:
                _realization_iteration = self.realization_initial.filter(**filter_kwargs)
                if verbose: print('optimization ' + str(1))
            else:
                _real = self.realization_initial.filter(**filter_kwargs)
                _realization_iteration = _real.join(_realization_iteration)

            N_foreground_halos = _realization_iteration.number_of_halos_before_redshift(self.lens_system.zlens)
            N_subhalos = _realization_iteration.number_of_halos_at_redshift(self.lens_system.zlens)
            N_background = _realization_iteration.number_of_halos_before_redshift(self.lens_system.zsource) - N_subhalos - N_foreground_halos

            self.lens_system.update_realization(_realization_iteration)
            self.lens_system.clear_static_lensmodel()

            if verbose:
                print('aperture size (front): ', aperture_sizes_front[run])
                print('aperture size (back): ', aperture_sizes_back[run])
                print('log10 minimum mass anywhere (front): ', mass_global_front[run])
                print('log10 minimum mass anywhere (back): ', mass_global_back[run])
                print('log10 minimum mass in aperture (front): ', aperture_mass_list_front[run])
                print('log10 minimum mass in aperture (back): ', aperture_mass_list_back[run])
                print('N foreground halos: ', N_foreground_halos)
                print('N subhalos: ', N_subhalos)
                print('N background halos: ', N_background)

            kwargs_lens_final, lens_model_full, [source_x, source_y] = self.fit(data_to_fit, param_class_name,
                                                                                    constrain_params, verbose=verbose,
                                                                                    include_substructure=True,
                                                                                    realization=_realization_iteration,
                                                                                    re_optimize=re_optimize,
                                                                                    re_optimize_scale=1.0,
                                                                                    particle_swarm=use_pso,
                                                                                    z_mass_sheet_max=self.lens_system.zsource,
                                                                                    log_mlow_mass_sheet=log_mlow_mass_sheet,
                                                                                    subtract_exact_mass_sheets=subtract_exact_mass_sheets)


            self.lens_system.clear_static_lensmodel()
            self.lens_system.update_realization(_realization_iteration)
            self.lens_system.set_lensmodel_static(lens_model_full, kwargs_lens_final)
            self.lens_system.update_kwargs_macro(kwargs_lens_final)
            reoptimized_realizations.append(_realization_iteration)

        kwargs_return = {'reoptimized_realizations': reoptimized_realizations}

        return self.return_results(
            [source_x, source_y], kwargs_lens_final, lens_model_full,
            _realization_iteration, kwargs_return
        )

