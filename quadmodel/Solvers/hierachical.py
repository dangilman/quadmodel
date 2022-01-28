from quadmodel.Solvers.settings import *
from quadmodel.Solvers.brute import BruteOptimization
import numpy as np
from quadmodel.util import interpolate_ray_paths_system

class HierarchicalOptimization(BruteOptimization):

    def __init__(self, lens_system, n_particles=None, simplex_n_iter=None, settings_class='default',
                 kwargs_settings_class={}):

        if settings_class == 'default':
            settings_class = HierarchicalSettingsDefault()
        elif settings_class == 'uldm':
            settings_class = HierarchicalSettingsULDM()
        elif settings_class == 'custom':
            settings_class = SettingsClass(**kwargs_settings_class)

        if n_particles is None:
            n_particles = settings_class.n_particles
        if simplex_n_iter is None:
            simplex_n_iter = settings_class.n_iterations

        self.settings = settings_class

        super(HierarchicalOptimization, self).__init__(lens_system, n_particles, simplex_n_iter)

    def optimize(self, data_to_fit, param_class_name, constrain_params, verbose=False,
                 threadCount=1, check_bad_fit=False):

        realization = self.realization_initial

        if realization is not None:
            foreground_realization, background_realization = realization.split_at_z(self.lens_system.zlens)
        else:
            foreground_realization, background_realization = None, None

        lens_model_full, kwargs_lens_final, foreground_realization_filtered, [source_x, source_y] = \
            self._fit_foreground(data_to_fit, foreground_realization, param_class_name, constrain_params,
                                 threadCount, verbose)

        if check_bad_fit:
            sx, sy = lens_model_full.ray_shooting(data_to_fit.x, data_to_fit.y, kwargs_lens_final)
            if np.std(sx) > 0.001 or np.std(sy) > 0.001:
                if verbose:
                    print('was not able to fit image positions, quitting...')
                return None, None, None

        lens_model_full, kwargs_lens_final, realization_filtered, \
        [source_x, source_y], reoptimized_realizations = \
            self._fit_background(data_to_fit, foreground_realization_filtered, background_realization,
                                 param_class_name, constrain_params, lens_model_full, source_x, source_y,
                                 threadCount, verbose)

        kwargs_return = {'reoptimized_realizations': reoptimized_realizations}
        return self.return_results(
            [source_x, source_y], kwargs_lens_final, lens_model_full,
            realization_filtered, kwargs_return
        )

    def _fit_foreground(self, data_to_fit, realization_foreground, param_class, constrain_params, threadCount, verbose=False):

        aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, \
        re_optimize_iteration, aperture_units = self.settings.foreground_settings

        N_foreground_halos_last = 0

        lens_model_full, kwargs_lens_final = self.lens_system.get_lensmodel(include_substructure=False)
        source_x, source_y = None, None

        z_mass_sheet_max = self.lens_system.zlens

        for run in range(0, self.settings.n_iterations_foreground):

            if run == 0:

                ray_x_interp, ray_y_interp = interpolate_ray_paths_system(data_to_fit.x, data_to_fit.y, self.lens_system,
                                                                   include_substructure=False)

            else:

                ray_x_interp, ray_y_interp = interpolate_ray_paths_system(data_to_fit.x, data_to_fit.y, self.lens_system,
                                                                   realization=realization_filtered)

            filter_kwargs = {'aperture_radius_front': window_sizes[run],
                             'aperture_radius_back': 0.,
                             'log_mass_allowed_in_aperture_front': aperture_masses[run],
                             'log_mass_allowed_in_aperture_back': 12,
                             'log_mass_allowed_global_front': globalmin_masses[run],
                             'log_mass_allowed_global_back': 10.,
                             'interpolated_x_angle': ray_x_interp,
                             'interpolated_y_angle': ray_y_interp,
                             'zmax': self.lens_system.zlens,
                             'aperture_units': aperture_units
                             }

            if run == 0:
                if realization_foreground is not None:
                    realization_filtered = realization_foreground.filter(**filter_kwargs)
                else:
                    realization_filtered = None

                if verbose: print('optimization ' + str(1))

            else:
                if realization_foreground is not None:
                    real = realization_foreground.filter(**filter_kwargs)
                    realization_filtered = real.join(realization_filtered)
                if verbose: print('optimization ' + str(run + 1))

            if realization_foreground is not None:
                N_foreground_halos = realization_filtered.number_of_halos_before_redshift(self.lens_system.zlens)
                N_subhalos = realization_filtered.number_of_halos_at_redshift(self.lens_system.zlens)
            else:
                N_foreground_halos = 0
                N_subhalos = 0

            self.lens_system.update_realization(realization_filtered)
            self.lens_system.clear_static_lensmodel()

            if verbose:
                print('aperture size: ', window_sizes[run])
                print('minimum mass in aperture: ', aperture_masses[run])
                print('minimum global mass: ', globalmin_masses[run])
                print('N foreground halos: ', N_foreground_halos)
                print('N subhalos: ', N_subhalos)

            do_optimization = True

            if run > 0:
                if N_foreground_halos == 0:
                    do_optimization = False
                if N_foreground_halos == N_foreground_halos_last:
                    do_optimization = False
            if optimize_iteration[run] is False:
                do_optimization = False

            if do_optimization:

                # kwargs_lens_final, lens_model_full, [source_x, source_y] = self.fit(data_to_fit, opt_routine, constrain_params,
                #    verbose=verbose, include_substructure=True, realization=realization_filtered,
                #    opt_kwargs={'re_optimize_scale': scale[run]}, re_optimize=re_optimize_iteration[run],
                #  particle_swarm=particle_swarm_reopt[run])

                kwargs_lens_final, lens_model_full, [source_x, source_y] = self.fit(data_to_fit, param_class, constrain_params, verbose=verbose,
                 include_substructure=True, realization=realization_filtered, re_optimize=re_optimize_iteration[run],
                                              re_optimize_scale=scale[run], particle_swarm=particle_swarm_reopt[run],
                                             threadCount=threadCount, z_mass_sheet_max=z_mass_sheet_max)

                N_foreground_halos_last = N_foreground_halos

                self.lens_system.clear_static_lensmodel()
                self.lens_system.update_realization(realization_filtered)
                self.lens_system.set_lensmodel_static(lens_model_full, kwargs_lens_final)
                self.lens_system.update_kwargs_macro(kwargs_lens_final)

            else:

                self.lens_system.clear_static_lensmodel()
                self.lens_system.update_realization(realization_filtered)
                lens_model_full, kwargs_lens_final = self.lens_system.get_lensmodel(
                    substructure_realization=realization_filtered)
                self.lens_system.set_lensmodel_static(lens_model_full, kwargs_lens_final)
                self.lens_system.update_kwargs_macro(kwargs_lens_final)
                N_foreground_halos_last = N_foreground_halos

        return lens_model_full, kwargs_lens_final, realization_filtered, [source_x, source_y]

    def _fit_background(self, data_to_fit, foreground_realization_filtered, realization_background,
                                 param_class, constrain_params, lens_model_full, source_x, source_y,
                                 threadCount, verbose):

        aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, \
        re_optimize_iteration, aperture_units = self.settings.background_settings

        N_background_halos_last = 0

        realization_filtered = None

        backx, backy, background_Tzs, background_zs, reoptimized_realizations = [], [], [], [], []

        for run in range(0, self.settings.n_iterations_background):

            if run == 0:

                ray_x_interp, ray_y_interp = interpolate_ray_paths_system(data_to_fit.x, data_to_fit.y, self.lens_system,
                                                                   realization=foreground_realization_filtered)
            else:

                ray_x_interp, ray_y_interp = interpolate_ray_paths_system(data_to_fit.x, data_to_fit.y, self.lens_system,
                                                                   realization=realization_filtered)

            filter_kwargs = {'aperture_radius_front': 10.,
                             'aperture_radius_back': window_sizes[run],
                             'log_mass_allowed_in_aperture_front': 10.,
                             'log_mass_allowed_in_aperture_back': aperture_masses[run],
                             'log_mass_allowed_global_front': 10.,
                             'log_mass_allowed_global_back': globalmin_masses[run],
                             'interpolated_x_angle': ray_x_interp,
                             'interpolated_y_angle': ray_y_interp,
                             'zmin': self.lens_system.zlens,
                             'aperture_units': aperture_units
                             }

            if run == 0:

                if foreground_realization_filtered is not None:
                    N_foreground_halos = foreground_realization_filtered.number_of_halos_before_redshift(self.lens_system.zlens)
                    N_subhalos = foreground_realization_filtered.number_of_halos_at_redshift(self.lens_system.zlens)
                    real = realization_background.filter(**filter_kwargs)
                    realization_filtered = foreground_realization_filtered.join(real)

                else:
                    N_foreground_halos = 0
                    N_subhalos = 0
                    realization_filtered = None

                if verbose: print('optimization ' + str(1))

            else:

                if verbose: print('optimization ' + str(run + 1))

                if realization_filtered is not None:
                    real = realization_background.filter(**filter_kwargs)
                    realization_filtered = realization_filtered.join(real)

            if realization_filtered is None:
                N_background_halos = 0
            else:

                N_background_halos = realization_filtered.number_of_halos_after_redshift(self.lens_system.zlens)

            self.lens_system.update_realization(realization_filtered)
            self.lens_system.clear_static_lensmodel()

            if realization_filtered is not None:
                ntotal_halos = realization_filtered.number_of_halos_after_redshift(0)

                assert ntotal_halos == N_foreground_halos + N_background_halos + N_subhalos

            if verbose:
                print('nhalos: ', N_background_halos+N_foreground_halos)
                print('aperture size: ', window_sizes[run])
                print('minimum mass in aperture: ', aperture_masses[run])
                print('minimum global mass: ', globalmin_masses[run])
                print('N foreground halos: ', N_foreground_halos)
                print('N subhalos: ', N_subhalos)
                print('N background halos: ', N_background_halos)

            do_optimization = True

            if run > 0:
                if N_background_halos == 0:
                    do_optimization = False
                if N_background_halos == N_background_halos_last:
                    do_optimization = False
            if optimize_iteration[run] is False:
                do_optimization = False

            if do_optimization:

                kwargs_lens_final, lens_model_full, [source_x, source_y] = self.fit(data_to_fit, param_class,
                                                                                    constrain_params, verbose=verbose,
                                                                                    include_substructure=True,
                                                                                    realization=realization_filtered,
                                                                                    re_optimize=re_optimize_iteration[
                                                                                        run],
                                                                                    re_optimize_scale=scale[run],
                                                                                    particle_swarm=particle_swarm_reopt[
                                                                                        run], threadCount=threadCount)

                reoptimized_realizations.append(realization_filtered)

                self.lens_system.clear_static_lensmodel()
                self.lens_system.update_realization(realization_filtered)
                self.lens_system.set_lensmodel_static(lens_model_full, kwargs_lens_final)
                self.lens_system.update_kwargs_macro(kwargs_lens_final)
                self.lens_system.update_source_centroid(source_x, source_y)

                N_background_halos_last = N_background_halos

            else:

                reoptimized_realizations.append(realization_filtered)
                self.lens_system.clear_static_lensmodel()
                self.lens_system.update_realization(realization_filtered)
                lens_model_full, kwargs_lens_final = self.lens_system.get_lensmodel(
                    include_substructure=True,
                    substructure_realization=realization_filtered
                )
                self.lens_system.set_lensmodel_static(lens_model_full, kwargs_lens_final)
                self.lens_system.update_kwargs_macro(kwargs_lens_final)
                self.lens_system.update_source_centroid(source_x, source_y)

        return lens_model_full, kwargs_lens_final, realization_filtered, \
               [source_x, source_y], reoptimized_realizations
