class HierarchicalSettingsDefault(object):

    """
    Good for dealing with dark matter halos between 10^5 - 10^10 M_sun
    with small source sizes <10 pc
    """

    def __init__(self):
        self.set_aperture_units('ANGLES')

    def set_aperture_units(self, units):
        self._aperture_units = units

    @property
    def aperture_units(self):
        return self._aperture_units

    @property
    def log_mass_cut_global(self):
        return 7.

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 350

    @property
    def n_iterations_background(self):
        return 2

    @property
    def n_iterations_foreground(self):
        return 2

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0.]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global] * 2
        # window size
        window_sizes = [100, 0.2]
        # controls starting points for re-optimizations
        scale = [1, 0.1, 0.1]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, \
               re_optimize_iteration, self.aperture_units

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global] * 2
        # window size
        window_sizes = [100, 0.2]
        # controls starting points for re-optimizations
        scale = [1, 0.5]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, \
               re_optimize_iteration, self.aperture_units

class SettingsClass(object):

    def __init__(self, foreground_kwargs, background_kwargs, n_particles=30, n_iterations=350, aperture_units='ANGLES'):

        self._names = ['aperture_masses', 'globalmin_masses', 'window_sizes', 'scale', 'particle_swarm_reopt',
                 'optimize_iteration', 're_optimize_iteration']

        self.n_particles = n_particles
        self.n_iterations = n_iterations

        self.n_iterations_foreground = self._set_foreground(foreground_kwargs)
        self.n_iterations_background = self._set_background(background_kwargs)

        self.set_aperture_units(aperture_units)

    def set_aperture_units(self, units):
        self._aperture_units = units

    @property
    def aperture_units(self):
        return self._aperture_units

    @property
    def foreground_settings(self):
        out = self._foreground
        return out[0], out[1], out[2], out[3], out[4], out[5], out[6], self.aperture_units

    @property
    def background_settings(self):
        out = self._background
        return out[0], out[1], out[2], out[3], out[4], out[5], out[6], self.aperture_units

    def _set_foreground(self, kwargs):

        self._check(kwargs)
        self._foreground = [kwargs[name] for name in self._names]

        return len(kwargs['aperture_masses'])

    def _set_background(self, kwargs):

        self._check(kwargs)
        self._background = [kwargs[name] for name in self._names]
        return len(kwargs['aperture_masses'])

    def _check(self, kwargs):

        L = len(kwargs['aperture_masses'])
        for name in self._names:
            assert L == len(kwargs[name])
