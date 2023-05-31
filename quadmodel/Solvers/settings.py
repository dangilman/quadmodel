class HierarchicalSettings(object):

    def __init__(self, log_m_cut=7.0, aperture_size_small_front=0.3,
                 aperture_size_small_back=0.2):
        """
        This class renders halos everywhere with m>10^log_m_cut,
        and only generates halos smaller than this near a lensed image
        :param log_m_cut:
        """
        self.mass_global_front = [log_m_cut, log_m_cut, log_m_cut]
        self.mass_global_back = [12, log_m_cut, log_m_cut]
        self.aperture_mass_list_front = [log_m_cut, -10, -10]
        self.aperture_mass_list_back = [12, log_m_cut, -10]
        self.aperture_sizes_front = [100, aperture_size_small_front, aperture_size_small_front]
        self.aperture_sizes_back = [100, 100, aperture_size_small_back]
        self.re_optimize_list = [True, True, True]

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 350

class HierarchicalSettingsNoSubstructure(object):

    def __init__(self, *args, **kargs):
        """
        This class renders halos everywhere with m>10^log_m_cut,
        and only generates halos smaller than this near a lensed image
        :param log_m_cut:
        """
        self.mass_global_front = [100]
        self.mass_global_back = [100]
        self.aperture_mass_list_front = [100]
        self.aperture_mass_list_back = [100]
        self.aperture_sizes_front = [100]
        self.aperture_sizes_back = [100]
        self.re_optimize_list = [True]

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 50
