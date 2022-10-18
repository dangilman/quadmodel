from lenstronomy.LensModel.QuadOptimizer.param_manager import PowerLawFixedShear, \
    PowerLawFixedShearMultipole, PowerLawFreeShear, PowerLawFreeShearMultipole, PowerLawParamManager
from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian
from lenstronomy.Util.param_util import ellipticity2phi_q
import numpy as np

class PowerLawFixedShearMultipole_34(PowerLawFixedShear):
    """
    This class implements a fit of EPL + external shear + two multipole terms (m=3 and m=4) with every parameter except the
    power law slope, shear strength, and multipole moments free to vary. The mass centroid and orientation of the
    m=4 multipole term are fixed to that of the EPL profile, and the orientation of m=3 multiple term is
    fixed to some user-defined angle.
    """

    def __init__(self, kwargs_lens_init, shear_strength, delta_phi_m3):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        :param shear_strength: the strenght of the external shear to be kept fixed
        :param the orientation of the m=3 multipole relative to the EPL position angle
        """
        self._delta_phi_m3 = delta_phi_m3
        super(PowerLawFixedShearMultipole_34, self).__init__(kwargs_lens_init, shear_strength)

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 4 because the first four lens models
        are EPL, SHEAR, MULTIPOLE, and MULTIPOLE, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 4

    def args_to_kwargs(self, args):

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']

        # handle the EPL profile
        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        self.kwargs_lens[0] = kwargs_epl

        # determine the orientation of external shear
        phi, _ = shear_cartesian2polar(g1, g2)
        gamma1, gamma2 = shear_polar2cartesian(phi, self._shear_strength)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}
        self.kwargs_lens[1] = kwargs_shear

        # fix m=4 multipole centroid to EPL centroid
        self.kwargs_lens[2]['center_x'] = center_x
        self.kwargs_lens[2]['center_y'] = center_y
        # fix m=4 multipole orientation to EPL orientation
        phi, _ = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[2]['phi_m'] = phi

        # fix m=3 multipole centroid to EPL centroid
        self.kwargs_lens[3]['center_x'] = center_x
        self.kwargs_lens[3]['center_y'] = center_y
        # fix m=3 multipole orientation to EPL orientation
        phi, _ = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[3]['phi_m'] = phi + self._delta_phi_m3

        return self.kwargs_lens

class PowerLawFreeShearMultipole_34(PowerLawParamManager):

    """
    This class implements a fit of EPL + external shear + a multipole term with every parameter except the
    power law slope and multipole moment free to vary. The mass centroid and orientation of the multipole term are
    fixed to that of the EPL profile

    """

    def __init__(self, kwargs_lens_init, delta_phi_m3):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        :param shear_strength: the strenght of the external shear to be kept fixed
        :param the orientation of the m=3 multipole relative to the EPL position angle
        """
        self._delta_phi_m3 = delta_phi_m3
        super(PowerLawFreeShearMultipole_34, self).__init__(kwargs_lens_init)

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 3 because the first three lens models
        are EPL, SHEAR, and MULTIPOLE, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 4

    def args_to_kwargs(self, args):
        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']

        # handle the EPL profile
        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        self.kwargs_lens[0] = kwargs_epl

        kwargs_shear = {'gamma1': g1, 'gamma2': g2}
        self.kwargs_lens[1] = kwargs_shear

        # fix m=4 multipole centroid to EPL centroid
        self.kwargs_lens[2]['center_x'] = center_x
        self.kwargs_lens[2]['center_y'] = center_y
        # fix m=4 multipole orientation to EPL orientation
        phi, _ = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[2]['phi_m'] = phi

        # fix m=3 multipole centroid to EPL centroid
        self.kwargs_lens[3]['center_x'] = center_x
        self.kwargs_lens[3]['center_y'] = center_y
        # fix m=3 multipole orientation to EPL orientation
        phi, _ = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[3]['phi_m'] = phi + self._delta_phi_m3

        return self.kwargs_lens
