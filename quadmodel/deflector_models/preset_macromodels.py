from quadmodel.deflector_models.power_law_shear import PowerLawShear
from quadmodel.deflector_models.multipole import Multipole
from quadmodel.deflector_models.sis import SIS
import numpy as np

class MacroBase(object):

    def __init__(self, components):

        self.component_list = components

    def add_SIS_satellite(self, redshit, theta_E, center_x, center_y):

        kwargs = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(redshit, kwargs)
        self.component_list += satellite

class EPLShear(MacroBase):

    def __init__(self, zlens, gamma_macro, shear_amplitude,
                 r_ein_approx=None, center_x=None, center_y=None, e1=None, e2=None):

        if shear_amplitude is None:
            shear_amplitude = np.random.uniform(0.02, 0.25)

        if e1 is None or e2 is None:
            theta_e = np.random.uniform(0, 2 * np.pi)
            e = np.random.uniform(0.001, 0.8)
            e1, e2 = e * np.cos(theta_e), e * np.sin(theta_e)

        if center_x is None or center_y is None:
            center_x = 0.0
            center_y = 0.0

        if r_ein_approx is None:
            r_ein_approx = np.random.uniform(0.8, 1.2)

        theta = np.random.uniform(0, 2 * np.pi)
        g1, g2 = shear_amplitude * np.cos(2 * theta), shear_amplitude * np.sin(2 * theta)
        kwargs_epl_shear = [{'theta_E': r_ein_approx, 'center_x': center_x, 'center_y': center_y, 'e1': e1, 'e2': e2,
                             'gamma': gamma_macro},
                            {'gamma1': g2, 'gamma2': g2}]
        component_1 = PowerLawShear(zlens, kwargs_epl_shear)

        super(EPLShear, self).__init__([component_1])

class EPLShearMultipole(MacroBase):

    def __init__(self, zlens, gamma_macro, shear_amplitude, multipole_amplitude,
                 r_ein_approx=None, center_x=None, center_y=None, e1=None, e2=None):

        _epl_shear = EPLShear(zlens, gamma_macro, shear_amplitude,
                 r_ein_approx, center_x, center_y, e1, e2)

        phi = np.random.uniform(0, 2*np.pi)

        kwargs_multipole = [
            {'m': 4, 'a_m': multipole_amplitude, 'center_x': center_x, 'center_y': center_y, 'phi_m': phi}]
        component_2 = Multipole(zlens, kwargs_multipole)
        component_list = _epl_shear.component_list
        component_list += [component_2]

        super(EPLShearMultipole, self).__init__(component_list)
