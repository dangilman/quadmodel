from quadmodel.data.quad_base import Quad
import numpy as np
from quadmodel.deflector_models.sis import SIS

class PSJ1606(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        _zvalues = [0.061757, 0.072423, 0.083195, 0.094076, 0.105066, 0.116167, 0.127379, 0.138704, 0.150142, 0.161695, 0.173365, 0.185151, 0.197056, 0.209081, 0.221226, 0.233494, 0.245884, 0.258399, 0.27104, 0.283808, 0.296704, 0.309729, 0.322886, 0.336174, 0.349596, 0.363153, 0.376846, 0.390677, 0.404646, 0.418756, 0.433008, 0.447402, 0.461942, 0.476627, 0.49146, 0.506442, 0.521574, 0.536859, 0.552297, 0.56789, 0.583639, 0.599547, 0.615615, 0.631844, 0.648236, 0.664793, 0.681516, 0.698407, 0.715467, 0.732699]
        _zpdf = [0.02615367293314519, 0.03838602176725219, 0.05347645756511757, 0.0725306235841679, 0.09630886832562517, 0.12623407934656633, 0.16417518562622155, 0.20884805880315607, 0.26301023498988496, 0.3214934787583879, 0.38287282202034273, 0.45397191772355616, 0.542488486707241, 0.639484077204843, 0.7251307369119288, 0.7938328158521795, 0.8598735281771241, 0.9143369408113338, 0.9567347408331182, 0.9874969862892992, 1.0, 0.9900518786585224, 0.9536506592224443, 0.9028834939217891, 0.8479129212004911, 0.7868556931289844, 0.7136885493679468, 0.6312642654072518, 0.5448556417276241, 0.46266070574067525, 0.3912921200491005, 0.3311464094314153, 0.27933310406694906, 0.23641032481988106, 0.20044085233245912, 0.16754488746161728, 0.1374436390204859, 0.11158009005028767, 0.0903198755597547, 0.07278415467403583, 0.057768275325266585, 0.04469269941586026, 0.03454031885977097, 0.02691135345900558, 0.02085861076809542, 0.015429038586511548, 0.010818096705539966, 0.007348759452648944, 0.004766781014295698, 0.003029982902928531]
        zlens = [np.array(_zvalues), np.array(_zpdf)]
        #zlens = 0.31
        zsource = 1.7
        x = [0.838, -0.784, 0.048, -0.289]
        y = [0.378, -0.211, -0.527, 0.528]
        m = [1.0, 1.0, 0.59, 0.79]
        delta_m = [0.03, 0.03, 0.02/0.6, 0.02/0.78]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]
        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3

        kwargs_macromodel = {'shear_amplitude_min': 0.1, 'shear_amplitude_max': 0.28}

        super(PSJ1606, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                      kwargs_macromodel, keep_flux_ratio_index)

    def generate_macromodel(self):
        """
        Used only if lens-specific data class has no satellite galaxies; for systems with satellites, add them in the
        lens-specific data class and override this method
        :return:
        """

        model, constrain_params, optimization_routine, params_sampled, param_names_macro = self._generate_macromodel()
        model_satellite, params_satellite, param_names_satellite = self.satellite_galaxy()
        model.add_satellite(model_satellite)
        params_sampled = np.append(params_sampled, params_satellite)
        param_names_macro += param_names_satellite
        return model, constrain_params, optimization_routine, params_sampled, param_names_macro

    def satellite_galaxy(self, sample=True):
        """
        If the deflector system has no satellites, return an empty list of lens components (see macromodel class)
        """

        theta_E = 0.27
        center_x = -0.307
        center_y = -1.153

        if sample:
            theta_E = abs(np.random.normal(theta_E, 0.05))
            center_x = np.random.normal(center_x, 0.05)
            center_y = np.random.normal(center_y, 0.05)

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        satellite = SIS(self.zlens, kwargs_init)
        params = np.array([theta_E, center_x, center_y])
        param_names = ['theta_E', 'center_x', 'center_y']
        return [satellite], params, param_names
