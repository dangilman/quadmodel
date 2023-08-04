from quadmodel.data.quad_base import Quad
import numpy as np

class WGDJ0405(Quad):

    def __init__(self, sourcemodel_type='NARROW_LINE_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE', sample_zlens_pdf=True):

        _zvalues = [0.116167, 0.127379, 0.138704, 0.150142, 0.161695, 0.173365, 0.185151, 0.197056, 0.209081, 0.221226, 0.233494, 0.245884, 0.258399, 0.27104, 0.283808, 0.296704, 0.309729, 0.322886, 0.336174, 0.349596, 0.363153, 0.376846, 0.390677, 0.404646, 0.418756, 0.433008, 0.447402, 0.461942, 0.476627, 0.49146]
        _zpdf = [0.042240683221597256, 0.06460813001247, 0.09637957460918829, 0.14066189737918483, 0.19721638701780192, 0.26608814362892685, 0.35091355490060167, 0.45468559565825517, 0.5712616917213, 0.6856177056270185, 0.7882543949231077, 0.8796303449443614, 0.9485193866950286, 0.9901317727321632, 1.0, 0.9758383733625418, 0.9194117408428446, 0.8364488915463322, 0.7373281407363117, 0.6279329618792122, 0.5143822943729816, 0.4063505249920189, 0.3131098964479039, 0.23310139498131266, 0.1705376560292507, 0.12534819855333965, 0.09247740144383829, 0.0679177403736765, 0.05018819636624036, 0.037156823837971226]
        zlens = [np.array(_zvalues), np.array(_zpdf)]
        zsource = 1.7
        x = [0.708, -0.358, 0.363, -0.515]
        y = [-0.244, -0.567, 0.592, 0.454]
        m = [0.8, 0.52, 1.0, 0.94]

        delta_m = [0.04, 0.04/0.65, 0.03/1.25, 0.04/1.17]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]
        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3
        kwargs_macromodel = {'shear_amplitude_min': 0.0025, 'shear_amplitude_max': 0.12}
        super(WGDJ0405, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                      kwargs_macromodel, keep_flux_ratio_index,
                                       sample_zlens_pdf=sample_zlens_pdf)

class WGDJ0405_JWST(WGDJ0405):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE', sample_zlens_pdf=True):

        super(WGDJ0405_JWST, self).__init__(sourcemodel_type, macromodel_type, sample_zlens_pdf)

        # now replace the data with the JWST measurements
        normalized_fluxes = [1.00, 0.70, 1.07, 1.28]
        self.m = np.array(normalized_fluxes)
        flux_uncertainties = [0.01] * 4  # percent uncertainty
        self.delta_m = np.array(flux_uncertainties)
