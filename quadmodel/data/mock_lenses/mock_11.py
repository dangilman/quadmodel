from quadmodel.data.quad_base import Quad

class Mock11(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FREE_SHEAR'):

        zlens = 0.5
        zsource = 2.0
        x =[ 0.25518624,  0.8679765 ,  1.0197466 , -0.79429581]
        y =[ 1.01603704, -0.63323242,  0.29967318, -0.26641724]
        m =[0.88721472, 0.86050555, 1., 0.30379165]
        delta_m = [0.03] * len(m)
        delta_xy = [0.005] * len(x)
        kwargs_macromodel = {'shear_amplitude_min': 0.01, 'shear_amplitude_max': 0.1}
        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3
        keep_flux_ratio_index = [0, 1, 2]
        super(Mock11, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                   kwargs_macromodel, keep_flux_ratio_index)
