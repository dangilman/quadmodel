from quadmodel.quadmodel import QuadLensSystem
from quadmodel.deflector_models.preset_macromodels import EPLShear
from quadmodel.macromodel import MacroLensModel
from pyHalo.preset_models import CDM
import numpy as np
import matplotlib.pyplot as plt
from quadmodel.Solvers.hierachical import HierarchicalOptimization

class DummyData(object):

    def __init__(self, x, y):
        self.x, self.y = x, y

zlens = 0.42
zsource = 2.2
theta_ellip = np.pi/5
ellip = 0.27
e1, e2 = ellip * np.cos(theta_ellip), ellip * np.sin(theta_ellip)
shear_amplitude = 0.06
theta_shear = np.pi/4
gamma_macro = 2.02
rein_approx = 0.95
macro_model = EPLShear(zlens, gamma_macro, shear_amplitude, rein_approx, 0.0, 0.0, e1, e2, theta_shear)
macromodel = MacroLensModel(macro_model.component_list)
lens_system = QuadLensSystem(macromodel, zsource)

source_x, source_y = 0.05, 0.05
x_image, y_image = lens_system.solve_lens_equation(source_x, source_y)
plt.scatter(x_image, y_image)
plt.show()
a=input('continue')

kwargs_realization = {'sigma_sub': 0.04, 'LOS_normalization': 1.0}
realization = CDM(zlens, zsource, **kwargs_realization)
data_to_fit = DummyData(x_image, y_image)
lens_system = QuadLensSystem.shift_background_auto(data_to_fit, macromodel, zsource, realization)
opt = HierarchicalOptimization(lens_system)
verbose = False
kwargs_lens_final, lens_model_full, kwargs_return = opt.optimize(data_to_fit, 'free_shear_powerlaw',
                                                                 constrain_params=None, verbose=verbose)
lens_model, kwargs_lens = lens_system.get_lensmodel()
source_size_pc_midIR = np.random.uniform(1.0, 10.0)
source_size_pc_NL = np.random.uniform(25, 60)
mags_midIR = lens_system.quasar_magnification(data_to_fit.x, data_to_fit.y, source_size_pc_midIR, lens_model, kwargs_lens)
lens_system.plot_images(data_to_fit.x, data_to_fit.y, source_size_pc_midIR, lens_model, kwargs_lens, grid_resolution_rescale=5.0)
mags_NL = lens_system.quasar_magnification(data_to_fit.x, data_to_fit.y, source_size_pc_NL, lens_model, kwargs_lens)
lens_system.plot_images(data_to_fit.x, data_to_fit.y, source_size_pc_NL, lens_model, kwargs_lens, grid_resolution_rescale=5.0)

print('zlens, zsource: ', zlens, zsource)
print('X IMAGE: ', repr(np.round(x_image + np.random.normal(0, 0.005, 4), 4)))
print('Y IMAGE: ', repr(np.round(y_image+ np.random.normal(0, 0.005, 4), 4)))
delta_midir = 0.02
delta_NL = 0.04
mags_midIR += np.random.normal(0, mags_midIR*delta_midir)
mags_NL += np.random.normal(0, mags_NL*delta_NL)
print('fluxes midIR: ', repr(np.round(mags_midIR, 4)))
print('fluxes NL: ', repr(np.round(mags_NL, 4)))
