from quadmodel.inference.forward_model import forward_model
import os
import sys
import numpy as np

import matplotlib.pyplot as plt

lens_name = 'RXJ0911'
output_path = os.getenv('HOME') + '/Code/quadmodel/notebooks/'+lens_name+'ULDMtest/'
job_index = 1
n_keep = 20
realization_priors = {}
realization_priors['PRESET_MODEL'] = 'ULDM'
#ULDM specific parameters
realization_priors['log10_m_uldm'] = ['UNIFORM', -22., -16.5]
realization_priors['uldm_plaw'] = ['UNIFORM', 0.2, 0.5]
realization_priors['log10_fluc_amplitude'] = ['UNIFORM', -2.5, -0.5]
# general parameters
realization_priors['sigma_sub'] = ['UNIFORM', 0.0, 0.1]
realization_priors['LOS_normalization'] = ['UNIFORM', 0.8, 1.2]
realization_priors['power_law_index'] = ['UNIFORM', -1.95, -1.85]
# 15, -0.3
# realization_priors['c_power'] = ['FIXED', -0.17]
# realization_priors['c_scale'] = ['FIXED', 100]
tolerance = 1e10

forward_model(output_path, job_index, lens_name, n_keep, realization_priors, tolerance=tolerance,
                  verbose=True, readout_steps=4, test_mode=True)
