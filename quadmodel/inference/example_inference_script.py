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
realization_priors['log10_m_uldm'] = ['UNIFORM', -22., -19.5]
realization_priors['uldm_plaw'] = ['UNIFORM', 0.2, 0.5]
realization_priors['log10_fluc_amplitude'] = ['UNIFORM', -1.5, -0.5]
# general parameters
realization_priors['sigma_sub'] = ['FIXED', 0.0]
realization_priors['LOS_normalization'] = ['FIXED', 0.0]
realization_priors['power_law_index'] = ['FIXED', -1.9]
tolerance = 1e10

forward_model(output_path, job_index, lens_name, n_keep, realization_priors, tolerance=tolerance,
                  verbose=True, readout_steps=4, test_mode=True)
