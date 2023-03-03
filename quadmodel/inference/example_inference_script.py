from quadmodel.inference.forward_model import forward_model
import os
import sys
import numpy as np

import matplotlib.pyplot as plt

lens_name = 'PSJ1606'
output_path = os.getenv('HOME') + '/Code/quadmodel/notebooks/'+lens_name+'ULDMtest/'
job_index = 1
n_keep = 2
tolerance = 0.05

realization_priors = {}
realization_priors['PRESET_MODEL'] = 'ULDM' # see inference/realization_setup
#ULDM specific parameters
#ULDM specific parameters
realization_priors['log10_m_uldm'] = ['UNIFORM', -22.5, -16.5]
realization_priors['flucs'] = ['FIXED', False]
realization_priors['uldm_plaw'] = ['FIXED', 0.2]

# general parameters
realization_priors['sigma_sub'] = ['UNIFORM', 0.0, 0.1]
realization_priors['LOS_normalization'] = ['UNIFORM', 0.8, 1.2]
realization_priors['power_law_index'] = ['FIXED', -1.95, -1.85]
verbose = True

macromodel_priors = {'m4_amplitude_prior': [np.random.normal, 0.0, 0.01],
                     'gamma_macro_prior': [np.random.uniform, 1.8, 2.3],
                     'shear_strength_prior': [np.random.uniform, 0.01, 0.2]}

forward_model(output_path, job_index, lens_name, n_keep, realization_priors, macromodel_priors,
              tolerance=tolerance,
              verbose=verbose, readout_steps=2, test_mode=True)
