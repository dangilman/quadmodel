from quadmodel.inference.forward_model import forward_model
import os
import sys
import numpy as np
from quadmodel.data.hs0810 import HS0810
import matplotlib.pyplot as plt

lens_name = 'HS0810'
lens_data = HS0810(macromodel_type='EPL_FREE_SHEAR_MULTIPOLE_34',
                   sourcemodel_type='EFFECTIVE_POINT_SOURCE')
output_path = os.getenv('HOME') + '/Code/quadmodel/notebooks/HS0810_test/'
job_index = 1
n_keep = 2
tolerance = 0.05

realization_priors = {}
realization_priors['PRESET_MODEL'] = 'CDM' # see inference/realization_setup

# general parameters
realization_priors['sigma_sub'] = ['FIXED', 0.0, 0.1]
realization_priors['LOS_normalization'] = ['FIXED', 0.0]
realization_priors['shmf_log_slope'] = ['FIXED', -1.95, -1.85]
verbose = True

macromodel_priors = {'m4_amplitude_prior': [np.random.normal, 0.0, 0.02],
                    'm3_amplitude_prior': [np.random.normal, 0.0, 0.02],
                     'gamma_macro_prior': [np.random.uniform, 1.8, 2.3]}

forward_model(output_path, job_index, lens_data, n_keep, realization_priors, macromodel_priors,
              tolerance=tolerance,
              verbose=verbose, readout_steps=2, test_mode=True)
