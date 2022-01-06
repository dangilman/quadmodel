from quadmodel.inference.forward_model import forward_model
import os

output_path = os.getenv('HOME') + '/Code/quadmodel/notebooks/b1422_inf/'
job_index = 1
n_keep = 12
lens_name = 'B1422'
realization_priors = {}
realization_priors['PRESET_MODEL'] = 'WDM_x'
realization_priors['sigma_sub'] = ['UNIFORM', 0.0, 0.1]
realization_priors['log_mc'] = ['UNIFORM', 3.0, 9.0]
realization_priors['x_wdm'] = ['UNIFORM', 0.6, 1.1]
realization_priors['LOS_normalization'] = ['UNIFORM', 0.01, 0.02]
realization_priors['power_law_index'] = ['FIXED', -1.9]
realization_priors['log_m_host'] = ['GAUSSIAN', 13.3, 0.3]

forward_model(output_path, job_index, lens_name, n_keep, realization_priors, tolerance=0.2,
                  verbose=True, readout_steps=2)
