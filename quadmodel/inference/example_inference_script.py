from quadmodel.inference.forward_model import forward_model
import os

lens_name = 'HE0435'
output_path = os.getenv('HOME') + '/Code/quadmodel/notebooks/'+lens_name+'_SIDMinf/'
job_index = 1
n_keep = 6
realization_priors = {}
realization_priors['PRESET_MODEL'] = 'SIDM_CORE_COLLAPSE'
realization_priors['f_68'] = ['UNIFORM', 0.8, 1.0]
realization_priors['f_810'] = ['UNIFORM', 0.8, 1.0]
realization_priors['sigma_sub'] = ['UNIFORM', 0.005, 0.08]
realization_priors['x_match'] = ['UNIFORM', 2.0, 5.0]
realization_priors['x_core_halo'] = ['FIXED', 0.05]
realization_priors['log_slope_halo'] = ['UNIFORM', 2.8, 3.2]
realization_priors['LOS_normalization'] = ['UNIFORM', 0.8, 1.2]
realization_priors['log_m_host'] = ['GAUSSIAN', 13.3, 0.3]
tolerance = 1.0

forward_model(output_path, job_index, lens_name, n_keep, realization_priors, tolerance=tolerance,
                  verbose=True, readout_steps=3, test_mode=True)
