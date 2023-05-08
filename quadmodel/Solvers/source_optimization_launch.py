import os
import sys
from quadmodel.Solvers.source_optimization import run_optimization
import numpy as np

N_jobs = 1
cluster = 'LAPTOP'
filename_suffix = ''
lens_data_name = 'simulated_lens_data_1'
initial_lens_fit_name = 'initial_smooth_lens_fit_1'
initialize_from_fit = True

job_name = 'output_mock_1_fixed_realization'
run_index_list = np.arange(1, 10)
job_index = int(sys.argv[1])

if cluster == 'LAPTOP':

    path_to_simulation_output = os.getenv('HOME')+'/Code/quadmodel/notebooks/fluxratio_arc_inference/' \
                                                  'inference_output/' + job_name+'/' \
                                                  'job_'+str(job_index)+'/'
    path_to_data = os.getenv('HOME')+'/Code/quadmodel/notebooks/fluxratio_arc_inference/simulated_datasets/'
    path_to_smooth_lens_fits = os.getenv('HOME') + '/Code/quadmodel/notebooks/fluxratio_arc_inference/initial_smooth_lens_fits/'

elif cluster == 'NIAGARA':
    job_index = int(sys.argv[1])
    path_to_simulation_output = os.getenv('SCRATCH') + '/chains/'+job_name+'/job_' + str(job_index) + '/'
    path_to_data = os.getenv('HOME') + '/mock_data/'
    path_to_smooth_lens_fits = path_to_data

elif cluster == 'HOFFMAN2':
    job_index = int(sys.argv[1])
    path_to_simulation_output = os.getenv('SCRATCH') + '/chains/'+job_name+'/job_' + str(job_index) + '/'
    path_to_data = os.getenv('HOME') + '/mock_data/'
    path_to_smooth_lens_fits = path_to_data
else:
    raise Exception('cluster must be either LAPTOP or NIAGARA')
path_to_smooth_lens_fit = path_to_smooth_lens_fits + initial_lens_fit_name

overwrite = True
plot_results = False
save_fitting_seq_kwargs = True
add_shapelets_source = False
random_seed = None
npix_mask_images = 0

n_max_source = 8
n_pso_particles = 100
n_pso_iterations = 200
n_run = 350
n_threads = 1

fitting_kwargs_list = [
            ['PSO', {'sigma_scale': 1.0, 'n_particles': n_pso_particles, 'n_iterations': n_pso_iterations, 'threadCount': n_threads}],
            ['MCMC', {'n_burn': 0, 'n_run': n_run, 'walkerRatio': 4, 'sigma_scale': 0.1, 'threadCount': n_threads}]
            ]

print('fitting realization '+path_to_simulation_output+' ...')
run_optimization(N_jobs, lens_data_name, filename_suffix, path_to_simulation_output, path_to_data,
                 fitting_kwargs_list, initialize_from_fit, path_to_smooth_lens_fit, add_shapelets_source, n_max_source,
                     plot_results, overwrite, random_seed, npix_mask_images, run_index_list)
