import os
import sys
from quadmodel.Solvers.source_optimization import run_optimization

N_jobs = 6
cluster = 'LAPTOP'
filename_suffix = ''
lens_data_name = 'simulated_lens_data_2'
initial_lens_fit_name = 'initial_smooth_lens_fit_2'
initialize_from_fit = True

job_name = 'extended_image_quad_sim_fixedshear_2'
run_index_list = None

if cluster=='LAPTOP':
    # For lens 2:
    # job 5, 5
    # job 6, 4
    # job 7, 2
    # job 8, 1
    job_index = 8
    run_index_list = [1]
    path_to_simulation_output = os.getenv('HOME')+'/Code/quadmodel/notebooks/fluxratio_arc_inference/inference_output/' + job_name+'/' \
                                                  'job_'+str(job_index)+'_local/'
    path_to_data = os.getenv('HOME')+'/Code/quadmodel/notebooks/fluxratio_arc_inference/'
elif cluster=='NIAGARA':
    job_index = int(sys.argv[1])
    path_to_simulation_output = os.getenv('SCRATCH') + '/chains/'+job_name+'/job_' + str(job_index) + '/'
    path_to_data = os.getenv('HOME') + '/mock_data/'
elif cluster=='HOFFMAN2':
    job_index = int(sys.argv[1])
    path_to_simulation_output = os.getenv('SCRATCH') + '/chains/'+job_name+'/job_' + str(job_index) + '/'
    path_to_data = os.getenv('HOME') + '/mock_data/'
else:
    raise Exception('cluster must be either LAPTOP or NIAGARA')
path_to_smooth_lens_fit = path_to_data + 'initial_smooth_lens_fit_2'

overwrite = True
plot_results = False
save_fitting_seq_kwargs = True
add_shapelets_source = False
random_seed = None
npix_mask_images = 0

n_max_source = 8
n_pso_particles = 2
n_pso_iterations = 2
n_run = 2
n_threads = 1

fitting_kwargs_list = [
            ['PSO', {'sigma_scale': 1.0, 'n_particles': n_pso_particles, 'n_iterations': n_pso_iterations, 'threadCount': n_threads}],
            #['update_settings', {'source_remove_fixed': [[0, ['e1', 'e2']]]}],
            #['PSO', {'sigma_scale': 0.1, 'n_particles': n_pso_particles, 'n_iterations': n_pso_iterations, 'threadCount': n_threads}],
            ['MCMC', {'n_burn': 0, 'n_run': n_run, 'walkerRatio': 2, 'sigma_scale': 0.01, 'threadCount': n_threads}]
            ]

run_optimization(N_jobs, lens_data_name, filename_suffix, path_to_simulation_output, path_to_data,
                 fitting_kwargs_list, initialize_from_fit, path_to_smooth_lens_fit, add_shapelets_source, n_max_source,
                     plot_results, save_fitting_seq_kwargs, overwrite, random_seed, npix_mask_images, run_index_list,
                 save_results=True)
