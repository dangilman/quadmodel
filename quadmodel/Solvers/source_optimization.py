import numpy as np
import os
import dill
import pickle
from lenstronomy.Plots.model_plot import ModelPlot
import matplotlib.pyplot as plt
from quadmodel.data.hst import HSTData, HSTDataModel
from quadmodel.Solvers.fit_wgd2038_light import fit_wgd2038_light
from quadmodel.Solvers.fit_wgdj0405_light import fit_wgdj0405_light
from quadmodel.Solvers.fit_psj1606_light import fit_psj1606_light
from quadmodel.Solvers.fit_mock import fit_mock


def run_optimization(launch_fuction, N_jobs, lens_data_name, filename_suffix, path_to_simulation_output, path_to_data,
                     fitting_kwargs_list, initialize_from_fit=False, path_to_smooth_lens_fit=None, add_shapelets_source=False,
                     n_max_source=10, plot_results=False, overwrite=False, random_seed=None,
                     run_index_list=None, astrometric_uncertainty=0.005, delta_x_offset_init=None,
                     delta_y_offset_init=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    if run_index_list is None:
        run_index_list = range(1, N_jobs + 1)

    f = open(path_to_data + lens_data_name, 'rb')
    hst_data = pickle.load(f)
    f.close()

    for idx in run_index_list:

        print('fitting light to realization ' + str(idx) + ' ...')
        fname_chi2 = path_to_simulation_output + 'chi2_image_data' + filename_suffix + '_' + str(idx) + '.txt'

        try:
            f = open(path_to_simulation_output + 'simulation_output_' + str(idx), 'rb')
            simulation_output = dill.load(f)
            f.close()
        except:
            print(
                'could not find simulation output file ' + path_to_simulation_output + 'simulation_output_' + str(idx))
            continue

        if os.path.exists(fname_chi2):
            if overwrite is False:
                print('logL computation already performed for file ' + str(idx))
                continue

        if callable(launch_fuction):
            fitting_seq, fitting_kwargs_class = launch_fuction(hst_data, simulation_output,
                                                                   astrometric_uncertainty, delta_x_offset_init,
                                                                   delta_y_offset_init, add_shapelets_source, n_max_source)

        elif launch_fuction == 'MOCK':
            fitting_seq, fitting_kwargs_class = fit_mock(fitting_kwargs_list, hst_data, simulation_output,
                                                        initialize_from_fit,
                                                        path_to_smooth_lens_fit, add_shapelets_source,
                                                        n_max_source, astrometric_uncertainty,
                                                        delta_x_offset_init, delta_y_offset_init)
        elif launch_fuction == 'WGDJ0405':
            fitting_seq, fitting_kwargs_class = fit_wgdj0405_light(hst_data, simulation_output,
                                                                   astrometric_uncertainty, delta_x_offset_init,
                                                                   delta_y_offset_init, add_shapelets_source, n_max_source)
        elif launch_fuction == 'PSJ1606':
            fitting_seq, fitting_kwargs_class = fit_psj1606_light(hst_data, simulation_output,
                                                                   astrometric_uncertainty, delta_x_offset_init,
                                                                   delta_y_offset_init, add_shapelets_source, n_max_source)
        elif launch_fuction == 'WGD2038':
            fitting_seq, fitting_kwargs_class = fit_wgd2038_light(hst_data, simulation_output,
                                                                  astrometric_uncertainty, delta_x_offset_init,
                                                                  delta_y_offset_init, add_shapelets_source,
                                                                  n_max_source)
        else:
            raise Exception('launch function not recognized')
        # modelPlot = ModelPlot(multi_band_list, kwargs_model,
        #                       kwargs_result, arrow_size=0.02, cmap_string="gist_heat")
        kwargs_best = fitting_seq.best_fit()
        neff = fitting_seq.likelihoodModule.effective_num_data_points(**kwargs_best)
        log_l = fitting_seq.best_fit_likelihood
        print('CHI2 FROM FIT: ', 2 * log_l / neff)

        f = open(path_to_simulation_output + 'kwargs_fitting_sequence_' + str(idx) + filename_suffix, 'wb')
        dill.dump(fitting_kwargs_class, f)
        f.close()
        np.savetxt(fname_chi2, X=np.atleast_1d(log_l))

        if plot_results:
            print('plotting... ')

            modelPlot = ModelPlot(**fitting_kwargs_class.kwargs_model_plot)
            #         chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
            #         for i in range(len(chain_list)):
            #             chain_plot.plot_chain_list(chain_list, i)

            f = plt.figure(1)
            f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

            modelPlot.data_plot(ax=axes[0, 0])
            modelPlot.model_plot(ax=axes[0, 1])
            modelPlot.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)
            modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100)
            modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
            modelPlot.magnification_plot(ax=axes[1, 2])
            f.tight_layout()
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
            plt.show()

            f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
            modelPlot.decomposition_plot(ax=axes[0, 0], text='Lens light', lens_light_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 0], text='Lens light convolved', lens_light_add=True)
            modelPlot.decomposition_plot(ax=axes[0, 1], text='Source light', source_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 1], text='Source light convolved', source_add=True)
            modelPlot.decomposition_plot(ax=axes[0, 2], text='All components', source_add=True, lens_light_add=True,
                                         unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 2], text='All components convolved', source_add=True,
                                         lens_light_add=True, point_source_add=True)
            f.tight_layout()
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
            plt.show()


