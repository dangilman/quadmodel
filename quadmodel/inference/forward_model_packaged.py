from quadmodel.inference.forward_model import forward_model
import numpy as np


class ForwardModelSimulation(object):

    def __init__(self, simulation_name, lens_data_class, kwargs_sample_realization,
                  kwargs_sample_macromodel, kwargs_realization_other={},
                  save_realizations=False, importance_weights_function=None,
                 readout_macromodel_samples=False, n_macro=None, realization_class=None):

        """

        :param simulation_name: The folder name where output is written, for example "B1422_WDM"
        :param lens_data_class: A class that contains the lens data, see data/quad_base for an example
        :param kwargs_sample_realization: keyword arguments that specify the priors for dark matter parameters
        :param kwargs_sample_macromodel: keyword arguments that specify the priors for macromodel parameters
        :param kwargs_realization_other: additional fixed keyword arguments to be passed to pyHalo
        :param save_realizations: bool; saves realizations, including the lens models
        :param importance_weights_function: an optional function that returns importance weights for samples
        :param readout_macromodel_samples: bool; reads out text files with the macromodel parameters only
        :param n_macro: the number of lens models corresponding to the macromodel, only has an effeect if
        readout_macromodel_samples is True
        :param realization_class: an optional fixed realization of halos
        """

        # these properties should not be changed after initialization, otherwise the output will either crash or be
        # nonsensical (for example, it would write output to the same text files for sims run on different data)
        if simulation_name[-1] != '/':
            simulation_name += '/'
        self._simulation_name = simulation_name
        self._lens_data_class = lens_data_class
        self._kwargs_sample_realization = kwargs_sample_realization
        self._kwargs_sample_macromodel = kwargs_sample_macromodel
        self._kwargs_realization_other = kwargs_realization_other
        self._save_realizations = save_realizations
        self._importance_weights_function = importance_weights_function
        self._readout_macromodel_samples = readout_macromodel_samples
        self._n_macro = n_macro
        self._realization_class = realization_class

    def print_settings(self):

        """
        This function prints the fixed settings of the class (for example, the dataset, priors, etc.)
        :return:
        """
        print('SIMULATION NAME: ', self.simulation_name)
        print('\n')
        print('LENS DATA: ', self.lens_data_class)
        print('LENS DATA (image positions): ', self.lens_data_class.x, self.lens_data_class.y)
        print('LENS DATA (astrometric uncertainty): ', self.lens_data_class.delta_xy)
        print('LENS DATA (normalized magnifications): ', self.lens_data_class.m)
        print('LENS DATA (flux uncertainty): ', self.lens_data_class.delta_m)
        print('LENS DATA (uncertainty in magnifications): ', self.lens_data_class.uncertainty_in_magnifications)
        print('LENS DATA (use flux ratios): ', self.lens_data_class.keep_flux_ratio_index)
        print('\n')
        print('REALIZATION PRIORS: ')
        for key in self.kwargs_sample_realization.keys():
            print(key+': ', self.kwargs_sample_realization[key])
        print('FIXED REALIZATION PRIORS', self.kwargs_realization_other)
        for key in self.kwargs_realization_other.keys():
            print(key+': ', self.kwargs_realization_other[key])
        print('FIXED SUBSTRUCTURE REALIZATION: ', self.realization_class)
        print('\n')
        print('MACROMODEL PRIORS: ')
        for key in self.kwargs_sample_macromodel.keys():
            print(key+': ', self.kwargs_sample_macromodel[key])
        print('MACROMODEL TYPE: ', self.lens_data_class.macromodel_type)
        print('SOURCE MODEL TYPE: ', self.lens_data_class.sourcemodel_type)
        print('\n')
        print('SAVE REALIZATIONS: ', self.save_realizations)
        print('IMPORTANCE WEIGHT FUNCTION: ', self.importance_weights_function)
        print('READOUT MACROMODEL SAMPLES: ', self.readout_macromodel_samples)
        print('n_macro: ', self.n_macro)

    def check_progress(self, output_path, n_cores_running=2000):
        """
        Prints thee number of accepted samples produced by the simulation
        :param output_path:
        :param i_max:
        :return:
        """
        folder = output_path + self.simulation_name
        done_folders = []
        tol = 1e10
        n = 0
        a = None
        s = None
        params = np.empty((1, 4))
        for i in range(1, n_cores_running):
            fname = folder + '/job_' + str(i) + '/'
            try:
                _s = np.loadtxt(fname + 'sampling_rate.txt')
                if s is None:
                    s = np.mean(_s)
                else:
                    s = np.mean(s, _s)
            except:
                pass
            try:
                _a = np.loadtxt(fname + 'acceptance_ratio.txt')
                if a is None:
                    a = [np.mean(_a)]
                else:
                    a.append(np.mean(_a))
            except:
                pass

            try:
                p = np.loadtxt(fname + 'parameters.txt', skiprows=1)
                # n += p.shape[0]
                # print(p[:,-1])
                inds = np.where(p[:, -1] < tol)[0]
                params = np.vstack((params, p[inds, 0:4]))
                n += np.sum(p[:, -1] < tol)
                done_folders.append(i)
            except:
                continue
        print(params)
        print('number completed: ', n)
        print('median acceptance rate [prob(accept) per realization]: ', np.median(a))
        print('sampling rate: (min per realization): ', s)
        print('folders with output: ', done_folders)

    def run(self, output_path, job_index, n_keep, abc_tolerance, verbose=False, readout_steps=2, ray_tracing_optimization='default',
            test_mode=False, crit_curves_in_test_mode=False, write_sampling_rate=False, shift_background_realization=True):
        """

        :param output_path: the folder where output will be created. For example, on the Niagara cluster at the
        University of Toronto, I would use output_path = os.getenv('SCRATCH') + '/chains/'
        :param job_index: the core index, output is written inside simulation_folder/job_job_index/
        :param n_keep: the number of realizations to keep per core
        :param abc_tolerance: the tolerance threshold for the ABC selection on flux ratios only
        :param verbose: bool; make print statements
        :param readout_steps: write output after every readout_steps samples are accepted
        :param ray_tracing_optimization: specifies the settings for HierarchicalOptimization
        :param test_mode: bool; make sure things are working as expected by showing substructure convergence maps
        and ray tracing to create lensed images
        :param crit_curves_in_test_mode: plot crit curves if test_mode=True
        :param write_sampling_rate: bool; create text files that approximate the sampling rate
        :param shift_background_realization: bool; apply shift background to source to realization
        """
        if output_path[-1]!='/':
            output_path += '/'
        output_path += self.simulation_name
        forward_model(output_path, job_index, self.lens_data_class, n_keep, self.kwargs_sample_realization,
                      self.kwargs_sample_macromodel, abc_tolerance, verbose, readout_steps, self.kwargs_realization_other,
                      ray_tracing_optimization, test_mode, self.save_realizations, crit_curves_in_test_mode, write_sampling_rate,
                      self.importance_weights_function, self.readout_macromodel_samples, self.n_macro, self.realization_class,
                      shift_background_realization)

    @property
    def simulation_name(self):
        return self._simulation_name

    @property
    def lens_data_class(self):
        return self._lens_data_class

    @property
    def kwargs_sample_realization(self):
        return self._kwargs_sample_realization

    @property
    def kwargs_sample_macromodel(self):
        return self._kwargs_sample_macromodel

    @property
    def kwargs_realization_other(self):
        return self._kwargs_realization_other

    @property
    def save_realizations(self):
        return self._save_realizations

    @property
    def importance_weights_function(self):
        return self._importance_weights_function

    @property
    def readout_macromodel_samples(self):
        return self._readout_macromodel_samples

    @property
    def n_macro(self):
        return self._n_macro

    @property
    def realization_class(self):
        return self._realization_class
