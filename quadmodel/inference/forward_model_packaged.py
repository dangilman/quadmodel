from quadmodel.inference.forward_model import forward_model
import numpy as np
from quadmodel.inference.util import filenames, FullSimulationContainer, delete_custom_logL
from copy import deepcopy
import pickle
import os

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
        print('************* INFORMATION FOR PACKAGED SIMULATION *************')
        print('SIMULATION NAME: ', self.simulation_name)
        print('\n')
        print('LENS DATA: ', self.lens_data_class)
        print('LENS/SOURCE REDSHIFTS: ', self.lens_data_class.zlens, self.lens_data_class.zsource)
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
        print('\n')
        print('\n')

    def check_progress(self, output_path, n_cores_running=2000):
        """
        Prints the number of accepted samples produced by the simulation in n_cores_running folders
        :param output_path: the directory where the output folder is created
        :param n_cores_running: the number of individual "job_" folders to search
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

    def compile_output(self, output_path, job_index_min, job_index_max, keep_realizations=False, keep_chi2=False,
                       filename_suffix=None, keep_kwargs_fitting_seq=False, keep_macromodel_samples=False,
                       save_subset_kwargs_fitting_seq=False):
        """
        This function compiles output from multiple jobs with output stored in different folders
        :param output_path: the path to the directory where job_1, job_2, ... are located
        :param job_index_min: starts at folder job_i where i=job_index_min
        :param job_index_max: ends at folder job_j where j=job_index_max
        :param keep_realizations: bool; flag to store the accepted realizations
        :param keep_chi2: bool; flag to search for and record the logL from a fit to imaging data
        :param filename_suffix: a string that appends to the end of a filename
        :param keep_kwargs_fitting_seq: bool; keep the FittingSequenceKwargs class for each sample
        :param keep_macromodel_samples: bool; keep macromodel samples
        :param save_subset_kwargs_fitting_seq: saves just 100 of the kwargs fitting sequence classes.
        The first 25 are the best 25, the middle 50 are randomly selected, and the last 25 are the worst 25

        :return: an instance of FullSimulationContainer that contains the compiled data for the simulation
        """

        if output_path[-1]!='/':
            output_path += '/'
        output_path += self.simulation_name

        if keep_realizations:
            realizations_and_lens_systems = []
        else:
            realizations_and_lens_systems = None

        if keep_kwargs_fitting_seq:
            fitting_seq_kwargs = []
        else:
            fitting_seq_kwargs = None

        init = True
        for job_index in range(job_index_min, job_index_max + 1):
            proceed = True
            filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio, \
            filename_macromodel_samples = filenames(output_path, job_index)

            try:
                _params = np.loadtxt(filename_parameters, skiprows=1)
            except:
                print('could not find file ' + filename_parameters)
                continue
            num_realizations = int(_params.shape[0])

            try:
                _fluxes = np.loadtxt(filename_mags)
            except:
                print('could not find file ' + filename_mags)
                continue

            if _fluxes.shape[0] != num_realizations:
                print('fluxes file has wrong shape')
                continue

            if keep_macromodel_samples:
                try:
                    _macro_samples = np.loadtxt(filename_macromodel_samples, skiprows=1)
                except:
                    print('could not find file ' + filename_macromodel_samples)
                    continue
                if _macro_samples.shape[0] != num_realizations:
                    print('macromodel file has wrong shape')
                    continue

            _chi2 = None
            if keep_chi2:
                proceed = True
                for n in range(1, 1 + int(_params.shape[0])):
                    filename_chi2 = output_path + 'job_' + str(job_index) + \
                                    '/chi2_image_data' + filename_suffix + '_' + str(n) + '.txt'
                    if os.path.exists(filename_chi2):
                        new = np.loadtxt(filename_chi2)
                        if _chi2 is None:
                            _chi2 = new
                        else:
                            _chi2 = np.append(_chi2, new)
                    else:
                        print('could not find chi2 file ' + filename_chi2)
                        proceed = False
                        break

            if proceed is False:
                continue

            if len(_chi2) != num_realizations:
                print('chi2 file has wrong shape')
                continue

            if keep_kwargs_fitting_seq:
                proceed = True
                _fitting_seq_kwargs = []
                for n in range(1, 1 + num_realizations):
                    filename_kwargs_fitting_seq = output_path + 'job_' + str(job_index) + \
                                                  '/kwargs_fitting_sequence_' + str(n) + filename_suffix
                    if os.path.exists(filename_kwargs_fitting_seq):
                        try:
                            f = open(filename_kwargs_fitting_seq, 'rb')
                            new = pickle.load(f)
                            f.close()
                            new = delete_custom_logL(new)
                        except:
                            raise Exception('could not open file ' + str(filename_kwargs_fitting_seq))
                        _fitting_seq_kwargs.append(new)
                    else:
                        print('could not find file ' + filename_kwargs_fitting_seq)
                        proceed = False
                        break
                if len(_fitting_seq_kwargs) != num_realizations:
                    print('number of saved fitting sequence classes not right')
                    proceed = False

            if proceed is False:
                continue

            if keep_realizations:
                for n in range(0, num_realizations):
                    try:
                        f = open(filename_realizations + 'simulation_output_' + str(n + 1), 'rb')
                        sim = pickle.load(f)
                        f.close()
                        realizations_and_lens_systems.append(sim)
                    except:
                        print(
                            'could not find pickled class ' + filename_realizations + 'simulation_output_' + str(n + 1))
                        proceed = False
                        break

            if proceed is False:
                continue
            print('compiling output for job ' + str(job_index) + '... ')
            if init:
                init = False
                params = deepcopy(_params)
                fluxes = deepcopy(_fluxes)
                if keep_chi2:
                    chi2_imaging_data = deepcopy(_chi2)
                if keep_macromodel_samples:
                    macro_samples = deepcopy(_macro_samples)
                if keep_kwargs_fitting_seq:
                    fitting_seq_kwargs += _fitting_seq_kwargs
            else:
                params = np.vstack((params, _params))
                fluxes = np.vstack((fluxes, _fluxes))
                if keep_chi2:
                    chi2_imaging_data = np.append(chi2_imaging_data, _chi2)
                if keep_macromodel_samples:
                    macro_samples = np.vstack((macro_samples, _macro_samples))
                if keep_kwargs_fitting_seq:
                    fitting_seq_kwargs += _fitting_seq_kwargs

        print('compiled ' + str(params.shape[0]) + ' realizations')
        assert params.shape[0] == fluxes.shape[0]
        if keep_macromodel_samples:
            assert macro_samples.shape[0] == params.shape[0]
        if keep_chi2:
            assert len(chi2_imaging_data) == params.shape[0]
        if keep_kwargs_fitting_seq:
            if save_subset_kwargs_fitting_seq:
                idx_sort = np.argsort(
                    chi2_imaging_data)  # this is actually the log-likelihood even though it's called chi2
                best_25 = idx_sort[0:25]
                worst_25 = idx_sort[25:]
                end_idx = len(fitting_seq_kwargs) - 25
                random_inds = np.random.randint(25, end_idx, 50)
                random_50 = idx_sort[random_inds]
                fitting_seq_kwargs_out = []
                for idx in best_25:
                    fitting_seq_kwargs_out.append(fitting_seq_kwargs[idx])
                for idx in random_50:
                    fitting_seq_kwargs_out.append(fitting_seq_kwargs[idx])
                for idx in worst_25:
                    fitting_seq_kwargs_out.append(fitting_seq_kwargs[idx])
                container = FullSimulationContainer(realizations_and_lens_systems, params,
                                                    fluxes, chi2_imaging_data, fitting_seq_kwargs_out, macro_samples)
            else:
                assert len(fitting_seq_kwargs) == params.shape[0]
                container = FullSimulationContainer(realizations_and_lens_systems, params,
                                                    fluxes, chi2_imaging_data, fitting_seq_kwargs, macro_samples)
        else:
            container = FullSimulationContainer(realizations_and_lens_systems, params,
                                                fluxes, chi2_imaging_data, fitting_seq_kwargs, macro_samples)

        return container
