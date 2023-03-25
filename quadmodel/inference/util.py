import numpy as np
import dill as pickle
from copy import deepcopy

class FullSimulationContainer(object):

    def __init__(self, individual_simulations, parameters, magnifications, chi2_imaging_data=None, modelplots=None):

        """
        A storage class for individual simulation containers
        :param individual_simulations: a list of SimulationOutputContainer classes
        :param parameters: a numpy array containing the samples accepted into the posterior
        :param magnifications a numpy array containing the magnifications corresponding to a particular set of parameters
        :param chi2_imaging_data: a numpy array containing reduced chi2 of the lens and source light models given
        the imaging data
        """
        self.simulations = individual_simulations
        self.parameters = parameters
        self.magnifications = magnifications
        self.chi2_imaging_data = chi2_imaging_data
        self.modelplots=modelplots

    @classmethod
    def join(cls, sim1, sim2):
        mags = np.vstack((sim1.magnifications, sim2.magnifications))
        params = np.vstack((sim1.parameters, sim2.parameters))
        simulations = [sim1, sim2]
        chi2 = None
        if sim1.chi2_imaging_data is not None:
            assert sim2.chi2_imaging_data is not None
            chi2 = np.append(sim1.chi2_imaging_data, sim2.chi2_imaging_data)
        combined = FullSimulationContainer(simulations, params, mags, chi2)
        return combined

class SimulationOutputContainer(object):

    """
    This class contains the output of a forward modeling simulation for a single accepted set of parameters.
    It includes the lens data class, the accepted lens system, and the corresponding set of parameters
    """

    def __init__(self, lens_data, lens_system, magnifications, parameters, chi2_imaging_data=None):

        self.data = lens_data
        self.lens_system = lens_system
        self.parameters = parameters
        self.magnifications = magnifications
        self.chi2_imaging_data = chi2_imaging_data

def filenames(output_path, job_index):
    """
    Creates the names for output files in a certain format
    :param output_path: the directly where output will be produced; individual jobs (indexed by job_index) will be created
    in directories output_path/job_1, output_path/job_2, etc. where the 1, 2 are set by job_index
    :param job_index: a unique integer that specifies the output folder number
    :return: the output filenames
    """
    filename_parameters = output_path + 'job_' + str(job_index) + '/parameters.txt'
    filename_mags = output_path + 'job_' + str(job_index) + '/fluxes.txt'
    filename_realizations = output_path + 'job_' + str(job_index) + '/'
    filename_sampling_rate = output_path + 'job_' + str(job_index) + '/sampling_rate.txt'
    filename_acceptance_ratio = output_path + 'job_' + str(job_index) + '/acceptance_ratio.txt'
    filename_macromodel_samples = output_path + 'job_' + str(job_index) + '/macromodel_samples.txt'
    return filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, \
           filename_acceptance_ratio, filename_macromodel_samples

def compile_output(output_path, job_index_min, job_index_max, keep_realizations=False, record_chi2=False,
                   filename_suffix_chi2=None, keep_modelplot=False):

    """
    This function complies the result from a simulation into a single pickled python class
    :param output_path: the directly where output will be produced; individual jobs (indexed by job_index) will be created
    in directories output_path/job_1, output_path/job_2, etc. where the 1, 2 are set by job_index
    :param job_index_min: the starting index for output folders
    :param job_index_max: the ending index for output folders
    :param keep_realizations: bool, whether or not to store the accepted realizations and the full lens system for each
    set of accepted parameters
    :param record_chi2: bool, whether or not to search for/save the chi2 fit to imaging data for each realization.
    NOTE - if the chi2 file is searched for but not found, then the corresponding flux ratios and parameters
    will not be saved
    :param filename_suffix_chi2: an optional string to append on a filename; format is
    'output_foler/job_$job_index$/chi2_$index$_$filename_suffix$.txt'
    :param keep_modelplot: bool; whether or not to save pickled classes of the ModelPlot in lenstronomy
    :return: an instance of FullSimulationContainer
    """

    if keep_realizations:
        realizations_and_lens_systems = []
    else:
        realizations_and_lens_systems = None

    if keep_modelplot:
        modelplots = []
    else:
        modelplots = None

    params, fluxes, chi2_imaging_data = None, None, None
    n_kept = 0
    for job_index in range(job_index_min, job_index_max + 1):

        filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio = \
            filenames(output_path, job_index)
        # np.loadtxt(filename_parameters, skiprows=1)
        try:
            _params = np.loadtxt(filename_parameters, skiprows=1)
        except:
            print('could not find file ' + filename_parameters)
            continue
        try:
            _fluxes = np.loadtxt(filename_mags)
        except:
            print('could not find file ' + filename_mags)
            continue
        if record_chi2:
            _chi2 = []
            for n in range(1, 1+int(_params.shape[0])):
                filename_chi2 = output_path + 'job_' + str(job_index) + \
                                '/chi2_'+ str(n) + filename_suffix_chi2 +'.txt'
                try:
                    __chi2 = np.loadtxt(filename_chi2)
                except:
                    print('could not find chi2 file ' + filename_chi2)
                    continue
                _chi2.append(__chi2)
            _chi2 = np.array(_chi2)

        number = _fluxes.shape[0]

        if keep_realizations:
            for n in range(0, number):
                try:
                    f = open(filename_realizations + 'simulation_output_' + str(n + 1), 'rb')
                    sim = pickle.load(f)
                    f.close()
                except:
                    print('could not find pickled class ' + filename_realizations + 'simulation_output_' + str(n + 1))
                    continue
            realizations_and_lens_systems.append(sim)

        if keep_modelplot:
            for n in range(0, number):
                filename_modelplot = output_path + 'job_' + str(job_index) + \
                            '/modelplot_'+ str(n+1) + filename_suffix_chi2 + '.txt'

                try:
                    f = open(filename_modelplot, 'rb')
                    _modelplot = pickle.load(f)
                    f.close()
                    modelplots.append(_modelplot)
                except:
                    print('could not find pickled class ' + filename_modelplot)
                    continue

        if params is None:
            params = deepcopy(_params)
            fluxes = deepcopy(_fluxes)
            if record_chi2:
                chi2_imaging_data = deepcopy(_chi2)
        else:
            if params.shape[1] != _params.shape[1]:
                print('shape mismatch on file '+str(job_index))
                print(params[-1,:])
                print(_params)
                print(_params.shape)
                print(params.shape)
                continue
            params = np.vstack((params, _params))
            fluxes = np.vstack((fluxes, _fluxes))
            chi2_imaging_data = np.append(chi2_imaging_data, _chi2)
            n_kept += _params.shape[0]
    print('compiled ' + str(n_kept) + ' realizations')
    container = FullSimulationContainer(realizations_and_lens_systems, params, fluxes, chi2_imaging_data, modelplots)
    return container




