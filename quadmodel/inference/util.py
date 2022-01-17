import numpy as np
import pickle


class FullSimulationContainer(object):

    def __init__(self, individual_simulations, parameters, magnifications):

        """
        A storage class for individual simulation containers
        :param individual_simulations: a list of SimulationOutputContainer classes
        :param parameters: a numpy array containing the samples accepted into the posterior
        :param magnifications a numpy array containing the magnifications corresponding to a particular set of parameters
        """
        self.simulations = individual_simulations
        self.parameters = parameters
        self.magnifications = magnifications

class SimulationOutputContainer(object):

    """
    This class contains the output of a forward modeling simulation for a single accepted set of parameters.
    It includes the lens data class, the accepted lens system, and the corresponding set of parameters
    """

    def __init__(self, lens_data, lens_system, magnifications, parameters):

        self.data = lens_data
        self.lens_system = lens_system
        self.parameters = parameters
        self.magnifications = magnifications

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
    return filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio

def compile_output(output_path, job_index_min, job_index_max, keep_realizations=False):

    """
    This function complies the result from a simulation into a single pickled python class
    :param output_path: the directly where output will be produced; individual jobs (indexed by job_index) will be created
    in directories output_path/job_1, output_path/job_2, etc. where the 1, 2 are set by job_index
    :param job_index_min: the starting index for output folders
    :param job_index_max: the ending index for output folders
    :param keep_realizations: bool, whether or not to store the accepted realizations and the full lens system for each
    set of accepted parameters
    :return: an instance of FullSimulationContainer
    """

    if keep_realizations:
        realizations_and_lens_systems = []
    else:
        realizations_and_lens_systems = None

    params, fluxes = None, None

    for job_index in range(job_index_min, job_index_max + 1):

        filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio = \
            filenames(output_path, job_index)

        try:
            _params = np.loadtxt(filename_parameters, skiprows=1)
        except:
            print('could not find file '+filename_parameters)
            continue
        try:
            _fluxes = np.loadtxt(filename_mags)
        except:
            print('could not find file '+filename_mags)
            continue

        number = _fluxes.shape[0]

        if keep_realizations:
            for n in range(0, number):
                try:
                    f = open(filename_realizations + 'simulation_output_' + str(n+1), 'rb')
                    sim = pickle.load(f)
                except:
                    print('could not find pickled class ' + filename_realizations + 'simulation_output_' + str(n+1))
                    continue
            realizations_and_lens_systems.append(sim)

        if params is None:
            params = _params
            fluxes = _fluxes
        else:
            params = np.vstack((params, _params))
            fluxes = np.vstack((fluxes, _fluxes))

    container = FullSimulationContainer(realizations_and_lens_systems, params, fluxes)
    return container




