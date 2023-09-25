import numpy as np
import dill as pickle
from copy import deepcopy
import os


class FullSimulationContainer(object):

    def __init__(self, individual_simulations, parameters, magnifications,
                 chi2_imaging_data=None, kwargs_fitting_seq=None, macromodel_samples=None, bic=None,
                 kappa_gamma_stats=None, curved_arc_stats=None):

        """
        A storage class for individual simulation containers
        :param individual_simulations: a list of SimulationOutputContainer classes
        :param parameters: a numpy array containing the samples accepted into the posterior
        :param magnifications a numpy array containing the magnifications corresponding to a particular set of parameters
        :param chi2_imaging_data: a numpy array containing reduced chi2 of the lens and source light models given
        the imaging data
        :param bic: the Bayesian information criterion computed for the fit
        :param kappa_gamma_stats: the convergence and shear at the image positions
        :param curved_arc_stats: the curved arc properties at the image positions
        """
        self.simulations = individual_simulations
        self.parameters = parameters
        self.magnifications = magnifications
        self.chi2_imaging_data = chi2_imaging_data
        self.kwargs_fitting_seq = kwargs_fitting_seq
        self.macromodel_samples = macromodel_samples
        self.bic = bic
        self.kappa_gamma_stats = kappa_gamma_stats
        self.curved_arc_stats = curved_arc_stats

    def cut_on_logL(self, percentile_cut):
        """

        :param percentile_cut:
        :return:
        """
        logL = self.chi2_imaging_data
        inds_sorted = np.argsort(logL)
        idx_cut = int((100 - percentile_cut) / 100 * len(logL))
        logL_cut = logL[inds_sorted[idx_cut]]
        inds_keep = np.where(logL > logL_cut)[0]

        if len(self.simulations) > 0:
            simulations = []
            for idx in inds_keep:
                simulations.append(self.simulations[idx])
        else:
            simulations = []
        parameters = self.parameters[inds_keep,:]
        mags = self.magnifications[inds_keep,:]
        if self.chi2_imaging_data is None:
            chi2 = None
        else:
            chi2 = self.chi2_imaging_data[inds_keep,:]
        if self.macromodel_samples is None:
            macro_samples = None
        else:
            macro_samples = self.macromodel_samples[inds_keep,:]
        if self.bic is None:
            bic = None
        else:
            bic = np.array(self.bic)[inds_keep]
        return FullSimulationContainer(simulations, parameters, mags, chi2, self.kwargs_fitting_seq,
                                       macro_samples, bic)

    def cut_on_S(self, keep_best_N=None, percentile_cut=None, idx_s_statistic=-2):
        """

        :param percentile_cut:
        :return:
        """
        sorted_inds = np.argsort(self.parameters[:, idx_s_statistic])
        if keep_best_N is None:
            idxcut = int(self.parameters.shape[0] * percentile_cut/100)
            inds_keep = sorted_inds[0:idxcut]
        else:
            inds_keep = sorted_inds[0:keep_best_N]

        if len(self.simulations) > 0:
            simulations = []
            for idx in inds_keep:
                simulations.append(self.simulations[idx])
        else:
            simulations = []
        parameters = self.parameters[inds_keep,:]
        mags = self.magnifications[inds_keep,:]
        if self.chi2_imaging_data is None:
            chi2 = None
        else:
            chi2 = self.chi2_imaging_data[inds_keep,:]
        if self.macromodel_samples is None:
            macro_samples = None
        else:
            macro_samples = self.macromodel_samples[inds_keep,:]
        if self.bic is None:
            bic = None
        else:
            bic = np.array(self.bic)[inds_keep]
        if self.kappa_gamma_stats is None:
            kappa_gamma_stats = None
        else:
            kappa_gamma_stats = self.kappa_gamma_stats[inds_keep, :]
        if self.curved_arc_stats is None:
            curved_arc_stats = None
        else:
            curved_arc_stats = self.curved_arc_stats[inds_keep, :]
        return FullSimulationContainer(simulations, parameters, mags, chi2, self.kwargs_fitting_seq,
                                       macro_samples, bic, kappa_gamma_stats, curved_arc_stats)


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

    def __init__(self, lens_data, lens_system, magnifications, parameters, chi2_imaging_data=None, kwargs_fitting_seq=None):

        self.data = lens_data
        self.lens_system = lens_system
        self.parameters = parameters
        self.magnifications = magnifications
        self.chi2_imaging_data = chi2_imaging_data
        self.kwargs_fitting_seq = kwargs_fitting_seq


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
    filename_kappagamma_stats = output_path + 'job_' + str(job_index) + '/kappa_gamma_statistics.txt'
    filename_curvedarc_stats = output_path + 'job_' + str(job_index) + '/curvedarc_statistics.txt'
    return filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio, \
           filename_macromodel_samples, filename_kappagamma_stats, filename_curvedarc_stats

def delete_custom_logL(kwargs_fitting_seq):
    """
    This function deletes fuctions from inside kwargs_likelihood because they cannot be pickled
    :param kwargs_fitting_seq:
    :return:
    """
    if 'custom_logL_addition' in kwargs_fitting_seq.kwargs_fitting_sequence['kwargs_likelihood'].keys():
        del kwargs_fitting_seq.kwargs_fitting_sequence['kwargs_likelihood']['custom_logL_addition']
    return kwargs_fitting_seq

def compile_output(output_path, job_index_min, job_index_max, keep_realizations=False, keep_chi2=False,
                   filename_suffix=None, keep_kwargs_fitting_seq=False, keep_macromodel_samples=False,
                   save_subset_kwargs_fitting_seq=False, keep_kappagamma_stats=False, keep_curvedarc_stats=False,
                   keep_best_N=None):
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
    :return: an instance of FullSimulationContainer that contains the data for the simulation
    :param keep_kappagamma_stats: bool; compile the convergence and shear values at image positions
    :param keep_curvedarc_stats: bool; compiles the curved arc properties at image positions
    :param keep_best_N: retains the top N samples as ranked by the summary statistic computed from the flux ratios;
    the default setting keeps all samples
    """

    if keep_realizations:
        realizations_and_lens_systems = []
    else:
        realizations_and_lens_systems = None

    if keep_kwargs_fitting_seq:
        fitting_seq_kwargs = []
    else:
        fitting_seq_kwargs = None
    kappa_gamma_stats = None
    curved_arc_stats = None
    init = True
    for job_index in range(job_index_min, job_index_max + 1):
        proceed = True
        filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio, \
        filename_macromodel_samples, filename_kappagamma_stats, filename_curvedarc_stats = filenames(output_path, job_index)

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
            print('fluxes file has wrong shape!')
            continue

        if keep_macromodel_samples:
            try:
                _macro_samples = np.loadtxt(filename_macromodel_samples, skiprows=1)
            except:
                print('could not find file ' + filename_macromodel_samples)
                continue
            if _macro_samples.shape[0] != num_realizations:
                print('macromodel file has wrong shape!')
                continue

        if keep_kappagamma_stats:
            try:
                _kappagammastatistics = np.loadtxt(filename_kappagamma_stats, skiprows=1)
            except:
                print('could not find file ' + filename_kappagamma_stats)
                continue
            if _kappagammastatistics.shape[0] != num_realizations:
                print('kappa/gamma file has wrong shape!')
                continue

        if keep_curvedarc_stats:
            try:
                _curvedarcstatistics = np.loadtxt(filename_curvedarc_stats, skiprows=1)
            except:
                print('could not find file ' + filename_curvedarc_stats)
                continue
            if _curvedarcstatistics.shape[0] != num_realizations:
                print('curved arc stats file has wrong shape!')
                continue

        _chi2 = None
        if keep_chi2:
            proceed = True
            for n in range(1, 1+int(_params.shape[0])):
                filename_chi2 = output_path + 'job_' + str(job_index) + \
                                '/chi2_image_data' + filename_suffix + '_' + str(n) + '.txt'
                if os.path.exists(filename_chi2):
                    new = np.loadtxt(filename_chi2)
                    if _chi2 is None:
                        _chi2 = new
                    else:
                        _chi2 = np.vstack((_chi2, new))
                else:
                    print('could not find chi2 file '+filename_chi2)
                    proceed = False
                    break
            if proceed:
                if _chi2 is None:
                    proceed = False
                elif _chi2.shape[0] != num_realizations:
                    print('chi2 file has wrong shape')
                    proceed = False

        if proceed is False:
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
                        raise Exception('could not open file '+str(filename_kwargs_fitting_seq))
                    _fitting_seq_kwargs.append(new)
                else:
                    print('could not find file '+filename_kwargs_fitting_seq)
                    proceed = False
                    break
            if len(_fitting_seq_kwargs) != num_realizations:
                print('number of saved fitting sequence classes not right')
                proceed = False

        if proceed is False:
            continue

        if keep_realizations:
            proceed = True
            # check that all the lens systmes exist
            for n in range(0, num_realizations):
                if os.path.exists(filename_realizations + 'simulation_output_' + str(n + 1)):
                    pass
                else:
                    print('could not find pickled class ' + filename_realizations + 'simulation_output_' + str(n + 1))
                    proceed = False

            if proceed is True:
                for n in range(0, num_realizations):
                    f = open(filename_realizations + 'simulation_output_' + str(n + 1), 'rb')
                    sim = pickle.load(f)
                    f.close()
                    realizations_and_lens_systems.append(sim)

        if proceed is False:
            continue

        print('compiling output for job '+str(job_index)+'... ')
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
            if keep_kappagamma_stats:
                kappa_gamma_stats = deepcopy(_kappagammastatistics)
            if keep_curvedarc_stats:
                curved_arc_stats = deepcopy(_curvedarcstatistics)
        else:
            params = np.vstack((params, _params))
            fluxes = np.vstack((fluxes, _fluxes))
            if keep_chi2:
                chi2_imaging_data = np.vstack((chi2_imaging_data, _chi2))
            if keep_macromodel_samples:
                macro_samples = np.vstack((macro_samples, _macro_samples))
            if keep_kwargs_fitting_seq:
                fitting_seq_kwargs += _fitting_seq_kwargs
            if keep_kappagamma_stats:
                kappa_gamma_stats = np.vstack((kappa_gamma_stats, _kappagammastatistics))
            if keep_curvedarc_stats:
                curved_arc_stats = np.vstack((curved_arc_stats, _curvedarcstatistics))

    print('compiled ' + str(params.shape[0]) + ' realizations')
    assert params.shape[0] == fluxes.shape[0]
    if keep_chi2:
        assert chi2_imaging_data.shape[0] == params.shape[0]
    else:
        chi2_imaging_data = None

    if keep_macromodel_samples:
        assert macro_samples.shape[0] == params.shape[0]
    else:
        macro_samples = None

    if keep_realizations:
        assert len(realizations_and_lens_systems) == params.shape[0]

    if keep_kwargs_fitting_seq:
        if save_subset_kwargs_fitting_seq:
            idx_sort = np.argsort(chi2_imaging_data[:,0]) # this is actually the log-likelihood even though it's called chi2
            n_keep_best_worst = 25
            end_idx = len(fitting_seq_kwargs) - n_keep_best_worst
            best_25 = idx_sort[0:n_keep_best_worst]
            worst_25 = idx_sort[end_idx:]
            random_inds = np.random.randint(n_keep_best_worst, end_idx, 50)
            random_50 = idx_sort[random_inds]
            fitting_seq_kwargs_out = []
            saved_inds = []
            for idx in best_25:
                fitting_seq_kwargs_out.append(fitting_seq_kwargs[idx])
                saved_inds.append(idx)
            for idx in random_50:
                fitting_seq_kwargs_out.append(fitting_seq_kwargs[idx])
                saved_inds.append(idx)
            for idx in worst_25:
                fitting_seq_kwargs_out.append(fitting_seq_kwargs[idx])
                saved_inds.append(idx)
            bic = chi2_imaging_data[:, 1]
            container = FullSimulationContainer(realizations_and_lens_systems, params,
                                                fluxes, chi2_imaging_data[:,0],
                                                fitting_seq_kwargs_out, macro_samples,
                                                bic)
            container.kwargs_fitting_seq_saved_inds = saved_inds
        else:
            assert len(fitting_seq_kwargs) == params.shape[0]
            container = FullSimulationContainer(realizations_and_lens_systems, params,
                                        fluxes, chi2_imaging_data[:,0], fitting_seq_kwargs, macro_samples, bic)
            container.kwargs_fitting_seq_saved_inds = None
    else:
        if chi2_imaging_data is None:

            container = FullSimulationContainer(realizations_and_lens_systems, params,
                                                fluxes, None, fitting_seq_kwargs, macro_samples,
                                                None, kappa_gamma_stats, curved_arc_stats)
        else:
            bic = chi2_imaging_data[:, 1]

            container = FullSimulationContainer(realizations_and_lens_systems, params,
                                            fluxes, chi2_imaging_data[:,0], fitting_seq_kwargs, macro_samples,
                                            bic, kappa_gamma_stats, curved_arc_stats)

    if keep_best_N is not None:
        container = container.cut_on_S(keep_best_N)

    return container




