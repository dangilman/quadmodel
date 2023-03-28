import numpy as np

class MacromodelInitSampler(object):

    def __init__(self, kde):
        self._kde = kde

    def __call__(self):
        param_names_in_order = ['theta_E', 'center_x', 'center_y', 'e1', 'e2']
        sample = self._kde.resample(1)
        kwargs_macro_init = {}
        for i, param_name in enumerate(param_names_in_order):
            kwargs_macro_init[param_name] = float(sample[i])
        return kwargs_macro_init

class ImportanceWeightFunction(object):

    def __init__(self, interpolated_likelihood_function):

        self._interpolated_likelihood_function = interpolated_likelihood_function
        self._param_names = interpolated_likelihood_function.param_names

    def __call__(self, realization_samples, realization_param_names,
                 macromodel_samples, macromodel_param_names,
                 source_samples, source_param_names,
                 verbose):

        idx_keep_realization = []
        idx_keep_macro = []
        idx_keep_source = []
        for param in self._param_names:
            if param in realization_param_names:
                idx = realization_param_names.index(param)
                idx_keep_realization.append(idx)
                continue
            elif param in macromodel_param_names:
                idx = macromodel_param_names.index(param)
                idx_keep_macro.append(idx)
                continue
            elif param in source_param_names:
                idx = source_param_names.index(param)
                idx_keep_source.append(idx)
                continue
            else:
                raise Exception('param ' + str(param) + ' missing from list of sampled parameters')
        samples_realization = np.array([])
        samples_macro = np.array([])
        samples_source = np.array([])
        if len(idx_keep_realization) > 0:
            samples_realization = np.array(realization_samples)[idx_keep_realization]
        if len(idx_keep_macro) > 0:
            samples_macro = np.array(macromodel_samples)[idx_keep_macro]
        if len(idx_keep_source) > 0:
            samples_source = np.array(source_samples)[idx_keep_source]
        samples = np.append(np.append(samples_realization, samples_macro), samples_source)
        return self._interpolated_likelihood_function(tuple(samples))
