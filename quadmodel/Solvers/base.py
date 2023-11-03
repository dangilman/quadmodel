from quadmodel.Solvers.optimization_param_managers import PowerLawFixedShear, \
    PowerLawFixedShearMultipole, PowerLawFreeShear, PowerLawFreeShearMultipole, PowerLawFixedShearMultipole_34, \
    PowerLawFreeShearMultipole_34


class OptimizationBase(object):

    def __init__(self, lens_system):

        self.lens_system = lens_system

    @staticmethod
    def set_param_class(param_class_name, constrain_params):

        if param_class_name == 'free_shear_powerlaw':
            return PowerLawFreeShear, None
        elif param_class_name == 'fixed_shear_powerlaw':
            return PowerLawFixedShear, [constrain_params['shear']]
        elif param_class_name == 'free_shear_powerlaw_multipole':
            return PowerLawFreeShearMultipole, None
        elif param_class_name == 'fixed_shear_powerlaw_multipole':
            return PowerLawFixedShearMultipole, [constrain_params['shear']]
        elif param_class_name == 'fixed_shear_powerlaw_multipole_34':
            return PowerLawFixedShearMultipole_34, [constrain_params['shear'], constrain_params['delta_phi_m3']]
        elif param_class_name == 'free_shear_powerlaw_multipole_34':
            return PowerLawFreeShearMultipole_34, [constrain_params['delta_phi_m3']]

        else:
            raise Exception('did not recognize param_class_name = ' + param_class_name)

    def return_results(self, source, kwargs_lens_final, lens_model_full, realization_final,
                       kwargs_return=None):

        self.update_lens_system(source, kwargs_lens_final, lens_model_full,
                                realization_final)

        return kwargs_lens_final, lens_model_full, kwargs_return

    def update_lens_system(self, source_centroid, new_kwargs, lens_model_full, realization_final):

        self.lens_system.clear_static_lensmodel()

        self.lens_system.update_source_centroid(source_centroid[0], source_centroid[1])

        index_max = self.lens_system.macromodel.n_lens_models
        self.lens_system.update_kwargs_macro(new_kwargs[0:index_max])

        self.lens_system.update_light_centroid(new_kwargs[0]['center_x'], new_kwargs[0]['center_y'])

        self.lens_system.update_realization(realization_final)

        self.lens_system.set_lensmodel_static(lens_model_full, new_kwargs)
