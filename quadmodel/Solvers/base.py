class OptimizationBase(object):

    def __init__(self, lens_system):

        self.lens_system = lens_system

    def return_results(self, source, kwargs_lens_final, lens_model_full, realization_final,
                       kwargs_return=None):

        self.update_lens_system(source, kwargs_lens_final, lens_model_full, realization_final)

        return kwargs_lens_final, lens_model_full, kwargs_return

    def update_lens_system(self, source_centroid, new_kwargs, lens_model_full, realization_final):

        self.lens_system.clear_static_lensmodel()

        self.lens_system.update_source_centroid(source_centroid[0], source_centroid[1])

        self.lens_system.update_kwargs_macro(new_kwargs)

        self.lens_system.update_light_centroid(new_kwargs[0]['center_x'], new_kwargs[0]['center_y'])

        self.lens_system.update_realization(realization_final)

        self.lens_system.set_lensmodel_static(lens_model_full, new_kwargs)
