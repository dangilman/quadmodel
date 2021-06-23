from quadmodel.deflector_models.model_base import ComponentBase

class SIS(ComponentBase):

    def __init__(self, redshift, kwargs_init=None):

        self._redshift = redshift

        super(SIS, self).__init__(self.lens_model_list, [redshift], kwargs_init)

    @property
    def n_models(self):
        return 1

    @property
    def lens_model_list(self):
        return ['SIS']

    @property
    def redshift_list(self):
        return [self._redshift] * len(self.lens_model_list)
