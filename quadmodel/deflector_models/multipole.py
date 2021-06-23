from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

pi = 3.14159265359

class Multipole(ComponentBase):

    def __init__(self, redshift, kwargs_init=None):


        """
        kwargs_init include 'm', 'a_m', 'phi_m', 'center_x', 'center_y'
        """
        self._redshift = redshift

        self._m = kwargs_init[0]['m']
        self._am = kwargs_init[0]['a_m']

        super(Multipole, self).__init__(self.lens_model_list, [redshift], kwargs_init)

    @property
    def n_models(self):
        return 1

    @property
    def lens_model_list(self):
        return ['MULTIPOLE']

    @property
    def redshift_list(self):
        return [self._redshift] * len(self.lens_model_list)
