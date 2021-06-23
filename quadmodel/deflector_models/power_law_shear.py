from quadmodel.deflector_models.model_base import ComponentBase

class PowerLawShear(ComponentBase):

    def __init__(self, redshift, kwargs_init=None):

        """
        This class defines an ellipsoidal power law mass profile plus external shear
        one of the more commmon models used in lensing.

        :param redshift: the redshift of the mass profile
        :param kwargs_init: the key word arguments for lenstronomy
        Example:
        = [{power law profile}, {external shear}]
        = [{'theta_E': 1, 'center_x': 0., 'center_y':, 0., 'e1': 0.1, 'e2': -0.2, 'gamma': 2.03},
        {'gamma1': 0.04, 'gamma2': -0.02}]
        """

        super(PowerLawShear, self).__init__(self.lens_model_list, [redshift]*self.n_models,
                                            kwargs_init)

    @property
    def n_models(self):
        return 2

    @property
    def lens_model_list(self):
        return ['EPL', 'SHEAR']

    @property
    def redshift_list(self):
        return [self.zlens] * len(self.lens_model_list)
