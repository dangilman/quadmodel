class MacroLensModel(object):

    def __init__(self, components):

        """
        This class defines a 'macromodel'. In lensing terminology this is the global mass profile
        for the main deflector, satellite galaxies, and galaxies along the line of sight (everything
        except substructure).

        :param components: a list of macromodel components

        example:
        components = [PowerLawShear(zlens, kwargs), SISsatellite(zlens, kwargs), ... etc.]

        For description of the component classes, see the classes in LensComponents
        """

        if not isinstance(components, list):
            components = [components]
        self.components = components
        self.n_lens_models = self._count_models(components)

    def add_component(self, new_component):

        if not isinstance(new_component, list):
            new_component = [new_component]

        self.components += new_component
        self.n_lens_models = self._count_models(self.components)

    @property
    def centroid(self):
        main = self.components[0]
        x_center, y_center = main.kwargs[0]['center_x'], main.kwargs[0]['center_y']
        return x_center, y_center

    @property
    def zlens(self):
        return self.components[0].zlens

    def update_kwargs(self, new_kwargs):

        if len(new_kwargs) != self.n_lens_models:
            raise Exception('New and existing keyword arguments must be the same length.')

        count = 0
        for model in self.components:
            n = model.n_models
            new = new_kwargs[count:(count+n)]
            model.update_kwargs(new)
            count += n

    def get_lenstronomy_args(self):

        lens_model_list, redshift_list, kwargs = [], [], []
        for component in self.components:

            #model_names, model_redshifts, model_kwargs, model_convention_index = component.lenstronomy_args()
            lens_model_list += component.lens_model_list
            redshift_list += component.redshift_list
            kwargs += component.kwargs

        return lens_model_list, redshift_list, kwargs

    @staticmethod
    def _count_models(components):

        n = 0
        for component in components:
            n += component.n_models
        return n

    @property
    def lens_model_list(self):
        lens_model_list, _, _ = self.get_lenstronomy_args()
        return lens_model_list

    @property
    def redshift_list(self):
        _, redshift_list, _ = self.get_lenstronomy_args()
        return redshift_list

    @property
    def kwargs(self):
        _, _, kwargs = self.get_lenstronomy_args()
        return kwargs

