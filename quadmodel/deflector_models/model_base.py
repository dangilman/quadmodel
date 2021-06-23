class ComponentBase(object):

    def __init__(self, lens_model_names, redshifts, kwargs):

        self.zlens = redshifts[0]
        self.redshifts = redshifts
        self.lens_model_names = lens_model_names
        self.update_kwargs(kwargs)
        self.x_center, self.y_center = kwargs[0]['center_x'], kwargs[0]['center_y']

    def update_kwargs(self, kwargs):
        self._kwargs = kwargs

    @property
    def kwargs(self):
        return self._kwargs
