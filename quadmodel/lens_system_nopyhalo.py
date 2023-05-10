from lenstronomy.LensModel.lens_model import LensModel
from copy import deepcopy


class LensSystem(object):

    def __init__(self, z_lens, z_source, x_image, y_image, kwargs_lens_model, kwargs_lens, astropy):

        self.z_lens = z_lens
        self.z_source = z_source
        self.x = x_image
        self.y = y_image
        if 'cosmo' in kwargs_lens_model.keys():
            del kwargs_lens_model['cosmo']
        self.kwargs_lens_model = kwargs_lens_model
        self.kwargs_lens = kwargs_lens
        self.astropy = astropy

    def get_lensmodel(self):

        kwargs_lens_model = deepcopy(self.kwargs_lens_model)
        kwargs_lens_model['cosmo'] = self.astropy
        lens_model = LensModel(**self.kwargs_lens_model)
        return lens_model, self.kwargs_lens
