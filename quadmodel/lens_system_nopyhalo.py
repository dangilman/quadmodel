from lenstronomy.LensModel.lens_model import LensModel
from copy import deepcopy


class LensSystem(object):

    def __init__(self, z_lens, z_source, x_image, y_image, kwargs_lens_model, kwargs_lens, astropy):

        self.zlens = z_lens
        self.zsource = z_source
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

    def get_macro_lensmodel(self, n_macro):

        kwargs_lens_model = deepcopy(self.kwargs_lens_model)
        kwargs_lens_model['cosmo'] = self.astropy
        lens_model_list = self.kwargs_lens_model['lens_model_list'][0:n_macro]
        lens_redshift_list = self.kwargs_lens_model['lens_redshift_list'][0:n_macro]
        lens_model = LensModel(lens_model_list, lens_redshift_list=lens_redshift_list,
                               multi_plane=True, z_source=self.zsource, cosmo=kwargs_lens_model['cosmo'])
        return lens_model, self.kwargs_lens[0:n_macro]
