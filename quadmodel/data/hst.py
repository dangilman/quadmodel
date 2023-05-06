import numpy as np
from lenstronomy.Data.psf import PSF
from quadmodel.data.quad_base import Quad
from lenstronomy.Data.coord_transforms import Coordinates

class HSTDataModel(object):

    def __init__(self, hst_data, param_class, param_names, mcmc_samples, kwargs_result, source_model_list,
                 lens_light_model_list):
        self.hst_data = hst_data
        self.param_class = param_class
        self.param_names = param_names
        self.mcmc_samples = mcmc_samples
        self.kwargs_best_fit = kwargs_result
        self.source_model_list = source_model_list
        self.lens_light_model_list = lens_light_model_list
        self.kwargs_source_init = kwargs_result['kwargs_source']
        self.kwargs_lens_light_init = kwargs_result['kwargs_lens_light']

class HSTData(object):

    def __init__(self, zlens, zsource, image_data, x_image, y_image, astrometric_uncertainties,
                 fluxes, flux_uncertainties, deflector_centroid,
                 ra_at_xy_0, dec_at_xy_0, deltaPix, transform_pix2angle, background_rms, exposure_time,
                 mask, psf_estimate, psf_error_map, psf_symmetry, satellite_centroid_list=None, delta_x_offset=None,
                 delta_y_offset=None):

        #### quantities that describe lens geometry and coordinate system ####
        self.zlens = zlens
        self.zsource = zsource
        self.image_data = image_data
        self.x = x_image
        self.y = y_image
        self.delta_xy = astrometric_uncertainties
        self.deflector_centroid = deflector_centroid
        self.ra_at_xy_0, self.dec_at_xy_0 = ra_at_xy_0, dec_at_xy_0
        self.transform_pix2angle = transform_pix2angle
        self.deltaPix = deltaPix

        # fluxes and uncertainties
        normalized_fluxes = fluxes/np.max(fluxes)
        self.m = normalized_fluxes
        self.delta_m = flux_uncertainties

        #### data info #####
        self.background_rms = background_rms
        self.exposure_time = exposure_time
        self.likelihood_mask = mask

        #self.noise_map = None

        ### POINT SPREAD FUNCTION ####
        self.psf_estimate = psf_estimate
        self.psf_error_map = psf_error_map
        self.psf_symmetry = psf_symmetry

        #self.psf_class = PSF(**self.kwargs_psf)

        self.satellite_centroid_list = satellite_centroid_list
        self.custom_mask = None

        self.delta_x_offset = delta_x_offset
        self.delta_y_offset = delta_y_offset

    @property
    def pixel_coordinates(self):
        coords = Coordinates(self.transform_pix2angle, self.ra_at_xy_0, self.dec_at_xy_0)
        x_image_pixels, y_image_pixels = coords.map_coord2pix(self.x + self.delta_x_offset,
                                                              self.y + self.delta_y_offset)
        return x_image_pixels, y_image_pixels

    @property
    def arcsec_coordinates(self):
        coords = Coordinates(self.transform_pix2angle, self.ra_at_xy_0, self.dec_at_xy_0)
        x_image_pixels, y_image_pixels = self.pixel_coordinates
        return coords.map_pix2coord(x_image_pixels, y_image_pixels)

    def get_lensed_image(self, mask=False):

        if mask:
            N = len(self.image_data)
            data = self.image_data * self.likelihood_mask.reshape(N, N)
        else:
            data = self.image_data
        return data

    def update_psf(self, new_kps, new_error_map):

        self.updated_psf, self.updated_psf_error_map = new_kps, new_error_map

    @property
    def kwargs_psf(self):

        psf_estimate, error_map = self.best_psf_estimate

        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': psf_estimate,
                      'psf_error_map': error_map}

        return kwargs_psf

    @property
    def best_psf_estimate(self):

        if self.updated_psf is None:
            print('using PSF estimate from initial star construction')
            return self.kernel_point_source_init, self.psf_error_map_init
        else:
            print('using PSF estimate from lenstronomy iteration during fitting sequence')
            return self.updated_psf, self.updated_psf_error_map

    @property
    def kwargs_data(self):

        kwargs_data = {'image_data': self.image_data,
                       'background_rms': self.background_rms,
                       'noise_map': self.noise_map,
                       'exposure_time': self.exposure_time,
                       'ra_at_xy_0': self.ra_at_xy_0,
                       'dec_at_xy_0': self.dec_at_xy_0,
                       'transform_pix2angle': self.transform_pix2angle
                       }

        return kwargs_data
