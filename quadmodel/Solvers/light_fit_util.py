from copy import deepcopy
import numpy as np
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from scipy.interpolate import RegularGridInterpolator

__all__ = ['source_params_sersic_ellipse', 'lens_light_params_sersic_ellipse',
           'lensmodel_params', 'ps_params', 'mask_images',
           'source_params_shapelets', 'FittingSequenceKwargs', 'customized_mask', 'FixedLensModel']


def customized_mask(x_image, y_image, ra_grid, dec_grid, mask_image_arcsec, r_semi_major_arcsec, q, rotation,
                    thickness_arcsec, shift_x=0.0, shift_y=0.0):

    baseline_mask = np.ones_like(ra_grid)
    for (xi, yi) in zip(x_image, y_image):
        dx = abs(xi - ra_grid)
        dy = abs(yi - dec_grid)
        dr = np.hypot(dx, dy)
        inds_mask = np.where(dr <= mask_image_arcsec)
        baseline_mask[inds_mask] = 0.0
    dr_mins = []
    theta = np.linspace(0, 2 * np.pi, 1000)

    x_ellipse = r_semi_major_arcsec * np.cos(theta) - shift_y
    y_ellipse = r_semi_major_arcsec * np.sin(theta) / q - shift_x
    rotation *= np.pi / 180
    x_ellipse, y_ellipse = x_ellipse * np.cos(rotation) + y_ellipse * np.sin(rotation), -x_ellipse * np.sin(
        rotation) + y_ellipse * np.cos(rotation)
    for (xi, yi) in zip(ra_grid.ravel(), dec_grid.ravel()):
        dx = xi - x_ellipse
        dy = yi - y_ellipse
        dr = np.sqrt(dx ** 2 + (dy) ** 2)
        dr_min = np.min(dr)
        dr_mins.append(dr_min)
    dr_mins = np.array(dr_mins).reshape(ra_grid.shape)
    ring_mask = np.ones_like(baseline_mask)
    ring_mask[np.where(dr_mins > thickness_arcsec)] = 0.0
    baseline_mask *= ring_mask
    return baseline_mask

def source_params_sersic_ellipse(source_x, source_y, kwargs_init):

    kwargs_sigma = [{'amp': 50000,
                    'R_sersic': 0.1,
                     'n_sersic': 2.0,
                     'e1': 0.25, 'e2': 0.25,
                        'center_x': 0.1, 'center_y': 0.1}]
    kwargs_lower = [{'amp': 1e-9, 'R_sersic': 0.001, 'n_sersic': 1.0, 'e1': -0.4, 'e2': -0.4, 'center_x': -10, 'center_y': -10}]
    kwargs_upper = [{'amp': 1e9, 'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.4, 'e2': 0.4, 'center_x': 10, 'center_y': 10}]
    kwargs_fixed = [{'center_x': source_x, 'center_y': source_y,
                     #'e1': kwargs_init[0]['e1'],
                     #'e2': kwargs_init[0]['e2']
                    }
                    ]
    return kwargs_sigma, kwargs_lower, kwargs_upper, kwargs_fixed

def source_params_shapelets(n_max, source_x, source_y):
    kwargs_sigma = [{'beta': 0.2, 'amp': 10.0, 'n_max': 1.0, 'center_x': 0.1, 'center_y': 0.1}]
    kwargs_lower = [{'beta': 1e-9, 'amp': 0.0, 'n_max': 1, 'center_x': -10, 'center_y': -10}]
    kwargs_upper = [{'beta': 1e9, 'amp': 1e9, 'n_max': 1000, 'center_x': 10, 'center_y': 10}]
    kwargs_fixed = [{'n_max': n_max, 'center_x': source_x, 'center_y': source_y}]
    return kwargs_sigma, kwargs_lower, kwargs_upper, kwargs_fixed

def lens_light_params_sersic_ellipse(kwargs_lens_light_init):

    kwargs_sigma = [{'amp': 100, 'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1,
                     'center_x': 0.1, 'center_y': 0.1}]
    kwargs_lower = [{'amp': 1e-9, 'R_sersic': 0.001, 'n_sersic': 1.0, 'e1': -0.4, 'e2': -0.4, 'center_x': -10, 'center_y': -10}]
    kwargs_upper = [{'amp': 1e9, 'R_sersic': 5.0, 'n_sersic': 10.0, 'e1': 0.4, 'e2': 0.4, 'center_x': 10, 'center_y': 10}]
    kwargs_fixed = [{}]
    return kwargs_sigma, kwargs_lower, kwargs_upper, kwargs_fixed

def lensmodel_params(lens_model_list, kwargs_lens_fixed):

    kwargs_sigma = deepcopy(kwargs_lens_fixed)
    kwargs_lower = deepcopy(kwargs_lens_fixed)
    kwargs_upper = deepcopy(kwargs_lens_fixed)
    kwargs_fixed = deepcopy(kwargs_lens_fixed)
    for index_shear, lens_model_name in enumerate(lens_model_list):
        if lens_model_name=='SHEAR':
            kwargs_fixed[index_shear]['ra_0'] = 0.0
            kwargs_fixed[index_shear]['dec_0'] = 0.0
            break
    else:
        raise Exception('lens model list must contain SHEAR')
    return kwargs_sigma, kwargs_lower, kwargs_upper, kwargs_fixed

def ps_params(x_image, y_image):

    kwargs_sigma = [{'ra_image': [0.01]*len(x_image), 'dec_image': [0.01]*len(y_image)}]
    kwargs_lower = [{'ra_image': x_image - 0.1, 'dec_image': y_image - 0.1}]
    kwargs_upper = [{'ra_image': x_image + 0.1, 'dec_image': y_image + 0.1}]
    kwargs_fixed = [{'ra_image': x_image, 'dec_image': y_image}]
    return kwargs_sigma, kwargs_lower, kwargs_upper, kwargs_fixed

def mask_images(x_image, y_image, radius, likelihood_mask, coordinate_system):
    new_mask = deepcopy(likelihood_mask)
    (n, n) = likelihood_mask.shape
    for (xi, yi) in zip(x_image, y_image):
        xi_pix, yi_pix = coordinate_system.map_coord2pix(xi, yi)
        _r = np.linspace(0, n, n)
        _xx, _yy = np.meshgrid(_r, _r)
        _rr = np.hypot(_xx - xi_pix, _yy - yi_pix)
        inds = np.where(_rr < radius)
        new_mask[inds] = 0.0
    return new_mask

class FittingSequenceKwargs(object):

    def __init__(self, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params,
                 kwargs_result):

        multi_band_list = kwargs_data_joint['multi_band_list']
        multi_band_type = kwargs_data_joint['multi_band_type']
        image_likelihood_mask_list = kwargs_likelihood['image_likelihood_mask_list']

        self.kwargs_fitting_sequence = {'kwargs_data_joint': kwargs_data_joint,
                      'kwargs_model': kwargs_model,
                       'kwargs_constraints': kwargs_constraints,
                        'kwargs_likelihood': kwargs_likelihood,
                        'kwargs_params': kwargs_params}

        self.kwargs_model_plot = {'multi_band_list': multi_band_list,
                                  'kwargs_model': kwargs_model,
                                  'kwargs_params': kwargs_result,
                                  'image_likelihood_mask_list': image_likelihood_mask_list,
                                  'multi_band_type': multi_band_type}

    @property
    def modelplot(self):
        return ModelPlot(**self.kwargs_model_plot)

    @property
    def fitting_sequence(self):
        return FittingSequence(**self.kwargs_fitting_sequence)

    @property
    def log_likelihood(self):
        fitting_sequence = self.fitting_sequence
        log_like = fitting_sequence.best_fit_likelihood
        return log_like

    @property
    def reduced_chi2(self):
        fitting_sequence = self.fitting_sequence
        log_like = fitting_sequence.best_fit_likelihood
        n_dof = self.fitting_sequence.likelihoodModule.num_data
        reduced_chi2 = -2 * log_like / n_dof
        return reduced_chi2

class FixedLensModel(object):

    def __init__(self, ra_coords, dec_coords, lens_model, kwargs_lens, super_sample_factor=1.0):

        nx_0 = int(np.sqrt(len(ra_coords.ravel())))
        ny_0 = int(np.sqrt(len(dec_coords.ravel())))
        nx = int(nx_0 * super_sample_factor)
        ny = int(ny_0 * super_sample_factor)
        _ra_coords = np.linspace(np.min(ra_coords), np.max(ra_coords), nx)
        _dec_coords = np.linspace(np.min(dec_coords), np.max(dec_coords), ny)
        ra_coords, dec_coords = np.meshgrid(_ra_coords, _dec_coords)

        alpha_x, alpha_y = lens_model.alpha(ra_coords.ravel(), dec_coords.ravel(), kwargs_lens)
        points = (ra_coords[0, :], dec_coords[:, 0])
        self._interp_x = RegularGridInterpolator(points, alpha_x.reshape(nx, ny), bounds_error=False, fill_value=None)
        self._interp_y = RegularGridInterpolator(points, alpha_y.reshape(nx, ny), bounds_error=False, fill_value=None)

    def __call__(self, x, y, *args, **kwargs):

        point = (y, x)
        alpha_x = self._interp_x(point)
        alpha_y = self._interp_y(point)

        if isinstance(x, float) or isinstance(x, int) and isinstance(y, float) or isinstance(y, int):
            alpha_x = float(alpha_x)
            alpha_y = float(alpha_y)
        else:
            alpha_x = np.squeeze(alpha_x)
            alpha_y = np.squeeze(alpha_y)

        return alpha_x, alpha_y
