from copy import deepcopy
import numpy as np
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from scipy.interpolate import RegularGridInterpolator
from lenstronomy.ImSim.Numerics.grid import RegularGrid
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from tqdm import tqdm


__all__ = ['source_params_sersic_ellipse', 'lens_light_params_sersic_ellipse',
           'lensmodel_params', 'ps_params', 'mask_images',
           'source_params_shapelets', 'FittingSequenceKwargs', 'customized_mask', 'FixedLensModel', 'FixedLensModelNew']

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

    def __init__(self, ra_coords, dec_coords, lens_model, kwargs_lens):

        shape0 = ra_coords.shape
        alpha_x, alpha_y = lens_model.alpha(ra_coords.ravel(), dec_coords.ravel(), kwargs_lens)
        points = (ra_coords[0, :], dec_coords[:, 0])
        self._interp_x = RegularGridInterpolator(points, alpha_x.reshape(shape0), bounds_error=False, fill_value=None)
        self._interp_y = RegularGridInterpolator(points, alpha_y.reshape(shape0), bounds_error=False, fill_value=None)

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

class FixedLensModelNew(object):

    def __init__(self, nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0,
                 lens_model, kwargs_lens, super_sample_factor=1):

        grid = RegularGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, super_sample_factor)
        ra_coords, dec_coords = grid.coordinates_evaluate

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

def extract_lens_models(simulation_output, index_max=None):

    if index_max is None:
        index_max = len(simulation_output.simulations)
        assert  len(simulation_output.simulations) == simulation_output.parameters.shape[0]

    lens_models = []
    kwargs_lens_list = []
    ximg_list = []
    yimg_list = []
    zd_list = []
    for index_realization in tqdm(range(0, index_max), desc="Extracting lens models..."):
        lens_model, kwargs_lens, ximg, yimg, zd = _extract_lens_model(simulation_output, index_realization)
        lens_models.append(lens_model)
        kwargs_lens_list.append(kwargs_lens)
        ximg_list.append(ximg)
        yimg_list.append(yimg)
        zd_list.append(zd)
    return lens_models, kwargs_lens_list, ximg_list, yimg_list, zd_list

def curved_arc_statistics(lens_models, kwargs_lens_list, ximg_list, yimg_list, zd_list, index_image, diff=None):

    index_max = len(lens_models)
    radial_stretch = np.empty(index_max)
    tangential_stretch = np.empty(index_max)
    curvature = np.empty(index_max)
    direction = np.empty(index_max)
    dtan_dtan = np.empty(index_max)
    for index_realization in tqdm(range(0, index_max), desc="Computing arc statistics..."):
        rs, ts, c, d, dtdt = curved_arc_statistics_single(lens_models[index_realization],
                                                    kwargs_lens_list[index_realization],
                                                    ximg_list[index_realization][index_image],
                                                     yimg_list[index_realization][index_image],
                                                    zd_list[index_realization],
                                                    diff)
        radial_stretch[index_realization] = rs
        tangential_stretch[index_realization] = ts
        curvature[index_realization] = c
        direction[index_realization] = d
        dtan_dtan[index_realization] = dtdt
    return radial_stretch, tangential_stretch, curvature, direction, dtan_dtan

def curved_arc_statistics_single(lens_model, kwargs_lens, x_image, y_image, z_lens, diff=None):
    ext = LensModelExtensions(lens_model)
    d_c_lens = lens_model.cosmo.comoving_distance(z_lens).value
    xi_comoving, yi_comoving, _, _ = lens_model.lens_model.ray_shooting_partial(
        0.0, 0.0, x_image, y_image, 0.0, z_lens, kwargs_lens)
    xi_lensed, yi_lensed = xi_comoving / d_c_lens, yi_comoving / d_c_lens
    kwargs_arc = ext.curved_arc_estimate(xi_lensed, yi_lensed, kwargs_lens, smoothing=diff,
                                         smoothing_3rd=diff, tan_diff=True)
    radial_stretch = kwargs_arc['radial_stretch']
    tangential_stretch = kwargs_arc['tangential_stretch']
    curvature = kwargs_arc['curvature']
    direction = kwargs_arc['direction']
    dtan_dtan = kwargs_arc['dtan_dtan']
    return radial_stretch, tangential_stretch, curvature, direction, dtan_dtan

def kappa_gamma_statistics(lens_models, kwargs_lens_list, ximg_list, yimg_list, zd_list, index_image, diff=None):
    index_max = len(lens_models)
    kappa = np.empty(index_max)
    g1 = np.empty(index_max)
    g2 = np.empty(index_max)
    for index_realization in tqdm(range(0, index_max), desc="Computing kappa/gamma statistics..."):
        k, g_1, g_2 = kappa_gamma_single(lens_models[index_realization],
                                                          kwargs_lens_list[index_realization],
                                                          ximg_list[index_realization][index_image],
                                                          yimg_list[index_realization][index_image],
                                                          zd_list[index_realization],
                                                          diff)
        kappa[index_realization] = k
        g1[index_realization] = g_1
        g2[index_realization] = g_2
    return kappa, g1, g2

def kappa_gamma_single(lens_model, kwargs_lens, x_image, y_image, z_lens, diff=None):

    d_c_lens = lens_model.cosmo.comoving_distance(z_lens).value
    xi_comoving, yi_comoving, _, _ = lens_model.lens_model.ray_shooting_partial(
        0.0, 0.0, x_image, y_image, 0.0, z_lens, kwargs_lens)
    xi_lensed, yi_lensed = xi_comoving / d_c_lens, yi_comoving / d_c_lens
    fxx, fxy, fyx, fyy = lens_model.hessian(xi_lensed, yi_lensed, kwargs_lens, diff=diff)
    kappa = 1. / 2 * (fxx + fyy)
    gamma1 = 1. / 2 * (fxx - fyy)
    gamma2 = 1. / 2 * (fxy + fyx)
    return kappa, gamma1, gamma2

def _extract_lens_model(simulation_output, index):
    lens_model, kwargs_lens = simulation_output.simulations[index].lens_system.get_lensmodel()
    ximg = simulation_output.simulations[index].data.x
    yimg = simulation_output.simulations[index].data.y
    zd = simulation_output.simulations[index].lens_system.zlens
    return lens_model, kwargs_lens, ximg, yimg, zd


def constrain_arc_params(param_list, stat_array, truths, truth_sigmas, sigma_scale=1.0):
    column_index = {'rs1': 0, 'ts1': 1, 'curv1': 2, 'dir1': 3, 'dtdt1': 4,
                    'rs2': 5, 'ts2': 6, 'curv2': 7, 'dir2': 8, 'dtdt2': 9,
                    'rs3': 10, 'ts3': 11, 'curv3': 12, 'dir3': 13, 'dtdt3': 14,
                    'rs4': 15, 'ts4': 16, 'curv4': 17, 'dir4': 18, 'dtdt4': 19}
    N = len(stat_array[:, 0])
    while True:
        weight = 1.
        for param in param_list:
            index = column_index[param]
            stat = stat_array[:, index]
            weight *= np.exp(
                -0.5 * (stat_array[:, index] - truths[param]) ** 2 / (sigma_scale * truth_sigmas[param]) ** 2)
        weight *= np.max(weight) ** -1
        effective_sample_size = np.sum(weight)
        if effective_sample_size / N < 0.1:
            sigma_scale += 0.01
        else:
            break
    print('sigma scale:', sigma_scale)
    return weight

def constrain_kappagamma_params(param_list, stat_array, truths, truth_sigmas, sigma_scale=1.0):
    column_index = {'kappa1': 0, 'gamma1_1': 1, 'gamma2_1': 2,
                    'kappa2': 3, 'gamma1_2': 4, 'gamma2_2': 5,
                    'kappa3': 6, 'gamma1_3': 7, 'gamma2_3': 8,
                    'kappa4': 9, 'gamma1_4': 10, 'gamma2_4': 11}
    N = len(stat_array[:, 0])
    while True:
        weight = 1.
        for param in param_list:
            index = column_index[param]
            stat = stat_array[:, index]
            dx = stat_array[:, index] - truths[param]
            sig = sigma_scale * truth_sigmas[param]
            exp_arg = (dx / sig) ** 2
            weight *= np.exp(-0.5 * dx ** 2 / sig ** 2)
        #             print(param, exp_arg)
        #             a=input('continue')
        weight *= np.max(weight) ** -1
        effective_sample_size = np.sum(weight)
        if effective_sample_size / N < 0.1:
            sigma_scale += 0.01
        else:
            break
    print('sigma scale:', sigma_scale)
    return weight


def load_kappagamma_stats(fname):
    x = np.loadtxt(fname, unpack=False).reshape(4, 1000, 3)
    x_med = {}
    x_stdev = {}
    x_med['kappa1'] = np.median(x[0, :, 0])
    x_med['gamma1_1'] = np.median(x[0, :, 1])
    x_med['gamma2_1'] = np.median(x[0, :, 2])
    x_med['kappa2'] = np.median(x[1, :, 0])
    x_med['gamma1_2'] = np.median(x[1, :, 1])
    x_med['gamma2_2'] = np.median(x[1, :, 2])
    x_med['kappa3'] = np.median(x[2, :, 0])
    x_med['gamma1_3'] = np.median(x[2, :, 1])
    x_med['gamma2_3'] = np.median(x[2, :, 2])
    x_med['kappa4'] = np.median(x[3, :, 0])
    x_med['gamma1_4'] = np.median(x[3, :, 1])
    x_med['gamma2_4'] = np.median(x[3, :, 2])

    x_stdev['kappa1'] = np.std(x[0, :, 0])
    x_stdev['gamma1_1'] = np.std(x[0, :, 1])
    x_stdev['gamma2_1'] = np.std(x[0, :, 2])
    x_stdev['kappa2'] = np.std(x[1, :, 0])
    x_stdev['gamma1_2'] = np.std(x[1, :, 1])
    x_stdev['gamma2_2'] = np.std(x[1, :, 2])
    x_stdev['kappa3'] = np.std(x[2, :, 0])
    x_stdev['gamma1_3'] = np.std(x[2, :, 1])
    x_stdev['gamma2_3'] = np.std(x[2, :, 2])
    x_stdev['kappa4'] = np.std(x[3, :, 0])
    x_stdev['gamma1_4'] = np.std(x[3, :, 1])
    x_stdev['gamma2_4'] = np.std(x[3, :, 2])

    x_dict = {}
    x_dict['kappa1'] = x[0, :, 0]
    x_dict['gamma1_1'] = x[0, :, 1]
    x_dict['gamma2_1'] = x[0, :, 2]
    x_dict['kappa2'] = x[1, :, 0]
    x_dict['gamma1_2'] = x[1, :, 1]
    x_dict['gamma2_2'] = x[1, :, 2]
    x_dict['kappa3'] = x[2, :, 0]
    x_dict['gamma1_3'] = x[2, :, 1]
    x_dict['gamma2_3'] = x[2, :, 2]
    x_dict['kappa4'] = x[3, :, 0]
    x_dict['gamma1_4'] = x[3, :, 1]
    x_dict['gamma2_4'] = x[3, :, 2]
    return x_med, x_stdev, x_dict


def load_arc_stats(fname):
    x = np.loadtxt(fname, unpack=False).reshape(4, 1000, 5)
    x_med = {}
    x_stdev = {}
    x_med['ts1'] = np.median(x[0, :, 0])
    x_med['rs1'] = np.median(x[0, :, 1])
    x_med['curv1'] = np.median(x[0, :, 2])
    x_med['dir1'] = np.median(x[0, :, 3])
    x_med['dtdt1'] = np.median(x[0, :, 4])
    x_med['ts2'] = np.median(x[1, :, 0])
    x_med['rs2'] = np.median(x[1, :, 1])
    x_med['curv2'] = np.median(x[1, :, 2])
    x_med['dir2'] = np.median(x[1, :, 3])
    x_med['dtdt2'] = np.median(x[1, :, 4])
    x_med['ts3'] = np.median(x[2, :, 0])
    x_med['rs3'] = np.median(x[2, :, 1])
    x_med['curv3'] = np.median(x[2, :, 2])
    x_med['dir3'] = np.median(x[2, :, 3])
    x_med['dtdt3'] = np.median(x[2, :, 4])
    x_med['ts4'] = np.median(x[3, :, 0])
    x_med['rs4'] = np.median(x[3, :, 1])
    x_med['curv4'] = np.median(x[3, :, 2])
    x_med['dir4'] = np.median(x[3, :, 3])
    x_med['dtdt4'] = np.median(x[3, :, 4])

    x_stdev['ts1'] = np.std(x[0, :, 0])
    x_stdev['rs1'] = np.std(x[0, :, 1])
    x_stdev['curv1'] = np.std(x[0, :, 2])
    x_stdev['dir1'] = np.std(x[0, :, 3])
    x_stdev['dtdt1'] = np.std(x[0, :, 4])
    x_stdev['ts2'] = np.std(x[1, :, 0])
    x_stdev['rs2'] = np.std(x[1, :, 1])
    x_stdev['curv2'] = np.std(x[1, :, 2])
    x_stdev['dir2'] = np.std(x[1, :, 3])
    x_stdev['dtdt2'] = np.std(x[1, :, 4])
    x_stdev['ts3'] = np.std(x[2, :, 0])
    x_stdev['rs3'] = np.std(x[2, :, 1])
    x_stdev['curv3'] = np.std(x[2, :, 2])
    x_stdev['dir3'] = np.std(x[2, :, 3])
    x_stdev['dtdt3'] = np.std(x[2, :, 4])
    x_stdev['ts4'] = np.std(x[3, :, 0])
    x_stdev['rs4'] = np.std(x[3, :, 1])
    x_stdev['curv4'] = np.std(x[3, :, 2])
    x_stdev['dir4'] = np.std(x[3, :, 3])
    x_stdev['dtdt4'] = np.std(x[3, :, 4])

    x_dict = {}
    x_dict['ts1'] = x[0, :, 0]
    x_dict['rs1'] = x[0, :, 1]
    x_dict['curv1'] = x[0, :, 2]
    x_dict['dir1'] = x[0, :, 3]
    x_dict['dtdt1'] = x[0, :, 4]
    x_dict['ts2'] = x[1, :, 0]
    x_dict['rs2'] = x[1, :, 1]
    x_dict['curv2'] = x[1, :, 2]
    x_dict['dir2'] = x[1, :, 3]
    x_dict['dtdt2'] = x[1, :, 4]
    x_dict['ts3'] = x[2, :, 0]
    x_dict['rs3'] = x[2, :, 1]
    x_dict['curv3'] = x[2, :, 2]
    x_dict['dir3'] = x[2, :, 3]
    x_dict['dtdt3'] = x[2, :, 4]
    x_dict['ts4'] = x[3, :, 0]
    x_dict['rs4'] = x[3, :, 1]
    x_dict['curv4'] = x[3, :, 2]
    x_dict['dir4'] = x[3, :, 3]
    x_dict['dtdt4'] = x[3, :, 4]

    return x_med, x_stdev, x_dict
