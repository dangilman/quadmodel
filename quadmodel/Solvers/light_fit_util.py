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
    if lens_model.multi_plane:
        d_c_lens = lens_model.cosmo.comoving_distance(z_lens).value
        xi_comoving, yi_comoving, _, _ = lens_model.lens_model.ray_shooting_partial(
            0.0, 0.0, x_image, y_image, 0.0, z_lens, kwargs_lens)
        xi_lensed, yi_lensed = xi_comoving / d_c_lens, yi_comoving / d_c_lens
    else:
        xi_lensed, yi_lensed = x_image, y_image
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

def split_kappa_gamma_params(params, keep_scale=1, oneD=False):
    if oneD:
        params = params[np.newaxis, :]
    kappa_scale1 = params[:, 0:4]
    kappa_scale2 = params[:, 4:8]
    kappa_scale3 = params[:, 8:12]
    g1_scale1 = params[:, 12:16]
    g1_scale2 = params[:, 16:20]
    g1_scale3 = params[:, 20:24]
    g2_scale1 = params[:, 24:28]
    g2_scale2 = params[:, 28:32]
    g2_scale3 = params[:, 32:36]

    if keep_scale == 1:
        kappa = kappa_scale1
        gamma1 = g1_scale1
        gamma2 = g2_scale1
    elif keep_scale == 2:
        kappa = kappa_scale2
        gamma1 = g1_scale2
        gamma2 = g2_scale2
    elif keep_scale == 3:
        kappa = kappa_scale3
        gamma1 = g1_scale3
        gamma2 = g2_scale3
    kappa = np.squeeze(kappa)
    gamma1 = np.squeeze(gamma1)
    gamma2 = np.squeeze(gamma2)
    if oneD:
        image1 = np.array([kappa[0], gamma1[0], gamma2[0]])
        image2 = np.array([kappa[1], gamma1[1], gamma2[1]])
        image3 = np.array([kappa[2], gamma1[2], gamma2[2]])
        image4 = np.array([kappa[3], gamma1[3], gamma2[3]])
    else:
        image1 = np.array([kappa[:, 0], gamma1[:, 0], gamma2[:, 0]])
        image2 = np.array([kappa[:, 1], gamma1[:, 1], gamma2[:, 1]])
        image3 = np.array([kappa[:, 2], gamma1[:, 2], gamma2[:, 2]])
        image4 = np.array([kappa[:, 3], gamma1[:, 3], gamma2[:, 3]])
    return image1.T, image2.T, image3.T, image4.T

def chi2_kappa_gamma_statistics_single_image(stats, kap_true, g1_true, g2_true,
                                kappa_sigma=10000, g1_sigma=0.05, g2_sigma=0.05):
    d_kappa = stats[:,0] - kap_true
    d_g1 = stats[:,1] - g1_true
    d_g2 = stats[:,2] - g2_true
    w1 = np.exp(-d_kappa**2 / 2 / kappa_sigma**2)
    w2 = np.exp(-d_g1**2 / 2 / g1_sigma**2)
    w3 = np.exp(-d_g2**2 / 2 / g2_sigma**2)
    return w1 * w2 * w3

def split_curved_arc_params(params, keep_scale=1, oneD=False):
    if oneD:
        params = params[np.newaxis, :]
    rs_scale1 = params[:,0:4]
    rs_scale2 = params[:,4:8]
    rs_scale3 = params[:,8:12]
    ts_scale1 = params[:,12:16]
    ts_scale2 = params[:,16:20]
    ts_scale3 = params[:,20:24]
    curv_scale1 = params[:,24:28]
    curv_scale2 = params[:,28:32]
    curv_scale3 = params[:,32:36]
    dir_scale1 = params[:,36:40]
    dir_scale2 = params[:,40:44]
    dir_scale3 = params[:,44:48]
    dtandtan_scale1 = params[:,48:52]
    dtandtan_scale2 = params[:,52:56]
    dtandtan_scale3 = params[:,56:60]
    if keep_scale==1:
        rs = rs_scale1
        ts = ts_scale1
        curv = curv_scale1
        direction = dir_scale1
        dtan_dtan = dtandtan_scale1
    elif keep_scale==2:
        rs = rs_scale2
        ts = ts_scale2
        curv = curv_scale2
        direction = dir_scale2
        dtan_dtan = dtandtan_scale2
    elif keep_scale==3:
        rs = rs_scale3
        ts = ts_scale3
        curv = curv_scale3
        direction = dir_scale3
        dtan_dtan = dtandtan_scale3
    rs = np.squeeze(rs)
    ts = np.squeeze(ts)
    curv = np.squeeze(curv)
    direction = np.squeeze(direction)
    dtan_dtan = np.squeeze(dtan_dtan)
    if oneD:
        image1 = np.array([rs[0], ts[0], curv[0], direction[0], dtan_dtan[0]])
        image2 = np.array([rs[1], ts[1], curv[1], direction[1], dtan_dtan[1]])
        image3 = np.array([rs[2], ts[2], curv[2], direction[2], dtan_dtan[2]])
        image4 = np.array([rs[3], ts[3], curv[3], direction[3], dtan_dtan[3]])
    else:
        image1 = np.array([rs[:, 0], ts[:, 0], curv[:, 0], direction[:, 0], dtan_dtan[:, 0]])
        image2 = np.array([rs[:, 1], ts[:, 1], curv[:, 1], direction[:, 1], dtan_dtan[:, 1]])
        image3 = np.array([rs[:, 2], ts[:, 2], curv[:, 2], direction[:, 2], dtan_dtan[:, 2]])
        image4 = np.array([rs[:, 3], ts[:, 3], curv[:, 3], direction[:, 3], dtan_dtan[:, 3]])
    return image1.T, image2.T, image3.T, image4.T

def chi2_curved_arc_statistics_single_image(stats, rs_true, ts_true, curv_true, direction_true, dtandtan_true,
                                rs_sigma=10000, ts_sigma=10000, curv_sigma=0.05, direction_sigma=1000, dtan_dtan_sigma=1000):
    d_rs = stats[:,0] - rs_true
    d_ts = stats[:,1] - ts_true
    d_curv = stats[:,2] - curv_true
    d_dir = stats[:,3] - direction_true
    d_dtandtan = stats[:,4] - dtandtan_true
    w1 = np.exp(-d_rs**2 / 2 / rs_sigma**2)
    w2 = np.exp(-d_ts**2 / 2 / ts_sigma**2)
    w3 = np.exp(-d_curv**2 / 2 / curv_sigma**2)
    w4 = np.exp(-d_dir**2 / 2 / direction_sigma**2)
    w5 = np.exp(-d_dtandtan**2 / 2 / dtan_dtan_sigma**2)
    prod = w1 * w2 * w3 * w4 * w5
    return w1 * w2 * w3 * w4 * w5

def compute_weights_imaging_data(simulation_container, truths, include_images,
                                 statistic='KAPPA_GAMMA', sigmas={}, normalize_weights=True,
                                 keep_scale=3):

    if statistic == 'KAPPA_GAMMA':
        weights_image_1, weights_image_2, weights_image_3, weights_image_4 = \
            weights_from_kappa_gamma_statistics(simulation_container, truths,
                                                include_images, sigmas, keep_scale)
    elif statistic == 'CURVED_ARC':
        weights_image_1, weights_image_2, weights_image_3, weights_image_4 = \
            weights_from_curvedarc_statistics(simulation_container, truths,
                                                include_images, sigmas, keep_scale)

    weights_imaging_data = 1.0
    if 0 in include_images:
        weights_imaging_data *= weights_image_1
    if 1 in include_images:
        weights_imaging_data *= weights_image_2
    if 2 in include_images:
        weights_imaging_data *= weights_image_3
    if 3 in include_images:
        weights_imaging_data *= weights_image_4
    if normalize_weights:
        weights_imaging_data = weights_imaging_data / np.max(weights_imaging_data)
    return weights_imaging_data

def weights_from_kappa_gamma_statistics(simulation_container, truths_kappa_gamma,
                                        include_images, sigmas, keep_scale=1):

    [truths_image1, truths_image2, truths_image3, truths_image4] = split_truths_convergence_shear(truths_kappa_gamma, keep_scale)
    kappa_gamma_stats = simulation_container.kappa_gamma_stats
    stats_image1_scale1, stats_image2_scale1, stats_image3_scale1, \
        stats_image4_scale1 = split_kappa_gamma_params(kappa_gamma_stats)

    weights_image_1 = chi2_kappa_gamma_statistics_single_image(stats_image1_scale1,
                                                               truths_image1['kappa1'], truths_image1['gamma11'],
                                                               truths_image1['gamma12'], **sigmas)
    weights_image_2 = chi2_kappa_gamma_statistics_single_image(stats_image2_scale1,
                                                               truths_image2['kappa2'], truths_image2['gamma12'],
                                                               truths_image2['gamma22'], **sigmas)
    weights_image_3 = chi2_kappa_gamma_statistics_single_image(stats_image3_scale1,
                                                               truths_image3['kappa3'], truths_image3['gamma13'],
                                                               truths_image3['gamma23'], **sigmas)
    weights_image_4 = chi2_kappa_gamma_statistics_single_image(stats_image4_scale1,
                                                               truths_image4['kappa4'], truths_image4['gamma14'],
                                                               truths_image4['gamma24'], **sigmas)
    return weights_image_1, weights_image_2, weights_image_3, weights_image_4

def weights_from_curvedarc_statistics(simulation_container, truths_curvedarc,
                                        include_images, sigmas_list, keep_scale=3):

    truth_list = split_truths_curved_arc(truths_curvedarc, keep_scale)
    curved_arc_stats = simulation_container.curved_arc_stats
    stats_image1, stats_image2, stats_image3, \
    stats_image4 = split_curved_arc_params(curved_arc_stats, keep_scale)
    weight_list = []
    stats_list = [stats_image1, stats_image2, stats_image3, stats_image4]
    for idx in [1,2,3,4]:
        truth = truth_list[idx-1]
        sigmas = sigmas_list[idx-1]
        w = chi2_curved_arc_statistics_single_image(stats_list[idx-1],
                                                              truth['rs'+str(idx)], truth['ts'+str(idx)],
                                                              truth['curv'+str(idx)], truth['dir'+str(idx)],
                                                              truth['dtandtan'+str(idx)],
                                                              rs_sigma=sigmas['rs'+str(idx)], ts_sigma=sigmas['ts'+str(idx)],
                                                              curv_sigma=sigmas['curv'+str(idx)],
                                                            direction_sigma=sigmas['dir'+str(idx)],
                                                              dtan_dtan_sigma=sigmas['dtandtan'+str(idx)])
        weight_list.append(w)
    return weight_list[0], weight_list[1], weight_list[2], weight_list[3]

def extract_arcstatistics_lensmodeling(fname):
    arc_statistics_lensmodeling = np.loadtxt(fname)
    rs, ts, curv, direc, dtdt = [], [], [], [], []
    for img_index in [0,1,2,3]:
        _rs = arc_statistics_lensmodeling[:,img_index]
        _ts = arc_statistics_lensmodeling[:,img_index+4]
        _curv = arc_statistics_lensmodeling[:,img_index+8]
        _direc = arc_statistics_lensmodeling[:,img_index+12]
        _dtdt = arc_statistics_lensmodeling[:,img_index+16]
        rs.append(_rs)
        ts.append(_ts)
        curv.append(_curv)
        direc.append(_direc)
        dtdt.append(_dtdt)
    return rs, ts, curv, direc, dtdt

def arcstats_truths_from_lens_modeling(fname):

    rs, ts, curv, direction, dtdt = extract_arcstatistics_lensmodeling(fname)
    sigmas_list = []
    truth_list = []
    for image_index in [0,1,2,3]:
        t = {'rs'+str(image_index+1): np.median(rs[image_index]),
             'ts' + str(image_index + 1): np.median(ts[image_index]),
             'curv' + str(image_index + 1): np.median(curv[image_index]),
             'dir' + str(image_index + 1): np.median(direction[image_index]),
             'dtandtan'+str(image_index+1): np.median(dtdt[image_index])}
        s = {'rs'+str(image_index+1): np.std(rs[image_index]),
             'ts' + str(image_index + 1): np.std(ts[image_index]),
             'curv' + str(image_index + 1): np.std(curv[image_index]),
             'dir' + str(image_index + 1): np.std(direction[image_index]),
             'dtandtan'+str(image_index+1): np.std(dtdt[image_index])}
        truth_list.append(t)
        sigmas_list.append(s)
    return truth_list, sigmas_list

def extract_kappagammastatistics_lensmodeling(fname):
    kappa_gamma_stats_lensmodeling = np.loadtxt(fname)
    kappa, gamma1, gamma2 = [], [], []
    for img_index in [0,1,2,3]:
        _kap = kappa_gamma_stats_lensmodeling[:,img_index]
        _g1 = kappa_gamma_stats_lensmodeling[:,img_index+3]
        _g2 = kappa_gamma_stats_lensmodeling[:,img_index+6]
        kappa.append(_kap)
        gamma1.append(_g1)
        gamma2.append(_g2)
    return kappa, gamma1, gamma2

def kappagammastats_truths_from_lens_modeling(fname):

    kappa, gamma1, gamma2 = extract_kappagammastatistics_lensmodeling(fname)
    sigmas_list = []
    truth_list = []
    for image_index in [0,1,2,3]:
        t = {'kappa'+str(image_index+1): np.median(kappa[image_index]),
             'gamma1' + str(image_index + 1): np.median(gamma1[image_index]),
             'gamma2' + str(image_index + 1): np.median(gamma2[image_index])}
        s = {'kappa'+str(image_index+1): np.std(kappa[image_index]),
             'gamma1' + str(image_index + 1): np.std(gamma1[image_index]),
             'gamma2' + str(image_index + 1): np.std(gamma2[image_index])}
        truth_list.append(t)
        sigmas_list.append(s)
    return truth_list, sigmas_list

def split_truths_curved_arc(truths_curvedarc, keep_scale=1):

    if isinstance(truths_curvedarc, list) and isinstance(truths_curvedarc[0], dict):
        return truths_curvedarc

    img1, img2, img3, img4 = split_curved_arc_params(truths_curvedarc,
                                                     keep_scale=keep_scale, oneD=True)
    truths_image1 = {'rs1': img1[0],
                     'ts1': img1[1],
                     'curv1': img1[2],
                     'dir1': img1[3],
                     'dtandtan1': img1[4]}
    truths_image2 = {'rs2': img2[0],
                     'ts2': img2[1],
                     'curv2': img2[2],
                     'dir2': img2[3],
                     'dtandtan2': img2[4]}
    truths_image3 = {'rs3': img3[0],
                     'ts3': img3[1],
                     'curv3': img3[2],
                     'dir3': img3[3],
                     'dtandtan3': img3[4]}
    truths_image4 = {'rs4': img4[0],
                     'ts4': img4[1],
                     'curv4': img4[2],
                     'dir4': img4[3],
                     'dtandtan4': img4[4]}
    return [truths_image1, truths_image2, truths_image3, truths_image4]

def split_truths_convergence_shear(truths_kappa_gamma, keep_scale=1):

    if isinstance(truths_kappa_gamma, list) and isinstance(truths_kappa_gamma[0], dict):
        return truths_kappa_gamma

    img1, img2, img3, img4 = split_kappa_gamma_params(truths_kappa_gamma, keep_scale=keep_scale,
                                                      oneD=True)
    truths_image1 = {'kappa1': img1[0],
                     'gamma11': img1[1],
                     'gamma21': img1[2]}
    truths_image2 = {'kappa2': img2[0],
                     'gamma12': img2[1],
                     'gamma22': img2[2]}
    truths_image3 = {'kappa3': img3[0],
                     'gamma13': img3[1],
                     'gamma23': img3[2]}
    truths_image4 = {'kappa4': img4[0],
                     'gamma14': img4[1],
                     'gamma24': img4[2]}
    return [truths_image1, truths_image2, truths_image3, truths_image4]
