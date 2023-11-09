import numpy as np
from scipy.interpolate import interp1d
from quadmodel.Solvers.light_fit_util import kappa_gamma_single, curved_arc_statistics_single
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Util.decouple_multi_plane_util import setup_grids, coordinates_and_deflections, class_setup, \
    setup_lens_model
from lenstronomy.LightModel.light_model import LightModel

def approx_theta_E(ximg,yimg):

    dis = []
    xinds,yinds = [0,0,0,1,1,2],[1,2,3,2,3,3]

    for (i,j) in zip(xinds,yinds):

        dx,dy = ximg[i] - ximg[j], yimg[i] - yimg[j]
        dr = (dx**2+dy**2)**0.5
        dis.append(dr)
    dis = np.array(dis)

    greatest = np.argmax(dis)
    dr_greatest = dis[greatest]
    dis[greatest] = 0

    second_greatest = np.argmax(dis)
    dr_second = dis[second_greatest]

    return 0.5*(dr_greatest*dr_second)**0.5

def ray_angles(alpha_x, alpha_y, lens_model, kwargs_lens, zsource):
    redshift_list = lens_model.redshift_list + [zsource]
    redshift_list_finely_sampled = np.arange(0.02, zsource, 0.02)

    full_redshift_list = np.unique(np.append(redshift_list, redshift_list_finely_sampled))
    full_redshift_list_sorted = full_redshift_list[np.argsort(full_redshift_list)]

    x_angle_list, y_angle_list, tz = [alpha_x], [alpha_y], [0.]

    try:
        cosmo_calc = lens_model.lens_model._multi_plane_base._cosmo_bkg.T_xy
    except:
        cosmo_calc = lens_model.astropy.comoving_transverse_distance

    x0, y0 = 0., 0.
    zstart = 0.

    for zi in full_redshift_list_sorted:

        assert len(lens_model.lens_model_list) == len(kwargs_lens)

        if hasattr(lens_model, 'lens_model'):

            x0, y0, alpha_x, alpha_y = lens_model.lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y, zstart, zi,
                                                                                  kwargs_lens)
            d = cosmo_calc(0., zi)

        elif hasattr(lens_model, 'ray_shooting_partial'):
            x0, y0, alpha_x, alpha_y = lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y, zstart, zi,
                                                                       kwargs_lens)
            d = cosmo_calc(zi).value

        else:
            raise Exception('the supplied lens model class does not have a ray shooting partial method')

        x_angle_list.append(x0 / d)
        y_angle_list.append(y0 / d)
        tz.append(d)

        zstart = zi

    return x_angle_list, y_angle_list, tz

def interpolate_ray_paths(x_image, y_image, lens_model, kwargs_lens, zsource,
                          terminate_at_source=False, source_x=None, source_y=None):
    """
    :param x_image: x coordinates to interpolate (arcsec)
    :param y_image: y coordinates to interpolate (arcsec)
    :param lens_model: instance of LensModel
    :param kwargs_lens: keyword arguments for lens model
    :param zsource: source redshift
    :param terminate_at_source: fix the final angular coordinate to the source coordinate
    :param source_x: source x coordinate (arcsec)
    :param source_y: source y coordinate (arcsec)
    :return: Instances of interp1d (scipy) that return the angular coordinate of a ray given a
    comoving distance
    """

    ray_angles_x = []
    ray_angles_y = []

    # print('coordinate: ', (x_image, y_image))
    for (xi, yi) in zip(x_image, y_image):

        angle_x, angle_y, tz = ray_angles(xi, yi, lens_model, kwargs_lens, zsource)

        if terminate_at_source:
            angle_x[-1] = source_x
            angle_y[-1] = source_y

        ray_angles_x.append(interp1d(tz, angle_x))
        ray_angles_y.append(interp1d(tz, angle_y))

    return ray_angles_x, ray_angles_y

def interpolate_ray_paths_system(x_image, y_image, lens_system,
                                 include_substructure=True, realization=None, terminate_at_source=False,
                                 source_x=None, source_y=None):

    lens_model, kwargs_lens = lens_system.get_lensmodel(include_substructure, realization)

    zsource = lens_system.zsource

    return interpolate_ray_paths(x_image, y_image, lens_model, kwargs_lens, zsource,
                                 terminate_at_source, source_x, source_y)

def kappa_gamma_statistics(lens_model, kwargs_lens, x_image, y_image, diff_scale, z_lens):
    """
    Computes the convergence and shear at positions x_image and y_image at angular scales set by diff_scale
    """
    if not isinstance(diff_scale, list):
        diff_scale = [diff_scale]
    kappa_list = []
    g1_list = []
    g2_list = []
    param_names = []
    for diff_counter, diff in enumerate(diff_scale):
        for i in range(0, 4):
            kap, g1, g2 = kappa_gamma_single(lens_model, kwargs_lens, x_image[i], y_image[i], z_lens, diff=diff)
            kappa_list.append(kap)
            g1_list.append(g1)
            g2_list.append(g2)
            param_name1 = 'kappa'+str(i+1)+'_'+str(diff)
            param_name2 = 'g1' + str(i + 1) + '_' + str(diff)
            param_name3 = 'g2' + str(i + 1) + '_' + str(diff)
            param_names.append(param_name1)
            param_names.append(param_name2)
            param_names.append(param_name3)
    kappagamma_stats = np.hstack((np.array(kappa_list), np.array(g1_list), np.array(g2_list)))
    return kappagamma_stats, param_names

def curved_arc_statistics_parallel(lens_model, kwargs_lens, x_coord_list, y_coord_list,
                                   z_lens, diff=None, nproc=10):
    """
    Computes the curved arc properties at different image positions in parallel
    """
    args = []
    for (xi, yi) in zip(x_coord_list, y_coord_list):
        new = (lens_model, kwargs_lens, xi, yi, z_lens, diff)
        args.append(new)

    from multiprocessing.pool import Pool
    pool = Pool(nproc)
    results = pool.starmap(curved_arc_statistics_single, args)
    pool.close()
    result_array = np.empty((len(results), 5))
    for i, result in enumerate(results):
        result_array[i,:] = np.array(result)
    return result_array

def curved_arc_statistics(lens_model, kwargs_lens, x_image, y_image, diff_scale, z_lens):
    """
    Computes the curved arc properties at positions x_image and y_image at angular scales set by diff_scale
    """
    if not isinstance(diff_scale, list):
        diff_scale = [diff_scale]
    rs = []
    ts = []
    curv = []
    dir = []
    dtan_dtan = []
    param_names = []
    for diff_counter, diff in enumerate(diff_scale):
        for i in range(0, 4):
            radial_stretch, tangential_stretch, curvature, \
            direction, dtdt = curved_arc_statistics_single(lens_model, kwargs_lens,
                                                                x_image[i], y_image[i], z_lens, diff=diff)
            rs.append(radial_stretch)
            ts.append(tangential_stretch)
            curv.append(curvature)
            dir.append(direction)
            dtan_dtan.append(dtdt)
            param_name1 = 'rs'+str(i+1)+'_'+str(diff)
            param_name2 = 'ts' + str(i + 1) + '_' + str(diff)
            param_name3 = 'curv' + str(i + 1) + '_' + str(diff)
            param_name4 = 'dir' + str(i + 1) + '_' + str(diff)
            param_name5 = 'dtandtan' + str(i + 1) + '_' + str(diff)
            param_names.append(param_name1)
            param_names.append(param_name2)
            param_names.append(param_name3)
            param_names.append(param_name4)
            param_names.append(param_name5)
    curvedarc_stats = np.hstack((np.array(rs), np.array(ts), np.array(curv), np.array(dir), np.array(dtan_dtan)))
    return curvedarc_stats, param_names

def magnification_finite_decoupled(source_model, kwargs_source, x_image, y_image,
                                   lens_model_init, kwargs_lens_init, kwargs_lens, index_lens_split,
                                   grid_size, grid_resolution, r_step_factor=10.0):
    """
    """
    lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free, z_source, z_split, cosmo_bkg = \
        setup_lens_model(lens_model_init, kwargs_lens_init, index_lens_split)
    grid_x_large, grid_y_large, interp_points_large, npix_large = setup_grids(grid_size, grid_resolution,
                                                      0.0, 0.0)
    grid_r = np.sqrt(grid_x_large**2 + grid_y_large**2)
    grid_r = grid_r.ravel()
    grid_x_large = grid_x_large.ravel()
    grid_y_large = grid_y_large.ravel()
    r_step = grid_size / r_step_factor
    magnifications = []
    flux_arrays = []
    for (x_img, y_img) in zip(x_image, y_image):
        mag, flux_array = mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x_img, y_img, grid_x_large, grid_y_large,
                            grid_r, r_step, grid_resolution, grid_size, z_split, z_source)
        magnifications.append(mag)
        flux_arrays.append(flux_array.reshape(npix_large, npix_large))
    return np.array(magnifications), flux_arrays

def mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x_image, y_image, grid_x_large, grid_y_large,
                            grid_r, r_step, grid_resolution, grid_size_max, zlens, zsource):
    """

    """
    # initalize flux array
    flux_array = np.zeros(len(grid_x_large))
    # setup ray tracing info
    xD = np.zeros_like(flux_array)
    yD = np.zeros_like(flux_array)
    alpha_x_foreground = np.zeros_like(flux_array)
    alpha_y_foreground = np.zeros_like(flux_array)
    alpha_x_background = np.zeros_like(flux_array)
    alpha_y_background = np.zeros_like(flux_array)
    r_min = 0.0
    r_max = r_min + r_step
    magnification_last = 0.0
    inds_compute = np.array([])
    Td = cosmo_bkg.T_xy(0, zlens)
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)
    while True:
        # select new coordinates to ray-trace through
        inds_compute, inds_outside_r, inds_computed = _inds_compute_grid(grid_r, r_min, r_max, inds_compute)
        x_points_temp = grid_x_large[inds_compute] + x_image
        y_points_temp = grid_y_large[inds_compute] + y_image

        # compute lensing stuff at these coordinates
        _xD, _yD, _alpha_x_foreground, _alpha_y_foreground, _alpha_x_background, _alpha_y_background = \
            coordinates_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                        x_points_temp, y_points_temp, z_split, z_source, cosmo_bkg)
        # update the master grids with the new information
        xD[inds_compute] = _xD
        yD[inds_compute] = _yD
        alpha_x_foreground[inds_compute] = _alpha_x_foreground
        alpha_y_foreground[inds_compute] = _alpha_y_foreground
        alpha_x_background[inds_compute] = _alpha_x_background
        alpha_y_background[inds_compute] = _alpha_y_background

        # ray trace to source plane
        x = xD[inds_computed]
        y = yD[inds_computed]
        # compute the deflection angles from the main deflector
        deflection_x_main, deflection_y_main = lens_model_free.alpha(
            x / Td, y / Td, kwargs_lens
        )
        deflection_x_main *= reduced_to_phys
        deflection_y_main *= reduced_to_phys

        # add the main deflector to the deflection field
        alpha_x = alpha_x_foreground[inds_computed] - deflection_x_main
        alpha_y = alpha_y_foreground[inds_computed] - deflection_y_main

        # combine deflections
        alpha_background_x = alpha_x + alpha_x_background[inds_computed]
        alpha_background_y = alpha_y + alpha_y_background[inds_computed]

        # ray propagation to the source plane with the small angle approximation
        beta_x = x / Ts + alpha_background_x * Tds / Ts
        beta_y = y / Ts + alpha_background_y * Tds / Ts

        sb = source_model.surface_brightness(beta_x, beta_y, kwargs_source)
        flux_array[inds_computed] = sb
        flux_array[inds_outside_r] = 0.0
        magnification_temp = np.sum(flux_array) * grid_resolution ** 2
        diff = (
            abs(magnification_temp - magnification_last) / magnification_temp
        )
        r_min += r_step
        r_max += r_step
        if r_max >= grid_size_max:
            break
        elif diff < 0.001 and magnification_temp > 0.0001:  # we want to avoid situations with zero flux
            break
        else:
            magnification_last = magnification_temp
    return magnification_temp, flux_array

def _inds_compute_grid(grid_r, r_min, r_max, inds_compute):
    condition1 = grid_r >= r_min
    condition2 = grid_r < r_max
    condition = np.logical_and(condition1, condition2)
    inds_compute_new = np.where(condition)[0]
    inds_outside_r = np.where(grid_r > r_max)[0]
    inds_computed = np.append(inds_compute, inds_compute_new).astype(int)
    return inds_compute_new, inds_outside_r, inds_computed

def setup_gaussian_source(source_fwhm_pc, source_x, source_y, astropy_cosmo, z_source):

    kpc_per_arcsec = 1/astropy_cosmo.arcsec_per_kpc_proper(z_source).value
    source_sigma = 1e-3 * source_fwhm_pc / 2.354820 / kpc_per_arcsec
    kwargs_source_light = [{'amp': 1.0, 'center_x': source_x, 'center_y': source_y, 'sigma': source_sigma}]
    return LightModel(['GAUSSIAN']), kwargs_source_light
