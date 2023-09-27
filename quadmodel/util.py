import numpy as np
from scipy.interpolate import interp1d
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from quadmodel.Solvers.light_fit_util import kappa_gamma_single, curved_arc_statistics_single

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
    ext = LensModelExtensions(lens_model)
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
