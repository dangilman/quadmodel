from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.Plots.plot_quasar_images import plot_quasar_images
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from quadmodel.Solvers.brute import BruteOptimization
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import numpy as np
from scipy.interpolate import interp1d
from quadmodel.util import interpolate_ray_paths_system
from quadmodel.Solvers.light_fit_util import kappa_gamma_single, curved_arc_statistics_single

class QuadLensSystem(object):

    def __init__(self, macromodel, z_source, substructure_realization=None, pyhalo_cosmology=None):

        """

        :param macromodel: an instance of MacroLensModel (see LensSystem.macrolensmodel)
        :param z_source: source redshift
        :param substructure_realization: an instance of Realization (see pyhalo.single_realization)
        :param pyhalo_cosmology: an instance of Cosmology() from pyhalo
        """

        if pyhalo_cosmology is None:
            # the default cosmology in pyHalo, currently WMAP9
            pyhalo_cosmology = Cosmology()
        self.macromodel = macromodel
        self.zlens = macromodel.zlens
        self.zsource = z_source
        self.pyhalo_cosmology = pyhalo_cosmology
        self.astropy = pyhalo_cosmology.astropy
        self.update_realization(substructure_realization)
        self.pc_per_arcsec_zsource = 1000 * pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value ** -1
        self.clear_static_lensmodel()

    @classmethod
    def shift_background_auto(cls, lens_data_class, macromodel, zsource,
                              realization, cosmo=None, particle_swarm_init=False,
                              opt_routine='free_shear_powerlaw', constrain_params=None, verbose=False,
                              re_optimize=False):

        """
        This method takes a macromodel and a substructure realization, fits a smooth model to the data
        with the macromodel only, and then shifts the halos in the realization such that they lie along
        a path traversed by the light. For simple 1-deflector lens models this path is close to a straight line
        between the observer and the source, but for more complicated lens models with satellite galaxies and
        LOS galaxies the path can be complicated and it is important to shift the line of sight halos;
        often, when line of sight galaxies are included, the source is not even close to being
        behind directly main deflector.
        :param lens_data_class: class that contaains the lens data (see data.quad_base)
        :param macromodel: an instance of MacroLensModel (see LensSystem.macrolensmodel)
        :param zsource: source redshift
        :param realization: an instance of Realization (see pyhalo.single_realization)
        :param cosmo: an instance of Cosmology() from pyhalo
        :param particle_swarm_init: whether or not to use a particle swarm algorithm when fitting the macromodel.
        You should use a particle swarm routine if you're starting the lens model from scratch
        :param opt_routine: the optimization routine to use... more documentation coming soon
        :param constrain_params: keywords to be passed to optimization routine
        :param verbose: whether to print stuff
        :param re_optimize: bool; determines the prior volume explored by the particle swarm. Set to True if starting from
        a good initial guess for the macromodel

        This routine can be immediately followed by doing a lens model fit to the data, for example:

        1st:
        lens_system = QuadLensSystem.shift_background_auto(data, macromodel, zsource,
                            realization, particle_swarm_init=True)

        2nd:
        lens_system.initialize(data, include_substructure=True)
        # will fit the lens system while including every single halo in the computation

        Other optimization routines are detailed in Optimization.quad_optimization

        """

        lens_system_init = QuadLensSystem(macromodel, zsource, None,
                                          pyhalo_cosmology=cosmo)

        lens_system_init.initialize(lens_data_class, opt_routine=opt_routine, constrain_params=constrain_params,
                                    kwargs_optimizer={'particle_swarm': particle_swarm_init, 're_optimize': re_optimize}, verbose=verbose)

        source_x, source_y = lens_system_init.source_centroid_x, lens_system_init.source_centroid_y

        ray_interp_x, ray_interp_y = interpolate_ray_paths_system(
            lens_data_class.x, lens_data_class.y, lens_system_init,
            include_substructure=False, terminate_at_source=True, source_x=source_x,
            source_y=source_y)

        ### Now compute the centroid of the light cone as the coordinate centroid of the individual images
        z_range = np.linspace(0, lens_system_init.zsource, 100)
        distances = [lens_system_init.pyhalo_cosmology.D_C_transverse(zi) for zi in z_range]
        angular_coordinates_x = []
        angular_coordinates_y = []
        for di in distances:
            x_coords = [ray_x(di) for i, ray_x in enumerate(ray_interp_x)]
            y_coords = [ray_y(di) for i, ray_y in enumerate(ray_interp_y)]
            x_center = np.mean(x_coords)
            y_center = np.mean(y_coords)
            angular_coordinates_x.append(x_center)
            angular_coordinates_y.append(y_center)

        ray_interp_x = [interp1d(distances, angular_coordinates_x)]
        ray_interp_y = [interp1d(distances, angular_coordinates_y)]

        realization = realization.shift_background_to_source(ray_interp_x[0], ray_interp_y[0])

        macromodel = lens_system_init.macromodel

        lens_system = QuadLensSystem(macromodel, zsource,
                                          realization, lens_system_init.pyhalo_cosmology)

        lens_system.update_source_centroid(source_x, source_y)

        return lens_system

    @classmethod
    def addRealization(cls, quad_lens_system, realization):

        """
        This routine creates a new instance of QuadLensSystem that is identical to quad_lens_system,
        but includes a different substructure realization
        """
        macromodel = quad_lens_system.macromodel
        z_source = quad_lens_system.zsource
        pyhalo_cosmo = quad_lens_system.pyhalo_cosmology
        new_quad = QuadLensSystem(macromodel, z_source, realization, pyhalo_cosmo)
        if hasattr(quad_lens_system, 'source_centroid_x'):
            source_x, source_y = quad_lens_system.source_centroid_x, quad_lens_system.source_centroid_y
            new_quad.update_source_centroid(source_x, source_y)
        return new_quad

    def initialize(self, data_to_fit, opt_routine='free_shear_powerlaw',
                   constrain_params=None, verbose=False,
                   include_substructure=False, kwargs_optimizer={}):

        """
        This routine fits a smooth macromodel profile defined by self.macromodel to the image positions in data_to_fit
        :param data_to_fit: an instanced of LensedQuasar (see LensSystem.BackgroundSource.lensed_quasar)

        """

        optimizer = BruteOptimization(self)
        kwargs_lens_final, lens_model_full, _ = optimizer.optimize(
            data_to_fit, opt_routine, constrain_params, verbose,
            include_substructure, kwargs_optimizer
        )
        self.clear_static_lensmodel()
        return

    def get_smooth_lens_system(self):

        """
        Returns a lens system with only the smooth component of the lens model (i.e. no substructure)
        """
        smooth_lens = QuadLensSystem(self.macromodel, self.zsource,
                                                  None, self.pyhalo_cosmology)

        if hasattr(self, 'source_centroid_x'):
            smooth_lens.update_source_centroid(self.source_centroid_x, self.source_centroid_y)

        return smooth_lens

    def get_lensmodel(self, include_substructure=True, substructure_realization=None,
                      include_macromodel=True, log_mlow_mass_sheet=7.0, subtract_exact_mass_sheets=False):

        if self._static_lensmodel and include_substructure is True:

            _, _, _, numercial_alpha_class = self._get_lenstronomy_args(
                True)

            self._numerical_alpha_class = numercial_alpha_class

            return self._lensmodel_static, self._kwargs_static

        names, redshifts, kwargs, numercial_alpha_class = self._get_lenstronomy_args(
            include_substructure, substructure_realization, log_mlow_mass_sheet, subtract_exact_mass_sheets)

        if include_macromodel is False:
            n_macro = self.macromodel.n_lens_models
            names = names[n_macro:]
            kwargs = kwargs[n_macro:]
            redshifts = list(redshifts)[n_macro:]

        self._numerical_alpha_class = numercial_alpha_class

        lensModel = LensModel(names, lens_redshift_list=redshifts, z_lens=self.zlens, z_source=self.zsource,
                              multi_plane=True, numerical_alpha_class=numercial_alpha_class, cosmo=self.astropy)
        return lensModel, kwargs

    def _get_lenstronomy_args(self, include_substructure=True, realization=None, log_mlow_mass_sheet=7.0,
                              subtract_exact_mass_sheets=False):

        lens_model_names, macro_redshifts, macro_kwargs = \
            self.macromodel.get_lenstronomy_args()

        if realization is None:
            realization = self.realization

        if realization is not None and include_substructure:
            kwargs_mass_sheet = {'log_mlow_sheets': log_mlow_mass_sheet, 'subtract_exact_sheets': subtract_exact_mass_sheets}
            halo_names, halo_redshifts, kwargs_halos, numerical_alpha_class = \
                realization.lensing_quantities(kwargs_mass_sheet=kwargs_mass_sheet)

        else:
            halo_names, halo_redshifts, kwargs_halos, numerical_alpha_class = [], [], [], None

        halo_redshifts = list(halo_redshifts)
        names = lens_model_names + halo_names
        redshifts = macro_redshifts + halo_redshifts
        kwargs = macro_kwargs + kwargs_halos

        return names, redshifts, kwargs, numerical_alpha_class

    def update_light_centroid(self, light_x, light_y):

        self.light_centroid_x = light_x
        self.light_centroid_y = light_y

    def update_realization(self, realization):

        self.realization = realization

    def update_kwargs_macro(self, new_kwargs):

        self.macromodel.update_kwargs(new_kwargs[0:self.macromodel.n_lens_models])

    def get_kwargs_macro(self, include_substructure=True):

        return self._get_lenstronomy_args(include_substructure)[2]

    def set_lensmodel_static(self, lensmodel, kwargs):

        self._static_lensmodel = True
        self._lensmodel_static = lensmodel
        self._kwargs_static = kwargs

    def clear_static_lensmodel(self):

        self._static_lensmodel = False
        self._lensmodel_static = None
        self._kwargs_static = None

    def update_source_centroid(self, source_x, source_y):

        self.source_centroid_x = source_x
        self.source_centroid_y = source_y

    def quasar_magnification(self, x, y, source_fwhm_pc,
                             lens_model,
                             kwargs_lensmodel, point_source=False,
                             grid_axis_ratio=0.5, grid_rmax=None,
                             grid_resolution=None,
                             normed=True, grid_resolution_rescale=1.0,
                             source_model='GAUSSIAN', grid_size_rescale=1.0,
                             **kwargs_magnification_finite):

        """
        Computes the magnifications (or flux ratios if normed=True)

        :param x: x image position
        :param y: y image position
        :param source_fwhm_pc: size of background quasar emission region in parcsec
        :param lens_model: an instance of LensModel (see lenstronomy.lens_model)
        :param kwargs_lensmodel: key word arguments for the lens_model
        :param point_source: computes the magnification of a point source
        :param grid_axis_ratio: axis ratio of ray tracing ellipse
        :param grid_rmax: sets the radius of the ray tracing aperture; if None, a default value will be estimated
        from the source size
        :param normed: If True, normalizes the magnifications such that the brightest image has a magnification of 1
        """
        if source_fwhm_pc == 0.0:
            point_source = True

        if point_source:
            mags = lens_model.magnification(x, y, kwargs_lensmodel)
            magnifications = abs(mags)

        else:
            if grid_rmax is None:
                from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size
                grid_rmax = auto_raytracing_grid_size(source_fwhm_pc)

            if grid_resolution is None:
                from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution
                grid_resolution = auto_raytracing_grid_resolution(source_fwhm_pc)

            grid_rmax *= grid_size_rescale
            grid_resolution *= grid_resolution_rescale
            extension = LensModelExtensions(lens_model)
            source_x, source_y = self.source_centroid_x, self.source_centroid_y

            if source_model == 'GAUSSIAN':
                magnifications = extension.magnification_finite_adaptive(x, y,
                                                    source_x, source_y, kwargs_lensmodel, source_fwhm_pc,
                                                                     self.zsource, self.astropy,
                                                                     grid_radius_arcsec=grid_rmax,
                                                                     grid_resolution=grid_resolution,
                                                                     axis_ratio=grid_axis_ratio)
            elif source_model == 'DOUBLE_GAUSSIAN':
                magnifications = extension.magnification_finite_adaptive(x, y,
                                                                         source_x, source_y, kwargs_lensmodel,
                                                                         source_fwhm_pc,
                                                                         self.zsource, self.astropy,
                                                                         grid_radius_arcsec=grid_rmax,
                                                                         grid_resolution=grid_resolution,
                                                                         axis_ratio=grid_axis_ratio,
                                                                         source_light_model=source_model,
                                                                         dx=kwargs_magnification_finite['dx'],
                                                                         dy=kwargs_magnification_finite['dy'],
                                                                         size_scale=kwargs_magnification_finite[
                                                                             'size_scale'],
                                                                         amp_scale=kwargs_magnification_finite[
                                                                             'amp_scale'])
            else:
                raise Exception('source model '+str(source_model)+ ' not recognized')

        if normed:
            magnifications *= max(magnifications) ** -1

        return magnifications

    def plot_images(self, x, y, source_fwhm_pc,
                             lens_model,
                             kwargs_lensmodel,
                             grid_rmax=None,
                             grid_resolution=None,
                             grid_resolution_rescale=1,
                             source_model='GAUSSIAN', **kwargs_magnification_finite):

        source_x, source_y = self.source_centroid_x, self.source_centroid_y

        if grid_rmax is None:
            from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size
            grid_rmax = auto_raytracing_grid_size(source_fwhm_pc)
        if grid_resolution is None:
            from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution
            grid_resolution = auto_raytracing_grid_resolution(source_fwhm_pc)
        grid_resolution *= grid_resolution_rescale

        if source_model == 'GAUSSIAN':
            plot_quasar_images(lens_model, x, y, source_x, source_y, kwargs_lensmodel,
                                                                     source_fwhm_pc,
                                                                     self.zsource, self.astropy,
                                                                     grid_radius_arcsec=grid_rmax,
                                                                     grid_resolution=grid_resolution)
        elif source_model == 'DOUBLE_GAUSSIAN':

            plot_quasar_images(lens_model, x, y, source_x, source_y, kwargs_lensmodel,
                                                                     source_fwhm_pc,
                                                                     self.zsource, self.astropy,
                                                                     grid_radius_arcsec=grid_rmax,
                                                                     grid_resolution=grid_resolution,
                                                                     source_light_model=source_model,
                               dx=kwargs_magnification_finite['dx'], dy=kwargs_magnification_finite['dy'],
                               size_scale=kwargs_magnification_finite['size_scale'], amp_scale=kwargs_magnification_finite['amp_scale'])

    def solve_lens_equation(self, source_x, source_y):

        """
        Solves the lens equation given a source coordinate
        :param source_x: source coordinate x [arcsec]
        :param source_y: source coordinate y [arcsec]
        :return: x and y image positions [arcsec]
        """
        lensmodel, kwargs = self.get_lensmodel()
        ext = LensEquationSolver(lensmodel)
        x_image, y_image = ext.findBrightImage(source_x, source_y, kwargs)
        return x_image, y_image

    def get_model_samples(self, n):

        """
        Returns a numpy array with the values of the keyword arguments corresponding to the first n lens models
        :param n: number of lens models for which to read out the values of the keyword arguments
        :return: an array of values
        """
        _, kw = self.get_lensmodel()
        kwargs = kw[0:n]
        kwargs_list = []
        param_names = []
        for kw in kwargs:
            for key in kw.keys():
                if key=='a_m':
                    if kw['m']==4:
                        param_names.append('a4_physical')
                        kwargs_list.append(kw[key])
                    elif kw['m']==3:
                        param_names.append('a3_physical')
                        kwargs_list.append(kw[key])
                    else:
                        raise Exception('if multipole is in lens model, must have m=3 or m=4')
                elif key=='gamma':
                    param_names.append('gamma_macro')
                    kwargs_list.append(kw[key])
                else:
                    kwargs_list.append(kw[key])
                    param_names.append(key)
        return np.array(kwargs_list), param_names

    def kappa_gamma_statistics(self, x_image, y_image, diff_scale):
        """
        Computes the convergence and shear at positions x_image and y_image at angular scales set by diff_scale
        """
        lens_model, kwargs_lens = self.get_lensmodel()
        if not isinstance(diff_scale, list):
            diff_scale = [diff_scale]
        kappa_list = []
        g1_list = []
        g2_list = []
        param_names = []
        for diff_counter, diff in enumerate(diff_scale):
            for i in range(0, 4):
                kap, g1, g2 = kappa_gamma_single(lens_model, kwargs_lens, x_image[i], y_image[i], self.zlens,
                                                 diff)
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

    def curved_arc_statistics(self, x_image, y_image, diff_scale):
        """
        Computes the curved arc properties at positions x_image and y_image at angular scales set by diff_scale
        """
        lens_model, kwargs_lens = self.get_lensmodel()
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
                                                                    x_image[i], y_image[i], self.zlens, diff=diff)

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
