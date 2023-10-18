#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:12:16 2023

@author: ryankeeley
"""

'''needs lens redshift, can a photometric redshift be calculated
'''

class J0608(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.6
        zsource = 2.346
        x = [-0.10151357,  0.62373664, -0.07702691, -0.44519615]
        y = [-0.48321348, -0.03697901,  0.76274544, -0.24255295]
        m = [1.0] * 4
        delta_m = [0.01] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.6
        self.log10_host_halo_mass_sigma = 0.35

        kwargs_macromodel = {'shear_amplitude_min': 0.05, 'shear_amplitude_max': 0.25}

        super(J0608, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)
