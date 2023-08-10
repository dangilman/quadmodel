#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:15:43 2023

@author: ryankeeley
"""

from quadmodel.data.quad_base import Quad
import numpy as np

class J0659(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 0.766
        zsource = 3.083 #ertl et al 2022
        x = [-1.73242364,  1.78707136,  0.9155554,  -0.97020312] 
        y = [-3.10002911,  0.23920932,  1.38420606,  1.47661373]
        m = [1.0] * 4
        delta_m = [0.01] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.6
        self.log10_host_halo_mass_sigma = 0.35

        kwargs_macromodel = {'shear_amplitude_min': 0.05, 'shear_amplitude_max': 0.25}

        super(J0659, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)