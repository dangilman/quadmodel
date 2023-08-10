#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:17:11 2023

@author: ryankeeley
"""

'''needs lens redshift, can a photometric redshift be calculated
'''

from quadmodel.data.quad_base import Quad
import numpy as np

class J0607(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        zlens = 
        zsource = 1.305
        x = [ 0.12410395,  0.64713852,  0.11607351, -0.88731598] 
        y = [-0.89758044,  0.22398861,  0.57591322,  0.09767862]
        m = [1.0] * 4
        delta_m = [0.01] * 4
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]

        self.log10_host_halo_mass = 13.6
        self.log10_host_halo_mass_sigma = 0.35

        kwargs_macromodel = {'shear_amplitude_min': 0.05, 'shear_amplitude_max': 0.25}

        super(J0607, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                    kwargs_macromodel, keep_flux_ratio_index)