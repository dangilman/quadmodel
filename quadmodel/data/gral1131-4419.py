#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:23:31 2023

@author: ryankeeley
"""

class GRAL1131m4419_JWST(Quad):

    def __init__(self, sourcemodel_type='midIR_Gaussian',
                 macromodel_type='EPL_FIXED_SHEAR_MULTIPOLE'):

        
        zsource = 
        zlense= 
        x = [-0.93397872,  0.00147215,  0.60869982,  0.32380675] 
        y = [ 0.10162072, -0.9665036,   0.24657854,  0.61830434]
        m = [0.8, 0.52, 1.0, 0.94]

        delta_m = [0.04, 0.04/0.65, 0.03/1.25, 0.04/1.17]
        delta_xy = [0.005] * 4
        keep_flux_ratio_index = [0, 1, 2]
        self.log10_host_halo_mass = 13.3
        self.log10_host_halo_mass_sigma = 0.3
        kwargs_macromodel = {'shear_amplitude_min': 0.0025, 'shear_amplitude_max': 0.12}
        super(WGDJ0405, self).__init__(zlens, zsource, x, y, m, delta_m, delta_xy, sourcemodel_type, {}, macromodel_type,
                                      kwargs_macromodel, keep_flux_ratio_index,
                                       sample_zlens_pdf=sample_zlens_pdf)