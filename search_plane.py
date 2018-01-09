#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Searches the closest points to a constant z-coordinate
 
Created on Tue Jan  9 11:23:00 2018

@author: antariksh
"""

import numpy as np
from surface3d import cylinder3D

blade_length= 5 # in metres
#surface= pickling.load_obj('KB6_surf_1000by1000') 
#surface= surface*blade_length 
N_c= 10
N_s= 6
surface= cylinder3D(N_s, N_c, 1, blade_length)
Ns_desired= 4

zc_vec= np.linspace(0, 1*blade_length, num= Ns_desired, endpoint= True)

#perturb

surface_perturb= surface
#perturb x
surface_perturb[span_low, :, 0]= surface[span_low, :, 0] + 0.01
surface_perturb[span_low, :, 1]= surface[span_low, :, 1] - 0.01
ind_perturb= np.arange(0, n_points, 2)
surface_perturb[span_low, ind_perturb, 2]= surface[span_low, ind_perturb, 2] - 0.01
surface_perturb[span_low, ind_perturb + 1, 2]= surface[span_low, ind_perturb + 1, 2] + 0.01 