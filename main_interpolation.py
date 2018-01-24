#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to find the points on the constant-z planes using bilinear interpolation.

Created on Mon Jan 15 13:21:17 2018

@author: antariksh
"""
import numpy as np
from parametric_space import bilinear_surface
from parametric_space import boundary_correction
import pickling 

optimization_file= '../../optimization.sqlite'
# set the iteration to choose from
iteration = 22
blade_length= 10.55 # in metres
#generate the lofted surface
#surface = kb6_loftedblade(optimization_file, iteration, N_c, N_s)
#surface_tmp= pickling.load_obj('KB6_surf_1000by1000') 
#surface_tmp= pickling.load_obj('KB6_surf_noshearsweep')
surface_tmp= pickling.load_obj('KB6_surface_S500_C10')
#surface_tmp= pickling.load_obj('KB1_surface_S100_C100')
#surface_tmp= pickling.load_obj('KB1_surface_S500_C10')

# set the number of chordwise N_c and spanwise sections N_s for the surface
N_c= surface_tmp.shape[0]
N_s= surface_tmp.shape[1]


surface_tmp= surface_tmp*blade_length 

# rearrange surface from (Nc, Ns, 3) to (Ns, Nc, 3)
surface_orig= np.zeros((N_s, N_c, 3), dtype= float)
for i in range(N_s):
    surface_orig[i,:,0]= surface_tmp[:,i,0]
    surface_orig[i,:,1]= surface_tmp[:,i,1]
    surface_orig[i,:,2]= surface_tmp[:,i,2]
    
Ns_desired= N_s ##do not change
Nc_desired= N_c

n_points= Nc_desired

surface_new= np.zeros((Ns_desired, Nc_desired, 3), dtype= float)
#set value of zc
zc_vec= np.linspace(0, 1*blade_length, Ns_desired)

Pk= np.zeros(2*n_points+1, dtype=float)

sin= np.arange(0, Ns_desired)
tin= np.arange(0, n_points)
ind_sin= np.arange(0, 2*n_points, 2)
ind_tin= np.arange(1, 2*n_points, 2)

#assign the cross-sectional initital guess
Pk[ind_tin]= tin
#----------------Step 2------------------------------------------------
# first guess of the s-t space
grid_s, grid_t= np.mgrid[0:N_s, 0:N_c]

span_low = 449
span_high = 450    

for i in range(span_low, span_high):
    
    Pk[ind_sin]= sin[i]
    
    S= Pk[ind_sin] 
    T= Pk[ind_tin]
    
    S, T= boundary_correction(S, T, Ns_desired, Nc_desired)
    
    Q, grid_map, val_map= bilinear_surface(surface_orig, grid_s, grid_t, S, T)
    
    