#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary correction by extending the parametric grid in S and T. This is
necessary to ensure C1-continuity at the boundaries. The present solution clips
S at the boundary leading to discontinuities at the boundaries. At the T boundaries
the solution loops for T values over-shooting the bounds of T=0 and T= N_c-1
by considering the cross-section as a closed surface. 

Extension to spanwise direction 'S': 
    The grid is extended to fit the profile of the cross-sections at the relevant
    boundary. So for all S < 0, the crosssections will be the same as that at
    the root. Whereas, for S > (N_s-1) it will be the same as that of 
    S = N_s.
    
Extension to chordwise direction 'T':
    The grid is extended such that a circular loop exists. So, for t<0 the     
    
Created on Tue Jan 16 12:52:36 2018

@author: antariksh
"""
import numpy as np
##------------------------Uncomment to run as a script----------------------
#from surface3d import cylinder3D

# generate a test case
#N_c= 2
#N_s= 2

#blade_length= 10 # in metres
#surface_orig= cylinder3D(N_s, N_c, 1, blade_length)

# set the multipliers for extending the grid
# multiplier for S
#a= 0
# multiplier for T
#b = 0
##-------------------------------------------------------------------------
def extended_grids(surface_orig, N_s, N_c, a, b):
    # generate grid
    grid_s, grid_t= np.mgrid[-a * N_s : (a+1) * N_s, 
                             - b * N_c : (b+1) * N_c]
    # extend the surface accordingly
    Ns_ext= grid_s.shape[0]
    Nc_ext= grid_s.shape[1]
    surface_ext= np.zeros((Ns_ext, Nc_ext, 3), dtype= float)
    s_ind1= N_s * a
    s_ind2= N_s * (a+1)
    t_ind1= N_c * b
    t_ind2= N_c * (b+1)

    # assign the original surface for S,T: ie S-->[0, Ns-1] and T-->[0, Nc-1]
    surface_ext[s_ind1 : s_ind1+N_s, t_ind1 : t_ind1+N_c, 0] = surface_orig[:, :, 0] #X
    surface_ext[s_ind1 : s_ind1+N_s, t_ind1 : t_ind1+N_c, 1] = surface_orig[:, :, 1] #Y
    surface_ext[s_ind1 : s_ind1+N_s, t_ind1 : t_ind1+N_c, 2] = surface_orig[:, :, 2] #Z

    # assign the extended surface for S: S<0, t--> [0, Nc -1] 
    surface_ext[0 : s_ind1, t_ind1 : t_ind1+N_c, 0] = np.tile(surface_orig[0, :, 0], 
                                                              (s_ind1, 1)) #X
    surface_ext[0 : s_ind1, t_ind1 : t_ind1+N_c, 1] = np.tile(surface_orig[0, :, 1], 
                                                              (s_ind1, 1)) #Y
    surface_ext[0 : s_ind1, t_ind1 : t_ind1+N_c, 2] = np.tile(surface_orig[0, :, 2], 
                                                              (s_ind1, 1)) #Z

    # assign the extended surface for S: S>Ns-1, t--> [0, Nc-1]
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, t_ind1 : t_ind1+N_c, 0] = np.tile(
                                          surface_orig[N_s-1, :, 0], (s_ind2 - N_s, 1)) #X
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, t_ind1 : t_ind1+N_c, 1] = np.tile(
                                      surface_orig[N_s-1, :, 1], (s_ind2 - N_s, 1)) #Y
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, t_ind1 : t_ind1+N_c, 2] = np.tile(
                                      surface_orig[N_s-1, :, 2], (s_ind2 - N_s, 1)) #Z


    #assign the extended surface for S: s-->[0, Ns-1] and T < 0
    surface_ext[s_ind1 : s_ind1+N_s, 0 : t_ind1, 0] = np.tile(
                                          surface_orig[:, :, 0], (1, b)) #X
    surface_ext[s_ind1 : s_ind1+N_s, 0 : t_ind1, 1] = np.tile(
                                          surface_orig[:, :, 1], (1, b)) #Y
    surface_ext[s_ind1 : s_ind1+N_s, 0 : t_ind1, 2] = np.tile(
                                          surface_orig[:, :, 2], (1, b)) #Z

    # assign the extended surface for S: S-->[0, Ns-1] and T >= N_c
    surface_ext[s_ind1 : s_ind1+N_s, t_ind1+N_c : t_ind2+t_ind1, 0] = np.tile(
                                          surface_orig[:, :, 0], (1, b)) #X
    surface_ext[s_ind1 : s_ind1+N_s, t_ind1+N_c : t_ind2+t_ind1, 1] = np.tile(
                                       surface_orig[:, :, 1], (1, b)) #Y
    surface_ext[s_ind1 : s_ind1+N_s, t_ind1+N_c : t_ind2+t_ind1, 2] = np.tile(
                                      surface_orig[:, :, 2], (1, b)) #Z

    # assign the extended surface for T: S<0 and T < 0
    surface_ext[0 : s_ind1, 0 : t_ind1, 0] = np.tile(
                                      surface_orig[0, :, 0], (s_ind1, b)) #X
    surface_ext[0 : s_ind1, 0 : t_ind1, 1] = np.tile(
                                      surface_orig[0, :, 1], (s_ind1, b)) #Y
    surface_ext[0 : s_ind1, 0 : t_ind1, 2] = np.tile(
                                      surface_orig[0, :, 2], (s_ind1, b)) #Z

    # assign the extended surface for T: S<0 and T >= Nc
    surface_ext[0 : s_ind1, t_ind1+N_c : t_ind2+t_ind1, 0] = np.tile(
                                      surface_orig[0, :, 0], (s_ind1, b)) #X
    surface_ext[0 : s_ind1, t_ind1+N_c : t_ind2+t_ind1, 1] = np.tile(
                                      surface_orig[0, :, 1], (s_ind1, b)) #Y
    surface_ext[0 : s_ind1, t_ind1+N_c : t_ind2+t_ind1, 2] = np.tile(
                                      surface_orig[0, :, 2], (s_ind1, b)) #Z

    # assign the extended surface for T: S>=Ns and T < 0
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, 0 : t_ind1, 0] = np.tile(
                                      surface_orig[N_s-1, :, 0], (s_ind2 - N_s, b)) #X
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, 0 : t_ind1, 1] = np.tile(
                                      surface_orig[N_s-1, :, 1], (s_ind2 - N_s, b)) #Y
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, 0 : t_ind1, 2] = np.tile(
                                      surface_orig[N_s-1, :, 2], (s_ind2 - N_s, b)) #Z

    # assign the extended surface for T: S>=Ns and T >= Nc
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, t_ind1+N_c : t_ind2+t_ind1, 0] = np.tile(
                                      surface_orig[N_s-1, :, 0], (s_ind2 - N_s, b)) #X
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, t_ind1+N_c : t_ind2+t_ind1, 1] = np.tile(
                                      surface_orig[N_s-1, :, 1], (s_ind2 - N_s, b)) #Y
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, t_ind1+N_c : t_ind2+t_ind1, 2] = np.tile(
                                      surface_orig[N_s-1, :, 2], (s_ind2 - N_s, b)) #Z


    return grid_s, grid_t, surface_ext