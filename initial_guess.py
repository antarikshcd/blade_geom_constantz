#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:52:07 2018

@author: antariksh
"""
import numpy as np

def search_plane(sin, tin, Ns, Nc, surface_orig, zc_vec, a):
    #Initialize the initial guess of the surface
    surface_in= np.zeros((Ns, Nc, 3), dtype= float)
    param_map_in= np.zeros((Ns, Nc, 2), dtype= int)
    t_ind = np.empty(0, dtype = int)
    Ns_orig = surface_orig.shape[0]
    Nc_orig = surface_orig.shape[1]
    tspace_orig = np.arange(0 , Nc_orig)
    
    # limit the search in the extended grid corresponding to S_orig >= 0
    #if a > 0 :
    #    s0_ind = int(Ns_orig/(3*a))   
    #    surface_orig[0: s0_ind , :, 0] = 9999
    #    surface_orig[0: s0_ind , :, 1] = 9999
    #    surface_orig[0: s0_ind , :, 2] = 9999
    
    #search in t direction and find closest t in original grid
    for j in range(Nc):
        delt = np.abs(tin[j] - tspace_orig)
        # t - index from the original grid where the search should occur
        t_ind = np.append(t_ind, np.argmin(delt))
    
    for i in range(Ns):
        for j in range(Nc):
            # vector that gives the minimum z-distance to the requested z plane
            delz = np.abs(zc_vec[i] - surface_orig[:, t_ind[j], 2])
            # for equidistant values take the value which is lower than current z-coordinate
            # this is automatically taken care of by the np.argmin() method    
            # Find the corresponding S-index
            s_ind = np.argmin(delz)
            #Use the (s_ind,t) value to obtain the (x,y,z) coordinate 
            x = surface_orig[s_ind, t_ind[j], 0]
            y = surface_orig[s_ind, t_ind[j], 1]
            z = surface_orig[s_ind, t_ind[j], 2]
            # and assign it to the new Surface matrix
            surface_in[i, j, 0] = x 
            surface_in[i, j, 1] = y
            surface_in[i, j, 2] = z
            # store the S,T info from the original surface
            param_map_in[i, j, 0] = s_ind
            param_map_in[i, j, 1] = t_ind[j]
            
    return surface_in, param_map_in