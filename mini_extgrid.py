#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:22:38 2018

@author: antariksh
"""
import numpy as np
from surface3d import cylinder3D

def ext_grid(surface_orig, N_s, N_c, a):
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

    """    
    # generate grid
    grid_s, grid_t= np.mgrid[ -a : N_s + a, 
                              0 : N_c]
    # extend the surface accordingly
    Ns_ext= grid_s.shape[0]
    Nc_ext= grid_s.shape[1]
    surface_ext= np.zeros((Ns_ext, Nc_ext, 3), dtype= float)
    s_ind1= a
    s_ind2= N_s + a
    S = grid_s[:, 0]
    # extend the grid in the negative S direction
    
    # store the indices of last 3 points to the root
    ind_root = np.zeros(3, dtype = int)
    ind_root[0] = S[2 + a] # index of the third span from the edge
    ind_root[1] = S[1 + a] # index of the second span from edge
    ind_root[2] = S[0 + a] # index of last span
    X_root, Y_root, Z_root = extrapolate_surface(S, surface_orig, ind_root)
    
    # store the indices of last 3 points to the tip
    ind_tip = np.zeros(3, dtype = int)
    ind_tip[0] = S[-3 - a] # index of the third span from the edge
    ind_tip[1] = S[-2 - a] # index of the second span from edge
    ind_tip[2] = S[-1 - a] # index of last span
    X_tip, Y_tip, Z_tip = extrapolate_surface(S, surface_orig, ind_tip)
    
    # assign the original surface for S,T: ie S-->[0, Ns-1] and T-->[0, Nc-1]
    surface_ext[s_ind1 : s_ind1+N_s, 
                0 : N_c, 0] = surface_orig[:, :, 0] #X
    surface_ext[s_ind1 : s_ind1+N_s, 
                0 : N_c, 1] = surface_orig[:, :, 1] #Y
    surface_ext[s_ind1 : s_ind1+N_s, 
                0 : N_c, 2] = surface_orig[:, :, 2] #Z

    
    # assign the extended surface for S: S<0, t--> [0, Nc -1] 
    surface_ext[0 : s_ind1, 0 : N_c, 0] = X_root
    surface_ext[0 : s_ind1, 0 : N_c, 1] = Y_root
    surface_ext[0 : s_ind1, 0 : N_c, 2] = Z_root

    # assign the extended surface for S: S>Ns-1, t--> [0, Nc-1]
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, 0 : N_c, 0] = X_tip
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, 0 : N_c, 1] = Y_tip
    surface_ext[s_ind1 + N_s : s_ind2 + s_ind1, 0 : N_c, 2] = Z_tip   

    return grid_s, grid_t, surface_ext

def extrapolate_surface(S, surface, ind):
    ind0 = ind[0] # index of the third span from the edge
    ind1 = ind[1] # index of the second span from edge
    ind2 = ind[2] # index of last span
    # define X1
    X1 = surface[ind1, :, 0]
    # X2
    X2 = surface[ind2, :, 0]
    #Y1
    Y1 = surface[ind1, :, 1]
    #Y2
    Y2 = surface[ind2, :, 1]
    #Z1
    Z1 = surface[ind1, :, 2]
    #Z2
    Z2 = surface[ind2, :, 2]
    # distance l2
    l2 = np.sqrt(np.power(X2-X1, 2) + np.power(Y2-Y1, 2) + np.power(Z2-Z1, 2))
    # calculate the gradient del(l)/del(x), del(l)/del(y) and del(l)/del(z)
    dXdl = np.divide(X2-X1, l2)
    dYdl = np.divide(Y2-Y1, l2)
    dZdl = np.divide(Z2-Z1, l2)
    
    #X0 = surface[ind0, :, 0]
    #Y0 = surface[ind0, :, 1]
    #Z0 = surface[ind0, :, 2]
    # calculate the distance l1
    #l1 = np.sqrt(np.power(X1-X0, 2) + np.power(Y1-Y0, 2) + np.power(Z1-Z0, 2))
    # calculate del(l)/del(s)
    dlds = l2/(S[ind2] - S[ind1])
    # calculate del(X)/del(s), del(Y)/ del(s) and del(Z)/ del(S)
    dXds = np.multiply(dXdl, dlds)
    dYds = np.multiply(dYdl, dlds)
    dZds = np.multiply(dZdl, dlds)
    # obtain the extrapolated position
    X3 = X2 + dXds*(S[ind2] - S[ind1])
    Y3 = Y2 + dYds*(S[ind2] - S[ind1])
    Z3 = Z2 + dZds*(S[ind2] - S[ind1])
    
    return X3, Y3, Z3

#surface = cylinder3D(5, 3, 1, 5)
#grid_s, grid_t, surface_ext = ext_grid(surface, 5, 3, 1)