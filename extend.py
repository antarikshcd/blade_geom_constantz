#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:22:38 2018

@author: antariksh
"""
import numpy as np
from surface3d import cylinder3D
from scipy import interpolate

def extrap_grid(surface_orig, N_s, N_c, a):
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

def extrap_grid2(surface_orig, rs=1, ts=1):
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
    # rs = number of sections beyond root
    # number of sections beyond tip
    # generate grid
    Ns_orig = surface_orig.shape[0]
    Nc_orig = surface_orig.shape[1]
    # S original
    S_orig = np.arange(0, Ns_orig)
    # generate the extended grid
    grid_sext, grid_text = np.mgrid[ -rs : Ns_orig + ts, 0 : Nc_orig]
    # extend S
    S_ext = grid_sext[:, 0]
    # extend the surface accordingly
    Ns_ext = grid_sext.shape[0]
    Nc_ext = grid_text.shape[1]
    surface_ext = np.zeros((Ns_ext, Nc_ext, 3), dtype= float)
    
    # extrapolate the root by 1 section using linear interpolation 
    ind = np.zeros(3, dtype = int)
    ind[0] = S_orig[2]
    ind[1] = S_orig[1]
    ind[2] = S_orig[0]
    # extrapolate
    Xroot, Yroot, Zroot = extrapolate_surface(S_ext, surface_orig, ind)
    # build the extrapolated surface at the root
    surface_ext[0, :, 0] = Xroot
    surface_ext[0, :, 1] = Yroot
    surface_ext[0, :, 2] = Zroot
    
    # extrude the tip
    ind[0] = S_orig[Ns_orig-3]
    ind[1] = S_orig[Ns_orig-2]
    ind[2] = S_orig[Ns_orig-1]
    #extrapolate at the thip but only the z-axes
    _, _, Ztip = extrapolate_surface(S_ext, surface_orig, ind)
    
    surface_ext[Ns_ext - 1, :, 0] = surface_orig[Ns_orig - 1, :, 0]
    surface_ext[Ns_ext - 1, :, 1] = surface_orig[Ns_orig - 1, :, 1]
    surface_ext[Ns_ext - 1, :, 2] = Ztip
    
    # assign the original surface for S,T: ie S-->[0, Ns-1] and T-->[0, Nc-1]
    surface_ext[1 : Ns_orig+1, 
                :, 0] = surface_orig[:, :, 0] #X
    surface_ext[1 : Ns_orig+1, 
                :, 1] = surface_orig[:, :, 1] #Y
    surface_ext[1 : Ns_orig+1, 
                 :, 2] = surface_orig[:, :, 2] #Z
    
    return grid_sext, grid_text, surface_ext

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
    # calculate unit direction vectors of l2
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
    dlds_x = (l2)/(S[ind2 - 1] - S[ind1 - 1])
    dlds_y = (l2)/(S[ind2 - 1] - S[ind1 - 1])
    dlds_z = (l2)/(S[ind2 - 1] - S[ind1 - 1])
    # calculate del(X)/del(s), del(Y)/ del(s) and del(Z)/ del(S)
    dXds = np.multiply(dXdl, dlds_x)
    dYds = np.multiply(dYdl, dlds_y)
    dZds = np.multiply(dZdl, dlds_z)
    # obtain the extrapolated position
    X3 = X2 + dXds*(S[ind2 - 1] - S[ind1 - 1])
    Y3 = Y2 + dYds*(S[ind2 - 1] - S[ind1 - 1])
    Z3 = Z2 + dZds*(S[ind2 - 1] - S[ind1 - 1])
    
    return X3, Y3, Z3

def extrap_np(surface_orig, rs, ts):
    """ Uses scipy's interp1D command to extraplate in x,y,z
       
    """
    # rs = number of sections beyond root
    # number of sections beyond tip
    # generate grid
    Ns_orig = surface_orig.shape[0]
    Nc_orig = surface_orig.shape[1]
    # S original
    S_orig = np.arange(0, Ns_orig)
    # generate the extended grid
    grid_sext, grid_text = np.mgrid[ -rs : Ns_orig + ts, 0 : Nc_orig]
    # extend S
    S_ext = grid_sext[:, 0]
    # extend the surface accordingly
    Ns_ext = grid_sext.shape[0]
    Nc_ext = grid_text.shape[1]
    surface_ext = np.zeros((Ns_ext, Nc_ext, 3), dtype= float)
    
    # extrapolate X for every T
    for i in range(Nc_orig):
        # original spanwise x,y,z distributions
        x_orig = surface_orig[:, i, 0]
        y_orig = surface_orig[:, i, 1]
        z_orig = surface_orig[:, i, 2]
        # interp1D functions for x, y, z
        fx = interpolate.interp1d(S_orig, x_orig, kind= 'linear', fill_value = 'extrapolate')
        fy = interpolate.interp1d(S_orig, y_orig, kind= 'linear', fill_value = 'extrapolate')
        fz = interpolate.interp1d(S_orig, z_orig,kind= 'linear', fill_value = 'extrapolate')
        # obtain the interpolated x, y,z spanwise vectors
        x_ext = fx(S_ext)
        y_ext = fy(S_ext)
        z_ext = fz(S_ext)
        # fill the surface extended array
        surface_ext[:, i, 0] = x_ext
        surface_ext[:, i, 1] = y_ext
        surface_ext[:, i, 2] = z_ext
    
    return grid_sext, grid_text, surface_ext
#-------------script to check the extrapolation of the grid-------------------
# load blade surface
surface_blade = np.load('./test_surfaces/KB6_surforig_100s100c_testing.npy')    
# load the cylinder surface with Ns=5, Nc = 3, radius = 1 and length = 5
Ns_cyl = 5
Nc_cyl = 10
r = 1
length_cyl = 5
#create the cylinder surface
surface_cyl = cylinder3D(Ns_cyl, Nc_cyl, r, length_cyl)
# get the extended cylinder surface
grid_s_cylext, grid_t_cylext, surface_cylext = extrap_grid(surface_cyl, Ns_cyl,
                                                           Nc_cyl, 1)

# get the extended blade surface
Ns_blade = surface_blade.shape[0]
Nc_blade = surface_blade.shape[1]
grid_s_blext, grid_t_blext, surface_blext = extrap_grid(surface_blade, 
                                                        Ns_blade, Nc_blade, 1)

#interpolate using splines---------------------------------------------------
grid_s_blext2, grid_t_blext2, surface_blext2 = extrap_np(surface_blade, 1, 1)

grid_s_cylext2, grid_t_cylext2, surface_cylext2 = extrap_np(surface_cyl, 1, 1)

grid_s_blext3, grid_t_blext3, surface_blext3 = extrap_grid2(surface_blade, 1, 1)
#-----------------------------------------------------------------------------
# plot check
# plot cylinder root with the extrpolates surface at the root
from matplotlib import pyplot as plt
fig = plt.figure('comp_root')
plt.plot(surface_cylext[0, : ,0], surface_cylext[0, : ,1], 'xr', 
         label = 'extrap_root')
plt.plot(surface_cylext[1, : ,0], surface_cylext[1, : ,1], 'b', 
         label = 'root')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('best')
plt.title('cylinder')
plt.show()
# show near tip extrapolation
fig = plt.figure('comp_tip')
plt.plot(surface_cylext[Ns_cyl+1, : ,0], surface_cylext[Ns_cyl+1, : ,1], 'xr', 
         label = 'extrap_tip')
plt.plot(surface_cylext[Ns_cyl, : ,0], surface_cylext[Ns_cyl, : ,1], 'b', 
         label = 'tip')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('best')
plt.title('cylinder')
plt.show()

# plot the blade root and extrapolated section
fig = plt.figure('comp_root2')
plt.plot(surface_blext[0, : ,0], surface_blext[0, : ,1], 'xr', 
         label = 'extrap_root')
plt.plot(surface_blext[1, : ,0], surface_blext[1, : ,1], 'b', 
         label = 'root')
plt.plot(surface_blext2[0, : ,0], surface_blext2[0, : ,1], 'g')

plt.xlabel('x')
plt.ylabel('y')
plt.legend('best')
plt.title('KB6 blade')
plt.show()
#plot the blade tip and the extrapolated section
fig = plt.figure('comp_tip2')
plt.plot(surface_blext[Ns_blade+1, : ,0], surface_blext[Ns_blade+1, : ,1], 'xr', 
         label = 'extrap_tip')
plt.plot(surface_blext[Ns_blade, : ,0], surface_blext[Ns_blade, : ,1], 'b', 
         label = 'tip')
plt.plot(surface_blext2[Ns_blade+1, : ,0], surface_blext2[Ns_blade+1, : ,1], 'g')
plt.plot(surface_blext3[Ns_blade+1, : ,0], surface_blext3[Ns_blade+1, : ,1], 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('best')
plt.title('KB6 blade')
plt.show()
