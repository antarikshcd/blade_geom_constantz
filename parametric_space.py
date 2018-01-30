#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:56:39 2017

@author: antariksh
"""
import numpy as np

def extended_grid(surface_orig, N_s, N_c, a, b):
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

#def search_plane(sin, tin, N_s, N_c, surface_orig, zc_vec):
    #Initialize the initial guess of the surface
#    surface_in= np.zeros((N_s, N_c, 3), dtype= float)
#    param_map_in= np.zeros((N_s, N_c, 2), dtype= int) 
#    for i in sin:
#        for j in tin:
#            # vector that gives the minimum z-distance to the requested z plane
#            delz= np.abs(zc_vec[i] - surface_orig[:, j, 2])
#            # for equidistant values take the value which is lower than current z-coordinate
#            # this is automatically taken care of by the np.argmin() method    
#            # Find the corresponding S-index
#            s_ind= np.argmin(delz)
#            #Use the (s_ind,t) value to obtain the (x,y,z) coordinate 
#            x= surface_orig[s_ind, j, 0]
#            y= surface_orig[s_ind, j, 1]
#            z= surface_orig[s_ind, j, 2]
#            # and assign it to the new Surface matrix
#            surface_in[i, j, 0]= x 
#            surface_in[i, j, 1]= y
#            surface_in[i, j, 2]= z
#            # store the S,T info from the original surface
#            param_map_in[i, j, 0]= s_ind
#            param_map_in[i, j, 1]= j
 #           
#    return surface_in, param_map_in 

def boundary_correction(S, T, N_s, N_c):
   #------------------treatment for S and/or T exceeding the bound-space---------
   # TODO: Solution for the spanwise  
   S[S<0] = 0
   S[S>=N_s] = N_s
   # T - correction
   # correct for T<0
   if np.min(T) < 0:
       ind_low = np.where(T < 0)[0]
       T_low = T[ind_low]
       quot = abs(T_low)/N_c
       quot = quot.astype(int)
       rem_low = abs(T_low) - quot*N_c
       T[ind_low]= N_c - rem_low
       # correct for T>= N_c    
   if np.max(T)>= N_c:
       ind_high = np.where(T >= N_c)[0]
       T_high = T[ind_high]
       quot= abs(T_high)/N_c
       quot= quot.astype(int)
       rem_high = abs(T_high) - quot*N_c
       T[ind_high] = rem_high
          
   return S, T 
#----------------------------------------------------------------------------   
       
def bilinear_surface(surface_orig, grid_s, grid_t, S, T):
   
   #Ncs= grid_t.shape[0] # number of cross-sections
   #Ns=  grid_s.shape[1] #number of spanwise sections 
   #grid_c, grid_s= np.mgrid[0:Ncs, 0:Ns]

# values in main body co-ordinate system atached at the root centre
   values_x= surface_orig[:,:, 0] #chordwise value
   values_y= surface_orig[:,:,1] #from pressure side to suction side
   values_z= surface_orig[:,:,2] #blade radius
       
#---------------------- perform bilinear interpolation----------------------
#1) find the 4 known data points closest to the point to be interpolated
   Ncs_desired= T.shape[0]
 #  Ns_desired= S.shape[0]
# stores the positions of the four neighbourhood points for each corresponding 
# interpolant grid location
   grid_map= np.empty((Ncs_desired), dtype= object)    
   val_map= np.empty((Ncs_desired), dtype= object)
   
   
   
   for i in range(Ncs_desired):
    #store the x-coordinate of the desired point
      x= S[i]
    #store the y-coordinate of the desired point
      y= T[i]
    #obtain the closest index of the x-coordinate in the original grid
      idx= (np.abs(x - grid_s)).argmin(axis=0)[0]
    # obtain the closest index of the y-coordinate in the desired grid
      idy= (np.abs(y - grid_t).argmin(axis=1))[0]
    # dictionary for storing indicies
      indices_x={}
      indices_y={}
        
   # point in the known grid closest to the desired point
      Px1= grid_s[idx, idy]
      Py1= grid_t[idx, idy]
        
      # store indices
      indices_x[Px1]= [idx, idy]
      indices_y[Py1]= [idx, idy]
        
      # obtain the neighbourhood
      up_bound_x= np.max(grid_s)
      up_bound_y= np.max(grid_t) 
      
      # obtain the second y-coordinate
      if Py1 == up_bound_y or y< Py1:
           
         Py2 = grid_t[idx, idy - 1]
           
         indices_y[Py2]= [idx, idy - 1]
                
      else:
         Py2 = grid_t[idx, idy + 1]
         
         indices_y[Py2]= [idx, idy + 1]       
      # obtain the second x-coordinate
      if Px1 == up_bound_x or x< Px1:
         
         Px2 = grid_s[idx - 1, idy]
          
         indices_x[Px2]= [idx - 1, idy]
      else:
         #print('(Px1=%i, Py1=%i), (Px2=%i, Py2=%i)'%(Px1, Py1, Px2, Py2)) 
         Px2 = grid_s[idx + 1, idy]
         
            
         indices_x[Px2]= [idx + 1, idy]
    # sort the neighbourhood in ascending order
      x1= min(Px1, Px2)
      ind_x1= indices_x[x1][0]
      x2= max(Px1, Px2)
      ind_x2= indices_x[x2][0]
      y1= min(Py1, Py2)
      ind_y1= indices_y[y1][1]
      y2= max(Py1, Py2)
      ind_y2= indices_y[y2][1]
        
      grid_map[i]= [(x1,y1), (x1, y2), (x2, y1), (x2,y2)]
      val_map[i]=  [(values_x[ind_x1, ind_y1], values_y[ind_x1, ind_y1], 
                     values_z[ind_x1, ind_y1]), (values_x[ind_x1, ind_y2],
                     values_y[ind_x1, ind_y2], values_z[ind_x1, ind_y2]),
                     (values_x[ind_x2, ind_y1], values_y[ind_x2, ind_y1], 
                     values_z[ind_x2, ind_y1]), (values_x[ind_x2, ind_y2],
                     values_y[ind_x2, ind_y2], values_z[ind_x2, ind_y2])]    
    
# obtain the corresponding values
   Q= np.zeros((Ncs_desired, 3), dtype= float)        
   for i in range(Ncs_desired):
       x1= grid_map[i][0][0]
       y1= grid_map[i][0][1]
       x2= grid_map[i][3][0]
       y2= grid_map[i][3][1]
        
       A= np.matrix([[1, x1, y1, x1*y1], 
                     [1, x1, y2, x1*y2],
                     [1, x2, y1, x2*y1],
                     [1, x2, y2, x2*y2]])
        #X- values
       X= np.matrix([[val_map[i][0][0]],
                     [val_map[i][1][0]],
                     [val_map[i][2][0]],
                     [val_map[i][3][0]]])
        #Y-values
       Y= np.matrix([[val_map[i][0][1]],
                     [val_map[i][1][1]],
                     [val_map[i][2][1]],
                     [val_map[i][3][1]]])
        #Z-values
       Z= np.matrix([[val_map[i][0][2]],
                     [val_map[i][1][2]],
                     [val_map[i][2][2]],
                     [val_map[i][3][2]]])
        
        # Coefficient matrix for X-values    
       Bx= np.linalg.inv(A) * X
        #Coefficient matrix for Y-values
       By= np.linalg.inv(A) * Y
        #Coefficient matrix for Z-values
       Bz= np.linalg.inv(A) * Z
        
       x_desired= S[i]
       y_desired= T[i]
        
        # X-value for the new surface
       Q[i, 0]= (float(Bx[0]) + float(Bx[1])*x_desired + 
                 float(Bx[2])*y_desired + float(Bx[3])*x_desired*y_desired)
        # Y-value for the new surface
       Q[i, 1]= (float(By[0]) + float(By[1])*x_desired + 
                 float(By[2])*y_desired + float(By[3])*x_desired*y_desired)
        
        # Y-value for the new surface
       Q[i, 2]= (float(Bz[0]) + float(Bz[1])*x_desired + 
                 float(Bz[2])*y_desired + float(Bz[3])*x_desired*y_desired)

   return Q, grid_map, val_map          