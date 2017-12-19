#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:56:39 2017

@author: antariksh
"""
import numpy as np

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
      x= T[i]
    #store the y-coordinate of the desired point
      y= S[i]
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
      up_bound_x= np.max(T)
      up_bound_y= np.max(S) 
      
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
        
       x_desired= T[i]
       y_desired= S[i]
        
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