#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow script to transfrom blade geoemtry in cross-section normal 
to the local blade running length vector and sample it in parallel planes to 
the X-Y global plane of the global coordinate system attched at the circular 
blade root centre.
 
Created on Thu Dec  7 16:58:33 2017

@author: antariksh

"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pickling #user-defined module to store and load python data as pickles

#format of surface_orig is [i, j, k] where i= Number of cross-sections,
# j= Number of spanwise sections, k= (x, y, z)
surface_orig= pickling.load_obj('KB6_surf_noshearsweep') 

Ncs= surface_orig.shape[0] # number of cross-sections
Ns= surface_orig.shape[1] #number of spanwise sections
#create grid between 0 and 1 from input data
#grid_c= cross-sectional discretisation
#grid_s= spanwise discretisation 
grid_c, grid_s= np.mgrid[0:1:complex(Ncs), 0:1: complex(Ns)]

# values in main body co-ordinate system atached at the root centre
values_x= surface_orig[:,:, 0] #chordwise
values_y= surface_orig[:,:,1] #pressure to suction side
values_z= surface_orig[:,:,2] #blade radius

# create a finer grid to interpolate the data points say 
#  (1000 cross-section, 500 span-wise)
Ncs_desired= 500
Ns_desired= 200
#create the desired grid
grid_c_desired, grid_s_desired= np.mgrid[0:1:
                             complex(Ncs_desired), 0:1: complex(Ns_desired)]

    
#---------------------- perform bilinear interpolation----------------------
#1) find the 4 known data points closest to the point to be interpolated

# stores the positions of the four neighbourhood points for each corresponding 
# interpolant grid location
grid_map= np.empty((Ncs_desired, Ns_desired), dtype= object)             
for i in range(Ncs_desired):
    #store the x-coordinate of the desired point
    x= grid_c_desired[i, 0]
    #obtain the closest index of the x-coordinate in the original grid
    idx= (np.abs(x - grid_c)).argmin(axis=0)[0]
    
    for j in range(Ns_desired):
        #store the y-coordinate of the desired point
        y= grid_s_desired[0, j]
        # obtain the closest index of the y-coordinate in the desired grid
        idy= (np.abs(y - grid_s).argmin(axis=1))[0]
        # point in the known grid closest to the desired point
        Px1= grid_c[idx, idy]
        Py1= grid_s[idx, idy]
        
        # obtain the neighbourhood
        low_bound= 0
        up_bound= 1
        
        # obtain the second y-coordinate
        if Py1 == up_bound or y< Py1:
           
           Py2 = grid_s[idx, idy - 1]
        
        #elif Py1 == low_bound:    
         #  Py2 = grid_s[idx, idy + 1]
        
        #elif y> Py1:
        else:
            Py2 = grid_s[idx, idy + 1]
     
       # obtain the second x-coordinate
        if Px1 == up_bound or x< Px1:
           Px2 = grid_s[idx - 1, idy]
        
       # elif Px1 == low_bound:    
           #Px2 = grid_s[idx + 1, idy]
        
        #elif x> Px1:
        #    Px2 = grid_s[idx +1 , idy]
        else:
            Px2 = grid_s[idx + 1, idy] 
    
        # sort the neighbourhood in ascending order
        x1= min(Px1, Px2)
        x2= max(Px1, Px2)
        y1= min(Py1, Py2)
        y2= max(Py1, Py2)
    
        grid_map[i, j]= [(x1,y1), (x2, y2)]    
    
    
    
# check grid
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(values_x, values_y, values_z)
#ax.set_zlabel('blade radius')
#plt.show()