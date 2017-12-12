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
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from parametric_space import bilinear_surface #user-defined method to perform bilinear
                                         #grid interpolation
#from generate_loftedblade import kb6_loftedblade
import pickling #user-defined module to store and load python data as pickles
from vector_operations import calculate_distance


#format of surface_orig is [i, j, k] where i= Number of cross-sections,
# j= Number of spanwise sections, k= (x, y, z)
#surface_orig= pickling.load_obj('KB6_surf_noshearsweep') 

#create grid between 0 and 1 from input data
#grid_c= cross-sectional discretisation
#grid_s= spanwise discretisation 
#---------------------------------------------------------------------------

#--------------------Step 1-------------------------------------------------
# Set optimization file path
optimization_file= '../../optimization.sqlite'
# set the iteration to choose from
iteration = 22
# set the number of chordwise N_c and spanwise sections N_s for the surface
N_c= 1000
N_s= 1000
#generate the lofted surface
#surface = kb6_loftedblade(optimization_file, iteration, N_c, N_s)
surface= pickling.load_obj('KB6_surf_1000by1000') 


#----------------Step 2------------------------------------------------
# first guess of the s-t space
grid_s, grid_t= np.mgrid[0:N_s, 0:N_c]

#------------test for Q generation---------------------------------------------
S= np.zeros((10, 1), dtype= float) #spanwise section
S.fill(1)
T= np.zeros((10, 1), dtype= float) #chordwise section
T[0:, 0]= np.arange(0, 10)
#-------------------------------------------------------------------
# obtain the X,Y,Z points for the S and T vectors
# Q[N, 3] where N=number of points in the slice
Q= bilinear_surface(surface, grid_s, grid_t, S, T)
#----------------------------------------------------------------------------
#---------------------------------------------------------------------------

#------------------------Step 3---------------------------------------------
#calculate distance between consecutive x,y,z in the slice
D= calculate_distance(Q[:, 0], Q[:, 1], Q[:, 2])

#------------------------Step 4---------------------------------------------
# store the gradients of each stage

# gradient of S-T space
ST_space= S
ST_space= np.c_[ST_space[:,0], T[:,0]]
grad_ST= np.gradient(ST_space, axis=1)

#gradients of Q(X,Y,Z) at the s-t points wrt x,y,z
grad_Q= np.gradient(Q, axis=1)

#gradients of D(d1,d2,d3....dn)
grad_D= np.gradient(D)

#------------------Step 5------------------------------------------------
#Newton Rhapson solver



#t0= time.time()

#surface_new= bilinear_surface(surface, Ncs_desired, Ns_desired)                                  
#t1= time.time()

#print('Time taken to interpolate for Ncs= %i, Ns= %i is %3.2f seconds.'
#       %(Ncs_desired, Ns_desired, (t1-t0)))
#print('Time taken to interpolate on Ncs= ' + repr(Ncs_desired) + ' and Ns= ' +
#        repr(Ns_desired)+ ' is ' + repr(t1-t0) + ' seconds.\n')



# check grid
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(Q[:,0], Q[:,1], Q[:,2])
#ax.set_zlabel('blade radius')
#plt.show()