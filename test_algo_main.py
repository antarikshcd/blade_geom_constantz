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
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import pyplot as plt
from parametric_space import bilinear_surface
import pickling 
from vector_operations import calculate_distance
from vector_operations import jacobian_Q
from vector_operations import jacobian_D
from vector_operations import jacobian_main
from surface3d import cylinder3D

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
N_c= 100
N_s= 11

blade_length= 10 # in metres
#surface= pickling.load_obj('KB6_surf_1000by1000') 
#surface= surface*blade_length 
surface= cylinder3D(N_s, N_c, 1, blade_length)
#initialize the residual vector constants
dc_in= 0 # distance constant that is dtermined by Newton method

#initialize the constant tc
tc= 0

# desired spanwise elements
Ns_desired= 11 ##do not change
Nc_desired= 100

n_points= Nc_desired

surface_new= np.zeros((Ns_desired, Nc_desired, 3), dtype= float)
#set value of zc
zc_vec= np.linspace(0, 1*blade_length, num= Ns_desired, endpoint= True)

#initialize the Pk vector with s and t points
Pk_in= np.zeros(2*n_points+1, dtype=float)

sin= np.arange(0, Ns_desired)
tin= np.arange(0, n_points)
ind_sin= np.arange(0, 2*n_points, 2)
ind_tin= np.arange(1, 2*n_points, 2)

#assign the cross-sectional initital guess
Pk_in[ind_tin]= tin
#----------------Step 2------------------------------------------------
# first guess of the s-t space
grid_s, grid_t= np.mgrid[0:N_s, 0:N_c]
#------------test for Q generation---------------------------------------------
#S= np.zeros((10, 1), dtype= float) #spanwise section
#S.fill(1)
#T= np.zeros((10, 1), dtype= float) #chordwise section
#T[0:, 0]= np.arange(0, 10)
#----------------------------------------------------------------------------
for i in range(5,6):#(Ns_desired):
  #flag for exiting the while loop
  exit_flag= 1
  # store initial zc 
  zc= zc_vec[i]
  #store initial dc
  dc= dc_in
  
  Pk_in[ind_sin]= sin[i]
  #initial guess for dc
  Pk_in[-1]= dc
  #initial guess for each span-wise section
  Pk= Pk_in
  
  #counter
  count=0 
  while exit_flag:
  #-------------------------------------------------------------------
    # obtain the X,Y,Z points for the S and T vectors 
    # Q[N, 3] where N=number of points in the slice
    S= Pk[ind_sin] 
    T= Pk[ind_tin]
    
    Q, grid_map, val_map= bilinear_surface(surface, grid_s, grid_t, S, T)
    #----------------------------------------------------------------------------
    #------------------------Step 3---------------------------------------------
    #calculate distance between consecutive x,y,z in the slice also add the final 
    #point and first point to close the loop
    D= calculate_distance(Q[:, 0], Q[:, 1], Q[:, 2], flag= True)

    #------------------------Step 4---------------------------------------------
    # calculate the analytic gradients of each stage

    # jacobian as a sparse matrix for Q-->(x,y,z) wrt P-->(s,t) of size 3Nx2N
    jac_qp, _, _, _, _, dZds, dZdt= jacobian_Q(S, T, grid_map, val_map)

    # jacobian as a sparse matrix for D-->(di) wrt Q-->(x,y,z) of size Nx3N
    jac_dq, _, _, _, _, _, _= jacobian_D(Q, D)

    # jacobian as a sparse matrix for D-->(di) wrt P-->(s,t) of size Nx2N
    jac_dp= jac_dq*jac_qp

    # construct the final jacobian matrix of order (2N+1)x(2N+1) with d-dc, z-zc, t-tc
    # partials
    n_points= S.shape[0] # number of slice points

    jac_main= jacobian_main(dZds, dZdt, jac_dp, n_points)

    #------------------Step 5------------------------------------------------
    #Newton Rhapson solver

    #construct the residual vector
    #NOTE: for the np.arrays use np.dot for matrix multiplication where column vectors
    # and row vectors are automatically treated.
    R= np.zeros((2*n_points + 1), dtype=float)
 
    # fill up the distance function di-dc
    #update dc
    dc= Pk[-1]
    R[:n_points]= D - dc
    #fill up the z coordinate function z-zc
    R[n_points:2*n_points]= Q[:, 2] - zc
    #fill up the t1-tc function 
    R[-1]= T[0] - tc


    #-------------------Step 6--------------------------------------------------
    # add a check to exit the newton method
    # ex: np.max(R)<1e-5 
    if np.max(R)<1e-5:
        # set exit flag as False
        exit_flag= 0
        # store the last Q(x,y,z) points as the final section
        surface_new[i, :, 0]= Q[:, 0]
        surface_new[i, :, 1]= Q[:, 1]
        surface_new[i, :, 2]= Q[:, 2]
        break
    
    #-----------------Step 7---------------------------------------------------
    #inverse the main jacobain array
    jac_main_inv= np.linalg.pinv(jac_main.toarray())
    # store the k+1 values of the P vector
    # P = [s1, t1, s2, t2...si,ti...., dc] order: (2N+1) x 1
    Pk1= Pk - np.dot(jac_main_inv, R) 
    #update P
    Pk=Pk1
    #increase count
    count+=1  
    
    
# check grid
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(Q[:,0], Q[:,1], Q[:,2])
#ax.set_zlabel('blade radius')
#plt.show()