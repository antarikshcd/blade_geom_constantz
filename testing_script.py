#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test workflow script to check the gradients being formed.
 
Created on Wed Jan 3 14:31:33 2018

@author: antariksh

"""
#import time
import numpy as np
from parametric_space import bilinear_surface #user-defined method to perform bilinear
                                         #grid interpolation
#from generate_loftedblade import kb6_loftedblade
import pickling #user-defined module to store and load python data as pickles
from vector_operations import calculate_distance
from vector_operations import jacobian_Q
from vector_operations import test_jacobian_Q
from vector_operations import jacobian_D
from vector_operations import test_jacobian_D
from vector_operations import jacobian_main
from vector_operations import test_jacobian_DP

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
 
# desired spanwise elements
Ns_desired= 1000 ##do not change
Nc_desired= 1000

n_points= Nc_desired

surface_new= np.zeros((Nc_desired, Ns_desired, 3), dtype= float)

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

# store initial zc 
zc= 0
#store initial dc
dc= 0
tc= 0  
Pk_in[ind_sin]= sin[0]
  #initial guess for dc
Pk_in[-1]= dc
  #initial guess for each span-wise section
Pk= Pk_in
  
  #counter
#-------------------------------------------------------------------
# obtain the X,Y,Z points for the S and T vectors
# Q[N, 3] where N=number of points in the slice
    
      
      
S= Pk[ind_sin] 
T= Pk[ind_tin]
    
Q, grid_map, val_map= bilinear_surface(surface, grid_s, grid_t, S, T)
#----------------------------------------------------------------------------
#---------------------------------------------------------------------------

#------------------------Step 3---------------------------------------------
#calculate distance between consecutive x,y,z in the slice also add the final 
#point and first point to close the loop
D= calculate_distance(Q[:, 0], Q[:, 1], Q[:, 2], flag= True)


# calculate the analytic gradients of each stage

# jacobian as a sparse matrix for Q-->(x,y,z) wrt P-->(s,t) of size 3Nx2N
jac_qp, dXds, dXdt, dYds, dYdt, dZds, dZdt= jacobian_Q(S, T, grid_map, val_map)

#------------------------test jacobian Q(s,t) wrt s,t------------------------
# get Qin
Qin= Q
h= 1e-5
S_f= S + h
T_f= T + h
Qf_s, grid_map_fs, val_map_fs= bilinear_surface(surface, grid_s, grid_t, S_f, T)
Qf_t, grid_map_ft, val_map_ft= bilinear_surface(surface, grid_s, grid_t, S, T_f)

jac_qp_test, dXds_test, dXdt_test, dYds_test, dYdt_test, dZds_test, dZdt_test = test_jacobian_Q(Qin, Qf_s, Qf_t, h, Ns_desired)

#-----------------------------------------------------------------------------

# jacobian as a sparse matrix for D-->(di) wrt Q-->(x,y,z) of size Nx3N
jac_dq, dDdX1, dDdX2, dDdY1, dDdY2, dDdZ1, dDdZ2= jacobian_D(Q, D)

#------------test jacobian D(x,y,z) wrt to Q ie X,Y,Z-------------------------
# dD/dx1
x= Q[:,0]
y= Q[:,1]
z= Q[:,2]

x_h= x + h
y_h= y + h
z_h= z + h

D_x1_h= np.power(np.power(x[1:] - x_h[:-1],2) + np.power(y[1:] - y[:-1],2) +
             np.power(z[1:] - z[:-1],2), 0.5)
D_x1_h= np.append(D_x1_h, np.power(np.power(x[0] - x_h[-1],2) + np.power(y[0] - 
                  y[-1],2) + np.power(z[0] - z[-1],2), 0.5))

D_x2_h= np.power(np.power(x_h[1:] - x[:-1],2) + np.power(y[1:] - y[:-1],2) +
             np.power(z[1:] - z[:-1],2), 0.5)
D_x2_h= np.append(D_x2_h, np.power(np.power(x_h[0] - x[-1],2) + np.power(y[0] - 
                  y[-1],2) + np.power(z[0] - z[-1],2), 0.5))

D_y1_h= np.power(np.power(x[1:] - x[:-1],2) + np.power(y[1:] - y_h[:-1],2) +
             np.power(z[1:] - z[:-1],2), 0.5)
D_y1_h= np.append(D_y1_h, np.power(np.power(x[0] - x[-1],2) + np.power(y[0] - 
                  y_h[-1],2) + np.power(z[0] - z[-1],2), 0.5))

D_y2_h= np.power(np.power(x[1:] - x[:-1],2) + np.power(y_h[1:] - y[:-1],2) +
             np.power(z[1:] - z[:-1],2), 0.5)
D_y2_h= np.append(D_y2_h, np.power(np.power(x[0] - x[-1],2) + np.power(y_h[0] - 
                  y[-1],2) + np.power(z[0] - z[-1],2), 0.5))

D_z1_h= np.power(np.power(x[1:] - x[:-1],2) + np.power(y[1:] - y[:-1],2) +
             np.power(z[1:] - z_h[:-1],2), 0.5)
D_z1_h= np.append(D_z1_h, np.power(np.power(x[0] - x[-1],2) + np.power(y[0] - 
                  y[-1],2) + np.power(z[0] - z_h[-1],2), 0.5))

D_z2_h= np.power(np.power(x[1:] - x[:-1],2) + np.power(y[1:] - y[:-1],2) +
             np.power(z_h[1:] - z[:-1],2), 0.5)
D_z2_h= np.append(D_z2_h, np.power(np.power(x[0] - x[-1],2) + np.power(y[0] - 
                  y[-1],2) + np.power(z_h[0] - z[-1],2), 0.5))

jac_dq_test, dDdX1_test, dDdX2_test, dDdY1_test, dDdY2_test, dDdZ1_test, dDdZ2_test= test_jacobian_D(D, D_x1_h, D_x2_h, D_y1_h, D_y2_h, D_z1_h, D_z2_h, h)
#-----------------------------------------------------------------------------


# jacobian as a sparse matrix for D-->(di) wrt P-->(s,t) of size Nx2N
jac_dp= jac_dq*jac_qp

#--------------jacobain of D wrt P test obtained numerically------------------
#jac_dp_test= jac_dq_test*jac_qp_test

jac_dp_test, dDds1, dDds2, dDdt1, dDdt2= test_jacobian_DP(Q, Qf_s, Qf_t, D, h)
#-------------------------------------------------------------------------- 
# construct the final jacobian matrix of order (2N+1)x(2N+1) with d-dc, z-zc, t-tc
#partials
n_points= S.shape[0] # number of slice points

jac_main= jacobian_main(dZds, dZdt, jac_dp, n_points)

#------------------Step 5------------------------------------------------
#Newton Rhapson solver

#construct the residual vector
#NOTE: for the np.arraysuse np.dot for matrix multiplication where column vectors
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

jac_main_inv= np.linalg.pinv(jac_main.toarray())
    
Pk1= Pk - np.dot(jac_main_inv, R) 
    
