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
from parametric_space import boundary_correction
from parametric_space import search_plane
import pickling 
from vector_operations import build_residual
from vector_operations import calculate_distance
from vector_operations import jacobian_Q
from vector_operations import jacobian_D
from vector_operations import jacobian_main

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
N_c= 150
N_s= 100
#generate the lofted surface
#surface = kb6_loftedblade(optimization_file, iteration, N_c, N_s)
blade_length= 1 # in metres
#surface_tmp= pickling.load_obj('KB6_surf_1000by1000') 
surface_tmp= pickling.load_obj('KB6_surf_noshearsweep')
#surface_tmp= pickling.load_obj('KB1_surface_S100_C100')

surface_tmp= surface_tmp*blade_length 

# rearrange surface from (Nc, Ns, 3) to (Ns, Nc, 3)
surface_orig= np.zeros((N_s, N_c, 3), dtype= float)
for i in range(N_s):
    surface_orig[i,:,0]= surface_tmp[:,i,0]
    surface_orig[i,:,1]= surface_tmp[:,i,1]
    surface_orig[i,:,2]= surface_tmp[:,i,2]
    
#initialize the residual vector constants
dc_in= 0 # distance constant that is dtermined by Newton method

#initialize the constant tc
tc= 0

# desired spanwise elements
Ns_desired= N_s ##do not change
Nc_desired= N_c

n_points= Nc_desired

surface_new= np.zeros((Ns_desired, Nc_desired, 3), dtype= float)
#set value of zc
zc_vec= np.linspace(0, 1*blade_length, Ns_desired)
#zc_vec= surface[:, 0, 2]
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

alpha= 1 # relaxation factor for the newton method
sor_flag= 0 #flag to trigger NEWTON SOR method
omega= 0.1 # relaxation factor for the SOR method

# testing for specific spans
span_low = 0
span_high = 1

# generate the intial surface with points closely arranged to z-zc=0 planes
surface_in = search_plane(sin, tin, N_s, N_c, surface_orig, zc_vec)
#----------------------------------------------------------------------------
for i in range(span_low, span_high):#(Ns_desired):
  #flag for exiting the while loop
  exit_flag= 1
  # store initial zc 
  zc= zc_vec[i]
  # store the current span value
  Pk_in[ind_sin]= sin[i]
  
  # initial guess for dc
  surface, _, _= bilinear_surface(surface_in, grid_s, grid_t, 
                                      Pk_in[ind_sin], Pk_in[ind_tin])
# guess for dc_in
  D_in= calculate_distance(surface[:, 0], 
                           surface[:, 1], 
                           surface[:, 2], flag= True)
  dc_in= np.sum(D_in)/n_points
  
  
  #store initial dc
  dc= dc_in
  #initial guess for dc
  Pk_in[-1]= dc
  #initial guess for each span-wise section
  Pk= Pk_in
  
  # initialize while loop counter
  count=0 
  
  while exit_flag:
  #-------------------------------------------------------------------
    # obtain the X,Y,Z points for the S and T vectors 
    # Q[N, 3] where N=number of points in the slice
    S= Pk[ind_sin] 
    T= Pk[ind_tin]
    
    S, T= boundary_correction(S, T, n_points)
    
    Q, grid_map, val_map= bilinear_surface(surface_in, grid_s, grid_t, S, T)
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
    jac_main= jacobian_main(dZds, dZdt, jac_dp, n_points)

    #------------------Step 5------------------------------------------------
    #Newton Rhapson solver
   
    # fill up the distance function di-dc
    #update dc
    dc= Pk[-1]
    # construct the residual vector
    R= build_residual(T, Q, D, zc, dc, tc, n_points)
    
    # take max of residual
    R_max= np.max(np.abs(R))
    #-------------------Step 6--------------------------------------------------
    # add a check to exit the newton method
    # ex: np.max(R)<1e-5 
    if np.max(R) < 1e-4:
        # set exit flag as False
        exit_flag= 0
        # store the last Q(x,y,z) points as the final section
        surface_new[i, :, 0] = Q[:, 0]
        surface_new[i, :, 1] = Q[:, 1]
        surface_new[i, :, 2] = Q[:, 2]
        break
    
    #-----------------Step 7---------------------------------------------------
    jac_main_array= jac_main.toarray()
    jac_main_inv= np.linalg.pinv(jac_main_array)
    
    delta= - np.linalg.solve(jac_main_array, R)
    # update the state
    if not sor_flag:
        Pk1= Pk + alpha*delta
    else:
        Pk1_tmp= Pk + delta
        Pk1= (omega)*Pk1_tmp + (1-omega)*Pk        
    #update P
    Pk=Pk1
       
    # print out the norm of residual, iteration and norm of delta
    R_norm= np.linalg.norm(R)
    delta_norm= np.linalg.norm(delta)
    jac_main_cond= np.linalg.cond(jac_main_array)
    
    print('----------------------------------------------------')
    print('\n Iteration= %i, dc= %3.5f, Main jac cond= %e'%(count, dc, jac_main_cond))
    print('\n Residual : R_max= %3.5f, R_norm= %3.5f \n'%(R_max, R_norm))
    print('\n Delta vector : delta_norm= %3.5f \n'%(delta_norm))
    print('----------------------------------------------------')
    time.sleep(0.3)
    
    #increase count
    count+=1 
    
    
# check
from matplotlib import pyplot as plt    
fig= plt.figure('compare')
plt.plot(surface_new[span_low, :, 0], surface_new[span_low, :, 1], 'xr', label='new')
plt.plot(surface_in[span_low, :, 0], surface_in[span_low, :, 1], 'b', label='orig')
plt.legend(loc='best')