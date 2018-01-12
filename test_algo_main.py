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
import copy
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
surface_orig= cylinder3D(N_s, N_c, 1, blade_length)


# desired spanwise elements
Ns_desired= 11 ##do not change
Nc_desired= 100

n_points= Nc_desired

#initialize the residual vector constants
dc_in= 0 # distance constant that is dtermined by Newton method

#initialize the constant tc
tc= 0 

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
# span sections where the surface is to be found
span_low= 0
span_high= 1
#
alpha= 1 # relaxation factor for the newton method
sor_flag= 0 #flag to trigger NEWTON SOR method
omega= 0.1 # relaxation factor for the SOR method
ls_flag= 1 # flag for line search

# generate the intial surface with points closely arranged to z-zc=0 planes
surface_in, param_map_in = search_plane(sin, tin, N_s, N_c, surface_orig, zc_vec)

#------------------
#----------------------------------------------------------------------------
for i in range(span_low, span_high):#(Ns_desired):
  
  #flag for exiting the while loop
  exit_flag= 1
  # store initial zc 
  zc= zc_vec[i]
  
  # store the current span value
  Pk_in[ind_sin]= param_map_in[span_low, :, 0]
  
  # perturbing the surface----------------
  Pk_in[ind_sin] += 100.
  Pk_in[ind_tin] +=100.
  surface_perturb, _, _= bilinear_surface(surface_in, grid_s, grid_t, 
                                      Pk_in[ind_sin], Pk_in[ind_tin])
# guess for dc_in
  D_in= calculate_distance(surface_perturb[:, 0], 
                           surface_perturb[:, 1], 
                           surface_perturb[:, 2], flag= True)
  dc_in= np.sum(D_in)/n_points
    
  #store initial dc
  dc= dc_in
  
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
    
    S, T= boundary_correction(S, T, n_points)
    
    Q, grid_map, val_map= bilinear_surface(surface_orig, grid_s, grid_t, S, T)
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

    #construct the residual vector
    #NOTE: for the np.arrays use np.dot for matrix multiplication where column vectors
    # and row vectors are automatically treated.
     
    #update dc
    dc= Pk[-1]
    
    R= build_residual(T, Q, D, zc, dc, tc, n_points)
    
    # take max of residual
    R_max= np.max(np.abs(R))
    #-------------------Step 6--------------------------------------------------
    # add a check to exit the newton method
    # ex: np.max(R)<1e-5 
    if R_max < 1e-5:
        # set exit flag as False
        exit_flag= 0
        # store the last Q(x,y,z) points as the final section
        surface_new[i, :, 0]= Q[:, 0]
        surface_new[i, :, 1]= Q[:, 1]
        surface_new[i, :, 2]= Q[:, 2]
        break
    
    #-----------------Step 7---------------------------------------------------
    #inverse the main jacobain array
    jac_main_array= jac_main.toarray()
    jac_main_inv= np.linalg.pinv(jac_main_array)
    # store the k+1 values of the P vector
    # P = [s1, t1, s2, t2...si,ti...., dc] order: (2N+1) x 1
    #delta= - np.dot(jac_main_inv, R)
    delta= - np.linalg.solve(jac_main_array, R)
    # update the state
    if not sor_flag:
        Pk1= Pk + alpha*delta
    else:
        Pk1_tmp= Pk + delta
        Pk1= (omega)*Pk1_tmp + (1-omega)*Pk        
    
    # ------Block to plot line-search----------------------------
    if ls_flag:
        # store R0
        R0_norm= np.linalg.norm(R)
        # create the relaxation factor as a GP
        alpha= np.geomspace(1e-6, 1, num=100, endpoint= True)
        #initialize numpy array
        num= alpha.shape[0]
        R1_norm= np.zeros(num, dtype= float)
        # for loop to get different R1s
        for k in range(num):
            
            Pk1= Pk + alpha[k]*delta
            # udate S and T
            S= Pk1[ind_sin] 
            T= Pk1[ind_tin]
    
            Q, _, _= bilinear_surface(surface_orig, grid_s, grid_t, S, T)
    
            D= calculate_distance(Q[:, 0], Q[:, 1], Q[:, 2], flag= True)
    
            #update dc
            dc= Pk1[-1]
            # construct the residual vector
            R1= build_residual(T, Q, D, zc, dc, tc, n_points)         
            # store  
            R1_norm[k]= np.linalg.norm(R1)
        
        # plot
        from matplotlib import pyplot as plt
        plt.figure()
        plt.semilogx(alpha, R1_norm/R0_norm, '-')
        plt.show()
        #exit the while loop
        exit_flag= 0
        break
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
plt.plot(surface_orig[span_low, :, 0], surface_orig[span_low, :, 1], 'b', label='orig')
plt.legend(loc='best')