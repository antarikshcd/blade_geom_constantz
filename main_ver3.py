#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow script to transfrom blade geoemtry in cross-section normal 
to the local blade running length vector and sample it in parallel 
planes to the X-Y global plane of the global coordinate system 
attached at the circular blade root centre.
 
Created on Thu Dec  7 16:58:33 2017

@author: antariksh

"""
import time
import numpy as np
from load_save import load_surface
from parametric_space import bilinear_surface 
from initial_guess import search_plane
from vector_operations import build_residual
from vector_operations import calculate_distance
from vector_operations import jacobian_Q
from vector_operations import jacobian_D
from vector_operations import jacobian_main
from numerical_method import newton_iteration
from extend import extend_grid
from interpolate import interp_surface, concat_surface
#from distributions import distro_cosine
#------------To be channged before each run----------------------------------
# relaxation factor for the newton method
alpha_max = 1 
# set tolerance for max of residual such that Rmax < tol
tol = 1e-5
# set tolerance for adaptive relaxation factor alpha
alpha_tol = 1e-8
# flag for the line search plot
flag_ls = 0 
# flag to trigger the script to generate surface within the workflow
flag_gensurf = 0 # 0: to load numpy data; 1: to load pickle data 2: generate surface
# flag to switch for closed surfaces from open surface
flag_surf = False # False: for open surface; True: for closed surface 
# flag to include the end point ie the tip
flag_endpoint = True
# filename to load data (Note: set optimization_file if flag_gensurf = 3)
filename = './input_surfaces/KB6_surface_S1000_C1000_fv1_shearsweep'
#filename = './input_surfaces/KB6_surface_S100_C200_final'
#filename = './input_surfaces/KB6_surface_S100_C100_fv2'
# initialize the desired slice points and spanwise sections
Ns = 100 # span
n_points = 100 # points on a slice (Note: also cross-sectional points)
# testing for specific spans
span_low = 0 # span to be calculated
span_high = Ns # upper limit excluded
# set blade length in metres
blade_length = 10.5538 # in metres for KB6
#blade_length= 11.0639 # in metres for KB1
# -------------------------------------------------------------------------- 
#
# load the surface in shape [Ns, Nc, 3]
surface_orig = load_surface(filename, flag_gensurf)
# scale surface by blade_length
surface_orig *= blade_length
# generate the Z plane position vector
zc_vec = np.linspace(0, 1*blade_length, Ns, endpoint = flag_endpoint)
#zc_vec = distro_cosine(Ns, 1*blade_length, end = False)
# initialize the array for the final surface being generated
surface_new = np.zeros((Ns, n_points, 3), dtype = float)
# initialize the initial state vector--> S_i, T_i, S_(i+1), T_(i+1) 
Pk_in = np.zeros(2*n_points+1, dtype = float)
# initialize the intended S points indices in state vector
ind_sin = np.arange(0, 2*n_points, 2)
# initialize the intended T points indices in state vector
ind_tin = np.arange(1, 2*n_points, 2)
# initialize alpha_prev
#alpha_prev = np.zeros(2*n_points+1, dtype = float)
#-------------- Create the parametric space ----------------------------------
# spanwise sections in original grid
Ns_orig = surface_orig.shape[0]
# chordwise sections in original grid
Nc_orig = surface_orig.shape[1]
# generate the extended parametric grid and the corresponding surface
grid_s, grid_t, surface_ext = extend_grid(surface_orig)
#-----------------------------------------------------------------------------
#-------initialize list and arrays to store state data of iterations---------
count_store = np.zeros(Ns, dtype = int) # stores the iterations per section
# list to store the state (S,T) data for each section and iteration
state_store = [] 
# sore the span index for uncoverged sections
delspan_ind = []
#---------------------------------------------------------------------------
#
# initialize the intended initial S points on slice
sin = np.linspace(0, Ns_orig - 1, Ns, endpoint = True)
# initialize the intended initial T points on slice
tin = np.linspace(0, Nc_orig - 1, n_points, endpoint = True)
# generate the intial surface with points closely arranged to z-zc=0 planes
surface_in, param_map_in = search_plane(sin, tin, Ns, n_points, 
                                        surface_ext, zc_vec)
#time the iteration
t0 = time.time()
#
for i in range(span_low, span_high):
    
    # flag for exiting the while loop
    exit_flag = 1
     
    # inititalize the list to store state at each iteration
    Pk_store = []
    
    # store initial zc 
    zc = zc_vec[i]
    
    # store the current span value
    Pk_in[ind_sin] = param_map_in[i, :, 0]
    # store the slice t-value
    Pk_in[ind_tin] = param_map_in[i, :, 1]
    # calculate the distances in the initial guess
    D_in= calculate_distance(surface_in[i, :, 0], 
                             surface_in[i, :, 1], 
                             surface_in[i, :, 2], flag= False)
    
    # calculate the initial constant distance
    dc_in = np.sum(D_in)/(n_points - 1)
    
    # initial guess for dc
    Pk_in[-1] = dc_in
    
    # initial guess for each span-wise section
    Pk = Pk_in
    
    # store the initial guess of state
    Pk_store.append(Pk)
    
    # initialize while loop counter
    count = 0 
    # initialize the Residual norm
    R_norm_prev = 1
    # set the value of alpha
    # execute the Newton - iteration
    while exit_flag == 1:
        # store s and t
        S = Pk[ind_sin] 
        T = Pk[ind_tin]
        # adjust for boundary correction in t
        #S, T = boundary_correction(S, T, Ns_orig, Nc_orig)
        # interpolate on the parametric space of the original fine grid
        Q, grid_map, val_map = bilinear_surface(surface_ext, grid_s, 
                                               grid_t, S, T)
        # calculate the distance between consecutive points
        D = calculate_distance(Q[:, 0], Q[:, 1], Q[:, 2], flag = flag_surf)
        # size of D vector (if flag = True, then n_D = n_points)
        n_D = D.shape[0]
        # jacobian as a sparse matrix for Q-->(x,y,z) wrt P-->(s,t) of size 3Nx2N
        jac_qp, _, _, _, _, dZds, dZdt = jacobian_Q(S, T, grid_map, val_map)
        # jacobian as a sparse matrix for D-->(di) wrt Q-->(x,y,z) of size (N-1)x3N
        jac_dq, _, _, _, _, _, _ = jacobian_D(Q, D, n_D, 
                                              n_points, flag = flag_surf)
        # jacobian as a sparse matrix for D-->(di) wrt P-->(s,t) of size (N-1)x2N
        jac_dp = jac_dq*jac_qp
        # construct the final jacobian matrix of order (2N+1)x(2N+1) with 
        # d-dc, z-zc, t-tc partials
        jac_main = jacobian_main(dZds, dZdt, jac_dp, 
                                 n_points, n_D, flag = flag_surf)
        # update dc
        dc = Pk[-1]
        # construct the residual vector
        R = build_residual(T, Q, D, zc, dc, tin, 
                           n_D, n_points, flag = flag_surf)
        # take max of residual
        R_max = np.max(np.abs(R))
        
        # check to exit newton iteration
        if R_max < tol:
            # set exit flag as False
            exit_flag = 0
            # store the last Q(x,y,z) points as the final section
            surface_new[i, :, 0] = Q[:, 0]
            surface_new[i, :, 1] = Q[:, 1]
            surface_new[i, :, 2] = Q[:, 2]
            break
            
        # obtain the updated state and status of iteration health
        Pk, alpha, R_norm, delta_norm, jac_main_cond = newton_iteration(Pk, 
                                                       jac_main, R, Nc_orig, 
                                                       n_points) 
        # print the health of the iteration
        print('--------------------------------------------------------------')
        print('\n Span location: S = %i, radius = %3.2f \n'%(i, zc))
        print('\n Iteration = %i, dc = %3.7f, Main jac cond = %e'%(count, dc, 
                                                                jac_main_cond))
        print('\n Residual : R_max = %3.7f, R_norm = %3.7f,' \
               'R_new/R_prev = %3.5f \n'%(R_max, R_norm, R_norm/R_norm_prev))
        print('\n Relaxation factor : alpha = %3.7f \n'%(alpha))
        print('\n Delta vector : delta_norm= %3.7f \n'%(delta_norm))
        print('-------------------------------------------------------------')
        time.sleep(0.1)
    
        #increase count
        count += 1 
        # store the current norm of R
        R_norm_prev = R_norm
        # store the state
        Pk_store.append(Pk)
        
        # escape clause for very low alphs
        if alpha < alpha_tol:
           # escape the section 
           delspan_ind.append(i)
           # print
           print('\n\n Skipping section %i, radius = %3.2f\n\n'%(i, zc))
           # set exit flag as False
           exit_flag = 0
           
    # store the count every iteration
    count_store[i] = count 
    # store the states per section
    state_store.append(Pk_store)


# interpolated across the missing surfaces

if delspan_ind:
    # get the concatenated surface
    surface_concat = concat_surface(surface_new, delspan_ind)    
# the end missing surfaces near the tip are currently extruded    
for i in delspan_ind:
    
    surface_new[i, :, :] = interp_surface(surface_concat, zc_vec[i], i)

    
#record the time    
tend = time.time()
t_elapsed = tend - t0

print('\n Geometry built from Z = %3.2f m to Z = %3.2f m with damping factor'\
      'of %0.2f\n'%(zc_vec[span_low], zc_vec[span_high-1], alpha))
print('Time taken = %3.4f s \n'%t_elapsed)
# check
if span_high - span_low == 1:    
    from matplotlib import pyplot as plt    
    fig = plt.figure('compare')
    plt.plot(surface_new[span_low, :, 0], surface_new[span_low, :, 1], 
             'xr', label = 'new')
    plt.plot(surface_in[span_low, :, 0], surface_in[span_low, :, 1], 
             'o-b', label = 'init- rearranged')
    #plt.plot(surface_orig[span_low, :, 0], surface_orig[span_low, :, 1], 
    #         'g', label = 'orig')
    plt.plot() 
    plt.legend(loc = 'best')
else:
    # save the file as a numpy readable
    np.save('KB6_parsurf_s%ic%i_test.npy'%(Ns, n_points), surface_new)