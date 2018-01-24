#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:42:49 2017

@author: antariksh
"""
import numpy as np
from scipy.sparse import coo_matrix
    
def build_residual(T, Q, D, zc, dc, tin, n_D, n_points, flag = False):
    """ Constructs the residual vector.
        
    Args:
        D (float): A float data type 1D numpy array of the distances in 
                  cartesian space.
        Q (float): A float data type 2D numpy array of the 
                  cartesian space.
        T (float): A float data type 1D numpy array of the t-coordinates in 
                   the parametric space.
        zc (float): Constant z plane where the cross-section is being constructed.
        dc (float): Constant distance between consecutive points on the cross-
                    sectional slice, being calculated by the Newton iteration.
        tc (int): The starting point of the slice in parametric space.            
        n_points: Number of points on the cross-sectional slice.
        
    Returns: 
        R:     Numpy vector of the residual terms.
       
     """        
    
    R= np.zeros((2*n_points + 1), dtype=float)
 
    # fill up the distance function di-dc
    #update dc
    R[:n_D]= D - dc
    #fill up the z coordinate function z-zc
    R[n_D : 2*n_D + 1*(n_points - n_D)]= Q[:, 2] - zc
    #fill up the t1-tc function 
    R[2*n_D + 1*(n_points - n_D)]= T[0] - tin[0]
    #
    if not flag:
        R[2*n_points] = T[-1] - tin[-1]
    
    return R
    
def calculate_distance(x, y, z, flag= False):
    """ Calculates length between consecutive points on a cross-sectional slice.
        
    Args:
        x (float): A float data type 1D numpy array of the x-coordinates in 
                  cartesian space.
        y (float): A float data type 1D numpy array of the y-coordinates in 
                  cartesian space.
        z (float): A float data type 1D numpy array of the z-coordinates in 
                  cartesian space.
        flag (bool): A flag which when set True closes treats the set of points
                     as describing the boundary of a surface.
     
    Returns: 
        float: Numpy array of distances between consecutive (x,y,z) locations.
               d1= distance between x1,y1,z1 and x2,y2,z2. If flag= True then
               dn= distance between xn,yn,zn and x1,y1,z1.
       
     """        
    
    delta_x= x[1:] - x[0:-1]
    delta_y= y[1:] - y[0:-1]
    delta_z= z[1:] - z[0:-1]
    
    if flag == True:
      delta_x= np.append(delta_x, x[0] - x[-1])
      delta_y= np.append(delta_y, y[0] - y[-1])
      delta_z= np.append(delta_z, z[0] - z[-1])
        
    distance= np.power(np.power(delta_x, 2) + np.power(delta_y, 2) + 
                       np.power(delta_z, 2), 0.5)
    return distance

def jacobian_Q(S, T, grid_map, val_map):
    """ Stores the jacobians of the obtained X,Y,Z points with respect to the 
        parametric s-t space (s= spanwise location, t= chordwise location). 
        The jacobian is calculated using analytic gradients derived from
        bilinear interpolation function.
        
    Args:
        S (float) : Numpy array of the s (spanwise) coordinate of the desired 
        constant z- point in the s-t parametric space
        
        T (float) : Numpy array of the t (chordwise) coordinate of the desired 
        constant z- point in the s-t parametric space
        
        grid_map (float): numpy array sequence with size equal to the number of
        points on the slice. Each point index has a sequence of 4 dependent 
        points from the original s-t space in the order (s1,t1), (s1,t2), 
        (s2,t1), (s2, t2) as required by the bilinear interpolation.
        
        val_map (float): numpy array sequence with size equal to th number of 
        points in the slice. Each slice point index has a sequence of cartesian 
        (X,Y,Z) values of at the 4 dependent points from the original s-t space
        in the order Q(s1,t1), Q(s1,t2), Q(s2,t1), Q(s2, t2) as required by the
        bilinear interpolation.
                              
    Returns:
        float: A sparse array of size 3Nx2N where N is the number of points on 
        the slice consisting of jacobians of the (x,y,z) with respect to s-t 
        space.
        
    """
    # number of points in the slice
    n_points= S.shape[0]
    
    #total sizeof the jacobian
    #size= 2*3*(n_points**2)
    
    # store dxds, dxdt, dyds, dydt, dzds, dzdt in that order
    # for every 1 point there will be 6 non-zero partials, for N --> 6*N
    data= np.zeros(6*n_points, dtype= float)
    row= np.zeros(6*n_points, dtype= int)
    col= np.zeros(6*n_points, dtype= int)
    
    # inititalize the row andcolumn indices for the partials being stored
    #row_x= 0
    #row_y= 1
    #row_z= 2
    #col_s= 0
    #col_t= 1
    
    #initialize the numpy arrays for the partials
    dXds= np.zeros(n_points, dtype= float)
    dXdt= np.zeros(n_points, dtype= float)
    dYds= np.zeros(n_points, dtype= float)
    dYdt= np.zeros(n_points, dtype= float)
    dZds= np.zeros(n_points, dtype= float)
    dZdt= np.zeros(n_points, dtype= float)
    
    for i in range(n_points):
       s= S[i]
       t= T[i]
       
       s1= grid_map[i][0][0]
       t1= grid_map[i][0][1]
       s2= grid_map[i][3][0]
       t2= grid_map[i][3][1] 
       
       #X- values
       X11= val_map[i][0][0]
       X12= val_map[i][1][0]
       X21= val_map[i][2][0]
       X22= val_map[i][3][0]
       
       #Y-values
       Y11= val_map[i][0][1]
       Y12= val_map[i][1][1]
       Y21= val_map[i][2][1]
       Y22= val_map[i][3][1]
       
       #Z-values
       Z11= val_map[i][0][2]
       Z12= val_map[i][1][2]
       Z21= val_map[i][2][2]
       Z22= val_map[i][3][2]
       
       # store partial derivatives of X wrt s and t
       dXds[i]= ((t - t2)*(X11 - X21) + (t - t1)*(X22 - X12))/((s2 -s1)*(t2 - t1))
       dXdt[i]= ((s - s2)*(X11 - X12) + (s - s1)*(X22 - X21))/((s2 -s1)*(t2 - t1))
       
       # store partial derivatives of Y wrt s and t
       dYds[i]= ((t - t2)*(Y11 - Y21) + (t - t1)*(Y22 - Y12))/((s2 -s1)*(t2 - t1))
       dYdt[i]= ((s - s2)*(Y11 - Y12) + (s - s1)*(Y22 - Y21))/((s2 -s1)*(t2 - t1))
       
       # store partial derivatives of Z wrt s and t
       dZds[i]= ((t - t2)*(Z11 - Z21) + (t - t1)*(Z22 - Z12))/((s2 -s1)*(t2 - t1))
       dZdt[i]= ((s - s2)*(Z11 - Z12) + (s - s1)*(Z22 - Z21))/((s2 -s1)*(t2 - t1))
       
       # store the partial derivatives in a vector as input to a sparse matrix
   #    ind= 6*i
       
   #    data[ind]= dXds
   #    data[ind+1]= dXdt
       
   #    data[ind+2]= dYds
   #    data[ind+3]= dYdt
       
   #    data[ind+4]= dZds
   #    data[ind+5]= dZdt
       
       #store the correpsonding row and column locations of the corresponding data points
   #    row[ind]=  row_x
   #    row[ind+1]= row_x
   #    row[ind+2]= row_y
   #    row[ind+3]= row_y
   #    row[ind+4]= row_z
   #    row[ind+5]= row_z
       
   #    col[ind]= col_s  
   #    col[ind+1]= col_t
   #    col[ind+2]= col_s
   #    col[ind+3]= col_t
   #    col[ind+4]= col_s
   #    col[ind+5]= col_t
       
       # increment the values of the row and coloumn indices
   #    row_x+= 3
   #    row_y+= 3
   #    row_z+= 3
   #    col_s+= 2
   #    col_t+= 2
       
      
    #store the partial derivatives
    #TODO without for loop
    ind= np.arange(0, 6*n_points, 6)
    
    #store the partial derivatives in the data vector
    data[ind]= dXds
    data[ind+1]= dXdt
    data[ind+2]= dYds
    data[ind+3]= dYdt
    data[ind+4]= dZds
    data[ind+5]= dZdt
    
    #store the row indices of corresponding partials
    row_x= np.arange(0, 3*n_points, 3)
    row_y= np.arange(1, 3*n_points, 3)
    row_z= np.arange(2, 3*n_points, 3)
    
    row[ind]= row_x
    row[ind+1]= row_x
    row[ind+2]= row_y
    row[ind+3]= row_y
    row[ind+4]= row_z
    row[ind+5]= row_z
    
    #store the coloumn indices of corrsponding partials
    col_s= np.arange(0, 2*n_points, 2)
    col_t= np.arange(1, 2*n_points, 2)
        
    col[ind]= col_s
    col[ind+1]= col_t
    col[ind+2]= col_s
    col[ind+3]= col_t
    col[ind+4]= col_s
    col[ind+5]= col_t
    
    #store the jacobian in the COO sparse format
    jac_q= coo_matrix((data,(row, col)), shape= (3*n_points, 2*n_points))
    
    return jac_q, dXds, dXdt, dYds, dYdt, dZds, dZdt

def test_jacobian_Q(Qin, Qf_s, Qf_t, h, Ns_desired):
    
    grad_Qs= (Qf_s-Qin)/h
    grad_Qt= (Qf_t-Qin)/h
    
    n_points= Ns_desired
    
    #total sizeof the jacobian
    #size= 2*3*(n_points**2)
    
    # store dxds, dxdt, dyds, dydt, dzds, dzdt in that order
    # for every 1 point there will be 6 non-zero partials, for N --> 6*N
    data= np.zeros(6*n_points, dtype= float)
    row= np.zeros(6*n_points, dtype= int)
    col= np.zeros(6*n_points, dtype= int)
    
    # inititalize the row andcolumn indices for the partials being stored
    #row_x= 0
    #row_y= 1
    #row_z= 2
    #col_s= 0
    #col_t= 1
    
    #initialize the numpy arrays for the partials
    dXds= grad_Qs[:,0]
    dXdt= grad_Qt[:,0]
    dYds= grad_Qs[:,1]
    dYdt= grad_Qt[:,1]
    dZds= grad_Qs[:,2]
    dZdt= grad_Qt[:,2]
    ind= np.arange(0, 6*n_points, 6)
    
    #store the partial derivatives in the data vector
    data[ind]= dXds
    data[ind+1]= dXdt
    data[ind+2]= dYds
    data[ind+3]= dYdt
    data[ind+4]= dZds
    data[ind+5]= dZdt
    
    #store the row indices of corresponding partials
    row_x= np.arange(0, 3*n_points, 3)
    row_y= np.arange(1, 3*n_points, 3)
    row_z= np.arange(2, 3*n_points, 3)
    
    row[ind]= row_x
    row[ind+1]= row_x
    row[ind+2]= row_y
    row[ind+3]= row_y
    row[ind+4]= row_z
    row[ind+5]= row_z
    
    #store the coloumn indices of corrsponding partials
    col_s= np.arange(0, 2*n_points, 2)
    col_t= np.arange(1, 2*n_points, 2)
        
    col[ind]= col_s
    col[ind+1]= col_t
    col[ind+2]= col_s
    col[ind+3]= col_t
    col[ind+4]= col_s
    col[ind+5]= col_t
    
    #store the jacobian in the COO sparse format
    jac_qp_test= coo_matrix((data,(row, col)), shape= (3*n_points, 2*n_points))
    
    return jac_qp_test, dXds, dXdt, dYds, dYdt, dZds, dZdt

def jacobian_D(Q, D, n_D, n_points, flag = False):
    """ Stores the jacobian of the distances between consecutive points on the
    cross-sectional slice with respect to the dependent cartesian coordiantes.
    The partial derivatives are calculated from analytic gradients.
    
    Args:
         Q (float) : Numpy array of the x,y,z coordinate points for the sampled 
        surface in the s-t parametric space.
    
    Returns:
        
    """
    # extract the x,y,z coordinates of the slice points
    x= Q[:, 0]
    y= Q[:, 1]
    z= Q[:, 2]
    
    # store the differences P(i+1)- P(i)
    delta_x= x[1:] - x[0:-1]
    delta_y= y[1:] - y[0:-1]
    delta_z= z[1:] - z[0:-1]
    
    if flag:
        #add the last element such that P(N+1) = P(0)
        delta_x= np.append(delta_x, x[0] - x[-1])
        delta_y= np.append(delta_y, y[0] - y[-1])
        delta_z= np.append(delta_z, z[0] - z[-1])
    
    # calculate the partials
    dDdX1= - np.divide(delta_x, D)
    dDdX2= - dDdX1
    
    dDdY1= - np.divide(delta_y, D)
    dDdY2= - dDdY1
    
    dDdZ1= - np.divide(delta_z, D)
    dDdZ2= - dDdZ1
    
    #total sizeof the jacobian
    # size= 3*(n_points**2)
    # store dxds, dxdt, dyds, dydt, dzds, dzdt in that order
    # for every 1 distance, there will be 6 non-zero partials
    data= np.zeros(6*n_D, dtype= float)
    row= np.zeros(6*n_D, dtype= int)
    col= np.zeros(6*n_D, dtype= int)
    
        
    #store the partial derivatives
    #TODO without for loop
    ind= np.arange(0, 6*n_D, 6)
    
    #store the partial derivatives in the data vector
    data[ind]= dDdX1
    data[ind+1]= dDdY1
    data[ind+2]= dDdZ1
    data[ind+3]= dDdX2
    data[ind+4]= dDdY2
    data[ind+5]= dDdZ2
    
    #store the row indices of corresponding partials
    row_d= np.arange(0, n_D, 1)
    row[ind]= row_d
    row[ind+1]= row_d
    row[ind+2]= row_d
    row[ind+3]= row_d
    row[ind+4]= row_d
    row[ind+5]= row_d
    
    col_x= np.arange(0, 3*n_D, 3)
    col_y= np.arange(1, 3*n_D, 3)
    col_z= np.arange(2, 3*n_D, 3)
    
    col[ind]= col_x
    col[ind+1]= col_y
    col[ind+2]= col_z
    col[ind+3]= col_x + 3
    col[ind+4]= col_y + 3
    col[ind+5]= col_z + 3
    
    if flag:
        #overwriting the final three indices to close the loop
        col[-1]= 2 #del(dn)/del(z1)
        col[-2]= 1 #del(dn)/del(z2)
        col[-3]= 0 #del(dn)/del(z3)
    
    # construct the jacobian
    jac_d= coo_matrix((data,(row, col)), shape= (n_D, 3*n_points))
    
    return jac_d, dDdX1, dDdX2, dDdY1, dDdY2, dDdZ1, dDdZ2

def test_jacobian_D(D, D_x1_h, D_x2_h, D_y1_h, D_y2_h, D_z1_h, D_z2_h, h):
        
    # calculate the partials
    dDdX1= (D_x1_h - D)/h
    dDdX2= (D_x2_h - D)/h
    
    dDdY1= (D_y1_h - D)/h
    dDdY2= (D_y2_h - D)/h
    
    dDdZ1= (D_z1_h - D)/h
    dDdZ2= (D_z2_h - D)/h
    
    # number of points on the slice
    n_points= D.shape[0]
    
    #total sizeof the jacobian
   # size= 3*(n_points**2)
    # store dxds, dxdt, dyds, dydt, dzds, dzdt in that order
    # for every 1 distance, there will be 6 non-zero partials
    data= np.zeros(6*n_points, dtype= float)
    row= np.zeros(6*n_points, dtype= int)
    col= np.zeros(6*n_points, dtype= int)
    
        
    #store the partial derivatives
    #TODO without for loop
    ind= np.arange(0, 6*n_points, 6)
    
    #store the partial derivatives in the data vector
    data[ind]= dDdX1
    data[ind+1]= dDdY1
    data[ind+2]= dDdZ1
    data[ind+3]= dDdX2
    data[ind+4]= dDdY2
    data[ind+5]= dDdZ2
    
    #store the row indices of corresponding partials
    row_d= np.arange(0, n_points, 1)
    row[ind]= row_d
    row[ind+1]= row_d
    row[ind+2]= row_d
    row[ind+3]= row_d
    row[ind+4]= row_d
    row[ind+5]= row_d
    
    col_x= np.arange(0, 3*n_points, 3)
    col_y= np.arange(1, 3*n_points, 3)
    col_z= np.arange(2, 3*n_points, 3)
    
    col[ind]= col_x
    col[ind+1]= col_y
    col[ind+2]= col_z
    col[ind+3]= col_x + 3
    col[ind+4]= col_y + 3
    col[ind+5]= col_z + 3
    #overwriting the final three indices to close the loop
    col[-1]= 2 #del(dn)/del(z1)
    col[-2]= 1 #del(dn)/del(z2)
    col[-3]= 0 #del(dn)/del(z3)
    
    jac_dq_test= coo_matrix((data,(row, col)), shape= (n_points, 3*n_points))
    
    return jac_dq_test, dDdX1, dDdX2, dDdY1, dDdY2, dDdZ1, dDdZ2

def test_jacobian_DP(Q, Qf_s, Qf_t, D, h):
    
    # store the Q values by perturbing s+h,t 
    x_s_h= np.append(Qf_s[:, 0], Qf_s[0, 0])
    x_s= np.append(Q[:, 0], Q[0, 0])
    y_s_h= np.append(Qf_s[:, 1], Qf_s[0, 1])
    y_s= np.append(Q[:, 1], Q[0, 1])
    z_s_h= np.append(Qf_s[:, 2], Qf_s[0, 2])
    z_s= np.append(Q[:, 2], Q[0, 2])
    
    # store the Q values by perturbing s,t+h
    x_t_h= np.append(Qf_t[:, 0], Qf_t[0, 0])
    x_t= x_s
    y_t_h= np.append(Qf_t[:, 1], Qf_t[0, 1])
    y_t= y_s
    z_t_h= np.append(Qf_t[:, 2], Qf_t[0, 2])
    z_t= z_s

    # caclulate distances for Di(s1+h, s2, t1, t2)
    D_s1= np.power(np.power(x_s[1:] - x_s_h[:-1], 2) + np.power(y_s[1:] - 
                   y_s_h[:-1], 2) + np.power(z_s[1:] - z_s_h[:-1], 2), 0.5)
    
    # caclulate distances for Di(s1, s2+h, t1, t2)
    D_s2= np.power(np.power(x_s_h[1:] - x_s[:-1], 2) + np.power(y_s_h[1:] - 
                   y_s[:-1], 2) + np.power(z_s_h[1:] - z_s[:-1], 2), 0.5)
    
    # caclulate distances for Di(s1, s2, t1+h, t2)
    D_t1= np.power(np.power(x_t[1:] - x_t_h[:-1], 2) + np.power(y_t[1:] - 
                   y_t_h[:-1], 2) + np.power(z_t[1:] - z_t_h[:-1], 2), 0.5)
    
    # caclulate distances for Di(s1, s2, t1, t2+h)
    D_t2= np.power(np.power(x_t_h[1:] - x_t[:-1], 2) + np.power(y_t_h[1:] - 
                   y_t[:-1], 2) + np.power(z_t_h[1:] - z_t[:-1], 2), 0.5)
    
    # gradient D wrt s1
    dDds1= (D_s1 - D)/h
    # gradient D wrt s2
    dDds2= (D_s2 - D)/h
    # gradient D wrt t1
    dDdt1= (D_t1 - D)/h
    # gradient D wrt t2
    dDdt2= (D_t2 - D)/h
    
    # number of points on the slice   
    n_points= D.shape[0]
    #initialize nump arrays for the sparese matric data storage
    data= np.zeros(4*n_points, dtype= float)
    row= np.zeros(4*n_points, dtype= int)
    col= np.zeros(4*n_points, dtype= int)
    # indices for storing the data
    ind= np.arange(0, 4*n_points, 4)
    #store the data
    data[ind]= dDds1
    data[ind+1]= dDdt1
    data[ind+2]= dDds2
    data[ind+3]= dDdt2
    #define the row and column indices to store the row and column info
    row_ind= np.arange(0, n_points)
    row[ind]= row_ind 
    row[ind+1]= row_ind
    row[ind+2]= row_ind
    row[ind+3]= row_ind
    # assign the column indices
    col_ind_s1= np.arange(0, 2*n_points, 2) 
    col_ind_t1= col_ind_s1 + 1
    col_ind_s2= col_ind_s1 + 2
    col_ind_t2= col_ind_s1 + 3
    #adjust the last line to reflect a loop
    col_ind_s2[-1]= 0
    col_ind_t2[-1]= 1
    #store the column indices
    col[ind]= col_ind_s1
    col[ind+1]= col_ind_t1
    col[ind+2]= col_ind_s2
    col[ind+3]= col_ind_t2
    
    #build the sparse jacobian
    jac_dp_test= coo_matrix((data,(row, col)), shape= (n_points, 2*n_points))
    
    return jac_dp_test, dDds1, dDds2, dDdt1, dDdt2

def jacobian_main(dZds, dZdt, jac_dp, n_points, n_D, flag = False):
    """ Constructs the main jacobian matrix of size 2N x (2N+1)
    """
    #
    #jac_main= np.zeros((2*n_points, 2*n_points + 1), dtype= float)
      
    #data= np.zeros(6*n_points, dtype= float)
    #row= np.zeros(6*n_points, dtype= float)
    #col= np.zeros(6*n_points, dtype= float)

# format: first fill in jac_dp (Nx2N), then jac(Z) (Nx2N) followed by grad(t-tc) (1X2N)
# and lastly grad(dc) (2N+1 X 1)
    
    data= jac_dp.data
    
    row= jac_dp.nonzero()[0]
    col= jac_dp.nonzero()[1]  # ind= 0 to ind 2N-1
    
    
    # adding delz/delP
    ind_dzdp= np.arange(0, 2*n_points, 2)
    row_dzdp= np.zeros(2*n_points, dtype=int)
    col_dzdp= np.zeros(2*n_points, dtype=int)
    data_dzdp= np.zeros(2*n_points, dtype=float)
    # fill in data 
    data_dzdp[ind_dzdp]= dZds
    data_dzdp[ind_dzdp + 1]= dZdt
    #fill in row
    row_ind_dzdp= np.arange(n_D, 2*n_D + 1*(n_points-n_D), 1)
    row_dzdp[ind_dzdp]= row_ind_dzdp
    row_dzdp[ind_dzdp + 1]= row_ind_dzdp
    #fill in column
    col_ind_dzds= np.arange(0, 2*n_points, 2)
    col_ind_dzdt= np.arange(1, 2*n_points, 2)
    col_dzdp[ind_dzdp]= col_ind_dzds
    col_dzdp[ind_dzdp + 1]= col_ind_dzdt
    # append it to the main jacobian row, column and data
    data= np.append(data, data_dzdp)
    row= np.append(row, row_dzdp)
    col= np.append(col, col_dzdp)
    
    #fill in the gradient of (d-dc) wrt to (dc) = -1
    data_dc= np.zeros(n_D, dtype = float)
    row_dc= np.zeros(n_D, dtype = int)
    col_dc= np.zeros(n_D, dtype = int)
    # fill in data
    data_dc[:]= -1
    # fill in row indices
    row_ind_dDdp= np.arange(0, n_D)
    row_dc= row_ind_dDdp
    # fill in column indices
    col_dc[:]= 2*n_points
    # append it to the main jacobian row, column and data
    data= np.append(data, data_dc)
    row= np.append(row, row_dc)
    col= np.append(col, col_dc)
    
    # make the del(t1-tc)/del(P). only add non zero entities.
    data_dt1dp= 1.0
    row_dt1dp= 2*n_D + 1*(n_points - n_D)
    col_dt1dp= 1
    #append it to the main jacobian row, column and data
    data= np.append(data, data_dt1dp)
    row= np.append(row, row_dt1dp)
    col= np.append(col, col_dt1dp)
    
    if not flag:
        # make the del(tn-tc)/del(P). only add non zero entities.
        data_dtNdp= 1.0
        row_dtNdp= 2*n_points
        col_dtNdp= 2*n_points - 1
        #append it to the main jacobian row, column and data
        data= np.append(data, data_dtNdp)
        row= np.append(row, row_dtNdp)
        col= np.append(col, col_dtNdp)
    # construct the main jacobian
    jac_main= coo_matrix((data,(row, col)), shape= (2*n_points + 1, 2*n_points+1))
    
    return jac_main