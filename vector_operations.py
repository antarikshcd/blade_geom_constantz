#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:42:49 2017

@author: antariksh
"""
import numpy as np
from scipy.sparse import coo_matrix

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
    
    return jac_q, dZds, dZdt

def jacobian_D(Q, D):
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
    # inititalize the row andcolumn indices for the partials being stored
   # col_x= 0
   # col_y= 1
   # col_z= 2
   # row_d= 0
    
   # for i in range(n_points):
   #     ind= 6*i
        # store the partial derivatives in the data vector
   #     data[ind]= dDdX1[i]
   #     data[ind+1]= dDdY1[i]
   #     data[ind+2]= dDdZ1[i]
   #     data[ind+3]= dDdX2[i]
   #     data[ind+4]= dDdY2[i]
   #     data[ind+5]= dDdZ2[i]
        
        # store the row indices of coresponding partials
    #    row[ind]= row_d
    #    row[ind+1]= row_d
    #    row[ind+2]= row_d
    #    row[ind+3]= row_d
    #    row[ind+4]= row_d
    #    row[ind+5]= row_d
        
        #store the column indices of corresponding partials
    #    col[ind]= col_x
    #    col[ind+1]= col_y
    #    col[ind+2]= col_z
    #    col[ind+3]= col_x + 3
    #    col[ind+4]= col_y + 3
    #    col[ind+5]= col_z + 3
        
        #increment the indices
     #   col_x+= 6
     #   col_y+= 6
     #   col_z+= 6
     #   row_d+= 1
    
    jac_d= coo_matrix((data,(row, col)), shape= (n_points, 3*n_points))
    
    return jac_d

def jacobian_main(dZds, dZdt, jac_dp, n_points):
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
    row_ind_dzdp= np.arange(n_points, 2*n_points, 1)
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
    data_dc= np.zeros(n_points, dtype=float)
    row_dc= np.zeros(n_points, dtype=int)
    col_dc= np.zeros(n_points, dtype=int)
    # fill in data
    data_dc[:]= -1
    # fill in row indices
    row_dc[:]= row_ind_dzdp
    # fill in column indices
    col_dc[:]= 2*n_points
    # append it to the main jacobian row, column and data
    data= np.append(data, data_dc)
    row= np.append(row, row_dc)
    col= np.append(col, col_dc)
    
    # make the del(t-tc)/del(P). only add non zero entities.
    data_dtdp= 1.0
    row_dtdp= 2*n_points
    col_dtdp= 1
    #append it to the main jacobian row, column and data
    data= np.append(data, data_dtdp)
    row= np.append(row, row_dtdp)
    col= np.append(col, col_dtdp)
    
    jac_main= coo_matrix((data,(row, col)), shape= (2*n_points+1, 2*n_points+1))
    
    return jac_main