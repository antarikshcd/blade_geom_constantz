#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the supporting methods for newton iteration.
Created on Fri Jan 26 08:27:55 2018

@author: antariksh
"""
import numpy as np

def newton_iteration(Pk, jac_main, R, Nc_orig, n_points):
    """
        Outputs the updated state vector and status of various parameters
        essential to determining the health of the iteration.
        
        Args:
            Pk (float) : A (2n+1) numpy array representing the state of the 
                         problem at the kth stage.
            jac_main (float): A (2n+1) X (2n+1) numpy array representing the 
                              jacobian of the state Pk.
            R (float) : A (2n+1) numpy array representing the residual of the 
                        problem.
            Nc_orig (int) : The number of cross-sectional points in the 
                            input surface.
            n_points (int) : The number of cross-sectional points in the final
                             desired output.                  
                  
        Returns:
            Pk1 : A (2n+1) floating point numpy array representing the state
                  at the (k+1)th iteration.
            R_norm : Second norm of the residual.
            delta_norm : Second norm of the state change.
            jac_main_cond : Condition number of the main jacobian.
            alpha : Relaxation factor
            
    """    
    # convert the main jacobian from sparse to dense
    jac_main_array = jac_main.toarray()
    # solve the change in state delta(P)= P(k+1) - Pk
    delta = - np.linalg.solve(jac_main_array, R)
    #
    alpha = adaptive_alpha(Pk, delta, Nc_orig, n_points)
    #alpha = 0.1
    ##
    # update the state
    Pk1 = Pk + alpha*delta
        
    # print out the norm of residual, iteration and norm of delta
    R_norm = np.linalg.norm(R)
    delta_norm = np.linalg.norm(delta)
    jac_main_cond = np.linalg.cond(jac_main_array)
    
    return Pk1, alpha, R_norm, delta_norm, jac_main_cond
    
    
def adaptive_alpha(Pk, delta, Nc_orig, n_points):
    """ 
        Value for alpha is obtained that maintains the order of t-space ie
        t0 < t1 < t2 .....< tn and ensures that 0<= t <= tn.
        
        Args:
            Pk (float) : The state vector comprising of [Si,Ti, S(i+1), 
                        T(i+1)] values.
            delta (float): The change in state vector for the newton step.\
            
        Returns:
            alpha : A floating point value of the relaxation factor
                    between 0 and 1.
                    
    """
    tin = np.arange(1, 2*n_points, 2)
    # T space at the kth iteration
    Tk = Pk[tin]
    # change in the T-space
    delta_T = delta[tin]
    
    # specify the T-limits
    T0 = 0
    Tend = Nc_orig - 1
    
    #  alpha limits
    alpha_max = 1
    
    # initialize alpha
    alpha = alpha_max
    
    #--------------Part1-----------------------------------------------------
    # while loop with conditions to satisfy t+delta*alpha<100 and t+delta*alpha>0
    # initialize T_high and T_low
    T_high = Tk + alpha*delta_T
    T_low = Tk - alpha*delta_T
    
    while max(T_high) > Tend or min(T_low) < T0:
        # decrease alpha by 10%
        alpha -= 0.1*alpha
        # re-evaluate T_high and T_low
        T_high = Tk + alpha*delta_T
        T_low = Tk - alpha*delta_T 
        
    #---------------------Part2----------------------------------------------
    # ensure that no points cross over each other
    
    diff_Tk = Tk[1:] - Tk[0:-1]
    diff_deltaT = delta_T[1:] - delta_T[0:-1]
    
    diff_deltaT_Tk = alpha*np.abs(diff_deltaT) - diff_Tk
    
    for i in range(n_points - 1):
        
        if diff_deltaT[i] < 0 and diff_deltaT_Tk[i] > 0:
            
            alpha_max = diff_Tk[i] / abs(diff_deltaT[i])
            if alpha > alpha_max:
                alpha = 0.5*alpha_max
                
    return alpha