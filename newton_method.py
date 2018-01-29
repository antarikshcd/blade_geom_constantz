#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the supporting methods for newton iteration.
Created on Fri Jan 26 08:27:55 2018

@author: antariksh
"""
import numpy as np

def adaptive_alpha(Pk, delta, Nc_orig, tin, n_points, alpha_prev):
    """ Value for alpha is obtained that maintains the order of t-space ie
        t0 < t1 < t2 .....< tn and ensures that 0<= t <= tn.
        
        Args:
            Pk (float) : The state vector comprising of [Si,Ti, S(i+1), 
                        T(i+1)] values.
            delta (float): The change in state vector for the newton step.\
            
        Returns:
            alpha : A floating point value of the relaxation factor
                    between 0 and 1.
                    
    """
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

def adaptive_alpha_old(Pk, delta, Nc_orig, alpha_prev, sin, tin, n_points):
    """ Value for alpha is obtained that maintains the order of t-space ie
        t0 < t1 < t2 .....< tn and ensures that 0<= t <= tn.
        
        Args:
            Pk (float) : The state vector comprising of [Si,Ti, S(i+1), 
                        T(i+1)] values.
            delta (float): The change in state vector for the newton step.\
            
        Returns:
            alpha : A floating point value of the relaxation factor
                    between 0 and 1.
                    
    """
    # T space at the kth iteration
    Tk = Pk[tin]
    # change in the T-space
    delta_T = delta[tin]
    
    # specify the T-limits
    T0 = 0
    Tend = Nc_orig - 1
    
    # store the original alpha vector
    alpha_orig = alpha_prev
    # concatenate the alpha vector
    alpha_t = alpha_prev[tin]
    # set the alpha limits
    alpha_max = alpha_t
    alpha_min = 1e-4
    
    # initialize alpha
    alpha = alpha_max
    
    # while loop with conditions to satisfy t+delta*alpha<100 and t+delta*alpha>0
    # initialize T_high and T_low
    T_high = Tk + np.multiply(alpha, delta_T)
    T_low = Tk - np.multiply(alpha, delta_T) 
    
    for i in range(n_points):
        while T_high[i] > Tend or T_low[i] < T0:
            # decrease alpha by 10%
            alpha[i] -= 0.1*alpha[i]
            # re-evaluate T_high and T_low
            T_high[i] = Tk[i] + alpha[i]*delta_T[i]
            T_low[i] = Tk[i] - alpha[i]*delta_T[i] 
    
    # ensure that no points cross over each other
    
    alpha_deltaT0 = np.multiply(alpha[0:-1], delta_T[0:-1])
    alpha_deltaT1 = np.multiply(alpha[1:], delta_T[1:])
    
    diff_Tk = Tk[1:] - Tk[0:-1]
    diff_Tk1 = diff_Tk + alpha_deltaT1 - alpha_deltaT0
    
    for i in range(n_points-1):
        while diff_Tk1[i] < 0:
            # variable alphas
        
            # decrease value of alpha for delta0 by 10%
            if delta_T[i+1] > delta_T[i]:
                alpha[i] -= 0.1*alpha[i]
            else:
                alpha[i+1] -= 0.1*alpha[i+1] 
            # re-evaluate
            diff_Tk1[i] = diff_Tk[i] + alpha[i+1] * delta_T[i+1] - alpha[i] * delta_T[i]
        
    
    # make the complete alpha vector
    alpha_orig[sin] = alpha
    alpha_orig[tin] = alpha
    
    return alpha        