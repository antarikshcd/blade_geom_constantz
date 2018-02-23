#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:14:12 2018

@author: antariksh
"""
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


def concat_surface(surface_new, span_ind):
    """
    Stores the new surface without the unfilled section.
    
    Args:
        surface_new (float): Cross-sectional surface definition of order NsXNcX3

        span_ind (int): Spanwise index of the un-solved cross-section    
        
    Return:
        surf_concat: interpolated cross-sectional co-ordinates at the 
                      requested section
    """
    # get the number of spanwise indices that need to be filled
    size = len(span_ind)
    # desired spanwise sections
    Ns = surface_new.shape[0]
    # desired chordwise sections
    Nc = surface_new.shape[1]
    # spanwise sections in the concatenated surface
    Ns_concat = Ns - size
    # initialise the concatenated surface
    surf_concat = np.zeros((Ns_concat, Nc, 3), dtype = float)
    # stores previous index to be excluded
    ind_prev = 0
    # stores previous index for the concatenated surface
    indc_prev = 0
    for ind in span_ind:
        
        # update the number of spanwise elements between two unfilled sections   
        indc_len = ind - ind_prev
        indc_new = indc_prev + indc_len
        # concatenate the surface
        surf_concat[indc_prev : indc_new, :, :] = surface_new[ind_prev:ind, :, :]
        # update the previous indices
        indc_prev = indc_new
        ind_prev = ind + 1
    
    # fill in the last index
    surf_concat[indc_prev : , :, :] = surface_new[ind_prev: , :, :]
    
    return surf_concat


def interp_surface(surface, zc, span_ind, interp_order = 3):
    """ 
    Interpolates at the unfilled spanwise section of the final surface
    
    Args:
        surface (float): Cross-sectional surface definition of order NsXNcX3
        zc (float): the spanwise z-location where the x,y co-ordiantes 
                    are to be obtained
        span_ind (int): Spanwise index of the un-solved cross-section    
        interp_order (int): Interpolation order with 
                            1: linear
                            2: quadratic
                            3: cubic (default)
            
    Return:
        surf: interpolated cross-sectional co-ordinates at the 
                      requested section
        
    """
    # chordwise sections
    Nc = surface.shape[1]
    # initialize output surf
    surf = np.zeros((Nc, 3), dtype = float)
    
    for i in range(Nc):
        # spanwise x and y as function of z
        x = surface[:, i, 0]
        y = surface[:, i, 1]
        z = surface[:, i, 2]
        
        # interp1D functions for x, y, z
        fx = InterpolatedUnivariateSpline(z, x, k= interp_order)
        fy = InterpolatedUnivariateSpline(z, y, k= interp_order)
        
        # obtain the interpolated x and y
        # obtain the interpolated x, y,z spanwise vectors
        surf[i, 0] = fx(zc)
        surf[i, 1] = fy(zc)
        surf[i, 2] = zc
    
    return surf