#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains distributions that are utilised to discretize
the rotor radius in a variable fashion. 
ex: using cosine distribution will give a fine discretization near
the root and the tip but more coarser distribution mid-span.

Created on Tue Jan 23 14:04:53 2018

@author: antariksh
"""
import numpy as np
    
def distro_cosine(Ndiv, blade_length, end = False):
    """
    """
    #Ndiv = 100
    #blade_length = 10

    # initialize the angle start
    theta_start = 0
    # initialize the angle stop
    theta_stop = np.pi
    # generate the theta vector
    theta_vec = np.linspace(theta_start, theta_stop, 
                            num = Ndiv, endpoint = end)
    # divide the cosine by 2 so that the sum of the divisions is equal to unity
    cos_theta = np.cos(theta_vec) * 0.5
    # get the spatial differences
    delta = abs(cos_theta[1:] - cos_theta[0:-1])
    # initialize the radial vector
    r = 0
    radius = np.empty(0, dtype = float)
    # append the starting point
    radius = np.append(radius, r)
    # construct the radial distribution
    for d in delta:
        r += d*blade_length
        radius = np.append(radius, r)
    
    return radius