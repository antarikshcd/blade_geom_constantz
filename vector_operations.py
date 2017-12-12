#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:42:49 2017

@author: antariksh
"""
import numpy as np

def calculate_distance(x, y, z):
    
    delta_x= x[1:] - x[0:-1]
    delta_y= y[1:] - y[0:-1]
    delta_z= z[1:] - z[0:-1]
    
    distance= np.power(np.power(delta_x, 2) + np.power(delta_y, 2) + 
                       np.power(delta_z, 2), 0.5)
    return distance