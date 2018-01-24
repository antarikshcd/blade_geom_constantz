#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to load and save surface data

Created on Fri Jan 19 14:10:49 2018

@author: antariksh
"""
import numpy as np
from pickling import load_obj
from pickling import save_obj

def load_surface(filename, flag, N_c = 1000, N_s = 1000, iteration = 22):
    # load the surface data
    if flag == 0:
        surface_tmp = np.load(filename + '.npy')
    elif flag == 1:
        surface_tmp = load_obj(filename)
    else:
        surface_tmp = kb6_loftedblade(filename, iteration, N_c, N_s)
    # restructure it from [Nc, Ns, 3] to [Ns, Nc, 3]   
    N_c= surface_tmp.shape[0]
    N_s= surface_tmp.shape[1] 
    # intialize surface array
    surface= np.zeros((N_s, N_c, 3), dtype= float)
    for i in range(N_s):
        surface[i, :, 0]= surface_tmp[:, i, 0]
        surface[i, :, 1]= surface_tmp[:, i, 1]
        surface[i, :, 2]= surface_tmp[:, i, 2]
    # return the surface
    return surface
    