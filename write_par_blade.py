#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:02:05 2018

@author: antariksh
"""

from PGL.main.domain import Domain, Block
import numpy as np
import os

# flag to sample only 35 spanwise cross-sections
flag = 0
# enter the version of the file being stored
ver = 6
#load the surface
surface_orig = np.load('./input/KB6_parsurf_s100c100_fv2.npy')
Ns = surface_orig.shape[0]
Nc = surface_orig.shape[1]

surface= np.zeros((Nc, Ns, 3), dtype= float)
for i in range(Ns):
	surface[:, i, 0] = surface_orig[i, :, 0]
	surface[:, i, 1] = surface_orig[i, :, 1]
	surface[:, i, 2] = surface_orig[i, :, 2]

x = surface[:, :, 0]
y = surface[:, :, 1]
z = surface[:, :, 2]
 
# make folder for saving the file
folder = 'output/S%i_C%i_v%i/'%(Ns, Nc, ver)
os.makedirs(folder)
os.chdir(folder)

domain = Domain()
domain.add_blocks(Block(x , y, z))
domain.write_plot3d('KB6_S%i_C%i_v%i.xyz'%(Ns, Nc, ver))
# save ascii format
np.savetxt('KB6_S%i_C%i_v%i_X.dat'%(Ns, Nc, ver), x)
np.savetxt('KB6_S%i_C%i_v%i_Y.dat'%(Ns, Nc, ver), y)
np.savetxt('KB6_S%i_C%i_v%i_Z.dat'%(Ns, Nc, ver), z)

if flag:
    # concatenate the spanwise sections to 30 including the first and last

    s_ind = np.arange(0, Ns, int(Ns/30))
    surface_cut = surface_orig[s_ind,:, :]
    x_cut = surface_cut[:, :, 0]
    y_cut = surface_cut[:, :, 1]
    z_cut = surface_cut[:, :, 2]

    filename = 'concat30_KB6_S%i_C%i_v%i.xyz'%(Ns, Nc, ver)

    domain = Domain()
    domain.add_blocks(Block(x , y, z))
    domain.write_plot3d(filename)
    # save ascii format
    np.savetxt('concat30_KB6_S%i_C%i_v%i_X.dat'%(Ns, Nc, ver), x_cut)
    np.savetxt('concat30_KB6_S%i_C%i_v%i_Y.dat'%(Ns, Nc, ver), y_cut)
    np.savetxt('concat30_KB6_S%i_C%i_v%i_Z.dat'%(Ns, Nc, ver), z_cut)