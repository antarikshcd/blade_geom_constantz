#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:02:05 2018

@author: antariksh
"""

from PGL.main.domain import Domain, Block
import numpy as np
#load the surface
surface = np.load('KB6_surface_wlast.npy')
x = surface[:, :, 0]
y = surface[:, :, 1]
z = surface[:, :, 2]
#enter the filename to be saved
filename = 'KB6_surf_parplanes_ver1.xyz' 

domain = Domain()
domain.add_blocks(Block(x , y, z))
domain.write_plot3d(filename)
# save ascii format
np.savetxt('KB6_surf_parplanes_ver1_X.dat', m.surface[:,:,0].T)
np.savetxt('KB6_surf_parplanes_ver1_Y.dat', m.surface[:,:,1].T)
np.savetxt('KB6_surf_parplanes_ver1_Z.dat', m.surface[:,:,2].T)