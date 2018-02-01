#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:04:27 2018

@author: antariksh
"""

#-------------script to check the extrapolation of the grid-------------------
# load blade surface
surface_blade = np.load('./test_surfaces/KB6_surforig_100s100c_testing.npy')    
# load the cylinder surface with Ns=5, Nc = 3, radius = 1 and length = 5
Ns_cyl = 5
Nc_cyl = 10
r = 1
length_cyl = 5
#create the cylinder surface
surface_cyl = cylinder3D(Ns_cyl, Nc_cyl, r, length_cyl)
# get the extended cylinder surface
grid_s_cylext, grid_t_cylext, surface_cylext = extrap_grid(surface_cyl, Ns_cyl,
                                                           Nc_cyl, 1)

# get the extended blade surface
Ns_blade = surface_blade.shape[0]
Nc_blade = surface_blade.shape[1]
grid_s_blext, grid_t_blext, surface_blext = extrap_grid(surface_blade, 
                                                        Ns_blade, Nc_blade, 1)

#interpolate using splines---------------------------------------------------
grid_s_blext2, grid_t_blext2, surface_blext2 = extrap_np(surface_blade, 1, 1)

grid_s_cylext2, grid_t_cylext2, surface_cylext2 = extrap_np(surface_cyl, 1, 1)

# extend grid by extrap at root and normal protrusion at tip
grid_s_blext3, grid_t_blext3, surface_blext3 = extend_grid(surface_blade)
grid_s_clext3, grid_t_clext3, surface_clext3 = extend_grid(surface_cyl)
#-----------------------------------------------------------------------------
# get the normal
norm_cyl  = normal_vec(surface_cyl[-1, :, :])

# plot check
# plot cylinder root with the extrpolates surface at the root
from matplotlib import pyplot as plt
fig = plt.figure('comp_root')
plt.plot(surface_cylext[0, : ,0], surface_cylext[0, : ,1], 'xr', 
         label = 'extrap_root')
plt.plot(surface_cylext[1, : ,0], surface_cylext[1, : ,1], 'b', 
         label = 'root')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('best')
plt.title('cylinder')
plt.show()
# show near tip extrapolation
fig = plt.figure('comp_tip')
plt.plot(surface_cylext[Ns_cyl+1, : ,0], surface_cylext[Ns_cyl+1, : ,1], 'xr', 
         label = 'extrap_tip')
plt.plot(surface_cylext[Ns_cyl, : ,0], surface_cylext[Ns_cyl, : ,1], 'b', 
         label = 'tip')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('best')
plt.title('cylinder')
plt.show()

# plot the blade root and extrapolated section
fig = plt.figure('comp_root2')
plt.plot(surface_blext[0, : ,0], surface_blext[0, : ,1], 'xr', 
         label = 'extrap_root')
plt.plot(surface_blext[1, : ,0], surface_blext[1, : ,1], 'b', 
         label = 'root')
plt.plot(surface_blext2[0, : ,0], surface_blext2[0, : ,1], 'g')

plt.xlabel('x')
plt.ylabel('y')
plt.legend('best')
plt.title('KB6 blade')
plt.show()
#plot the blade tip and the extrapolated section
fig = plt.figure('comp_tip2')
plt.plot(surface_blext[Ns_blade+1, : ,0], surface_blext[Ns_blade+1, : ,1], 'xr', 
         label = 'extrap_tip')
plt.plot(surface_blext[Ns_blade, : ,0], surface_blext[Ns_blade, : ,1], 'b', 
         label = 'tip')
plt.plot(surface_blext2[Ns_blade+1, : ,0], surface_blext2[Ns_blade+1, : ,1], 'g')
plt.plot(surface_blext3[Ns_blade+1, : ,0], surface_blext3[Ns_blade+1, : ,1], 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('best')
plt.title('KB6 blade')
plt.show()
