#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates point cloud of 3D surfaces built using parametric information.
Created on Mon Jan  8 12:58:33 2018

@author: antariksh
"""
import numpy as np

def cylinder3D(Ns, Nc, radius, length):
  """ Calculates length between consecutive points on a cross-sectional slice.
        
    Args:
        Nc (int): Number of cross-sectional sections.
        Ns (int): Number of length wise sections.
        length (float): Length of the cylinder in metres.
        radius (float): Radius of the cylindrical cross-sections in metres.
        
    Returns: 
        float: Numpy array of surface point cloud of the shape (Nc, Ns, (x,y,z))
     """     
#Number of crossectional sections
#Nc= 100
#Number of spanwise (or length wise sections)
#Ns=11
#radius= 1
#length= 10
  theta_vec= np.linspace(0, 2*np.pi, num= Nc, endpoint= False)

  #obtian the x,y,z vectors
  x= radius*np.cos(theta_vec)
  y= radius*np.sin(theta_vec)
  z= np.linspace(0, length, num= Ns, endpoint= True)

  surface= np.zeros((Ns, Nc, 3), dtype= float)
  z_cs= np.zeros(Nc, dtype= float)

  for i in range(Ns):
     z_cs[:]= z[i]
     surface[i, :, 0]= x
     surface[i, :, 1]= y
     surface[i, :, 2]= z_cs
       
  # plot and check
  #2-D plot
  dim=''
  if dim=='2D':
      from matplotlib import pyplot as plt
      plt.figure('test_surface')
      plt.plot(x, y)
      plt.show()
  #3-D figure
  elif dim=='3D':
      from matplotlib import pyplot as plt
      from mpl_toolkits.mplot3d import Axes3D
      fig = plt.figure('test_3d')
      ax = fig.add_subplot(111, projection='3d') 
      ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2])
      ax.set_zlabel('length')
      plt.show()
      
  return surface    