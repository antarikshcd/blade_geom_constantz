#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:28:53 2018

@author: antariksh
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# given values
xi = np.array([0.2, 0.5, 0.7, 0.9])
yi = np.array([0.3, -0.1, 0.2, 0.1])
# positions to inter/extrapolate
x = np.linspace(0, 2, 100)
# spline order: 1 linear, 2 quadratic, 3 cubic ... 
order = 1
# do inter/extrapolation
s = InterpolatedUnivariateSpline(xi, yi, k=order)
y = s(x)

# example showing the interpolation for linear, quadratic and cubic interpolation
plt.figure()
plt.plot(xi, yi)
for order in range(1, 4):
    s = InterpolatedUnivariateSpline(xi, yi, k=order)
    y = s(x)
    plt.plot(x, y, label= str(order))
    plt.legend(loc='best')
plt.show()