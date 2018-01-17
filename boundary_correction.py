#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary correction by extending the parametric grid in S and T. This is
necessary to ensure C1-continuity at the boundaries. The present solution clips
S at the boundary leading to discontinuities at the boundaries. At the T boundaries
the solution loops for T values over-shooting the bounds of T=0 and T= N_c-1
by considering the cross-section as a closed surface. 

Extension to spanwise direction 'S': 
    The grid is extended to fit the profile of the cross-sections at the relevant
    boundary. So for all S < 0, the crosssections will be the same as that at
    the root. Whereas, for S > (N_s-1) it will be the same as that of 
    S = N_s.
    
Extension to chordwise direction 'T':
    The grid is extended such that a circular loop exists. So, for t<0 the     
    
Created on Tue Jan 16 12:52:36 2018

@author: antariksh
"""

