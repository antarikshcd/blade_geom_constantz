#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:02:05 2018

@author: antariksh
"""

from PGL.main.domain import Domain, Block
from pickling import load_obj

surface = load_obj('./solved_surfaces/KB6_zc_s30_c100_wlast')
filename = 'KB6_surf_parplanes.xyz' 
domain = Domain()
domain.add_blocks(Blocks(x , y, z))
domain.write_plot3d(filename)