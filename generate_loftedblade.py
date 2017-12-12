# -*- coding: utf-8 -*-
""" generate_loftedblade_kb6
   
This script generates the KB6 (renamed officially as KB5) lofted blade geometry with 
the swept cross-sections rotated according to Euler rotation theorem and is not sheared.
The blade geometry is inititally generated with sampling of the running length normal
cross-sections in the main -body coordinate attached at the centre of the circular root.
All local coordinates are attached to the blade axis at a distance of p_le*c from the leading
edge. 
The generated blade geometry does not have a constant z-coordiante for a cross-section and is 
thus sampled to reflect this.

Example:
      The script is run using the command::
                 $ python KB6_loftedblade_noshear_constz.py
                 
"""

import numpy as np
import copy
#from PGL.components.blademesher import BladeMesher
from PGL.components.loftedblade import LoftedBladeSurface
from PGL.main.planform import read_blade_planform, \
                              redistribute_planform
from PGL.main.bezier import BezierCurve
from PGL.main.curve import Curve
from PGL.main.curve import Circle
from fusedwind.turbine.recorders import get_recorded_planform
#from pickling import save_obj
from sqlitedict import SqliteDict


# path to the planform
def kb6_loftedblade(optimization_file, iteration,  N_s= 100, N_c= 200):
# read reference values

  db = SqliteDict(optimization_file, 'iterations', flag='r')

# ITERATION TO USE
  it = iteration

  us = db[db.keys()[it]]['Unknowns']

  pf0 = get_recorded_planform(db, db.keys()[it])
# get_recorded_planform recomputes s based on x,y,z which we seem not to do
# in the fusedwind geometry classes, so to get consistency with what HAWTOpt2
# has been optimizing on we overwrite s with the recorded one 
  pf0['s'] = us['s']
  pf0['dy'] = np.zeros(pf0['x'].shape[0])
#spanwise sections
  nsec_ae = N_s
   
  dist = np.array([[0., 1./nsec_ae, 1], [0.14, 1./nsec_ae, 15.], [1., 1./nsec_ae/4., nsec_ae]])
  pf = redistribute_planform(pf0.copy(), dist=dist)
  s_st = pf['s']

# Smoothen the root chord 
  factor = 0.7
  h = Curve(points=np.array([pf['s'], pf['chord']]).T)
# root
  p0 = np.array([pf['s'][0], 0.43/us['blade_length']])
# max chord
  imax = pf['chord'].argmax() 
  p3 = np.array([pf['s'][imax+3], pf['chord'][imax+3]])
  p1 = factor * p0 + (1-factor) * p3
  p1[1] = p0[1]   # same y-coordinate for first/secord CP
  factor = 0.3
  p2 = p3 + 8 * h.ds[imax+3] * h.dp[imax+3] * -1.
# p2 = factor * p3 + (1-factor) * p1
# p2[1] = p3[1]   # same y-coordinate for secord/last CP

  b = BezierCurve()
  b.add_control_point(p0)
  b.add_control_point(p1)
  b.add_control_point(p2)
  b.add_control_point(p3)
  b.ni = imax + 4
  b.update()

  bc = copy.deepcopy(b)
  b.points[:,1] = np.interp(pf['s'][:imax+4], b.points[:,0], b.points[:,1])
# replace chord in pf
  pf['chord'][:imax+4] = b.points[:,1]

# Smoothen the root twist
# create helper curve
  h = Curve(points=np.array([pf['s'], pf['rot_z']]).T)
  factor = 0.9
# root twist hard-coded to -15. deg
  p0 = np.array([pf['s'][0], -15.])
# max chord
  imax = pf['chord'].argmax()
  p3 = np.array([pf['s'][imax+3], pf['rot_z'][imax+3]])
  p1 = factor * p0 + (1-factor) * p3
  p1[1] = p0[1]   # same y-coordinate for first/secord CP
  factor = 0.3
  p2 = p3 + 3 * h.ds[imax+3] * h.dp[imax+3] * -1.
# p2[1] = p3[1]   # same y-coordinate for secord/last CP

  b = BezierCurve()
  b.add_control_point(p0)
  b.add_control_point(p1)
  b.add_control_point(p2)
  b.add_control_point(p3)
  b.ni = imax + 4
  b.update()

  bb = copy.deepcopy(b)
  b.points[:,1] = np.interp(pf['s'][:imax+4], b.points[:,0], b.points[:,1])
# replace rot_z in pf
  pf['rot_z'][:imax+4] = b.points[:,1]

# adjust chordwise offset
  factor = 0.8
# root
  p0 = np.array([pf['s'][0], pf['p_le'][0]])
# max chord
  p3 = np.array([pf['s'][imax], pf['p_le'][imax]])
  p1 = factor * p0 + (1-factor) * p3
  p1[1] = p0[1]   # same y-coordinate for first/secord CP
  factor = 0.5
  p2 = factor * p3 + (1-factor) * p0
  p2[1] = p3[1]   # same y-coordinate for secord/last CP

  b = BezierCurve()
  b.add_control_point(p0)
  b.add_control_point(p1)
  b.add_control_point(p2)
  b.add_control_point(p3)
  b.ni = imax + 1
  b.update()
  b.points[:,1] = np.interp(pf['s'][:imax+1], b.points[:,0], b.points[:,1])

# replace p_le in pf
  pf['p_le'][:imax+1] = b.points[:,1].copy()

# modify athick
  factor = 0.65
  athick = pf['chord'] * pf['rthick']
# root
  p0 = np.array([pf['s'][0], pf['chord'][0]])
# max chord
  p3 = np.array([pf['s'][imax], athick[imax]]) 
  p1 = factor * p0 + (1-factor) * p3
  p1[1] = p0[1]   # same y-coordinate for first/secord CP
  factor = 0.5
  p2 = factor * p3 + (1-factor) * p1
  p2[1] = p3[1]   # same y-coordinate for secord/last CP

  b = BezierCurve()
  b.add_control_point(p0)
  b.add_control_point(p1)
  b.add_control_point(p2)
  b.add_control_point(p3)
  b.add_control_point(np.array([pf['s'][imax+1], athick[imax+1]]))
  b.add_control_point(np.array([pf['s'][imax+2], athick[imax+2]]))
  b.ni = imax + 3
  b.update()
  b.points[:,1] = np.interp(pf['s'][:imax+3], b.points[:,0], b.points[:,1])
  b.points[:,0] = pf['s'][:imax+3]

# replace pf
  pf['rthick'][:imax+3] = b.points[:,1] / pf['chord'][:imax+3]


# modify tip chord
  factor = 0.65

  b = BezierCurve()
  for i in range(-10,-2,1):
      b.add_control_point(np.array([pf['s'][i], pf['chord'][i]]))
  b.add_control_point(np.array([1., pf['chord'][-1]]))
  b.CPs[-1,1] = 0.001
  b.ni = 10
  b.update()
  b.points[:,1] = np.interp(pf['s'][-10:], b.points[:,0], b.points[:,1])
  b.points[:,0] = pf['s'][-10:]

# replace chord in pf
  pf['chord'][-10:] = b.points[:,1]

# create symmetric 50% airfoil to deal with root assymmetry
  t50 = np.loadtxt('../data/naca0030.dat')
  t50[:,1] *= 5. / 3.

# create lofted blade
  m = LoftedBladeSurface()
  m.pf = pf 
  m.blend_var = np.array([0.18, 0.21, 0.241, 0.301, 0.360, 0.5, 1.])
#m.minTE = 0.0002
  m.minTE=0.002 #in metres
  afs = []
  for f in ['../data/NACA_63-418.dat',
            '../data/ffaw3211.dat',
            '../data/ffaw3241.dat',
            '../data/ffaw3301.dat',
            '../data/ffaw3360.dat']:
#           'data/cylinder.dat']:

       afs.append(np.loadtxt(f))
  afs.insert(5, t50)

  c = Circle(0.5)
  c.points[:,0] += 0.5
  afs.append(c.points[::-1])
  m.base_airfoils = afs

# chordwise number of vertices
  m.ni_chord = N_c #200

# align cross-sections parallel with rotor axis (ie don't rotate them to be parallel to swept blade axis)
# however, since the blade chord is optimized with HAWC2, the chord should actually be scaled
# with cos(local_sweep_angle)
  m.shear_sweep = False#True

# redistribute points chordwise
  m.redistribute_flag = True
# number of points on blunt TE
  m.chord_nte = 0
# set min TE thickness (which will open TE using AirfoilShape.open_trailing_edge) - 2 mm
  m.minTE = 0.002

  m.update()

  m.domain.scale(us['blade_length'])

# store the loftedblade surface of type [n_cross_Sections, n_span_Sections, (x,y,z)]
  surface= m.surface

  #save_obj(surface, 'KB6_surf_noshearsweep')
# write the interpolation script and transfer it into a method

  return surface