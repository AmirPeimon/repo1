# In The Name of GOD
# Full P-Delta Analysis
# Units KN mm

from Stiffness_Matrix import * 
from S2 import *
from sympy import *
import numpy as np

# nStory: number of stories 
nStory = 2

# hStory: height of stories
hStory    = np.ones( (nStory,1) ) * 3000
hStory[0] = 3000
 
## Bays
#  wBay:  width of bays
#  nBay: number of bays 
wBay = [ 3000, 2000 ] 
nBay = len(wBay)
  
## Bracing (X) 
Braced_Bays    = [ 1 ]    
Braced_Stories = [  ]  
 
# Geometry (auto)
# Points: Dataframe of Points 
# CnC:    Dataframe of Connectivity of Columns
# CnB:    Dataframe of Connectivity of Beams
# CnD:    Dataframe of Connectivity of Diagonals(Braces) 
[ Points, CnC, CnB, CnD ] = PC( wBay, hStory, Braced_Bays, Braced_Stories ) 

# CheckShape (auto) 
Shape( Points, CnC, CnB, CnD, wBay, hStory )

# Static Analysis Coefficients
# NOP: Number Of Points
# NOD: Number of "Degrees of freedom"
# RD:  Restrained "Degrees of freedom"
# FD:  Free "Degrees of freedom"
# NFD: Number of Free "Degrees of Freedom"
# NC:  Number of Columns
# NB:  Number of Beams
# ND:  Number of Diagonals(braces)
# IndxC: ['near_x','near_y','near_z','far_x','far_y','far_z'] of all columns
# IndxB: ['near_x','near_y','near_z','far_x','far_y','far_z'] of all beams
# IndxD: ['near_x','near_y','near_z','far_x','far_y','far_z'] of all diagonals (braces)
[NOP, NOD, RD, FD, NFD, NC, NB, ND, IndxC, IndxB, IndxD] = Static_Coeff( Points, CnC, CnB, CnD, wBay )

# Elements   
#  Material Properties:   
#     E: Elasticity 
#  Shape Properties 
#     A: Area (mm2)
#     I: Inertia (mm6)  
#  Distributed Load (just for Beams)
#     W: Distributed Load of Beams (KN/mm)
E_Cols, E_Beams, E_Diags = 200,200,200
A_Cols, A_Beams, A_Diags = 1140,1140,300
I_Cols, I_Beams          = 11e6,11e6
W_Beams = 0.01  #KN/mm

[ Beams, Columns, Diagonals ] = Elements(
      NB, IndxB, E_Beams, A_Beams, I_Beams, W_Beams
    , NC, IndxC, E_Cols,  A_Cols,  I_Cols
    , ND, IndxD, E_Diags, A_Diags )
  
# Nodal_Forces, NF 
# Lbot = "Story 1" load from "Left"
# Ltop = "Roof"    load from "Left"
# Rbot = "Story 1" load from "Right"
# Rtop = "Roof"    load from "Right"
Lbot, Ltop, Rbot, Rtop = 0.0, 5.0, 0.0 ,0.00 

NF = Nodal_Forces( Lbot, Ltop, Rbot, Rtop, NOD, nStory, nBay )
 
# Ke Qf 
[ KE,QF, LC,TC,keC,KeC, LB,TB,keB,KeB,qfB, LD,TD,keD,KeD ] = KeQf( 
        Points,  NOD
      , CnB, NB, IndxB, Beams 
      , CnC, NC, IndxC, Columns
      , CnD, ND, IndxD, Diagonals )

# P-Delta Analysis
[ qC, qB, qD, R, U ] = P_Delta( 
      NF,  NOD,  FD, KE, QF   
    , NC, IndxC, TC, keC, KeC, Columns
    , NB, IndxB, TB, keB, KeB, Beams, qfB
    , ND, IndxD, TD, keD, KeD, Diagonals  )
 

# D2S
# Converting Directions (D.S.M. to Standard.Static) 
# qX  --->  qXs
[ qCs,qBs,qDs ] = D2S( NC,NB,ND, qC,qB,qD )
 
# xyzNVM
[ Uxyz, Rxyz, AxC,VC,MC, AxB,VB,MB, AxD ] = xyzNVM( 
    NC, NB, ND, U, R, qCs, qBs, qDs )


# Plot Deformed 
Scale = 100
Undeformed_YN = 'Y';
Plot_Deformed( Scale, Undeformed_YN, Points, Uxyz, CnC,CnB,CnD, NC,NB,ND, wBay,hStory )
 
