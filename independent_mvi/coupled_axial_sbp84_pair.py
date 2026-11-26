"""Opt-in research-only independent MVI paired axial differences.

Diagonal positive SBP(8,4) norm. C is the fourth-polynomial-exact
centered SBP matrix with C+C.T=B, and S=Z.T Z/512, where Z is the
fifth difference. The free closure coefficients are determined by
minimizing the exact rational squared norm of the 28 upper-triangle
left closure entries subject to the polynomial moment equations.
No import of any FDM/MKR numerical operator.

H P = C-S, H M = C+S; P.T H + H M = B.
Trial jets are I, P, M P, P M P. The dual steps are M,P,M.
S annihilates all quartics; it is spatial, NOT physical damping.
This candidate is not certified for general sixth-order equations.
"""
from itertools import combinations
import numpy as np

SCHEME="SBP84_PAIRED_V1"
LEGACY="SBP42"
_WEIGHTS=(1498139/5080320,1107307/725760,20761/80640,
    1304999/725760,299527/725760,103097/80640,
    670091/725760,5127739/5080320)
# Exact minimum-norm rational solution of the 40 moment equations
# in 28 skew entries, with three free entries before minimization.
_NUM=(206453678869,-8297331272,-62219441237,-4701462484,
    32824323755,-6900598660,-2250051531,57992582837,
    199996839952,16047367973,-92350104140,19087511835,
    5679480412,67665740933,-26032736156,8812367109,
    -78907276,-671213045,84863974923,165600371972,
    -40107426107,-4913781140,72281584165,-867954640,
    -129991573,212469480181,-35997043048,234869646581)
_DEN=309818234880
_CENTER=((-4,1/280),(-3,-4/105),(-2,1/5),(-1,-4/5),
         (1,4/5),(2,-1/5),(3,4/105),(4,-1/280))

def coupled_mvi_sbp84_pair(nx,length_m):
    if type(nx) is not int or not 21<=nx<=257:
        raise ValueError("MVI_PAIRED_SBP84_REQUIRES_21_TO_257_NODES")
    if isinstance(length_m,(bool,np.bool_)):
        raise ValueError("MVI_PAIRED_SBP84_INVALID_LENGTH")
    L=float(length_m)
    if not np.isfinite(L) or L<=0:
        raise ValueError("MVI_PAIRED_SBP84_INVALID_LENGTH")
    h=L/(nx-1)
    w=np.ones(nx);w[:8]=_WEIGHTS;w[-8:]=w[:8][::-1]
    C=np.zeros((nx,nx))
    C[0,0]=-.5;C[-1,-1]=.5
    for i in range(8,nx-8):
        for j,c in _CENTER:
            C[i,i+j]=c;C[i+j,i]=-c
    for (i,j),num in zip(combinations(range(8),2),_NUM):
        c=num/_DEN
        C[i,j]=c;C[j,i]=-c
        C[-1-i,-1-j]=-c;C[-1-j,-1-i]=c
    Z=np.zeros((nx-5,nx))
    for i in range(nx-5):
        Z[i,i:i+6]=(1.,-5.,10.,-10.,5.,-1.)
    S=(Z.T@Z)/512.
    H=w*h
    P=(C-S)/H[:,None];M=(C+S)/H[:,None]
    for a in (P,M,H):
        if not np.isfinite(a).all():
            raise ValueError("MVI_PAIRED_SBP84_NONFINITE_OPERATOR")
        a.setflags(write=False)
    return P,M,H
