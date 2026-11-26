"""Independent explicit matched axial FDM candidate (no shell solve).

Uniform nodal grid, diagonal SBP(6,3) norm and dual differences with a
fourth-difference correction. All flux derivatives are explicit products
of the two differences; no variational residual or MVI assembly is used.
This is an algebraic candidate, not a certified strong shell closure.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np


# Dimensionless left SBP norm and upper triangle of H D in first six rows.
# Obtain the latter by skew symmetry, the centered sixth-order interior
# stencil and exact differentiation of 1,x,x^2,x^3 at the first six nodes.
# The one free coefficient is chosen as 139/200 (near minimum l2 norm).
_H_LEFT=(13649/43200,12013/8640,2711/4320,5359/4320,7877/8640,43801/43200)
_CLOSURE=(5153/8100,-5087/259200,-2591/14400,15059/259200,137/25920,
          2909/8640,25157/51840,-1643/8640,377/86400,
          661/2592,1919/17280,-2117/43200,
          15103/25920,-10103/259200,139/200)
_INTERIOR=((-3,-1/60),(-2,3/20),(-1,-3/4),
           (1,3/4),(2,-3/20),(3,1/60))


@dataclass(frozen=True)
class MatchedAxialDifferences:
    """Dimensionless matrices on integer x/h; scale externally by h**k."""
    weights: np.ndarray
    boundary: np.ndarray
    kinematic: tuple
    coefficient: tuple
    # The q action for A2 uses P, and that for A3 uses P M;
    # the slope action for A3 uses M. This is NOT the old common F_k rule.
    second_displacement_flux: np.ndarray
    third_displacement_flux: np.ndarray
    third_slope_flux: np.ndarray


@lru_cache(maxsize=8)
def matched_axial_differences(nx):
    """Construct explicit dimensionless paired finite differences.

    Requires enough nodes for nonoverlapping boundary closures. This
    standalone dense algebraic prototype is capped at 257 nodes.
    """
    if type(nx) is not int or not 15<=nx<=257:
        raise ValueError("MATCHED_AXIAL_REQUIRES_15_TO_257_NODES")
    w=np.ones(nx)
    w[:6]=_H_LEFT
    w[-6:]=w[:6][::-1]
    B=np.zeros((nx,nx));B[0,0]=-1.;B[-1,-1]=1.
    C=B/2
    for i in range(6,nx-6):
        for shift,c in _INTERIOR:
            j=i+shift
            C[i,j]=c;C[j,i]=-c
    k=0
    for i in range(6):
        for j in range(i+1,6):
            c=_CLOSURE[k];k+=1
            C[i,j]=c;C[j,i]=-c
            C[-1-i,-1-j]=-c;C[-1-j,-1-i]=c
    # Undivided fourth difference E. E kills every cubic. The correction
    # retains the base consistency and detects the Nyquist mode.
    E=np.zeros((nx-4,nx))
    for i in range(nx-4):
        E[i,i:i+5]=(1.,-4.,6.,-4.,1.)
    S=(E.T@E)/128.
    P=(C-S)/w[:,None]
    M=(C+S)/w[:,None]
    I=np.eye(nx)
    K2=M@P
    K3=P@K2
    F3=K2@M
    A3=P@M
    for a in (w,B,I,P,M,K2,K3,F3,A3):
        if not np.isfinite(a).all():
            raise ValueError("MATCHED_AXIAL_NONFINITE")
        a.setflags(write=False)
    return MatchedAxialDifferences(w,B,(I,P,K2,K3),(I,M,K2,F3),P,A3,M)


def matched_axial_green_defects(nx):
    """Relative matrix residuals of the exact first/second/third identities."""
    op=matched_axial_differences(nx)
    H=np.diag(op.weights);B=op.boundary
    I,P,K2,K3=op.kinematic
    _,M,F2,F3=op.coefficient
    weak=(P.T@H,K2.T@H,K3.T@H)
    strong=(-H@M+B,
            H@F2-B@op.second_displacement_flux+P.T@B,
            -H@F3+B@op.third_displacement_flux
            -P.T@B@op.third_slope_flux+K2.T@B)
    return tuple(float(np.max(np.abs(a-b)))/max(1.,float(np.max(np.abs(a))),
                     float(np.max(np.abs(b)))) for a,b in zip(weak,strong))
