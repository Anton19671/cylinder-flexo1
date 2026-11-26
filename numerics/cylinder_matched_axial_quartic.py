"""Quartic-exact independent paired axial differences for strong FDM."""
from functools import lru_cache
import numpy as np
from cylinder_matched_axial import MatchedAxialDifferences

# Rational SBP(8,4) closure: skew part of H D at the left 8 nodes.
# Free upper-triangle entries C[5,6], C[5,7], C[6,7] are respectively
# 896463/1000000, -191603/1000000, 388289/500000. The other entries
# follow from exact polynomial conditions through degree four.
_H=(1498139/5080320,1107307/725760,20761/80640,1304999/725760,
    299527/725760,103097/80640,670091/725760,5127739/5080320)
_C_NUM=(130474756611,-19473414111,-19251594235,-5675070420,5352187365,
        8457531859,-4628397069,75807160506,45560431980,41396735905,
        -14328997074,-34154535240,16193960534,140814129330,-112891057335,
        1238726034,42420404376,-15248456010,161107772175,4973146290,
        6840232280,-5798183670,143626425705,-77928364395,18920719015,
        170786959056,-36502670736,147947427936)
_C_DEN=190512000000
_D=((-4,1/280),(-3,-4/105),(-2,1/5),(-1,-4/5),
    (1,4/5),(2,-1/5),(3,4/105),(4,-1/280))


@lru_cache(maxsize=8)
def quartic_matched_axial_differences(nx):
    """Dimensionless explicit differences, independent of MVI and energy FDM.

    Quartic reproduction also gives a consistent fourth-derivative end
    action for sixth-order equations, unlike the cubic-exact candidate.
    This is a dense research operator (20..257 axial nodes).
    """
    if type(nx) is not int or not 20<=nx<=257:
        raise ValueError("QUARTIC_MATCHED_AXIAL_REQUIRES_20_TO_257_NODES")
    w=np.ones(nx);w[:8]=_H;w[-8:]=w[:8][::-1]
    B=np.zeros((nx,nx));B[0,0]=-1.;B[-1,-1]=1.
    C=B/2
    for i in range(8,nx-8):
        for j,c in _D:
            C[i,i+j]=c;C[i+j,i]=-c
    k=0
    for i in range(8):
        for j in range(i+1,8):
            c=_C_NUM[k]/_C_DEN;k+=1
            C[i,j]=c;C[j,i]=-c
            C[-1-i,-1-j]=-c;C[-1-j,-1-i]=c
    # Fifth difference annihilates quartics, retaining the exactness of
    # C while giving a nonzero high-frequency first-derivative symbol.
    E=np.zeros((nx-5,nx))
    for i in range(nx-5):
        E[i,i:i+6]=(1.,-5.,10.,-10.,5.,-1.)
    S=(E.T@E)/512.
    P=(C-S)/w[:,None];M=(C+S)/w[:,None]
    I=np.eye(nx);K2=M@P;K3=P@K2;F3=K2@M;PM=P@M
    for a in (w,B,I,P,M,K2,K3,F3,PM):
        if not np.isfinite(a).all():
            raise ValueError("QUARTIC_MATCHED_AXIAL_NONFINITE")
        a.setflags(write=False)
    return MatchedAxialDifferences(w,B,(I,P,K2,K3),(I,M,K2,F3),P,PM,M)
