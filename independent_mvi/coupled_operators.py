"""Own axial SBP42 and odd-periodic Fourier operators for coupled MVI.

The operators and quadrature are numerically identical to the existing
independent linear MVI kernels, but live in an isolated coupled namespace.
No MKR stencil, old MVI Newton, rank seed, SVD, or external FD backend is
imported. This is only one-dimensional spatial differentiation, NOT a
certificate for all higher-order natural boundary closures.
"""
from math import pi
import numpy as np


def coupled_mvi_sbp42(nx,length_m):
    """Diagonal-norm SBP42 first derivative on the finite axial interval."""
    if type(nx) is not int or nx<9:
        raise ValueError("SBP42 requires nx>=9")
    if isinstance(length_m,(bool,np.bool_)):
        raise ValueError("positive finite axial length required")
    L=float(length_m)
    if not np.isfinite(L) or L<=0.:
        raise ValueError("positive finite axial length required")
    h=L/(nx-1)
    d=np.zeros((nx,nx))
    closure=(
        ((0,-24/17),(1,59/34),(2,-4/17),(3,-3/34)),
        ((0,-.5),(2,.5)),
        ((0,4/43),(1,-59/86),(3,59/86),(4,-4/43)),
        ((0,3/98),(2,-59/98),(4,32/49),(5,-4/49)),
    )
    for i in range(nx):
        if i<4:
            entries=closure[i]
        elif i>=nx-4:
            entries=tuple((nx-1-j,-v) for j,v in closure[nx-1-i])
        else:
            entries=((i-2,1/12),(i-1,-2/3),
                     (i+1,2/3),(i+2,-1/12))
        for j,value in entries:
            d[i,j]=value/h
    w=np.ones(nx)
    w[:4]=[17/48,59/48,43/48,49/48]
    w[-4:]=w[:4][::-1]
    return d,w*h


def coupled_mvi_odd_fourier(ns,radius_m):
    """Skew Fourier differentiation along the periodic physical arc s."""
    if type(ns) is not int or ns<5 or ns%2==0:
        raise ValueError("MVI odd Fourier circumference requires odd ns>=5")
    if isinstance(radius_m,(bool,np.bool_)):
        raise ValueError("positive finite radius required")
    R=float(radius_m)
    if not np.isfinite(R) or R<=0.:
        raise ValueError("positive finite radius required")
    circumference=2*pi*R
    freq=2*pi*np.fft.fftfreq(ns,d=circumference/ns)
    d=np.fft.ifft(1j*freq[:,None]*
                  np.fft.fft(np.eye(ns),axis=0),axis=0).real
    return .5*(d-d.T),np.full(ns,circumference/ns)


__all__=("coupled_mvi_sbp42","coupled_mvi_odd_fourier")
