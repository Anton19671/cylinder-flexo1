#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 finite-difference scaling helpers.

Implements the minimum V3 repair from the audit:

    typ_j = max(abs(x_j), characteristic_scale_j)
    h_j   = relative_step * typ_j
    J_j   = (R(x+h_j e_j)-R(x-h_j e_j))/(2 h_j)

For CUF Taylor fields q_tau, the characteristic coefficient scale is
h/L**tau.  Thus TE1 uses h for u0/v0/w0 and h/L for u1/v1/w1, exactly the
scale proposed in the audit.  For KL all three nodal displacement fields use h.

``fd_step`` remains the public input name for backward compatibility, but after
V3 it is a *dimensionless relative step*, not an absolute increment.
"""
from __future__ import annotations
import numpy as np


def _positive_finite(value, name):
    v=float(value)
    if not np.isfinite(v) or v<=0.0:
        raise ValueError(f"{name} must be finite and >0")
    return v


def thickness_from_quadrature(quadrature):
    """Recover total thickness from positive through-thickness weights."""
    w=np.asarray(quadrature.w,float)
    if w.ndim!=1 or w.size<1 or not np.all(np.isfinite(w)) or np.any(w<=0.0):
        raise ValueError("invalid thickness quadrature weights")
    return _positive_finite(np.sum(w),"quadrature thickness")


def cuf_characteristic_scale(state, ops, quadrature):
    """Per-DOF V3 scale for final Taylor CUF TE1..TE4 states.

    state shape: (3,m,ns,nx).  q_tau has scale h/L**tau, so TE1 is
    [h, h/L] for each displacement component.
    """
    q=np.asarray(state,float)
    if q.ndim!=4 or q.shape[0]!=3:
        raise ValueError("CUF state must have shape (3,m,ns,nx)")
    L=_positive_finite(ops.grid.length_x,"grid.length_x")
    h=thickness_from_quadrature(quadrature)
    out=np.empty_like(q)
    for tau in range(q.shape[1]):
        s=h/(L**tau)
        if not np.isfinite(s) or s<=0.0:
            raise ValueError("invalid CUF characteristic scale")
        out[:,tau,:,:]=s
    return out


def kl_characteristic_scale(state, h):
    """Per-DOF V3 scale for KL nodal u,v,w displacement fields."""
    U=np.asarray(state,float)
    if U.ndim!=3 or U.shape[0]!=3:
        raise ValueError("KL state must have shape (3,ns,nx)")
    hh=_positive_finite(h,"h")
    return np.full_like(U,hh,dtype=float)


def relative_steps(x, characteristic_scale, relative_step):
    """Return h_j = relative_step*max(|x_j|, characteristic_scale_j)."""
    x=np.asarray(x,float)
    c=np.asarray(characteristic_scale,float)
    if x.shape!=c.shape:
        raise ValueError("x/characteristic_scale shape mismatch")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(c)) or np.any(c<=0.0):
        raise ValueError("non-finite state or invalid characteristic scale")
    eta=_positive_finite(relative_step,"fd_step")
    hj=eta*np.maximum(np.abs(x),c)
    if not np.all(np.isfinite(hj)) or np.any(hj<=0.0):
        raise ValueError("invalid V3 finite-difference increments")
    return hj
