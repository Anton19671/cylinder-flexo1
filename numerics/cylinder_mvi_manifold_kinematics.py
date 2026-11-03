"""Exact factor-space kinematics and first-variation kernel for rank-fixed MVI.

For every physical displacement or electrical component the 2D field is
Q = sum_r X_r(x,t) * Y_r(s,t).  If the factors depend on time, then

    Qdot  = Xdot.T @ Y + X.T @ Ydot,
    Qddot = Xddot.T @ Y + 2*Xdot.T @ Ydot + X.T @ Yddot.

The mixed acceleration is mandatory: a solver that integrates X and Y
separately while omitting it is not the physical shell acceleration.

This module is NOT yet a Gauss IRK4 integrator. It supplies the exact
physical factor-coordinate kinematics and the first variation that a
variational, endpoint-closed low-rank integrator must preserve. No SVD,
Gram matrix, post-step truncation, FDM fallback or rank growth is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RankFactorMotion:
    field: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    rank: int


def _checked_pair(x, y):
    x=np.asarray(x,float)
    y=np.asarray(y,float)
    if (x.ndim!=3 or y.ndim!=3 or x.shape[0]!=y.shape[0] or
        x.shape[1]!=y.shape[1] or min(*x.shape,*y.shape)<=0 or
        x.shape[1]>min(x.shape[2],y.shape[2]) or
        not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("MVI_FACTOR_MOTION_INVALID_RANK_OR_FIELD")
    return x,y


def _same_shape(name, value, expected):
    v=np.asarray(value,float)
    if v.shape!=expected.shape or not np.isfinite(v).all():
        raise ValueError("MVI_FACTOR_MOTION_INVALID_"+name)
    return v


def rank_factor_motion(x, y, xdot, ydot, xddot, yddot):
    """Exactly reconstruct q, qdot and qddot for moving rank-r factors.

    Shapes: X and derivatives (ncomp,r,nx), Y (ncomp,r,ns).
    Each field remains a 3D (ncomp,nx,ns) physical coefficient array.
    """
    x,y=_checked_pair(x,y)
    xd=_same_shape("XDOT",xdot,x)
    yd=_same_shape("YDOT",ydot,y)
    xdd=_same_shape("XDDOT",xddot,x)
    ydd=_same_shape("YDDOT",yddot,y)
    field=np.einsum("crx,crs->cxs",x,y)
    velocity=np.einsum("crx,crs->cxs",xd,y)+np.einsum(
        "crx,crs->cxs",x,yd)
    acceleration=(np.einsum("crx,crs->cxs",xdd,y)
                  +2.*np.einsum("crx,crs->cxs",xd,yd)
                  +np.einsum("crx,crs->cxs",x,ydd))
    if not all(np.isfinite(item).all() for item in (field,velocity,acceleration)):
        raise ValueError("MVI_FACTOR_MOTION_NONFINITE_FIELD")
    return RankFactorMotion(field,velocity,acceleration,x.shape[1])


def rank_factor_tangent(x,y,delta_x,delta_y):
    """Variation dQ = dX.T Y + X.T dY, with no full-nodal correction."""
    x,y=_checked_pair(x,y)
    dx=_same_shape("DELTA_X",delta_x,x)
    dy=_same_shape("DELTA_Y",delta_y,y)
    return np.einsum("crx,crs->cxs",dx,y)+np.einsum(
        "crx,crs->cxs",x,dy)


def rank_factor_first_variation(residual,x,y):
    """Ritz/MVI factor variations for a physical field first variation R.

    For all dX,dY:
    <R,dQ> = <dPi/dX,dX> + <dPi/dY,dY>.
    No projection of R onto the full discrete nodal FDM solution occurs.
    """
    x,y=_checked_pair(x,y)
    r=np.asarray(residual,float)
    if r.shape!=(x.shape[0],x.shape[2],y.shape[2]) or not np.isfinite(r).all():
        raise ValueError("MVI_FACTOR_FIRST_VARIATION_INVALID_RESIDUAL")
    grad_x=np.einsum("cxs,crs->crx",r,y)
    grad_y=np.einsum("cxs,crx->crs",r,x)
    return grad_x,grad_y


def rank_factor_stationary_trace(matrix, x,y,xdot,ydot,xddot,yddot,
                                 atol=1.e-10):
    """Certify true stationary homogeneous boundary traces for q,v,a.

    This only checks; it never zeros or projects mechanical field values.
    """
    motion=rank_factor_motion(x,y,xdot,ydot,xddot,yddot)
    size=motion.field.size
    c=np.asarray(matrix,float)
    if c.ndim!=2 or c.shape[1]!=size or not np.isfinite(c).all():
        raise ValueError("MVI_FACTOR_TRACE_INVALID_MATRIX")
    if not np.isfinite(atol) or atol<=0.:
        raise ValueError("MVI_FACTOR_TRACE_INVALID_TOLERANCE")
    defects=tuple(float(np.max(np.abs(c@field.ravel()),initial=0.))
                  for field in (motion.field,motion.velocity,motion.acceleration))
    if any(value>atol for value in defects):
        raise ValueError("MVI_FACTOR_MOTION_STATIONARY_SUPPORT_VIOLATED")
    return defects


__all__=("RankFactorMotion","rank_factor_motion","rank_factor_tangent",
         "rank_factor_first_variation","rank_factor_stationary_trace")
