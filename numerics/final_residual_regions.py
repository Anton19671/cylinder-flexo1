#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V11.2 residual-region diagnostics for strong-form FDM.

A smoothed PATCH front is a deliberately narrow transition layer. Its rows are
reported separately for diagnostics, but every assembled equation remains in
the Newton stopping norm. A large front residual can therefore never be hidden
by the regional partition.

For every mask configuration the convergence norm is exactly the full infinity
norm: max(interior,boundary,load_front) == max(all rows).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ResidualRegionMasks:
    interior: np.ndarray
    boundary: np.ndarray
    load_front: np.ndarray
    exclude_load_front_from_convergence: bool = False

    def __post_init__(self):
        i=np.asarray(self.interior,bool).reshape(-1)
        b=np.asarray(self.boundary,bool).reshape(-1)
        f=np.asarray(self.load_front,bool).reshape(-1)
        if not (i.shape==b.shape==f.shape):
            raise ValueError("residual region mask size mismatch")
        if np.any(i & b) or np.any(i & f) or np.any(b & f):
            raise ValueError("residual region masks must be disjoint")
        if not np.all(i | b | f):
            raise ValueError("residual region masks must cover every residual row")
        object.__setattr__(self,"interior",i)
        object.__setattr__(self,"boundary",b)
        object.__setattr__(self,"load_front",f)

    @property
    def size(self):
        return int(self.interior.size)


@dataclass(frozen=True)
class ResidualRegionMetrics:
    residual_inf: float
    residual_inf_interior: float
    residual_inf_boundary: float
    residual_inf_load_front: float
    residual_convergence_inf: float
    residual_relative: float
    residual_relative_interior: float
    residual_relative_boundary: float
    residual_relative_load_front: float
    residual_convergence_relative: float


def _front_surface(mask,ns,nx):
    if mask is None:
        return np.zeros((ns,nx),dtype=bool)
    a=np.asarray(mask,bool)
    if a.shape!=(ns,nx):
        raise ValueError(f"load_front_surface_mask must have shape ({ns},{nx})")
    return a.copy()


def cuf_masks(ops,basis,load_front_surface_mask=None):
    ns,nx=int(ops.grid.ns),int(ops.grid.nx)
    m=int(basis.m)
    if nx<3:
        raise ValueError("CUF V11.2 requires nx>=3")
    shape=(3,m,ns,nx)
    boundary=np.zeros(shape,dtype=bool)
    boundary[...,0]=True
    boundary[...,-1]=True
    sf=_front_surface(load_front_surface_mask,ns,nx)
    front=np.broadcast_to(sf,(3,m,ns,nx)).copy()
    # Boundary equations always remain mandatory and take precedence over a
    # physical load transition touching an end.
    front &= ~boundary
    interior=~(boundary|front)
    return ResidualRegionMasks(interior.ravel(),boundary.ravel(),front.ravel())


def kl_masks(ops,load_front_surface_mask=None):
    ns,nx=int(ops.grid.ns),int(ops.grid.nx)
    if nx<5:
        raise ValueError("KL V11.2 requires nx>=5")
    shape=(3,ns,nx)
    boundary=np.zeros(shape,dtype=bool)
    # u,v: one end BC row at each x-end.
    boundary[0,:,0]=True; boundary[0,:,-1]=True
    boundary[1,:,0]=True; boundary[1,:,-1]=True
    # w: value + slope/moment end rows occupy two rows at each end in the
    # assembled strong residual ordering.
    boundary[2,:,0:2]=True; boundary[2,:,-2:]=True
    sf=_front_surface(load_front_surface_mask,ns,nx)
    front=np.broadcast_to(sf,(3,ns,nx)).copy()
    front &= ~boundary
    interior=~(boundary|front)
    return ResidualRegionMasks(interior.ravel(),boundary.ravel(),front.ravel())


def _masked_inf(r,mask):
    a=np.asarray(r,float).reshape(-1)
    m=np.asarray(mask,bool).reshape(-1)
    if a.shape!=m.shape:
        raise ValueError("residual/mask size mismatch")
    if not np.any(m):
        return 0.0
    return float(np.max(np.abs(a[m])))


def resolve_front_exclusion(masks,generalized_load):
    """Compatibility hook: load-front rows are always convergence rows.

    Older logic could exclude a smoothed load front from the Newton norm and
    report false convergence. The load is still shape-checked, but exclusion
    is now unconditionally disabled.
    """
    p=np.asarray(generalized_load,float).reshape(-1)
    if p.shape!=(masks.size,):
        raise ValueError("generalized_load size does not match residual region masks")
    return ResidualRegionMasks(
        masks.interior,masks.boundary,masks.load_front,
        exclude_load_front_from_convergence=False
    )


def metrics(residual,masks,scale):
    r=np.asarray(residual,float).reshape(-1)
    if r.shape!=(masks.size,):
        raise ValueError("residual size does not match region masks")
    if not np.all(np.isfinite(r)):
        raise ValueError("residual contains non-finite values")
    s=float(scale)
    if not np.isfinite(s) or s<=0.0:
        raise ValueError("residual scale must be finite and >0")
    rall=float(np.max(np.abs(r))) if r.size else 0.0
    ri=_masked_inf(r,masks.interior)
    rb=_masked_inf(r,masks.boundary)
    rf=_masked_inf(r,masks.load_front)
    # Every equation is mandatory, including the smoothed load front.
    rc=max(ri,rb,rf)
    return ResidualRegionMetrics(
        residual_inf=rall,
        residual_inf_interior=ri,
        residual_inf_boundary=rb,
        residual_inf_load_front=rf,
        residual_convergence_inf=rc,
        residual_relative=rall/s,
        residual_relative_interior=ri/s,
        residual_relative_boundary=rb/s,
        residual_relative_load_front=rf/s,
        residual_convergence_relative=rc/s,
    )


def converged(region_metrics,residual_limit):
    lim=float(residual_limit)
    if not np.isfinite(lim) or lim<0.0:
        raise ValueError("residual_limit must be finite and nonnegative")
    return bool(region_metrics.residual_convergence_inf <= lim)


def as_dict(region_metrics):
    return {
        "residual_inf":region_metrics.residual_inf,
        "residual_inf_interior":region_metrics.residual_inf_interior,
        "residual_inf_boundary":region_metrics.residual_inf_boundary,
        "residual_inf_load_front":region_metrics.residual_inf_load_front,
        "residual_convergence_inf":region_metrics.residual_convergence_inf,
        "residual_relative":region_metrics.residual_relative,
        "residual_relative_interior":region_metrics.residual_relative_interior,
        "residual_relative_boundary":region_metrics.residual_relative_boundary,
        "residual_relative_load_front":region_metrics.residual_relative_load_front,
        "residual_convergence_relative":region_metrics.residual_convergence_relative,
    }
