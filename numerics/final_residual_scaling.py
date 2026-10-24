#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 residual scaling for nonlinear strong-form FDM Newton solves.

The absolute residual has physical units and therefore changes when the same
problem is written in another consistent unit system.  V2 uses one fixed scale
per Newton solve, computed before the first Newton update:

    scale = max(||generalized_load||_inf, ||R_initial||_inf, 1)

The user tolerance is dimensionless and convergence is tested as

    ||R||_inf <= residual_tolerance_relative * scale.

The absolute residual is still retained for diagnostics and output.
"""
from __future__ import annotations
import numpy as np


def _finite_inf_norm(a, name: str) -> float:
    x = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return 0.0 if x.size == 0 else float(np.max(np.abs(x)))


def residual_scale(generalized_load, initial_residual) -> float:
    """Return the fixed V2 Newton residual scale prescribed by the audit."""
    load_inf = _finite_inf_norm(generalized_load, "generalized_load")
    initial_inf = _finite_inf_norm(initial_residual, "initial_residual")
    return max(load_inf, initial_inf, 1.0)


def relative_residual(residual_inf: float, scale: float) -> float:
    r = float(residual_inf)
    s = float(scale)
    if not np.isfinite(r) or r < 0.0:
        raise ValueError("residual_inf must be finite and nonnegative")
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("residual scale must be finite and positive")
    return r / s


def threshold(residual_tolerance_relative: float, scale: float) -> float:
    tol = float(residual_tolerance_relative)
    s = float(scale)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("residual_tolerance_relative must be finite and >0")
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("residual scale must be finite and positive")
    return tol * s


# ---------------------------------------------------------------------------
# V-4 reviewer fix: dimensionally homogeneous CUF residual MEASURES.
# These helpers MUST NOT be used to alter the assembled equations/Jacobian.
# They produce a left-scaled copy only for norms, stopping criteria and
# projected-residual measures.
# ---------------------------------------------------------------------------

def cuf_row_scale(basis_m, L):
    """Return the reviewer-prescribed interior CUF row multipliers L**(-tau)."""
    m=int(basis_m)
    LL=float(L)
    if m<1:
        raise ValueError("basis_m must be >=1")
    if not np.isfinite(LL) or LL<=0.0:
        raise ValueError("L must be finite and >0")
    out=np.array([LL**(-tau) for tau in range(m)],dtype=float)
    if not np.all(np.isfinite(out)) or np.any(out<=0.0):
        raise ValueError("invalid CUF row scale")
    return out


def cuf_residual_measure_multipliers(*, basis_m, ns, nx, L, bc, E0, h):
    """Per-row V-4 multipliers for a CUF assembled strong residual.

    Interior strong-form rows use L**(-tau), exactly as prescribed in V-4.
    End-BC rows are overridden separately, also exactly as prescribed:
      - GEOMETRIC: multiply by E0*h/L;
      - NATURAL:   divide by L;
      - ELASTIC:   divide by L (the assembled residual is traction-like).

    The returned array has shape (3,m,ns,nx) and is for MEASURES ONLY.
    """
    m=int(basis_m); nss=int(ns); nxx=int(nx)
    LL=float(L); EE=float(E0); hh=float(h)
    if nss<1 or nxx<2:
        raise ValueError("invalid ns/nx for CUF residual measure")
    if not np.isfinite(EE) or EE<=0.0:
        raise ValueError("E0 must be finite and >0")
    if not np.isfinite(hh) or hh<=0.0:
        raise ValueError("h must be finite and >0")

    row=cuf_row_scale(m,LL)
    mult=np.broadcast_to(row[None,:,None,None],(3,m,nss,nxx)).copy()

    # Boundary rows are not interior strong-form equations; use the separate
    # reviewer-prescribed characteristic multipliers.
    geo=EE*hh/LL
    nat=1.0/LL
    for ix,end in ((0,bc.left),(nxx-1,bc.right)):
        if int(end.m)!=m:
            raise ValueError("BC/basis m mismatch")
        for a in range(3):
            for tau in range(m):
                kind=str(end.conditions[a][tau].kind).upper()
                if kind=="GEOMETRIC":
                    mult[a,tau,:,ix]=geo
                elif kind in ("NATURAL","ELASTIC"):
                    # ELASTIC row: T + k(q-q0) - Tbar = 0 has the same
                    # physical units as a generalized end traction.  V-4 is
                    # measure-only scaling, so an elastic row must use the
                    # NATURAL characteristic multiplier rather than the
                    # displacement-like GEOMETRIC multiplier.
                    mult[a,tau,:,ix]=nat
                else:
                    raise ValueError(f"unsupported CUF end BC kind {kind!r}")
    return mult


def cuf_scaled_residual_for_measure(
    residual, *, basis_m, ns, nx, L, bc, E0, h
):
    """Return a V-4-scaled residual copy; never modifies the physical system."""
    r=np.asarray(residual,float)
    shape=(3,int(basis_m),int(ns),int(nx))
    if r.shape==(int(np.prod(shape)),):
        rr=r.reshape(shape)
    elif r.shape==shape:
        rr=r
    else:
        raise ValueError(f"CUF residual must have shape {shape} or flat equivalent")
    if not np.all(np.isfinite(rr)):
        raise ValueError("CUF residual contains non-finite values")
    mult=cuf_residual_measure_multipliers(
        basis_m=basis_m,ns=ns,nx=nx,L=L,bc=bc,E0=E0,h=h
    )
    return (rr*mult).reshape(-1)


def cuf_scaled_strong_load_for_measure(generalized_load, *, basis_m, ns, nx, L):
    """Scale CUF strong-form load blocks with L**(-tau) for the V-4 norm scale."""
    p=np.asarray(generalized_load,float)
    shape=(3,int(basis_m),int(ns),int(nx))
    if p.shape!=(shape):
        raise ValueError(f"generalized_load must have shape {shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")
    row=cuf_row_scale(basis_m,L)
    return (p*row[None,:,None,None]).reshape(-1)


def cuf_residual_inf_by_block(scaled_residual, *, basis_m, ns, nx):
    """Return per-(component,tau) inf norms from an already V-4-scaled residual."""
    m=int(basis_m)
    r=np.asarray(scaled_residual,float).reshape(3,m,int(ns),int(nx))
    out={}
    for c,name in enumerate("uvw"):
        for tau in range(m):
            out[f"{name}{tau}"]=float(np.max(np.abs(r[c,tau])))
    return out
