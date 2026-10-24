#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal x,s integration by parts for FSDT / Reddy G3D director theory.

Input is the *thickness-integrated* local first variation computed by
cylinder_director_g3d.thickness_weak_coefficients. The independent surface
fields are q=(u0,v0,w0,theta_x,theta_s) and electrical potential coefficients.
The period s=R*theta has NO circumferential edges. x=0,L are open ends.

Mechanical density:
  sum_{a+b<=3} P[a,b] * d_x^a d_s^b(delta q).
Electrical density:
  E0*delta psi + Ex*d_x(delta psi) + Es*d_s(delta psi).

The formulas below give the formal strong interior Euler residual and
three separate signed edge-work trace channels at each axial end:
  n_x [T0*delta q + T1*delta q_x + T2*delta q_xx + Ex*delta psi].
For FSDT/G3D (order <=2), T2=0; for Reddy/G3D (order <=3),
T2 generally DOES NOT vanish. Mixed derivatives appear once, not twice.

Differentiation of *coefficient fields* is delegated to an explicit supplied
callback D(field, dx_order, ds_order). This is algebraic operator plumbing,
NOT a validated FDM/MVI discretization, NOT a finite-element weak assembly,
and NOT a justification for prescribing all traces independently at a
physical clamp. Natural edge conditions require work-conjugate trace
independence and corner treatment to be established during PDE assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cylinder_director_g3d import (
    DirectorWeak, DirectorSurfaceJets, FIELDS, ELECTRICAL_FIELDS,
)

ORDERS = (
    (0,0), (1,0), (0,1), (2,0), (1,1),
    (0,2), (3,0), (2,1), (1,2), (0,3),
)


@dataclass(frozen=True)
class DirectorDifferentialBalance:
    mechanical_bulk: np.ndarray
    electrical_bulk: np.ndarray
    end_mechanical: np.ndarray  # (5,3,...): q, q_x, q_xx traces
    end_electrical: np.ndarray  # (P+1,...): potential trace
    x_outward_normal: str = "-1 at x=0, +1 at x=L"


def _coefficients(mechanical, electrical):
    m = np.asarray(mechanical)
    e = np.asarray(electrical)
    if m.ndim < 2 or m.shape[:2] != (5,len(ORDERS)):
        raise ValueError("mechanical coefficients must have shape (5,10,...grid)")
    if e.ndim < 2 or e.shape[1] != len(ELECTRICAL_FIELDS):
        raise ValueError("electrical coefficients must have shape (P+1,3,...grid)")
    if not 1 <= e.shape[0] <= 5 or m.shape[2:] != e.shape[2:]:
        raise ValueError("incompatible electrical and mechanical coefficient grids")
    if not np.isfinite(m).all() or not np.isfinite(e).all():
        raise ValueError("coefficient arrays must be finite")
    return m,e


def _d(derivative, array, nx, ns):
    if nx == 0 and ns == 0:
        return array
    value = np.asarray(derivative(array,nx,ns))
    if value.shape != array.shape or not np.isfinite(value).all():
        raise ValueError("derivative callback returned an invalid field")
    return value


def integrate_by_parts(mechanical, electrical, derivative):
    """Return interior residuals and oriented axial end-work coefficients.

    No axial integration, quadrature, boundary enforcement, or solver step
    occurs here. The supplied D must commute in x and periodic s for this
    formal continuum IBP identity to represent the requested discretization.
    """
    if not callable(derivative):
        raise TypeError("derivative must implement D(field, dx, ds)")
    m,e = _coefficients(mechanical,electrical)

    # Euler-Lagrange alternating derivatives of every independent jet.
    bulk_m = np.zeros_like(m[:,0])
    for k,(ax,ats) in enumerate(ORDERS):
        bulk_m += (-1)**(ax+ats)*_d(derivative,m[:,k],ax,ats)

    bulk_e = e[:,0] - _d(derivative,e[:,1],1,0) - _d(derivative,e[:,2],0,1)

    # First integrate all s derivatives by parts on a periodic circumference,
    # then integrate the remaining x derivatives by parts at the two ends.
    t0=(m[:,1] - _d(derivative,m[:,4],0,1)
        + _d(derivative,m[:,8],0,2)
        - _d(derivative,m[:,3],1,0)
        + _d(derivative,m[:,7],1,1)
        + _d(derivative,m[:,6],2,0))
    t1=m[:,3] - _d(derivative,m[:,7],0,1) - _d(derivative,m[:,6],1,0)
    t2=m[:,6].copy()
    mechanical_ends=np.stack((t0,t1,t2),axis=1)
    return DirectorDifferentialBalance(
        bulk_m,bulk_e,mechanical_ends,e[:,1].copy(),
    )


def local_virtual_work(weak: DirectorWeak, variation: DirectorSurfaceJets):
    """Contraction with independent virtual jets at one surface point.

    This evaluates a weak-form density; the area integral is the caller's
    responsibility. The potential basis must be identical in both inputs.
    """
    if not isinstance(weak,DirectorWeak):
        raise TypeError("Expected thickness-integrated DirectorWeak")
    if not isinstance(variation,DirectorSurfaceJets):
        raise TypeError("Expected virtual DirectorSurfaceJets")
    m,e=_coefficients(weak.mechanical,weak.electrical)
    if m.ndim != 2 or e.ndim != 2 or e.shape[0] != len(variation.psi):
        raise ValueError("local weak coefficients and variation do not match")
    result=0.
    for k,name in enumerate(FIELDS):
        result += float(m[:,k] @ getattr(variation,name))
    for k,name in enumerate(ELECTRICAL_FIELDS):
        result += float(e[:,k] @ getattr(variation,name))
    return result


def outward_axial_work(balance, side, *, delta_q, delta_qx, delta_qxx,
                        delta_psi):
    """Outward-oriented boundary density; integrate it over circumference.

    Variations must have shape (5,...grid) and (P+1,...grid). A physical
    condition MAY constrain some traces; this function never invents them.
    """
    if not isinstance(balance,DirectorDifferentialBalance):
        raise TypeError("Expected DirectorDifferentialBalance")
    if side not in ("LEFT","RIGHT"):
        raise ValueError("side must be LEFT or RIGHT")
    deltas=(np.asarray(delta_q),np.asarray(delta_qx),np.asarray(delta_qxx))
    ends=np.asarray(balance.end_mechanical)
    potential=np.asarray(delta_psi)
    if ends.ndim<2 or ends.shape[:2]!=(5,3):
        raise ValueError("end mechanical work requires shape (5,3,...grid)")
    if any(d.shape!=ends[:,0].shape for d in deltas):
        raise ValueError("mechanical trace variations have incorrect shape")
    if potential.shape != np.asarray(balance.end_electrical).shape:
        raise ValueError("potential trace variation has incorrect shape")
    arrays=(*deltas,potential)
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("virtual traces must be finite")
    power=sum(np.sum(ends[:,k]*deltas[k],axis=0) for k in range(3))
    power=power+np.sum(np.asarray(balance.end_electrical)*potential,axis=0)
    return power * (-1 if side=="LEFT" else 1)


__all__=("ORDERS","DirectorDifferentialBalance","integrate_by_parts",
         "local_virtual_work","outward_axial_work")
