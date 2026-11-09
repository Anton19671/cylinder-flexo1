"""Formal axial natural end actions from independent MVI weak jets.

Research-only: formal integration by parts in x and s of the local
through-thickness variational coefficients. The axial derivatives are
the OWN MVI SBP42 matrices, circumferential derivatives OWN odd Fourier.
This is a diagnostic of formal action signs, NOT a certified high-order
discrete edge closure, source microfree implementation, or CASE dispatch.

For coefficients A[c,a,b] multiplying delta(d_x^a d_s^b U_c), the periodic
circumferential reduction is B[c,a] = sum_b (-d_s)^b A[c,a,b].
The outward axial work conjugate to delta(d_x^k U_c) is
n_x * sum_(a=k+1)^amax (-d_x)^(a-k-1) B[c,a],
for k from 0 through amax-1. Physical inward/outward sign uses -1/+1.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from independent_mvi.coupled_surface import (
    MECHANICAL_MULTI, ELECTRIC_MULTI, CoupledMVIWork,
    NativeCoupledMVISurface,
)


@dataclass(frozen=True)
class MVIEndActions:
    left: tuple
    right: tuple
    circumferentially_reduced: tuple
    status: str = "FORMAL_END_ACTION_DIAGNOSTIC_NOT_BOUNDARY_CERTIFIED"


def formal_mvi_axial_end_actions(surface: NativeCoupledMVISurface,
                                 work: CoupledMVIWork):
    """Return per-component tuples (value, slope, curvature) end actions.

    Returned arrays have shape (ns,); each tuple length matches the highest
    axial variation order of its local physical kinematics (up to 3).
    Electrical modes have only the potential end action (one entry).
    """
    if not isinstance(surface, NativeCoupledMVISurface):
        raise TypeError("independent coupled MVI surface required")
    if not isinstance(work, CoupledMVIWork):
        raise TypeError("CoupledMVIWork required")
    a=work.coefficient_jets
    expected=(surface.n_fields,len(surface.multi),
              surface.grid.nx,surface.grid.ns)
    if a is None or np.shape(a)!=expected:
        raise ValueError("MVI_END_ACTIONS_REQUIRE_LOCAL_WEAK_COEFFICIENTS")
    if work.rank<1 or work.rank>8:
        raise ValueError("MVI_END_ACTIONS_INVALID_FIXED_RANK")
    if (np.shape(work.mechanical)!=(surface.n_mechanical,
                                    surface.grid.nx,surface.grid.ns) or
            (surface.electrical_on and
             np.shape(work.electrical)!=(surface.n_electric,
                                          surface.grid.nx,surface.grid.ns)) or
            (not surface.electrical_on and work.electrical is not None)):
        raise ValueError("MVI_END_ACTIONS_FIELD_SHAPE_MISMATCH")
    a=np.asarray(a,dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("MVI_NONFINITE_LOCAL_WEAK_COEFFICIENTS")
    dx=surface.grid.dx
    ds=surface.grid.ds
    results=[]
    for c in range(surface.n_fields):
        multi=(surface.mechanical_multi if c<surface.n_mechanical
               else ELECTRIC_MULTI)
        grouped={}
        for k,(ax,ang) in enumerate(multi):
            # d_s operates along the second grid axis, hence Ds.T.
            contribution=a[c,k]@surface._ds_power[ang].T
            grouped[ax]=grouped.get(ax,0.)+(-1)**ang*contribution
        max_order=max(grouped)
        reduced=tuple(np.array(grouped.get(j,np.zeros_like(a[c,0])),
                               copy=True) for j in range(max_order+1))
        sides=[]
        for node,normal in ((0,-1.),(-1,1.)):
            actions=[]
            for derivative_order in range(max_order):
                value=np.zeros(surface.grid.ns)
                for ax in range(derivative_order+1,max_order+1):
                    order=ax-derivative_order-1
                    value+=(-1)**order*(surface._dx_power[order]@
                                         reduced[ax])[node]
                actions.append(normal*value)
            sides.append(tuple(actions))
        results.append((sides[0],sides[1],reduced))
    return MVIEndActions(
        tuple(r[0] for r in results),
        tuple(r[1] for r in results),
        tuple(r[2] for r in results),
    )


__all__=("MVIEndActions","formal_mvi_axial_end_actions")
