"""Gauss-IRK4 MVI-owned physical loads and stationary essential trace checks.

This is intentionally self-contained in the MVI path: no import from the
FDM-specific Gauss numerical adapter, no FDM solver selection or calls.
The underlying CylinderFunctional/BoundarySpatialProblem is still a shared
weak formulation and DOES NOT make the two coupled spatial methods
scientifically independent.
"""
from __future__ import annotations

import numpy as np
from cylinder_gauss2_rk import GAUSS2_C


def _stage_loads(loads_at, time_s):
    """Require the physical mechanical load and optional free-charge pair."""
    loads=loads_at(float(time_s))
    if not isinstance(loads,tuple) or len(loads)!=2:
        raise TypeError("loads_at(t) must return (mechanical_load,free_charge)")
    return loads


def _physical_assembled(problem,q,p,*,time_s,density_kg_m3,
                        acceleration=None,velocity=None,
                        damping_rate=0.,eta_s=0.,loads_at):
    """Compile MVI stage traces using the physical boundary program.

    This is a boundary compiler/physical action; it is not a dense FDM
    spatial Newton stage or an FDM solution used as an MVI rank lift.
    """
    dead,charge=_stage_loads(loads_at,time_s)
    kwargs=dict(dead_load=dead,free_charge=charge,time_s=time_s)
    if acceleration is not None:
        kwargs.update(acceleration=acceleration,density_kg_m3=density_kg_m3)
        if damping_rate or eta_s or problem.functional.theory=="KL":
            kwargs.update(velocity=velocity,
                          mass_damping_rate_per_s=damping_rate,eta_s=eta_s)
    return problem.assemble(q,p,**kwargs)


def _mechanical_trace(assembled,mechanical_size):
    matrix=np.asarray(assembled.space.matrix,float)
    target=np.asarray(assembled.space.target,float)
    if matrix.ndim!=2 or matrix.shape[1]<mechanical_size:
        raise ValueError("invalid physical trace matrix")
    mechanical_only=~np.any(matrix[:,mechanical_size:]!=0.,axis=1)
    return matrix[mechanical_only,:mechanical_size],target[mechanical_only]


def _require_fixed_supports(problem,state,settings,density,loads_at):
    """Verify unchanged mechanical traces at t0, both Gauss nodes and end.

    Moving mechanical supports require a specifically derived collocation
    closure. Reject them rather than silently correcting an MVI endpoint.
    """
    q,p=state.mechanical,state.electrical
    t0=state.time_s
    times=(t0,*(t0+settings.dt_s*GAUSS2_C),t0+settings.dt_s)
    traces=[]
    for t in times:
        a=_physical_assembled(problem,q,p,time_s=float(t),
                              density_kg_m3=density,loads_at=loads_at)
        traces.append(_mechanical_trace(a,q.size))
    c0,d0=traces[0]
    for c,d in traces[1:]:
        if (c.shape!=c0.shape or d.shape!=d0.shape or
                not np.allclose(c,c0,rtol=0.,atol=1e-13) or
                not np.allclose(d,d0,rtol=0.,atol=1e-13)):
            raise ValueError("GAUSS2_MOVING_MECHANICAL_SUPPORT_REQUIRES_END_CLOSURE")
    q0=state.mechanical.ravel()
    v0=state.velocity.ravel()
    if (np.max(np.abs(c0@q0-d0),initial=0)>1e-10 or
            np.max(np.abs(c0@v0),initial=0)>1e-10):
        raise ValueError("GAUSS2_INITIAL_SUPPORT_INCONSISTENT")
    return c0,d0


__all__=("_stage_loads","_physical_assembled",
         "_mechanical_trace","_require_fixed_supports")
