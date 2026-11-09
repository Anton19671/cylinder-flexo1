"""Own fixed-rank nonlinear coupled MVI Y->X static iteration driver.

No MKR or reference spatial solver. Repeats independent frozen-partner
halfsteps until BOTH tangent stationarity tests and the reconstructed
field-change criterion hold. This only establishes stationarity on the
selected fixed-rank manifold; it DOES NOT certify unconstrained
two-dimensional equilibrium, high-order microfree boundary work, or a
physically well-posed connected G3D/Gz system.

The result is opt-in research-only. An unfinished sweep is never accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from independent_mvi.coupled_surface import (
    CoupledMVIFactors, NativeCoupledMVISurface,
)
from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram, CoupledMVIHalfstepSettings,
    _check_end_factors, _scales, _active_field_ids,
    solve_coupled_mvi_halfstep,
)


@dataclass(frozen=True)
class CoupledMVISweepSettings:
    max_sweeps: int=25
    relative_field_change: float=1.e-8
    halfstep: CoupledMVIHalfstepSettings=CoupledMVIHalfstepSettings()

    def __post_init__(self):
        if type(self.max_sweeps) is not int or self.max_sweeps<1:
            raise ValueError("MVI_SWEEP_COUNT_MUST_BE_POSITIVE")
        if not isinstance(self.halfstep,CoupledMVIHalfstepSettings):
            raise TypeError("CoupledMVIHalfstepSettings required")
        value=float(self.relative_field_change)
        if (isinstance(self.relative_field_change,(bool,np.bool_)) or
                not math.isfinite(value) or not 0<value<1):
            raise ValueError("MVI_INVALID_FIELD_CHANGE_TOLERANCE")
        object.__setattr__(self,"relative_field_change",value)


@dataclass(frozen=True)
class CoupledMVISweepResult:
    factors: CoupledMVIFactors
    converged: bool
    status: str
    sweeps: int
    field_change: float
    projected_norm_x: float
    projected_norm_y: float
    history: tuple
    full_nodal_norm: float=float("inf")


def _projected_norms(surface,factors,program,field_scales,energy_scale_j,
                     mechanical_load,free_charge,work_evaluator=None,
                     active_fields=None):
    scale,energy=_scales(surface,field_scales,energy_scale_j)
    free=_check_end_factors(surface,factors,program)
    components=_active_field_ids(surface,active_fields)
    work=(surface.weak_work(factors,mechanical_load=mechanical_load,
                            free_charge=free_charge)
          if work_evaluator is None else work_evaluator(factors))
    gx=work.gradient_x
    gy=work.gradient_y
    norm_x=max(float(np.max(np.abs(gx[c][:,ids]*scale[c]/energy),initial=0.))
               for c in components for ids in (free[c],))
    norm_y=max(float(np.max(np.abs(gy[c]*scale[c]/energy),initial=0.))
               for c in components)
    nodal=getattr(work,"nodal_gradient",None)
    if nodal is None:
        raise ValueError("MVI_OWN_FULL_NODAL_FIRST_VARIATION_REQUIRED")
    nodal=np.asarray(nodal,dtype=float)
    if nodal.shape!=surface.shape or not np.isfinite(nodal).all():
        raise ValueError("MVI_INVALID_FULL_NODAL_FIRST_VARIATION")
    # Restrict tests to the independently declared essential axial traces.
    # Each electric mode can have a different admissible endpoint subset.
    norm_full=max(float(np.max(np.abs(nodal[c,ids,:]*scale[c]/energy),
                               initial=0.))
                  for c in components for ids in (free[c],))
    return norm_x,norm_y,norm_full


def solve_coupled_mvi_static(
        surface:NativeCoupledMVISurface,
        factors:CoupledMVIFactors, *,
        program:CoupledMVIEndProgram,
        field_scales,energy_scale_j,
        mechanical_load=None,free_charge=None,
        settings=CoupledMVISweepSettings(),
        on_iteration=None,cancelled=None,work_evaluator=None,
        active_fields=None):
    """Explicit Y then X halfsteps; no rank growth or hidden full-node correction.

    A successful result means *projected* stationarity only. A publication-
    quality solution requires additional independent checks and microfree
    boundary certification not present in this module.
    """
    if not isinstance(surface,NativeCoupledMVISurface):
        raise TypeError("NativeCoupledMVISurface required")
    if not isinstance(settings,CoupledMVISweepSettings):
        raise TypeError("CoupledMVISweepSettings required")
    if on_iteration is not None and not callable(on_iteration):
        raise TypeError("on_iteration must be callable")
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable")
    if work_evaluator is not None and not callable(work_evaluator):
        raise TypeError("work_evaluator must be callable")
    _check_end_factors(surface,factors,program)
    _active_field_ids(surface,active_fields)
    _scales(surface,field_scales,energy_scale_j)
    history=[]
    accepted=factors
    previous=surface.reconstruct(accepted)
    change=float("inf")

    full_norm=float("inf")

    def finish(flag,status,sweep,nx,ny):
        return CoupledMVISweepResult(accepted,flag,status,sweep,
                                    change,nx,ny,tuple(history),full_norm)

    for sweep in range(1,settings.max_sweeps+1):
        if cancelled is not None and cancelled():
            return finish(False,"MVI_COUPLED_CANCELLED",sweep-1,float("inf"),
                          float("inf"))
        candidate=accepted
        for axis in ("Y","X"):
            response=solve_coupled_mvi_halfstep(
                surface,candidate,axis=axis,program=program,
                field_scales=field_scales,energy_scale_j=energy_scale_j,
                mechanical_load=mechanical_load,free_charge=free_charge,
                settings=settings.halfstep,
                on_iteration=(None if on_iteration is None else
                    lambda item,axis=axis,sweep=sweep:
                    on_iteration(dict(sweep=sweep,**item))),
                cancelled=cancelled,work_evaluator=work_evaluator,
                active_fields=active_fields)
            if not response.converged:
                return finish(False,"MVI_"+axis+"_"+response.status,
                              sweep-1,float("inf"),float("inf"))
            candidate=response.factors
        norm_x,norm_y,full_norm=_projected_norms(
            surface,candidate,program,field_scales,energy_scale_j,
            mechanical_load,free_charge,work_evaluator=work_evaluator,
            active_fields=active_fields)
        next_field=surface.reconstruct(candidate)
        change=float(np.linalg.norm(next_field-previous)/
                     max(np.linalg.norm(next_field),
                         np.linalg.norm(previous),1.e-30))
        # Full Y->X sweep becomes the ONLY accepted updated iterate.
        accepted=candidate
        previous=next_field
        row=dict(sweep=sweep,field_change=change,
                 projected_x=norm_x,projected_y=norm_y,
                 full_nodal=full_norm,rank=accepted.rank)
        history.append(row)
        if on_iteration is not None:
            on_iteration(dict(row))
        if max(norm_x,norm_y)<=settings.halfstep.tolerance_scaled and (
                change<=settings.relative_field_change):
            if full_norm>settings.halfstep.tolerance_scaled:
                return finish(False,"MVI_FIXED_RANK_INSUFFICIENT_FULL_NODAL_WORK",
                              sweep,norm_x,norm_y)
            return finish(True,"MVI_FIXED_RANK_FULL_NODAL_STATIONARY",
                          sweep,norm_x,norm_y)
    return finish(False,"MVI_FIXED_RANK_SWEEPS_EXHAUSTED",
                  settings.max_sweeps,norm_x,norm_y)


__all__=("CoupledMVISweepSettings","CoupledMVISweepResult",
         "solve_coupled_mvi_static")
