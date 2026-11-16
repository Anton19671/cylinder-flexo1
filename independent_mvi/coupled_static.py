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
from fractions import Fraction
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


class ProjectedBasisAmbiguous(ValueError):
    """Near-dependent directions cannot be safely classified in float64."""


@dataclass(frozen=True)
class CoupledMVISweepSettings:
    max_sweeps: int=25
    relative_field_change: float=1.e-8
    halfstep: CoupledMVIHalfstepSettings=CoupledMVIHalfstepSettings()
    allow_projected_convergence: bool=False

    def __post_init__(self):
        if type(self.allow_projected_convergence) is not bool:
            raise ValueError("MVI_PROJECTED_CONVERGENCE_FLAG_MUST_BE_BOOLEAN")
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
    nodal=getattr(work,"nodal_gradient",None)
    if nodal is None:
        raise ValueError("MVI_OWN_FULL_NODAL_FIRST_VARIATION_REQUIRED")
    nodal=np.asarray(nodal,dtype=float)
    if nodal.shape!=surface.shape or not np.isfinite(nodal).all():
        raise ValueError("MVI_INVALID_FULL_NODAL_FIRST_VARIATION")
    def basis(rows):
        # Only exact zero/proportional rows are known redundant without a
        # floating-point rank assumption. They remain intact in
        # the solver; this is solely a virtual-work diagnostic.
        def exactly_proportional(row,other):
            pivot=int(np.flatnonzero(other)[0])
            # Cheap screening before expensive exact binary-rational checks.
            if not np.allclose(row/np.max(np.abs(row)),
                               other/np.max(np.abs(other)),rtol=1.e-12,atol=1.e-14) and not np.allclose(
                               row/np.max(np.abs(row)),
                               -other/np.max(np.abs(other)),rtol=1.e-12,atol=1.e-14):
                return False
            factor=Fraction.from_float(float(row[pivot]))/Fraction.from_float(float(other[pivot]))
            return all(Fraction.from_float(float(a))==factor*Fraction.from_float(float(b))
                       for a,b in zip(row,other))
        unique=[]
        for row in rows:
            if not np.any(row):
                continue
            if any(exactly_proportional(row,other) for other in unique):
                continue
            unique.append(row)
        if not unique:
            return np.empty((rows.shape[1],0))
        values=np.stack(unique)
        amplitudes=np.max(np.abs(values),axis=1)
        matrix=(values/amplitudes[:,None]).T
        q,r=np.linalg.qr(matrix,mode="reduced")
        diagonal=np.abs(np.diag(r))
        cutoff=1.e-12*float(np.linalg.norm(r,ord="fro"))
        if len(diagonal)!=len(unique) or np.any(diagonal<=cutoff):
            raise ProjectedBasisAmbiguous("MVI_PROJECTED_BASIS_AMBIGUOUS")
        return q
    def rms(value):
        return float(np.linalg.norm(value)/np.sqrt(value.size)) if value.size else 0.
    x_values=[]
    y_values=[]
    for c in components:
        ids=free[c]
        residual=nodal[c,ids,:]*scale[c]/energy
        x_values.append(rms(residual@basis(factors.y[c])))
        y_values.append(rms(basis(factors.x[c][:,ids]).T@residual))
    norm_x=max(x_values)
    norm_y=max(y_values)
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
        active_fields=None,axis_sequence=("Y","X")):
    """Explicit independent frozen-partner halfsteps, no rank growth.

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
    if axis_sequence not in (("Y","X"),("X",)):
        raise ValueError("MVI_STATIC_INVALID_AXIS_SEQUENCE")
    _check_end_factors(surface,factors,program)
    components=_active_field_ids(surface,active_fields)
    scales,_=_scales(surface,field_scales,energy_scale_j)
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
        for axis in axis_sequence:
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
                # Keep the failing reduced X/Y linear solver's exact
                # condition/GMRES diagnostics in the static-sweep result.
                # Previously this branch discarded response.history and
                # collapsed every CUF TE3/4 failure to a generic status.
                history.append(dict(
                    sweep=sweep,axis=axis,
                    halfstep_status=response.status,
                    halfstep_iterations=response.iterations,
                    reduced_halfstep_history=list(response.history),
                    rank=candidate.rank,
                    rejected=True))
                return finish(False,"MVI_"+axis+"_"+response.status,
                              sweep-1,float("inf"),float("inf"))
            candidate=response.factors
        try:
            norm_x,norm_y,full_norm=_projected_norms(
                surface,candidate,program,field_scales,energy_scale_j,
                mechanical_load,free_charge,work_evaluator=work_evaluator,
                active_fields=active_fields)
        except ProjectedBasisAmbiguous:
            history.append(dict(sweep=sweep,rank=candidate.rank,rejected=True,
                                status="MVI_PROJECTED_BASIS_AMBIGUOUS"))
            return finish(False,"MVI_PROJECTED_BASIS_AMBIGUOUS",
                          sweep-1,float("inf"),float("inf"))
        next_field=surface.reconstruct(candidate)
        # Each active physical field has its own units and magnitude.
        # A large voltage or an inactive fixed field must not hide a
        # changing displacement (and vice versa).
        change=max(float(np.linalg.norm(next_field[c]-previous[c])/
                         max(np.linalg.norm(next_field[c]),
                             np.linalg.norm(previous[c]),1.e-30*scales[c]))
                   for c in components)
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
                if settings.allow_projected_convergence:
                    return finish(True,"CONVERGED_PROJECTED",sweep,norm_x,norm_y)
                return finish(False,"MVI_FIXED_RANK_INSUFFICIENT_FULL_NODAL_WORK",
                              sweep,norm_x,norm_y)
            return finish(True,"MVI_FIXED_RANK_FULL_NODAL_STATIONARY",
                          sweep,norm_x,norm_y)
    return finish(False,"MVI_FIXED_RANK_SWEEPS_EXHAUSTED",
                  settings.max_sweeps,norm_x,norm_y)


__all__=("CoupledMVISweepSettings","CoupledMVISweepResult",
         "solve_coupled_mvi_static")
