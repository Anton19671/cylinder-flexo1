"""Own frozen-partner Newton halfstep for nonlinear coupled fixed-rank MVI.

Research-only small-grid component. Works directly with the independent MVI
X(x)Y(s) weak-work gradient and its OWN SBP/Fourier surface operators.
Never calls MKR, a reference variational spatial solve, full-grid Newton,
SVD, a pseudoinverse, or a rank-changing correction.

Supported boundary subset is EXPLICIT: every mechanical generalized
displacement coefficient is immobile at both axial ends. Electric ON
requires all potential polynomial modes grounded at the left axial end
(optionally also the right); this physically fixes the potential datum
and does NOT claim insulated-end/source gauge equivalence. Higher-order
microfree closure is still unverified: this is not a production solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve, get_lapack_funcs

from independent_mvi.coupled_surface import (
    CoupledMVIFactors, NativeCoupledMVISurface,
)


@dataclass(frozen=True)
class CoupledMVIEndProgram:
    """Explicit supported homogeneous boundary program, no default BC.

    mechanical: both axial ends immobile for EVERY q coefficient.
    electric: all modes grounded on left; right may be grounded or natural.
    Face insulation and additional microfree traces remain unverified.
    """
    mechanical: str
    electric: str

    def validate(self, surface: NativeCoupledMVISurface):
        if not isinstance(surface, NativeCoupledMVISurface):
            raise TypeError("native coupled MVI surface required")
        if self.mechanical != "IMMOBILE_PARALLEL_ENDS":
            raise ValueError("MVI_REQUIRES_EXPLICIT_IMMOBILE_PARALLEL_ENDS")
        if surface.electrical_on:
            if self.electric not in ("GROUNDED_LEFT","GROUNDED_BOTH"):
                raise ValueError("MVI_FLOATING_ELECTRIC_GAUGE_NOT_IMPLEMENTED")
        elif self.electric != "OFF":
            raise ValueError("MVI_ELECTRIC_PROGRAM_MUST_BE_OFF")

    def free_axial_indices(self, surface):
        self.validate(surface)
        nx=surface.grid.nx
        free=[]
        for c in range(surface.n_fields):
            if c<surface.n_mechanical or self.electric=="GROUNDED_BOTH":
                free.append(np.arange(1,nx-1,dtype=int))
            else:
                free.append(np.arange(1,nx,dtype=int))
        return tuple(free)


@dataclass(frozen=True)
class CoupledMVIHalfstepSettings:
    max_iterations: int=10
    max_halvings: int=12
    max_coordinates: int=192
    difference_step: float=1.e-6
    tolerance_scaled: float=1.e-8
    minimum_reciprocal_condition: float=0.

    def __post_init__(self):
        for name in ("max_iterations","max_coordinates","max_halvings"):
            val=getattr(self,name)
            if type(val) is not int or val<(0 if name=="max_halvings" else 1):
                raise ValueError(name+": positive integer required")
        for name in ("difference_step","tolerance_scaled",
                     "minimum_reciprocal_condition"):
            v=getattr(self,name)
            if isinstance(v,(bool,np.bool_)):
                raise ValueError(name+": finite parameter required")
            try:v=float(v)
            except (TypeError,ValueError,OverflowError) as exc:
                raise ValueError(name+": finite parameter required") from exc
            if not math.isfinite(v) or (
                    v<0 if name=="minimum_reciprocal_condition" else v<=0):
                raise ValueError(name+": finite parameter required")
            if name=="difference_step" and v>=.1:
                raise ValueError("difference_step must be below .1")
            if name=="minimum_reciprocal_condition" and v>=1.:
                raise ValueError("minimum reciprocal condition must be below one")
            object.__setattr__(self,name,v)


@dataclass(frozen=True)
class CoupledMVIHalfstepResult:
    factors: CoupledMVIFactors
    converged: bool
    status: str
    axis: str
    iterations: int
    scaled_gradient_norm: float
    history: tuple


def _scales(surface, field_scales, energy_scale_j):
    scale=np.asarray(field_scales,dtype=float)
    if (scale.shape!=(surface.n_fields,) or
            not np.isfinite(scale).all() or np.any(scale<=0)):
        raise ValueError("one explicit positive physical scale per field required")
    if isinstance(energy_scale_j,(bool,np.bool_)):
        raise ValueError("energy scale must be positive finite")
    energy=float(energy_scale_j)
    if not math.isfinite(energy) or energy<=0:
        raise ValueError("energy scale must be positive finite")
    return scale.copy(),energy


def _check_end_factors(surface, factors, program, *, tolerance=1.e-11):
    surface._check_factors(factors)
    free=program.free_axial_indices(surface)
    nx=surface.grid.nx
    for c,ids in enumerate(free):
        if factors.rank>min(len(ids),surface.grid.ns):
            raise ValueError("MVI_RANK_EXCEEDS_ESSENTIAL_FREE_CAPACITY")
        blocked=np.ones(nx,dtype=bool)
        blocked[ids]=False
        if np.max(np.abs(factors.x[c,:,blocked]),initial=0.)>tolerance:
            raise ValueError("MVI_NONHOMOGENEOUS_OR_INCOMPATIBLE_X_TRACES")
    return free


def normalized_mvi_factors(surface, factors, program, axis):
    """QR change of profile coordinates; reconstructed physical field unchanged.

    Crucially, Y-halfstep normalization takes QR on each component's
    ADMISSIBLE axial coordinates, so inactive QR columns can never create
    displacement/voltage values at constrained ends.
    """
    if axis not in ("X","Y"):
        raise ValueError("MVI halfstep axis must be X or Y")
    free=_check_end_factors(surface,factors,program)
    x=factors.x.copy()
    y=factors.y.copy()
    for c,ids in enumerate(free):
        if axis=="X":
            Q,R=np.linalg.qr(y[c].T,mode="reduced")
            y[c]=Q.T
            x[c]=R@x[c]
        else:
            Q,R=np.linalg.qr(x[c][:,ids].T,mode="reduced")
            x[c]=0.
            x[c][:,ids]=Q.T
            y[c]=R@y[c]
    out=CoupledMVIFactors(x,y)
    _check_end_factors(surface,out,program)
    return out


def _checked_lu_newton(hessian,gradient,min_reciprocal):
    """Indefinite electromechanical Newton direction: LU, never SVD."""
    A=np.asarray(hessian,dtype=float)
    b=np.asarray(gradient,dtype=float)
    n=len(b)
    if A.shape!=(n,n) or not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise np.linalg.LinAlgError("MVI_INVALID_HALFSTEP_TANGENT")
    anorm=float(np.linalg.norm(A,ord=1))
    if not math.isfinite(anorm) or anorm<=0:
        raise np.linalg.LinAlgError("MVI_ZERO_HALFSTEP_TANGENT")
    with warnings.catch_warnings():
        warnings.simplefilter("error",LinAlgWarning)
        try:
            lu,piv=lu_factor(A,check_finite=False)
        except (LinAlgWarning,ValueError) as exc:
            raise np.linalg.LinAlgError("MVI_SINGULAR_HALFSTEP_TANGENT") from exc
    gecon=get_lapack_funcs("gecon",(A,))
    reciprocal,info=gecon(lu,anorm)
    threshold=max(100*n*np.finfo(float).eps,min_reciprocal)
    if info or not np.isfinite(reciprocal) or reciprocal<=threshold:
        raise np.linalg.LinAlgError("MVI_ILL_CONDITIONED_HALFSTEP_TANGENT")
    delta=lu_solve((lu,piv),-b,check_finite=False)
    if not np.isfinite(delta).all():
        raise np.linalg.LinAlgError("MVI_NONFINITE_HALFSTEP_INCREMENT")
    return delta


def solve_coupled_mvi_halfstep(
        surface: NativeCoupledMVISurface, factors: CoupledMVIFactors, *,
        axis: str, program: CoupledMVIEndProgram,
        field_scales, energy_scale_j,
        mechanical_load=None, free_charge=None,
        settings=CoupledMVIHalfstepSettings(),
        on_iteration=None, cancelled=None):
    """Solve ONLY the active frozen-partner X or Y coordinates.

    All coordinates belong to the selected separated family. A status
    other than converged means the caller must NOT accept the new factors.
    The energy is not minimized: coupled electric enthalpy is a saddle
    functional, so globalization checks its stationarity equations.
    """
    if not isinstance(surface,NativeCoupledMVISurface):
        raise TypeError("native coupled MVI surface required")
    if not isinstance(program,CoupledMVIEndProgram):
        raise TypeError("explicit coupled MVI boundary program required")
    if not isinstance(settings,CoupledMVIHalfstepSettings):
        raise TypeError("CoupledMVIHalfstepSettings required")
    if on_iteration is not None and not callable(on_iteration):
        raise TypeError("on_iteration must be callable")
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable")
    scales,energy_scale=_scales(surface,field_scales,energy_scale_j)
    # All field/end validation precedes the first nonlinear material call.
    _check_end_factors(surface,factors,program)
    working=normalized_mvi_factors(surface,factors,program,axis)
    allowed=program.free_axial_indices(surface)
    active=working.x if axis=="X" else working.y
    indices=(allowed if axis=="X" else
             tuple(np.arange(surface.grid.ns,dtype=int)
                   for _ in range(surface.n_fields)))
    sizes=[working.rank*len(i) for i in indices]
    count=sum(sizes)
    if count>settings.max_coordinates:
        raise ValueError("MVI_DENSE_HALFSTEP_COORDINATE_LIMIT")
    if any(n==0 for n in sizes):
        raise ValueError("MVI_EMPTY_HALFSTEP_FIELD")
    origin=np.concatenate([active[c][:,ids].ravel()/scales[c]
                           for c,ids in enumerate(indices)])

    def unpack(z):
        if z.shape!=(count,) or not np.isfinite(z).all():
            raise ValueError("MVI_NONFINITE_HALFSTEP_COORDINATES")
        x=working.x.copy()
        y=working.y.copy()
        array=x if axis=="X" else y
        offset=0
        for c,ids in enumerate(indices):
            n=sizes[c]
            array[c][:,ids]=scales[c]*z[offset:offset+n].reshape(
                working.rank,len(ids))
            offset+=n
        return CoupledMVIFactors(x,y)

    def evaluate(z):
        f=unpack(z)
        work=surface.weak_work(f,mechanical_load=mechanical_load,
                               free_charge=free_charge)
        derivative=(work.gradient_x if axis=="X" else work.gradient_y)
        g=np.concatenate([derivative[c][:,ids].ravel()*scales[c]/energy_scale
                          for c,ids in enumerate(indices)])
        if not np.isfinite(g).all():
            raise ValueError("MVI_NONFINITE_HALFSTEP_STATIONARITY")
        return f,g

    def result(converged,status,it,z,g,history):
        # Any unsuccessful halfstep returns ORIGINAL factors, never a
        # partly corrected field mistaken for an accepted physical step.
        output=unpack(z) if converged else factors
        return CoupledMVIHalfstepResult(
            output,converged,status,axis,it,
            float(np.max(np.abs(g),initial=0.)),tuple(history))

    z=origin.copy()
    _,g=evaluate(z)
    norm=float(np.max(np.abs(g),initial=0.))
    history=[dict(axis=axis,iteration=0,scaled_gradient_norm=norm)]
    if on_iteration is not None:
        on_iteration(dict(history[-1]))
    if norm<=settings.tolerance_scaled:
        return result(True,"MVI_HALFSTEP_PROJECTED_STATIONARY",0,z,g,history)

    for iteration in range(1,settings.max_iterations+1):
        if cancelled is not None and cancelled():
            return result(False,"MVI_HALFSTEP_CANCELLED",iteration-1,z,g,history)
        H=np.empty((count,count),dtype=float)
        for j in range(count):
            h=settings.difference_step*(1.+abs(z[j]))
            column=None
            for _ in range(settings.max_halvings+1):
                zp=z.copy(); zm=z.copy()
                zp[j]+=h;zm[j]-=h
                try:
                    _,gp=evaluate(zp)
                    _,gm=evaluate(zm)
                except ValueError as exc:
                    if "INVALID_DEFORMATION_MAPPING" not in str(exc):
                        raise
                    h*=.5
                    continue
                column=(gp-gm)/(2*h)
                break
            if column is None:
                return result(False,"MVI_HALFSTEP_TANGENT_PROBE_FAILED",
                              iteration-1,z,g,history)
            H[:,j]=column
        try:
            direction=_checked_lu_newton(H,g,settings.minimum_reciprocal_condition)
        except np.linalg.LinAlgError:
            return result(False,"MVI_HALFSTEP_TANGENT_NOT_SOLVED",
                          iteration-1,z,g,history)
        accepted=False
        for backtrack in range(settings.max_halvings+1):
            trial=z+2.**(-backtrack)*direction
            try:
                _,g_trial=evaluate(trial)
            except (ValueError,FloatingPointError,OverflowError):
                continue
            candidate_norm=float(np.max(np.abs(g_trial),initial=0.))
            if candidate_norm<norm or candidate_norm<=settings.tolerance_scaled:
                z,g,norm=trial,g_trial,candidate_norm
                accepted=True
                break
        history.append(dict(axis=axis,iteration=iteration,
                            scaled_gradient_norm=norm,accepted=accepted))
        if on_iteration is not None:
            on_iteration(dict(history[-1]))
        if not accepted:
            return result(False,"MVI_HALFSTEP_LINE_SEARCH_FAILED",
                          iteration,z,g,history)
        if norm<=settings.tolerance_scaled:
            return result(True,"MVI_HALFSTEP_PROJECTED_STATIONARY",
                          iteration,z,g,history)
    return result(False,"MVI_HALFSTEP_MAX_ITERATIONS",
                  settings.max_iterations,z,g,history)


__all__=("CoupledMVIEndProgram","CoupledMVIHalfstepSettings",
         "CoupledMVIHalfstepResult","normalized_mvi_factors",
         "solve_coupled_mvi_halfstep")
