"""Independent MVI initial/endpoint acceleration closure, no MKR/FDM.

For a prescribed fixed-rank mechanical field q, velocity v and a
quasi-static electric potential, solve
    M(q) a = -[elastic(q,phi) + KV(q,v) + mass_bias(q,v)]
on every unconstrained mechanical nodal direction. The MVI mass action
is linear in acceleration for the given q,v (including finite-rotation
KL terms), so its columns are obtained directly by applying the OWN
MVI inertial virtual-work operator to unit acceleration variations.

This is a linear acceleration reconstruction, NOT a full-grid spatial
displacement Newton solve, and never changes X/Y mechanical factors.
Potential equilibrium is solved independently at fixed q using the
method's own electric-only X/Y weak iteration, if electricity is active.
Electrical modes carry no mass, and SVD/pseudoinverse are forbidden.

Research only: source natural boundary/microfree/electric face closure
remains unverified. A max-coordinate cap prevents unsafe dense mass solves.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram, _check_end_factors, _checked_lu_newton, _scales,
)
from independent_mvi.coupled_newmark import (
    CoupledMVIDynamicState, NativeCoupledMVINewmark, _mechanical,
)
from independent_mvi.coupled_static import (
    CoupledMVISweepSettings, solve_coupled_mvi_static,
)
from independent_mvi.coupled_surface import CoupledMVIFactors


@dataclass(frozen=True)
class MVIAccelerationResult:
    state: CoupledMVIDynamicState | None
    status: str
    acceleration_condition: float | None
    electric_status: str | None


def close_mvi_instantaneous_state(
        configuration: NativeCoupledMVINewmark, factors: CoupledMVIFactors,
        velocity, *, time_s, field_scales, energy_scale_j,
        mechanical_load=None, free_charge=None,
        electric_settings=CoupledMVISweepSettings(),
        max_acceleration_coordinates=256,
        min_mass_reciprocal_condition=1.e-12,
        acceleration_linear_solver="AUTO",
        acceleration_dense_limit=192,
        acceleration_krylov_restart=32,
        acceleration_krylov_maxiter=200,
        acceleration_krylov_relative_tolerance=1.e-8,
        on_iteration=None, cancelled=None):
    """Close phi and a at prescribed q/v; fail closed without partial state."""
    if not isinstance(configuration,NativeCoupledMVINewmark):
        raise TypeError("NativeCoupledMVINewmark configuration required")
    surface=configuration.surface
    free=_check_end_factors(surface,factors,configuration.program)
    _scales(surface,field_scales,energy_scale_j)
    if not isinstance(electric_settings,CoupledMVISweepSettings):
        raise TypeError("CoupledMVISweepSettings required")
    if (type(max_acceleration_coordinates) is not int or
            not 1<=max_acceleration_coordinates<=300000):
        raise ValueError("MVI_ACCELERATION_COORDINATE_LIMIT_INVALID")
    if acceleration_linear_solver not in (
            "AUTO","DENSE_LU","MATRIX_FREE_GMRES"):
        raise ValueError("MVI_ACCELERATION_LINEAR_SOLVER_INVALID")
    for label,value in (
            ("DENSE_LIMIT",acceleration_dense_limit),
            ("KRYLOV_RESTART",acceleration_krylov_restart),
            ("KRYLOV_MAXITER",acceleration_krylov_maxiter)):
        if type(value) is not int or value<1:
            raise ValueError("MVI_ACCELERATION_"+label+"_INVALID")
    if isinstance(acceleration_krylov_relative_tolerance,
                  (bool,np.bool_)):
        raise ValueError("MVI_ACCELERATION_KRYLOV_TOLERANCE_INVALID")
    try:
        accuracy=float(acceleration_krylov_relative_tolerance)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("MVI_ACCELERATION_KRYLOV_TOLERANCE_INVALID") from exc
    if not math.isfinite(accuracy) or not 0<accuracy<.1:
        raise ValueError("MVI_ACCELERATION_KRYLOV_TOLERANCE_INVALID")
    if not isinstance(min_mass_reciprocal_condition,(int,float)) or (
            not math.isfinite(min_mass_reciprocal_condition) or
            min_mass_reciprocal_condition<0 or min_mass_reciprocal_condition>=1):
        raise ValueError("MVI_ACCELERATION_CONDITION_INVALID")
    if not math.isfinite(float(time_s)) or time_s<0:
        raise ValueError("MVI_ACCELERATION_TIME_INVALID")
    v=_mechanical(surface,velocity,"velocity")
    for c,ids in enumerate(free[:surface.n_mechanical]):
        blocked=np.ones(surface.grid.nx,dtype=bool)
        blocked[ids]=False
        if np.max(np.abs(v[c,blocked,:]),initial=0.)>configuration.stationary_tolerance:
            raise ValueError("MVI_INITIAL_MOVING_SUPPORT_NOT_IMPLEMENTED")
    if cancelled is not None and cancelled():
        return MVIAccelerationResult(None,"MVI_INITIAL_CANCELLED",None,None)

    # This is the declared physical mass-coordinate BUDGET, not a
    # requirement to allocate a dense mass matrix. Above the explicit
    # small dense threshold the independent MVI inertia acts as a
    # matrix-free Krylov operator. Reject budget violations before
    # any local material evaluation.
    free_count=sum(len(free[c])*surface.grid.ns
                   for c in range(surface.n_mechanical))
    if free_count>max_acceleration_coordinates:
        return MVIAccelerationResult(
            None,"MVI_ACCELERATION_DENSE_COORDINATE_LIMIT",None,None)

    current=factors
    electric_status=None
    if surface.electrical_on:
        electrical_ids=tuple(range(surface.n_mechanical,surface.n_fields))
        electric=solve_coupled_mvi_static(
            surface,current,program=configuration.program,
            field_scales=field_scales,energy_scale_j=energy_scale_j,
            mechanical_load=mechanical_load,free_charge=free_charge,
            settings=electric_settings,active_fields=electrical_ids,
            on_iteration=on_iteration,cancelled=cancelled)
        electric_status=electric.status
        if not electric.converged:
            return MVIAccelerationResult(
                None,"MVI_ENDPOINT_ELECTRIC_NOT_SOLVED",
                None,electric.status)
        current=electric.factors

    full=surface.reconstruct(current)
    q=full[:surface.n_mechanical]
    static=surface.weak_work(current,mechanical_load=mechanical_load,
                             free_charge=free_charge)
    base=np.asarray(static.nodal_gradient[:surface.n_mechanical],float).copy()
    if configuration.viscosity.eta_s:
        base+=configuration.viscosity.apply(q,v)
    z=np.zeros_like(q)
    bias=configuration.inertia.apply(q,v,z)
    base+=bias
    # Every free nodal acceleration is an independent mechanical degree
    # of freedom for this linear *mass* equation. No full-node Newton.
    coords=[(c,i,j)
            for c in range(surface.n_mechanical)
            for i in free[c]
            for j in range(surface.grid.ns)]
    n=len(coords)
    if n>max_acceleration_coordinates:
        return MVIAccelerationResult(
            None,"MVI_ACCELERATION_DENSE_COORDINATE_LIMIT",None,electric_status)
    rhs=np.array([base[c,i,j] for c,i,j in coords],float)
    if not np.isfinite(rhs).all():
        raise ValueError("MVI_ACCELERATION_NONFINITE_STATIC_FORCE")

    # Only the inertial MASS equation is solved in nodal accelerations.
    # Mechanical displacements stay exactly the original rank-R MVI
    # factors. No spatial displacement Newton, no MKR or SVD fallback.
    components=np.fromiter((c for c,i,j in coords),dtype=int,count=n)
    axial=np.fromiter((i for c,i,j in coords),dtype=int,count=n)
    circum=np.fromiter((j for c,i,j in coords),dtype=int,count=n)
    def action(values):
        if cancelled is not None and cancelled():
            raise InterruptedError("MVI_INITIAL_CANCELLED")
        values=np.asarray(values,dtype=float)
        if values.shape!=(n,) or not np.isfinite(values).all():
            raise np.linalg.LinAlgError("MVI_MASS_INVALID_KRYLOV_VECTOR")
        trial=np.zeros_like(q)
        trial[components,axial,circum]=values
        physical=configuration.inertia.apply(q,v,trial)-bias
        result=physical[components,axial,circum]
        if not np.isfinite(result).all():
            raise np.linalg.LinAlgError("MVI_MASS_NONFINITE_OPERATOR")
        return result

    method=(acceleration_linear_solver
            if acceleration_linear_solver!="AUTO" else
            "DENSE_LU" if n<=acceleration_dense_limit
            else "MATRIX_FREE_GMRES")
    if method=="DENSE_LU" and n>acceleration_dense_limit:
        return MVIAccelerationResult(
            None,"MVI_ACCELERATION_DENSE_METHOD_LIMIT",None,electric_status)
    try:
        if method=="DENSE_LU":
            mass=np.empty((n,n))
            for column in range(n):
                if cancelled is not None and cancelled():
                    raise InterruptedError("MVI_INITIAL_CANCELLED")
                unit=np.zeros(n)
                unit[column]=1.
                mass[:,column]=action(unit)
            if not np.isfinite(mass).all():
                raise ValueError("MVI_ACCELERATION_NONFINITE_MASS_OPERATOR")
            if np.linalg.norm(mass,1)==0:
                return MVIAccelerationResult(
                    None,"MVI_ACCELERATION_MASS_SINGULAR",0.,electric_status)
            delta=_checked_lu_newton(
                mass,rhs,min_mass_reciprocal_condition)
        else:
            operator=LinearOperator((n,n),matvec=action,dtype=float)
            delta,info=gmres(
                operator,-rhs,atol=0.,
                rtol=accuracy,
                restart=min(n,acceleration_krylov_restart),
                maxiter=acceleration_krylov_maxiter)
            if info!=0 or not np.isfinite(delta).all():
                return MVIAccelerationResult(
                    None,"MVI_ACCELERATION_KRYLOV_NOT_SOLVED",
                    None,electric_status)
    except InterruptedError:
        return MVIAccelerationResult(
            None,"MVI_INITIAL_CANCELLED",None,electric_status)
    except np.linalg.LinAlgError:
        return MVIAccelerationResult(
            None,"MVI_ACCELERATION_MASS_NOT_SOLVED",None,electric_status)
    # The accepted correction must actually close the SAME physical
    # nodal inertial equations. An approximate Krylov exit is not
    # sufficient by itself.
    try:
        correction=action(delta)+rhs
    except InterruptedError:
        return MVIAccelerationResult(
            None,"MVI_INITIAL_CANCELLED",None,electric_status)
    bound=1.e-8*max(1.,float(np.max(np.abs(rhs),initial=0.)))
    if (not np.isfinite(correction).all() or
            float(np.max(np.abs(correction),initial=0.))>bound):
        return MVIAccelerationResult(
            None,"MVI_ACCELERATION_EQUILIBRIUM_NOT_MET",
            None,electric_status)
    acceleration=np.zeros_like(q)
    for value,(c,i,j) in zip(delta,coords):
        acceleration[c,i,j]=value
    full_action=base+configuration.inertia.apply(q,v,acceleration)-bias
    defect=max((abs(full_action[c,i,j]) for c,i,j in coords),default=0.)
    scale=max(1.,max((abs(base[c,i,j]) for c,i,j in coords),default=0.))
    if defect>1.e-8*scale:
        return MVIAccelerationResult(None,"MVI_ACCELERATION_EQUILIBRIUM_NOT_MET",
                                     None,electric_status)
    state=CoupledMVIDynamicState(float(time_s),current,v,acceleration)
    configuration._validate_state(state)
    return MVIAccelerationResult(state,"MVI_INITIAL_EQUILIBRIUM_CLOSED",
                                 None,electric_status)


__all__=("MVIAccelerationResult","close_mvi_instantaneous_state")
