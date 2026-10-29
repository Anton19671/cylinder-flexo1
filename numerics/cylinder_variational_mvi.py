#!/usr/bin/env python3
"""Small-grid alternating variational solve of the coupled shell enthalpy.

All fields, including potential coefficients, have a separated representation
u/scales = sum_r X[c,r,x] Y[c,r,s]. Each Newton solve changes ONLY the
active family, with the partner frozen and essential traces eliminated. QR
changes profile coordinates without changing the represented field. There is
no full-grid equilibrium correction, rank growth or regularization.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr as pivoted_qr

from cylinder_boundary_spatial import affine_trace_space, separated_profile_directions
from cylinder_variational_fdm import (
    StaticFDMSettings, StaticFDMResult, solve_static_subspace, state_scales,
)
from cylinder_variational_boundary import combine_functional_and_boundary
from cylinder_residual_norms import admissible_nodal_inf_norm, admissible_nodal_metrics
from cylinder_mvi_cold_start import seed_inactive_mvi_partners
from cylinder_mvi_acceptance import certify_mvi_solution
from cylinder_mvi_continuation import continue_accepted_mvi_profiles


@dataclass(frozen=True)
class StaticMVISettings:
    rank: int = 1
    max_sweeps: int = 30

    def __post_init__(self):
        for key in ("rank", "max_sweeps"):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(key+" must be a positive integer")
        if self.rank>8:raise ValueError("MVI rank must be 1..8")


@dataclass(frozen=True)
class StaticMVIResult(StaticFDMResult):
    x_profiles: np.ndarray
    y_profiles: np.ndarray
    scales: np.ndarray
    sweeps: int
    projected_residual_scaled: float
    projected_equilibrium_satisfied: bool
    mvi_certificate: dict | None = None


def reconstruct_profiles(x,y):
    return np.einsum("crx,crs->cxs",x,y)


def _normalize_partner(x,y,axis):
    """Exact QR coordinate change, including completion of inactive columns.

    A zero field must not permanently freeze a family. Reduced QR supplies a
    full independent partner family; multiplication of the other family by R
    preserves its product, even when R is rank deficient. No singular values
    of the physical state are discarded.
    """
    x,y=x.copy(),y.copy()
    for c in range(len(x)):
        if axis=="X":
            q,r=np.linalg.qr(y[c].T,mode="reduced")
            y[c]=q.T;x[c]=r@x[c]
        else:
            q,r=np.linalg.qr(x[c].T,mode="reduced")
            x[c]=q.T;y[c]=r@y[c]
    return x,y


def _initial_profiles(field,rank,roundoff_scale):
    nc,nx,ns=field.shape
    if rank>min(nx,ns):
        raise ValueError("MVI rank exceeds grid dimensions; increase nx and ns")
    x=np.zeros((nc,rank,nx));y=np.zeros((nc,rank,ns))
    # Constant first partner activates the axisymmetric cold start. Higher
    # modes are deterministic and independent, not identically zero terms.
    bank=np.cos(np.pi*np.arange(rank)[:,None]*(np.arange(nx)+.5)/nx)
    bank=np.linalg.qr(bank.T,mode="reduced")[0].T
    for c in range(nc):
        noise=50*np.finfo(float).eps*max(nx,ns)*roundoff_scale[c]
        if np.linalg.norm(field[c])<=noise:
            # Sandwich MVI: all X factors are ready, Y starts at zero.
            # The represented field remains exactly zero, with no lift.
            x[c]=bank
        else:
            # Pivoted QR selects an independent row space without SVD.
            # The represented field must already fit the requested fixed rank.
            q,_,_=pivoted_qr(field[c].T,mode="economic",pivoting=True,
                             check_finite=False)
            y[c]=q[:,:rank].T
            x[c]=(field[c]@q[:,:rank]).T
            error=np.linalg.norm(x[c].T@y[c]-field[c])
            # Ignore only backward error of the affine projection/previous
            # profile reconstruction, scaled to this component's own units.
            if error>1.e-10*np.linalg.norm(field[c])+noise:
                raise ValueError("initial field cannot be represented at requested MVI rank")
    return x,y


def solve_static_mvi(problem, mechanical, electrical=None, *, dead_load=None,
                     free_charge=None, settings=None, mvi_settings=None,
                     on_iteration=None, previous_mvi_result=None):
    if problem.kind!="STATIC":raise ValueError("solve_static_mvi requires STATIC")
    settings=StaticFDMSettings() if settings is None else settings
    mvi_settings=StaticMVISettings() if mvi_settings is None else mvi_settings
    if not isinstance(settings,StaticFDMSettings):raise TypeError("invalid Newton settings")
    if not isinstance(mvi_settings,StaticMVISettings):raise TypeError("invalid MVI settings")
    f=problem.functional
    q=np.asarray(mechanical,float)
    p=None if electrical is None else np.asarray(electrical,float)
    first=problem.assemble(q,p,dead_load=dead_load,free_charge=free_charge)
    scales=state_scales(f,q,p,settings)
    energy_scale=float(f.material.young_pa*f.h*np.sum(f.W))
    c=first.space.matrix*scales[None,:]
    target=first.space.target
    # Constraints define admissible tests, NOT a physical additive field.
    full=affine_trace_space(c,target,np.zeros_like(scales))
    # Full-space admissibility is checked for the MVI certificate, not
    # solved as a full-nodal Newton system. Only each frozen-partner
    # X/Y subspace must respect the dense Newton DOF safety limit.
    state=first.state/scales
    nc=q.shape[0]+(0 if p is None else p.shape[0])
    shape=(nc,*q.shape[1:])
    initial_scaled=state.reshape(shape)
    roundoff_scale=np.linalg.norm(initial_scaled,axis=(1,2))
    if previous_mvi_result is None:
        x,y=_initial_profiles(initial_scaled,mvi_settings.rank,roundoff_scale)
        initialization="COLD_START"
    else:
        x,y,_=continue_accepted_mvi_profiles(
            previous_mvi_result,q,p,new_scales=scales,
            full_space=full,rank=mvi_settings.rank)
        initialization="CONTINUATION"

    def fields():
        v=reconstruct_profiles(x,y).ravel()
        u=scales*v
        return v,u[:q.size].reshape(q.shape),None if p is None else u[q.size:].reshape(p.shape)

    def evaluate():
        v,qm,pm=fields()
        energy,rm,re,_=combine_functional_and_boundary(
            f,qm,pm,bc_spec=problem.spec,dead_load=dead_load,free_charge=free_charge)
        r=rm.ravel() if re is None else np.concatenate((rm.ravel(),re.ravel()))
        if not np.isfinite(energy) or not np.isfinite(r).all():
            raise ValueError("nonfinite stationary functional/residual")
        g=full.basis.T@(scales*r)/energy_scale
        return v,qm,pm,float(energy),r,g

    def projected_components(v,r):
        # No extra nodal field may be inserted to force inhomogeneous traces.
        # Until X/Y impose them, use full residual as a conservative diagnostic.
        if np.max(np.abs(full.defect(v)),initial=0)>1.e-9:
            reduced=full.basis.T@(scales*r)/energy_scale
            value=admissible_nodal_inf_norm(full.basis,reduced)
            return {"X":value,"Y":value}
        norms={}
        for axis in ("Y","X"):
            xx,yy=_normalize_partner(x,y,axis)
            directions=separated_profile_directions(xx,yy,axis)
            space=affine_trace_space(c,target,v,directions=directions)
            reduced=space.basis.T@(scales*r)/energy_scale
            norms[axis]=admissible_nodal_inf_norm(space.basis,reduced)
        return norms

    history=[];iterations=0;sweeps=0;status="MAX_SWEEPS";converged=False
    previous_sweep_change=None
    v,qm,pm,energy,r,g=evaluate()
    # For an exact zero separated component, the original constant partner
    # cannot see a zero-mean circumferential load. Align its *inactive* Y
    # directions with the admissible residual. No field or rank changes.
    admissible=(full.basis@g).reshape(shape)
    x,y=seed_inactive_mvi_partners(x,y,admissible,first_axis="Y")
    v,qm,pm,energy,r,g=evaluate()
    initial=admissible_nodal_inf_norm(full.basis,g)
    tolerance=settings.absolute_tolerance_scaled+settings.relative_tolerance*initial
    projected_parts=projected_components(v,r)
    projected=max(projected_parts.values())
    field_change=0.
    change_tolerance=max(1.e-8,0.5*settings.relative_tolerance)

    def record_sweep(sweep,previous_v=None):
        nonlocal previous_sweep_change
        metrics=admissible_nodal_metrics(full.basis,g,q.size)
        current_change=field_change if sweep else None
        ratio=(current_change/previous_sweep_change if
               current_change is not None and previous_sweep_change is not None
               and previous_sweep_change>0 else None)
        if current_change is not None: previous_sweep_change=current_change
        row={"event":"sweep","sweep":sweep,"iteration":iterations,
             "mvi_initialization":initialization,
             "residual_scaled":metrics["full"],
             "mechanical_residual_scaled":metrics["mechanical"],
             "electrical_residual_scaled":metrics["electrical"],
             "basis_coordinate_residual_scaled":metrics["basis_coordinates"],
             "residual_relative_to_initial":metrics["full"]/initial if initial else 0.,
             "projected_residual_scaled":projected,
             "projected_x_residual_scaled":projected_parts["X"],
             "projected_y_residual_scaled":projected_parts["Y"],
             "sweep_field_change_relative":field_change,
             "sweep_field_change_tolerance":change_tolerance,
             "sweep_field_change_ratio":ratio,
             "sweep_stagnation_warning":bool(ratio is not None and ratio>=.95),
             "projected_diagnostic_applicable":bool(
                 np.max(np.abs(full.defect(v)),initial=0)<=1.e-9),
             "potential_j":energy,
             "sweep_field_increment_scaled":(
                 0. if previous_v is None else float(np.max(abs(v-previous_v),initial=0))),
             "constraint_defect_scaled":float(np.max(abs(full.defect(v)),initial=0))}
        history.append(row)
        if on_iteration is not None:on_iteration(dict(row))

    record_sweep(0)
    if initial<=tolerance and np.max(np.abs(full.defect(v)),initial=0)<=1.e-9:
        converged=True;status="CONVERGED_FULL"
    for sweep in range(1,mvi_settings.max_sweeps+1):
        if converged:break
        failed=False
        previous_v=v.copy()
        for axis in ("Y","X"):
            x,y=_normalize_partner(x,y,axis)
            v,qm,pm=fields()
            directions=separated_profile_directions(x,y,axis)
            def report(row):
                data={**row,"event":"halfstep","sweep":sweep,"direction":axis}
                if on_iteration is not None:on_iteration(dict(data))
            result=solve_static_subspace(problem,qm,pm,dead_load=dead_load,
                free_charge=free_charge,settings=settings,directions=directions,on_iteration=report)
            # Newton fills tangent/line-search diagnostics after its callback.
            history.extend({**row,"event":"halfstep","sweep":sweep,"direction":axis}
                           for row in result.history)
            u=result.mechanical.ravel() if p is None else np.concatenate(
                (result.mechanical.ravel(),result.electrical.ravel()))
            delta=u/scales-v
            # Frozen partners are orthonormal, hence T.T exactly recovers
            # active profile increments without an equilibrium FDM solve.
            increment=directions.T@delta
            if np.linalg.norm(delta-directions@increment)>1.e-10*max(1.,np.linalg.norm(delta)):
                raise ValueError("MVI correction escaped frozen-partner space")
            if axis=="X":x+=increment.reshape(x.shape)
            else:y+=increment.reshape(y.shape)
            iterations+=result.iterations
            if not result.converged:
                status="HALFSTEP_"+axis+"_"+result.status;failed=True;break
        sweeps=sweep
        v,qm,pm,energy,r,g=evaluate()
        projected_parts=projected_components(v,r)
        projected=max(projected_parts.values())
        # A relative physical-field test must not normalize small motion by 1.
        field_change=float(np.linalg.norm(v-previous_v)/max(
            np.linalg.norm(v),np.linalg.norm(previous_v),1.e-30))
        record_sweep(sweep,previous_v)
        if failed:break
        # Classical sandwich/Krysko finite-rank stop: both tangent
        # equations AND relative field change on the completed Y->X state.
        if (projected<=tolerance and field_change<=change_tolerance
                and np.max(np.abs(full.defect(v)),initial=0)<=1.e-9):
            converged=True
            status=("CONVERGED_FULL" if
                admissible_nodal_inf_norm(full.basis,g)<=tolerance else
                "CONVERGED_PROJECTED")
            break
    if not converged and status=="MAX_SWEEPS":
        status="MAX_SWEEPS_PROJECTED_NOT_CONVERGED"
    defect=float(np.max(abs(full.defect(v)),initial=0))
    if defect>1.e-9:converged=False;status="BOUNDARY_CONSTRAINT_DRIFT"
    certificate=certify_mvi_solution(
        x_profiles=x,y_profiles=y,scales=scales.reshape(shape),
        mechanical=qm,electrical=pm,
        full_basis=full.basis,reduced_residual=g,
        projected_residual_scaled=projected,
        full_tolerance_scaled=tolerance,constraint_defect_scaled=defect,
        solver_converged=converged,
        field_change_relative=field_change,
        field_change_tolerance=change_tolerance)
    if converged and not certificate["accepted"]:
        converged=False
        status=("MVI_PROFILE_RECONSTRUCTION_DRIFT"
                if not certificate["profile_reconstruction_satisfied"]
                else "BOUNDARY_CONSTRAINT_DRIFT"
                if not certificate["boundary_constraint_satisfied"]
                else "MVI_PROJECTED_OR_FIELD_CHANGE_NOT_CONVERGED")
    elif converged:
        status=certificate["solution_status"]
    scaled_r=scales*r
    reaction=(scaled_r-full.basis@(full.basis.T@scaled_r))/scales
    norm=certificate["full_admissible_residual_scaled"]
    return StaticMVIResult(converged,status,qm,pm,energy,r.copy(),reaction,norm,
        norm/initial if initial else 0.,defect,tuple(history),iterations,
        x.copy(),y.copy(),scales.reshape(shape),sweeps,
        projected,projected<=tolerance,certificate)


__all__=("StaticMVISettings","StaticMVIResult","solve_static_mvi","reconstruct_profiles")
