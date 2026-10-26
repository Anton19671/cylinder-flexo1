#!/usr/bin/env python3
"""Small-grid alternating variational solve of the coupled shell enthalpy.

All fields, including potential coefficients, have a separated representation
u/scales = lift + sum_r X[c,r,x] Y[c,r,s]. Each Newton solve changes ONLY the
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
from cylinder_mvi_cold_start import seed_inactive_mvi_partners


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
    lift_scaled: np.ndarray
    scales: np.ndarray
    sweeps: int
    projected_residual_scaled: float
    projected_equilibrium_satisfied: bool


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
    bank=np.cos(np.pi*np.arange(rank)[:,None]*(np.arange(ns)+.5)/ns)
    bank=np.linalg.qr(bank.T,mode="reduced")[0].T
    for c in range(nc):
        noise=50*np.finfo(float).eps*max(nx,ns)*roundoff_scale[c]
        if np.linalg.norm(field[c])<=noise:
            y[c]=bank
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
                     on_iteration=None):
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
    # Minimal boundary lift is independent of the equilibrium solution and
    # is the ONLY full-grid affine operation. It applies prescribed traces.
    full=affine_trace_space(c,target,np.zeros_like(scales))
    if full.basis.shape[1]>settings.max_free_dofs:
        raise ValueError("reference free-DOF limit exceeded")
    lift=full.origin.copy()
    state=first.state/scales
    roundoff=100*np.finfo(float).eps*max(c.shape)*max(
        np.max(abs(state),initial=0),np.max(abs(full.target),initial=0))
    if np.max(abs(full.defect(state)),initial=0)<=roundoff:
        # Preserve an accepted separated warm start. A second dense projector
        # needlessly mixes roundoff from large components into inactive ones.
        variation=state-lift
    else:
        variation=full.basis@(full.basis.T@state)
    nc=q.shape[0]+(0 if p is None else p.shape[0])
    shape=(nc,*q.shape[1:])
    initial_scaled=(first.state/scales).reshape(shape)
    roundoff_scale=np.maximum(np.linalg.norm(initial_scaled,axis=(1,2)),
                              np.linalg.norm(lift.reshape(shape),axis=(1,2)))
    x,y=_initial_profiles(variation.reshape(shape),mvi_settings.rank,roundoff_scale)

    def fields():
        v=lift+reconstruct_profiles(x,y).ravel()
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

    def projected_norm(v,r):
        norms=[]
        for axis in ("X","Y"):
            xx,yy=_normalize_partner(x,y,axis)
            directions=separated_profile_directions(xx,yy,axis)
            space=affine_trace_space(c,target,v,directions=directions)
            norms.append(float(np.max(abs(space.basis.T@(scales*r)/energy_scale),initial=0)))
        return max(norms)

    history=[];iterations=0;sweeps=0;status="MAX_SWEEPS";converged=False
    v,qm,pm,energy,r,g=evaluate()
    # For an exact zero separated component, the original constant partner
    # cannot see a zero-mean circumferential load. Align its *inactive* Y
    # directions with the admissible residual. No field or rank changes.
    admissible=(full.basis@g).reshape(shape)
    x,y=seed_inactive_mvi_partners(x,y,admissible)
    v,qm,pm,energy,r,g=evaluate()
    initial=float(np.max(abs(g),initial=0))
    tolerance=settings.absolute_tolerance_scaled+settings.relative_tolerance*initial
    projected=projected_norm(v,r)

    def record_sweep(sweep):
        row={"event":"sweep","sweep":sweep,"iteration":iterations,
             "residual_scaled":float(np.max(abs(g),initial=0)),
             "residual_relative_to_initial":float(np.max(abs(g),initial=0))/initial if initial else 0.,
             "projected_residual_scaled":projected,"potential_j":energy,
             "constraint_defect_scaled":float(np.max(abs(full.defect(v)),initial=0))}
        history.append(row)
        if on_iteration is not None:on_iteration(dict(row))

    record_sweep(0)
    if initial<=tolerance:converged=True;status="CONVERGED"
    for sweep in range(1,mvi_settings.max_sweeps+1):
        if converged:break
        failed=False
        for axis in ("X","Y"):
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
        projected=projected_norm(v,r)
        record_sweep(sweep)
        if failed:break
        if np.max(abs(g),initial=0)<=tolerance:
            converged=True;status="CONVERGED";break
        if projected<=tolerance:
            status="PROJECTED_ONLY_FULL_RESIDUAL_NOT_CONVERGED";break
    defect=float(np.max(abs(full.defect(v)),initial=0))
    if defect>1.e-9:converged=False;status="BOUNDARY_CONSTRAINT_DRIFT"
    scaled_r=scales*r
    reaction=(scaled_r-full.basis@(full.basis.T@scaled_r))/scales
    norm=float(np.max(abs(g),initial=0))
    return StaticMVIResult(converged,status,qm,pm,energy,r.copy(),reaction,norm,
        norm/initial if initial else 0.,defect,tuple(history),iterations,
        x.copy(),y.copy(),lift.reshape(shape),scales.reshape(shape),sweeps,
        projected,projected<=tolerance)


__all__=("StaticMVISettings","StaticMVIResult","solve_static_mvi","reconstruct_profiles")
