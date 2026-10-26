"""Fixed-rank matrix-free X/Y Newton-Krylov sweeps for coupled weak residual.

Evaluator returns (potential, full admissible scaled residual) of the SAME
variational physical model as full FDM. Never modify rank, drop electric
components, apply a full-grid Newton correction, use SVD or regularize.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from cylinder_mvi_direction_space import (
    SeparatedMVIDirections,SparseHalfstepTraceProjector)
from cylinder_mvi_cold_start import seed_inactive_mvi_partners

@dataclass(frozen=True)
class SweepResult:
    x_profiles: np.ndarray
    y_profiles: np.ndarray
    state_scaled: np.ndarray
    full_residual_scaled: np.ndarray
    converged: bool
    status: str
    sweeps: int
    iterations: int
    projected_residual_scaled: float
    history: tuple

def separated_krylov_sweeps(x,y,lift_scaled,trace_matrix,evaluator,*,
        relative_tolerance=1.e-7,absolute_tolerance=1.e-12,
        tangent_step=2.e-6,max_sweeps=30,
        max_halfstep_iterations=15,max_halvings=16,
        gmres_rtol=3.e-2,gmres_restart=40,
        gmres_max_matvecs=180,on_iteration=None):
    x=np.asarray(x,float).copy();y=np.asarray(y,float).copy()
    if (x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=y.shape[:2]
            or not 1<=x.shape[1]<=8 or x.shape[1]>min(x.shape[2],y.shape[2])
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("MVI_KRYLOV_INVALID_FIXED_RANK_FACTORS")
    shape=(x.shape[0],x.shape[2],y.shape[2])
    lift=np.asarray(lift_scaled,float)
    if lift.shape!=shape or not np.isfinite(lift).all():
        raise ValueError("MVI_KRYLOV_INVALID_LIFT")
    if any(not math.isfinite(float(a)) or float(a)<=0 for a in (
            relative_tolerance,tangent_step,gmres_rtol)):
        raise ValueError("MVI_KRYLOV_INVALID_POSITIVE_SETTINGS")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance<0:
        raise ValueError("MVI_KRYLOV_INVALID_ABSOLUTE_TOLERANCE")
    if (not 1<=max_sweeps or not 1<=max_halfstep_iterations
            or not 0<=max_halvings or not 1<=gmres_restart
            or not 1<=gmres_max_matvecs or not 0<gmres_rtol<1):
        raise ValueError("MVI_KRYLOV_INVALID_ITERATION_LIMITS")
    n=int(np.prod(shape))

    def state():
        return (lift+np.einsum("crx,crs->cxs",x,y,optimize=True)).ravel()

    def physics(v):
        energy,g=evaluator(np.asarray(v,float))
        g=np.asarray(g,float)
        if g.shape!=(n,) or not np.isfinite(g).all():
            raise ValueError("MVI_KRYLOV_NONFINITE_FULL_ADMISSIBLE_RESIDUAL")
        return energy,g

    def normalized(axis):
        xx,yy=x.copy(),y.copy()
        for c in range(shape[0]):
            if axis=="X":
                q,r=np.linalg.qr(yy[c].T,mode="reduced")
                yy[c]=q.T;xx[c]=r@xx[c]
            else:
                q,r=np.linalg.qr(xx[c].T,mode="reduced")
                xx[c]=q.T;yy[c]=r@yy[c]
        return xx,yy

    def make_direction(axis,xx,yy):
        op=SeparatedMVIDirections(xx,yy,axis)
        return op,SparseHalfstepTraceProjector(trace_matrix,op)

    def projected_norm(full_g):
        values=[]
        for axis in ("X","Y"):
            xx,yy=normalized(axis)
            op,pj=make_direction(axis,xx,yy)
            reduced=pj.project(op.pull(full_g))
            values.append(float(np.max(np.abs(op.push(reduced)),initial=0)))
        return max(values,default=0.)

    history=[];iterations=0
    def report(row):
        history.append(row)
        if on_iteration is not None:on_iteration(dict(row))

    v=state()
    _,g=physics(v)
    x,y=seed_inactive_mvi_partners(x,y,g.reshape(shape))
    v=state();_,g=physics(v)
    initial=float(np.max(np.abs(g),initial=0))
    tolerance=absolute_tolerance+relative_tolerance*initial
    projected=projected_norm(g)
    status="MAX_SWEEPS_FULL_RESIDUAL_NOT_CONVERGED"
    converged=initial<=tolerance
    if converged:status="CONVERGED"
    report(dict(event="sweep",iteration=0,sweep=0,
                residual_scaled=initial,projected_residual_scaled=projected))
    sweeps=0
    for sweep in range(1,max_sweeps+1):
        if converged:break
        before=v.copy()
        failed=False
        for axis in ("X","Y"):
            x,y=normalized(axis)
            op,pj=make_direction(axis,x,y)
            v=state()
            _,g=physics(v)
            gparam=pj.project(op.pull(g))
            norm0=float(np.max(np.abs(op.push(gparam)),initial=0))
            half_tolerance=absolute_tolerance+relative_tolerance*norm0
            inner_status="MAX_ITERATIONS"
            for it in range(max_halfstep_iterations+1):
                current=float(np.max(np.abs(op.push(gparam)),initial=0))
                report(dict(event="halfstep",iteration=iterations,
                    sweep=sweep,direction=axis,halfstep_iteration=it,
                    residual_scaled=current,
                    full_residual_scaled=float(np.max(np.abs(g),initial=0))))
                if current<=half_tolerance:
                    inner_status="CONVERGED";break
                if it==max_halfstep_iterations:break
                matvec_count=[0]
                def tangent_action(vec):
                    matvec_count[0]+=1
                    active=pj.project(vec)
                    normal=vec-active
                    length=float(np.linalg.norm(active))
                    if length<1.e-30:return normal
                    eps=tangent_step/length
                    dv=op.push(active)
                    for _ in range(6):
                        try:
                            _,gp=physics(v+eps*dv)
                            _,gm=physics(v-eps*dv)
                            break
                        except ValueError as exc:
                            if "INVALID_" not in str(exc):raise
                            eps*=.5
                    else:
                        raise ValueError("INVALID_TANGENT_PERTURBATION")
                    return pj.project(op.pull((gp-gm)/(2*eps)))+normal

                linear=LinearOperator((op.n_parameters,op.n_parameters),
                                      matvec=tangent_action,dtype=float)
                try:
                    correction,info=gmres(linear,-gparam,rtol=gmres_rtol,
                        atol=0.,restart=gmres_restart,
                        maxiter=gmres_max_matvecs,callback_type="legacy")
                except ValueError as exc:
                    if "INVALID_" not in str(exc):raise
                    inner_status="INVALID_TANGENT_PERTURBATION";break
                if matvec_count[0]>gmres_max_matvecs:
                    inner_status="KRYLOV_MATVEC_BUDGET_EXCEEDED";break
                if info!=0 or not np.isfinite(correction).all():
                    inner_status="KRYLOV_LINEAR_NOT_CONVERGED";break
                correction=pj.project(correction)
                for h in range(max_halvings+1):
                    alpha=2.**(-h)
                    trial=v+alpha*op.push(correction)
                    try: _,gt=physics(trial)
                    except ValueError as exc:
                        if "INVALID_" not in str(exc):raise
                        continue
                    reduced=pj.project(op.pull(gt))
                    new_norm=float(np.max(np.abs(op.push(reduced)),initial=0))
                    if new_norm<=(1.-1.e-4*alpha)*current or new_norm<=half_tolerance:
                        delta=alpha*correction.reshape(op.parameter_shape)
                        if axis=="X":x+=delta
                        else:y+=delta
                        v=state();_,g=physics(v)
                        gparam=pj.project(op.pull(g))
                        iterations+=1
                        break
                else:
                    inner_status="HALFSTEP_RESIDUAL_LINE_SEARCH_FAILED"
                    break
            if inner_status!="CONVERGED":
                status="HALFSTEP_"+axis+"_"+inner_status
                failed=True;break
        sweeps=sweep
        v=state();_,g=physics(v)
        norm=float(np.max(np.abs(g),initial=0))
        projected=projected_norm(g)
        report(dict(event="sweep",iteration=iterations,sweep=sweep,
                    residual_scaled=norm,
                    projected_residual_scaled=projected,
                    sweep_field_increment_scaled=float(
                        np.max(np.abs(v-before),initial=0))))
        if failed:break
        if norm<=tolerance:
            status="CONVERGED";converged=True;break
    return SweepResult(x.copy(),y.copy(),state(),g.copy(),converged,
                       status,sweeps,iterations,projected,tuple(history))

__all__=("SweepResult","separated_krylov_sweeps")
