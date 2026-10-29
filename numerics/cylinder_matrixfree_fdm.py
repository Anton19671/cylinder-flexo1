#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in sparse-trace, matrix-free Newton--Krylov FULL FDM shell solve.

This is an independent *full-nodal* nonlinear route. It never substitutes
MVI, drops G3D/flexoelectric terms, changes a physical boundary condition,
or builds the full n_free x n_free tangent. SciPy GMRES acts on centered
finite-difference Jacobian-vector products of the EXACT assembled weak
residual; the electric saddle variables remain in the same Krylov system.

All affine traces (including an insulated electric gauge) are stored as
sparse rows. A sparse factorization of the much smaller Gram matrix C C.T
eliminates constrained directions and supplies a minimum-norm lift:

    q_dimensionless = lift + P z,    P = I - C.T (C C.T)^(-1) C.

No physical tangent regularization or artificial electric penalty is
introduced. Dependent or ill-conditioned constraints fail closed.
The Newton step solves
    [P K P + (I-P)] delta = -P g
with line search on ||P g||_inf, where g is the scaled FULL residual.
This is an explicit experimental alternative to the dense FDM solver.
Its large-grid computational efficiency, preconditioning, physical
convergence and boundary accuracy have NOT been established.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

try:
    from scipy import sparse
    from scipy.sparse import linalg as sla
except ImportError as exc:
    raise ImportError(
        "Optional full FDM Newton-Krylov route requires scipy>=1.12") from exc

from cylinder_functional_assembly import _basis_moments
from cylinder_variational_boundary import compile_boundary_program
from cylinder_constraint_row import combined_constraint_entries
from cylinder_variational_fdm import StaticFDMResult, state_scales


@dataclass(frozen=True)
class MatrixFreeNewtonSettings:
    relative_tolerance: float = 1.e-7
    absolute_tolerance_scaled: float = 1.e-12
    tangent_step_scaled: float = 2.e-6
    max_iterations: int = 15
    max_halvings: int = 16
    potential_scale_v: float = 1.
    krylov_rtol: float = 3.e-2
    krylov_restart: int = 40
    krylov_max_matvecs: int = 180
    krylov_preconditioner: str = "NONE"
    ilu_drop_tolerance: float = 1.e-3
    ilu_fill_factor: float = 8.
    ilu_max_surface_nodes: int = 8192
    max_total_dofs: int = 100_000
    max_surface_nodes: int = 8192
    max_constraint_rows: int = 20_000

    def __post_init__(self):
        positive=("relative_tolerance","tangent_step_scaled",
                  "potential_scale_v","krylov_rtol",
                  "ilu_drop_tolerance","ilu_fill_factor")
        finite=("absolute_tolerance_scaled",)+positive
        for name in finite:
            raw=getattr(self,name)
            if isinstance(raw,(bool,np.bool_)):
                raise ValueError(name+": boolean not permitted")
            try:value=float(raw)
            except (TypeError,ValueError) as exc:
                raise ValueError(name+": finite number required") from exc
            if not math.isfinite(value) or value<0 or (name in positive and value==0):
                raise ValueError(name+": invalid sign/nonfinite")
            object.__setattr__(self,name,value)
        if self.krylov_preconditioner not in ("NONE","PHYSICS_BLOCK_ILU"):
            raise ValueError("krylov_preconditioner must be NONE or PHYSICS_BLOCK_ILU")
        if self.ilu_drop_tolerance>=1. or self.ilu_fill_factor<1.:
            raise ValueError("invalid ILU drop tolerance or fill factor")
        if self.krylov_rtol>=1.:
            raise ValueError("krylov_rtol must be strictly less than one")
        for name in ("max_iterations","max_halvings","krylov_restart",
                     "krylov_max_matvecs","max_total_dofs","max_surface_nodes",
                     "max_constraint_rows",
                     "ilu_max_surface_nodes"):
            val=getattr(self,name)
            if isinstance(val,(bool,np.bool_)) or not isinstance(val,int) or val<1:
                raise ValueError(name+": positive integer required")


class _KrylovMatvecBudgetExceeded(RuntimeError):
    """Hard Jv budget reached; never interpret as a successful solve."""


class SparseTraceProjector:
    """Sparse normalized C*u=d with full-row-rank Gram factorization.

    The projector only uses boundary constraints and a gauge, not
    the physical elastic or electric tangent. The matrix C acts on the
    dimensionless state u = physical_state/scales.
    """

    def __init__(self, matrix, target, *, max_rows):
        c=sparse.csr_matrix(matrix,dtype=float)
        d=np.asarray(target,dtype=float)
        if (c.ndim!=2 or c.shape[1]<1 or c.shape[0]>max_rows or
                d.shape!=(c.shape[0],) or not np.isfinite(d).all() or
                not np.isfinite(c.data).all()):
            raise ValueError("SPARSE_TRACE_INVALID_SHAPE")
        self.size=c.shape[1]
        self.count=c.shape[0]
        if not self.count:
            self.c=c
            self.target=d
            self._lu=None
            self.lift=np.zeros(self.size)
            return
        norms=np.sqrt(np.asarray(c.multiply(c).sum(axis=1)).reshape(-1))
        if not np.isfinite(norms).all() or np.any(norms<=0):
            raise ValueError("SPARSE_TRACE_ZERO_CONSTRAINT_ROW")
        self.c=(sparse.diags(1./norms)@c).tocsr()
        self.target=d/norms
        gram=(self.c@self.c.T).tocsc()
        try:self._lu=sla.splu(gram)
        except RuntimeError as exc:
            raise ValueError("SPARSE_TRACE_DEPENDENT_CONSTRAINTS") from exc
        pivots=np.abs(self._lu.U.diagonal())
        if (not len(pivots) or not np.isfinite(pivots).all() or
            min(pivots)<=1.e-12*max(pivots)):
            raise ValueError("SPARSE_TRACE_ILL_CONDITIONED_CONSTRAINTS")
        self.lift=np.asarray(self.c.T@self._lu.solve(self.target)).reshape(-1)
        if np.max(np.abs(self.defect(self.lift)))>2.e-10*max(
                1.,float(np.max(np.abs(self.target),initial=0))):
            raise ValueError("SPARSE_TRACE_UNREACHABLE_PRESCRIPTION")

    @classmethod
    def from_problem(cls,problem,q,p,scales,*,time_s,max_rows):
        functional=problem.functional
        edge=compile_boundary_program(
            functional,q,p,spec=problem.spec,
            calculation_kind=problem.kind,time_s=time_s,
            check_charge_balance=False)
        n=np.size(q)+(0 if p is None else np.size(p))
        triplet_i=[];triplet_j=[];triplet_v=[];targets=[]
        for row in edge.constraints:
            offset=0 if row.field=="mechanical" else np.size(q)
            size=np.size(q) if row.field=="mechanical" else np.size(p)
            indexes,weights=combined_constraint_entries(row.indices,row.weights,size,offset=offset)
            for j,weight in zip(indexes,weights):
                triplet_i.append(len(targets))
                triplet_j.append(int(j))
                triplet_v.append(float(weight)*float(scales[j]))
            targets.append(float(row.prescribed))
        if edge.gauge_required:
            if p is None:
                raise ValueError("SPARSE_TRACE_ELECTRICAL_GAUGE_WITHOUT_FIELD")
            moments=_basis_moments(len(p)-1,functional.R,functional.h)
            gauge=(moments[:,None,None]*functional.W[None,...]).ravel()
            for local,value in enumerate(gauge):
                if value:
                    j=np.size(q)+local
                    triplet_i.append(len(targets))
                    triplet_j.append(j)
                    triplet_v.append(float(value)*float(scales[j]))
            targets.append(0.)
        c=sparse.coo_matrix(
            (triplet_v,(triplet_i,triplet_j)),
            shape=(len(targets),n)).tocsr()
        return cls(c,np.asarray(targets,float),max_rows=max_rows)

    def project_null(self,value):
        v=np.asarray(value,float)
        if v.shape!=(self.size,) or not np.isfinite(v).all():
            raise ValueError("SPARSE_TRACE_NONFINITE_STATE")
        if not self.count:return v.copy()
        return v-np.asarray(self.c.T@self._lu.solve(self.c@v)).reshape(-1)

    def project_affine(self,value):
        v=np.asarray(value,float)
        if v.shape!=(self.size,) or not np.isfinite(v).all():
            raise ValueError("SPARSE_TRACE_NONFINITE_STATE")
        return self.lift+self.project_null(v-self.lift)

    def defect(self,value):
        v=np.asarray(value,float)
        if v.shape!=(self.size,) or not np.isfinite(v).all():
            raise ValueError("SPARSE_TRACE_NONFINITE_STATE")
        return np.asarray(self.c@v).reshape(-1)-self.target


def _solve(problem, q0, p0, *, time_s, dead_load, free_charge,
           settings, on_iteration, dynamic=None):
    if not isinstance(settings,MatrixFreeNewtonSettings):
        raise TypeError("matrix-free settings must be MatrixFreeNewtonSettings")
    f=problem.functional
    q=np.asarray(q0,float)
    p=None if p0 is None else np.asarray(p0,float)
    if (q.ndim!=3 or q.shape!=(f.n_mechanical,*f.ops.shape) or
            not np.isfinite(q).all() or
            (p is not None and (p.ndim!=3 or p.shape[1:]!=f.ops.shape
                                or not np.isfinite(p).all()))):
        raise ValueError("matrix-free fields must match functional")
    n=q.size+(0 if p is None else p.size)
    if np.prod(f.ops.shape)>settings.max_surface_nodes:
        raise ValueError("MATRIX_FREE_SURFACE_NODE_SAFETY_LIMIT")
    if n>settings.max_total_dofs:
        raise ValueError("MATRIX_FREE_NODAL_DOF_SAFETY_LIMIT")
    scales=state_scales(f,q,p,settings)
    if np.any(scales<=0) or not np.isfinite(scales).all():
        raise ValueError("MATRIX_FREE_INVALID_FIELD_SCALING")
    projector=SparseTraceProjector.from_problem(
        problem,q,p,scales,time_s=time_s,
        max_rows=settings.max_constraint_rows)
    e_scale=float(f.material.young_pa*f.h*np.sum(f.W))
    if not math.isfinite(e_scale) or e_scale<=0:
        raise ValueError("MATRIX_FREE_INVALID_ENERGY_SCALE")
    qsize=q.size
    pshape=None if p is None else p.shape
    qshape=q.shape

    if dynamic is not None:
        q_pred=np.asarray(dynamic["q_predictor"],float)
        v_pred=np.asarray(dynamic["velocity_predictor"],float)
        if (q_pred.shape!=qshape or v_pred.shape!=qshape or
            not np.isfinite(q_pred).all() or not np.isfinite(v_pred).all()):
            raise ValueError("MATRIX_FREE_NEWMARK_PREDICTOR_SHAPE")
        dt=float(dynamic["dt_s"]);beta=float(dynamic["beta"])
        gamma=float(dynamic["gamma"]);rho=float(dynamic["density_kg_m3"])
        if not all(math.isfinite(v) for v in (dt,beta,gamma,rho)) or (
                dt<=0 or beta<=0 or gamma<=0 or rho<=0):
            raise ValueError("MATRIX_FREE_NEWMARK_PARAMETERS")
        eta=float(dynamic["eta_s"]);alpha=float(dynamic["mass_damping_rate_per_s"])
        if not all(math.isfinite(v) and v>=0 for v in (eta,alpha)):
            raise ValueError("MATRIX_FREE_DAMPING_PARAMETERS")

    def evaluate(z):
        state=scales*z
        qm=state[:qsize].reshape(qshape)
        pm=None if pshape is None else state[qsize:].reshape(pshape)
        va=aa=None
        parameters={}
        if dynamic is not None:
            aa=(qm-q_pred)/(beta*dt**2)
            va=v_pred+gamma*dt*aa
            parameters=dict(acceleration=aa,density_kg_m3=rho,
                velocity=va if (alpha>0 or eta>0 or f.theory=="KL") else None,
                eta_s=eta,mass_damping_rate_per_s=alpha)
        energy,r,_=problem.evaluate_effective_residual(
            qm,pm,dead_load=dead_load,free_charge=free_charge,
            time_s=time_s,**parameters)
        if not np.isfinite(r).all() or not math.isfinite(energy):
            raise ValueError("MATRIX_FREE_NONFINITE_PHYSICAL_RESIDUAL")
        projected=projector.project_null(scales*r)/e_scale
        return float(energy),r,projected,qm,pm,va,aa

    initial_state=q.ravel() if p is None else np.concatenate(
        (q.ravel(),p.ravel()))
    z=projector.project_affine(initial_state/scales)
    data=evaluate(z)
    initial=float(np.max(np.abs(data[2]),initial=0))
    tol=settings.absolute_tolerance_scaled+settings.relative_tolerance*initial
    preconditioner=None
    if settings.krylov_preconditioner=="PHYSICS_BLOCK_ILU":
        from cylinder_krylov_block_preconditioner import PhysicsBlockILU
        preconditioner=PhysicsBlockILU(
            f,scales,e_scale,projector,
            drop_tolerance=settings.ilu_drop_tolerance,
            fill_factor=settings.ilu_fill_factor,
            max_surface_nodes=settings.ilu_max_surface_nodes,
            dynamic=dynamic)
    history=[]
    iterations=0
    converged=False
    status="MAX_ITERATIONS"
    for index in range(settings.max_iterations+1):
        g=data[2]
        norm=float(np.max(np.abs(g),initial=0))
        rec={"iteration":index,"residual_scaled":norm,
             "residual_relative_to_initial":norm/initial if initial else 0.,
             "linear_solver":"MATRIX_FREE_GMRES",
             "linear_preconditioner":settings.krylov_preconditioner,
             "constraint_defect_scaled":float(np.max(
                 np.abs(projector.defect(z)),initial=0))}
        history.append(rec)
        if on_iteration is not None:on_iteration(dict(rec))
        if norm<=tol:
            converged=True
            status="CONVERGED"
            break
        if index==settings.max_iterations:
            break
        tangent_count=[0]

        def action(x):
            if tangent_count[0]>=settings.krylov_max_matvecs:
                raise _KrylovMatvecBudgetExceeded("KRYLOV_MATVEC_BUDGET_EXCEEDED")
            tangent_count[0]+=1
            xp=projector.project_null(x)
            magnitude=float(np.linalg.norm(xp))
            # Identity on normal directions ensures that the GMRES
            # system is nonsingular even though P K P vanishes there.
            normal=x-xp
            if magnitude<1.e-30:
                return normal
            eps=settings.tangent_step_scaled/magnitude
            for _ in range(6):
                try:
                    gp=evaluate(z+eps*xp)[2]
                    gm=evaluate(z-eps*xp)[2]
                    break
                except ValueError as exc:
                    if "INVALID_" not in str(exc):raise
                    eps*=.5
            else:
                raise ValueError("INVALID_TANGENT_PERTURBATION")
            return projector.project_null((gp-gm)/(2*eps))+normal

        linear=sla.LinearOperator((n,n),matvec=action,dtype=float)
        linear_history=[]
        try:
            correction,info=sla.gmres(
                linear,-g,rtol=settings.krylov_rtol,atol=0.,
                M=None if preconditioner is None else preconditioner.linear_operator(),
                restart=settings.krylov_restart,
                # With pr_norm, maxiter counts restart cycles, NOT Jv calls.
                # The action guard above enforces the exact physical Jv cap.
                maxiter=max(1,math.ceil(settings.krylov_max_matvecs/settings.krylov_restart)),
                callback=lambda residual:linear_history.append(float(residual)),
                callback_type="pr_norm")
        except _KrylovMatvecBudgetExceeded:
            rec["krylov_matvecs"]=tangent_count[0]
            rec["krylov_iterations"]=len(linear_history)
            rec["krylov_last_preconditioned_relative_residual"]=(
                linear_history[-1] if linear_history else None)
            status="KRYLOV_MATVEC_BUDGET_EXCEEDED"
            break
        except ValueError as exc:
            if "INVALID_" not in str(exc):raise
            status="INVALID_TANGENT_PERTURBATION";break
        rec["krylov_matvecs"]=tangent_count[0]
        rec["krylov_iterations"]=len(linear_history)
        rec["krylov_last_preconditioned_relative_residual"]=(
            linear_history[-1] if linear_history else None)
        if info!=0 or not np.isfinite(correction).all():
            status="KRYLOV_LINEAR_SOLVE_NOT_CONVERGED"
            break
        correction=projector.project_null(correction)
        try:
            true_linear=float(np.linalg.norm(action(correction)+g))
        except _KrylovMatvecBudgetExceeded:
            rec["krylov_matvecs"]=tangent_count[0]
            status="KRYLOV_MATVEC_BUDGET_EXCEEDED"
            break
        rec["krylov_matvecs"]=tangent_count[0]
        rec["krylov_true_residual_scaled"]=true_linear
        rec["krylov_last_relative_residual"]=(
            true_linear/max(float(np.linalg.norm(g)),np.finfo(float).tiny))
        if true_linear>max(1.e-11,2.*settings.krylov_rtol*float(np.linalg.norm(g))):
            status="KRYLOV_TRUE_RESIDUAL_TOO_LARGE";break
        rec["increment_scaled"]=float(np.max(np.abs(correction),initial=0))
        merit=norm
        for halving in range(settings.max_halvings+1):
            factor=2.**(-halving)
            trial=projector.project_affine(z+factor*correction)
            try: candidate=evaluate(trial)
            except ValueError as exc:
                if "INVALID_" not in str(exc):raise
                continue
            next_norm=float(np.max(np.abs(candidate[2]),initial=0))
            if next_norm<=(1.-1.e-4*factor)*merit or next_norm<=tol:
                rec["step_factor"]=factor
                z,data=trial,candidate
                iterations+=1
                break
        else:
            status="RESIDUAL_LINE_SEARCH_FAILED";break
    energy,residual,gradient,qm,pm,velocity,acceleration=data
    defect=float(np.max(np.abs(projector.defect(z)),initial=0))
    if defect>1.e-9:
        converged=False
        status="BOUNDARY_CONSTRAINT_DRIFT"
    # Euclidean normal component of the full dimensionless physical
    # gradient defines the actual reactions, not the reduced GMRES RHS.
    sr=scales*residual
    reaction=(sr-projector.project_null(sr))/scales
    norm=float(np.max(np.abs(gradient),initial=0))
    return dict(
        converged=converged,status=status,mechanical=qm,electrical=pm,
        total_potential_j=energy,full_residual=residual.copy(),
        constraint_reaction=reaction,
        admissible_residual_scaled=norm,
        residual_relative_to_initial=norm/initial if initial else 0.,
        constraint_defect_scaled=defect,history=tuple(history),
        iterations=iterations,velocity=velocity,acceleration=acceleration)


def solve_matrixfree_static_fdm(problem,mechanical,electrical=None,*,
                                dead_load=None,free_charge=None,
                                settings=None,on_iteration=None):
    """FULL spatial FDM Newton without dense tangent or dense null basis."""
    if problem.kind!="STATIC":
        raise ValueError("matrix-free static solver requires STATIC")
    settings=MatrixFreeNewtonSettings() if settings is None else settings
    data=_solve(problem,mechanical,electrical,time_s=None,
                dead_load=dead_load,free_charge=free_charge,
                settings=settings,on_iteration=on_iteration)
    return StaticFDMResult(
        data["converged"],data["status"],data["mechanical"],
        data["electrical"],data["total_potential_j"],data["full_residual"],
        data["constraint_reaction"],data["admissible_residual_scaled"],
        data["residual_relative_to_initial"],data["constraint_defect_scaled"],
        data["history"],data["iterations"])


def solve_matrixfree_dynamic_fdm(problem,mechanical,electrical=None,*,
                                 q_predictor,velocity_predictor,time_s,
                                 dt_s,beta,gamma,density_kg_m3,
                                 dead_load=None,free_charge=None,
                                 mass_damping_rate_per_s=0.,eta_s=0.,
                                 settings=None,on_iteration=None):
    """FULL Newmark FDM step with exact saddle coupling and sparse traces.

    The caller still owns step acceptance, full Newmark residual,
    velocity/acceleration trace tests and persistence of accepted states.
    """
    if problem.kind!="DYNAMIC":
        raise ValueError("matrix-free dynamic solver requires DYNAMIC")
    settings=MatrixFreeNewtonSettings() if settings is None else settings
    arguments=dict(q_predictor=q_predictor,
        velocity_predictor=velocity_predictor,dt_s=dt_s,beta=beta,gamma=gamma,
        density_kg_m3=density_kg_m3,eta_s=eta_s,
        mass_damping_rate_per_s=mass_damping_rate_per_s)
    data=_solve(problem,mechanical,electrical,time_s=time_s,
                dead_load=dead_load,free_charge=free_charge,dynamic=arguments,
                settings=settings,on_iteration=on_iteration)
    from cylinder_variational_dynamic import DynamicSpatialResult
    return DynamicSpatialResult(
        data["converged"],data["status"],data["mechanical"],
        data["electrical"],data["velocity"],data["acceleration"],
        data["full_residual"],data["constraint_reaction"],
        data["admissible_residual_scaled"],
        data["residual_relative_to_initial"],data["constraint_defect_scaled"],
        data["history"],data["iterations"])


__all__=("MatrixFreeNewtonSettings","SparseTraceProjector",
         "solve_matrixfree_static_fdm","solve_matrixfree_dynamic_fdm")
