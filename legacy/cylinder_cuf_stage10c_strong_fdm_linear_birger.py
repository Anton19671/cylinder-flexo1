#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10C: independent full-grid STRONG-FORM CUF FDM,
LINEAR GEOMETRY + BIRGER MATERIAL.

Frozen GB inner problem:
    q[component,tau,s,x] full nodal unknowns,
    linear CUF kinematics,
    frozen plane 3D Birger/StVK secant material through Stage7F,
    strong Stage7A residual,
    Stage7C end BC,
    one global np.linalg.solve.

Outer Birger:
    reuses verified Stage5A solve_birger_outer_cycle unchanged.

No MVI, rank, X/Y profiles, weak projection, FEM, zones,
SVD/pinv/lstsq/QR/Cholesky, regularization, retry, line-search,
or automatic load subdivision.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import final_numeric_scaling as nscale
import final_residual_scaling as rscale
import final_v3_fd as v3fd
import final_k2_coloring as k2

import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7c_general_boundary as g9
import cylinder_cuf_stage7f_general_linear as gl
import cylinder_cuf_stage10a_strong_fdm_linear_elastic as fdl


class CUFStrongFDMLinearBirgerError(RuntimeError):
    pass


class CUFStrongFDMLinearBirgerSystemSingular(CUFStrongFDMLinearBirgerError):
    pass


@dataclass
class CUFFrozenGBLinearInnerResult:
    state: np.ndarray
    converged: bool
    status: str
    residual_inf: float
    residual_inf_by_block: dict | None = None


@dataclass
class CUFStrongFDMLinearBirgerCounter:
    calls: int = 0


def linear_frozen_gb_fullgrid_residual(
    q,ops,basis,bc,*,K0,GB,quadrature,generalized_load
):
    q=fdl._validate_q(q,ops,basis)
    if bc.m!=basis.m:
        raise ValueError("basis/BC m mismatch")
    bc.check_rigid_mode_requirement()

    p=np.asarray(generalized_load,float)
    if p.shape!=q.shape:
        raise ValueError(f"generalized_load must have shape {q.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    qx,qs=fdl.surface_first_derivatives(q,ops,basis)

    resultants=gl.general_linear_thickness_resultants(
        q,qx,qs,
        R=ops.grid.radius,
        K0=float(K0),
        GB=GB,
        quadrature=quadrature,
        basis=basis,
    )

    raw=g7.general_residuals(
        resultants,
        Dx=ops.Dx,
        Ds=ops.Ds,
        generalized_load=p,
    )

    assembled=g9.assemble_strong_residual_with_end_bcs(
        raw,q,resultants,bc,reject_fully_free=True
    )

    return assembled.reshape(-1),raw,resultants,qx,qs


def _assemble_frozen_gb_linear_system_dense_reference(
    ops,basis,bc,*,K0,GB,quadrature,generalized_load
):
    """Pre-K2 one-column frozen-GB assembly retained for self-checks."""
    n=3*basis.m*ops.grid.ns*ops.grid.nx
    q0=np.zeros((3,basis.m,ops.grid.ns,ops.grid.nx),float)
    r0,*_=linear_frozen_gb_fullgrid_residual(
        q0,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load,
    )
    A=np.empty((n,n),float)
    for j in range(n):
        e=np.zeros(n,float); e[j]=1.0
        qj=fdl.unflatten_q(e,ops,basis)
        rj,*_=linear_frozen_gb_fullgrid_residual(
            qj,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,
        )
        A[:,j]=rj-r0
    return A,r0


def assemble_frozen_gb_linear_system(
    ops,basis,bc,*,K0,GB,quadrature,generalized_load
):
    q0=np.zeros((3,basis.m,ops.grid.ns,ops.grid.nx),float)
    r0,*_=linear_frozen_gb_fullgrid_residual(
        q0,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load,
    )

    def evaluate(q):
        r,*_=linear_frozen_gb_fullgrid_residual(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,
        )
        return r

    A,_,stats=k2.assemble_cuf_colored_columns(
        q0,ops,evaluate,base_residual=r0,step=1.0,dependency_layers=2
    )
    assemble_frozen_gb_linear_system.last_k2_stats=stats.to_dict()
    return A,r0


assemble_frozen_gb_linear_system.last_k2_stats=None


def solve_frozen_gb_linear_strong_fdm(
    initial_q,ops,basis,bc,*,E0,K0,GB,quadrature,generalized_load
):
    # initial_q is part of the generic Birger interface.
    # The frozen LINEAR problem is solved directly and does not use the guess.
    fdl._validate_q(initial_q,ops,basis)

    A,r0=assemble_frozen_gb_linear_system(
        ops,basis,bc,
        K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load,
    )

    if np.max(np.abs(r0))==0.0:
        q=np.zeros((3,basis.m,ops.grid.ns,ops.grid.nx),float)
    else:
        try:
            x,_,_,_=nscale.equilibrated_solve(A,-r0)
        except np.linalg.LinAlgError as exc:
            raise CUFStrongFDMLinearBirgerSystemSingular(
                "SINGULAR_CUF_STRONG_FDM_LINEAR_BIRGER_SYSTEM"
            ) from exc
        q=fdl.unflatten_q(x,ops,basis)

    R,*_=linear_frozen_gb_fullgrid_residual(
        q,ops,basis,bc,
        K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load,
    )

    h_v4=v3fd.thickness_from_quadrature(quadrature)
    Rm=rscale.cuf_scaled_residual_for_measure(
        R,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,bc=bc,E0=E0,h=h_v4,
    )
    return CUFFrozenGBLinearInnerResult(
        state=q,
        converged=True,
        status="CONVERGED_CUF_STRONG_FDM_FROZEN_GB_LINEAR",
        residual_inf=float(np.max(np.abs(Rm))),
        residual_inf_by_block=rscale.cuf_residual_inf_by_block(
            Rm,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
        ),
    )


def evaluate_linear_material_fullgrid(
    q,ops,basis,*,K0,GB,quadrature
):
    """K2b batched material-point evaluation for linear CUF FDM."""
    q=fdl._validate_q(q,ops,basis)
    qx,qs=fdl.surface_first_derivatives(q,ops,basis)
    surf=(ops.grid.ns,ops.grid.nx)
    nz=quadrature.z.size
    eps=np.empty((nz,)+surf,float)
    sig=np.empty_like(eps)
    det=np.ones_like(eps)
    GBa,mode=g7._gb_mode(GB,nz,surf)
    R=float(ops.grid.radius); K0=float(K0)

    for g,z in enumerate(quadrature.z):
        if mode=="scalar": GBg=float(GBa)
        elif mode=="surface": GBg=GBa
        else: GBg=GBa[g]
        _H,_Ft,_Fp,abc=g7._batched_general_gradient_quantities(q,qx,qs,z,R,basis)
        E=g7._batched_linear_strain_from_abc(abc)
        S=g7._batched_birger_second_piola(E,K0,GBg)
        eps[g]=g7._batched_strain_intensity(E)
        sig[g]=g7._batched_stress_intensity(S)
    return b5.MaterialPointState(eps,sig,det)


def make_inner_solver(
    ops,basis,bc,*,E0,K0,quadrature,generalized_load,counter=None
):
    if counter is None:
        counter=CUFStrongFDMLinearBirgerCounter()

    def inner(state_guess,GB):
        counter.calls+=1
        result=solve_frozen_gb_linear_strong_fdm(
            state_guess,ops,basis,bc,
            E0=E0,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,
        )
        inner.last_result=result
        return result

    inner.counter=counter
    inner.last_result=None
    return inner


def make_material_evaluator(
    ops,basis,*,K0,quadrature
):
    def evaluator(q,GB):
        return evaluate_linear_material_fullgrid(
            q,ops,basis,K0=K0,GB=GB,quadrature=quadrature
        )
    return evaluator


def solve_linear_birger_strong_fdm(
    initial_q,initial_GB,
    *,
    ops,basis,bc,E0,nu0,quadrature,generalized_load,
    material_diagram,birger_settings,
    previous_accepted_epsilon_i=None,
    inner_counter=None,
):
    initial_q=fdl._validate_q(initial_q,ops,basis)
    ec=gl.lk.ElasticConstants(float(E0),float(nu0))

    expected=(quadrature.z.size,ops.grid.ns,ops.grid.nx)
    GB0=np.asarray(initial_GB,float)
    if GB0.shape!=expected:
        raise ValueError(f"initial_GB must have shape {expected}")
    if np.any(~np.isfinite(GB0)) or np.any(GB0<=0.0):
        raise ValueError("initial_GB must be finite and positive")

    inner=make_inner_solver(
        ops,basis,bc,
        E0=E0,K0=ec.K0,quadrature=quadrature,
        generalized_load=generalized_load,
        counter=inner_counter,
    )
    evaluator=make_material_evaluator(
        ops,basis,K0=ec.K0,quadrature=quadrature
    )

    result=b5.solve_birger_outer_cycle(
        initial_q,GB0,
        G0=ec.G0,
        material_diagram=material_diagram,
        inner_solve=inner,
        evaluate_material=evaluator,
        state_vector=lambda st: np.asarray(st,float).reshape(-1),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
    )
    # Reviewer V-4: expose the already-computed final frozen-GB block norms
    # for STEP_XXXX.json.  This is diagnostics only; no constitutive/mechanics
    # quantity is fed back into the Birger iteration.
    if inner.last_result is not None:
        result.residual_inf_by_block=inner.last_result.residual_inf_by_block
    return result,inner.counter
