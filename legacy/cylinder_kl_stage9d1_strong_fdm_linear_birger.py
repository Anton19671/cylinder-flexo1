#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 9D-1: independent STRONG-FORM KL FDM, LINEAR BIRGER.

Frozen-GB inner problem:
- full nodal U(s,x), no MVI/rank/X/Y;
- LINEAR KL kinematics;
- plane-stress Birger resultants integrated directly through thickness;
- strong collocation equations and direct end mechanical BC;
- fixed-GB problem is linear and solved once with np.linalg.solve.

Outer Birger:
the already verified generic Stage5A solve_birger_outer_cycle is reused
without rewriting its algorithm.

KL compatibility:
MaterialPointState.detF is identically one, exactly as in the verified
Stage8C KL Birger adapter.  It is NOT a physical KL deformation Jacobian.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import final_numeric_scaling as nscale

import cylinder_te1_stage5a_birger as b5
import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8b1_boundary as kb
import cylinder_kl_stage8c1_birger_material as bm
import cylinder_kl_stage9b_strong_fdm_linear as fl


class KLStrongFDMBirgerError(RuntimeError):
    pass


class KLStrongFDMBirgerLinearSystemSingular(KLStrongFDMBirgerError):
    pass


@dataclass
class KLFrozenGBInnerResult:
    state: np.ndarray
    converged: bool
    status: str
    residual_inf: float


@dataclass
class KLStrongFDMInnerCounter:
    calls: int = 0


def frozen_birger_strong_residual(
    U,ops,bc,*,K0,GB,quadrature,generalized_load
):
    fl._validate_grid(ops)
    U=fl._validate_U(U,ops)
    bc.check_rigid_mode_requirement()

    p=np.asarray(generalized_load,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    kin=kl.field_kinematics(U,ops,geometry="LINEAR")
    bout=bm.frozen_birger_resultants_from_kinematics(
        kin,K0=K0,GB=GB,quadrature=quadrature
    )
    nm=bout.resultants

    strong=kl.strong_residual_from_resultants(
        kin,nm,ops,geometry="LINEAR",generalized_load=p
    )

    ql=kb.end_mechanical_quantities(
        kin,nm,ops,geometry="LINEAR",side="LEFT"
    )
    qr=kb.end_mechanical_quantities(
        kin,nm,ops,geometry="LINEAR",side="RIGHT"
    )

    ns,nx=ops.grid.ns,ops.grid.nx
    lu=fl._bc_component_residual(bc.left.u,ql.u,ql.T_u,ns)
    ru=fl._bc_component_residual(bc.right.u,qr.u,qr.T_u,ns)
    lv=fl._bc_component_residual(bc.left.v,ql.v,ql.T_v,ns)
    rv=fl._bc_component_residual(bc.right.v,qr.v,qr.T_v,ns)
    lw=fl._bc_component_residual(bc.left.w,ql.w,ql.T_w,ns)
    rw=fl._bc_component_residual(bc.right.w,qr.w,qr.T_w,ns)
    ls=fl._bc_component_residual(
        bc.left.slope,ql.theta_n,ql.T_theta,ns
    )
    rs=fl._bc_component_residual(
        bc.right.slope,qr.theta_n,qr.T_theta,ns
    )

    rows=[]
    for js in range(ns):
        rows.append(lu[js])
        rows.extend(strong[0,js,1:nx-1])
        rows.append(ru[js])

    for js in range(ns):
        rows.append(lv[js])
        rows.extend(strong[1,js,1:nx-1])
        rows.append(rv[js])

    for js in range(ns):
        rows.append(lw[js])
        rows.append(ls[js])
        rows.extend(strong[2,js,2:nx-2])
        rows.append(rs[js])
        rows.append(rw[js])

    R=np.asarray(rows,float)
    n=3*ns*nx
    if R.shape!=(n,):
        raise KLStrongFDMBirgerError(
            f"residual size {R.shape}, expected ({n},)"
        )
    return R,strong,bout,kin


def assemble_frozen_gb_linear_system(
    ops,bc,*,K0,GB,quadrature,generalized_load
):
    n=3*ops.grid.ns*ops.grid.nx
    U0=np.zeros((3,ops.grid.ns,ops.grid.nx),float)

    r0,*_=frozen_birger_strong_residual(
        U0,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load
    )

    A=np.empty((n,n),float)
    for j in range(n):
        e=np.zeros(n,float)
        e[j]=1.0
        Uj=fl.unflatten_U(e,ops)
        rj,*_=frozen_birger_strong_residual(
            Uj,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
        A[:,j]=rj-r0
    return A,r0


def solve_frozen_gb_linear_strong_fdm(
    initial_U,ops,bc,*,K0,GB,quadrature,generalized_load
):
    # initial_U is accepted for the generic outer-cycle interface.
    # The frozen LINEAR problem has a unique direct linear solution;
    # therefore the initial guess is not used in the solve.
    fl._validate_U(initial_U,ops)

    A,r0=assemble_frozen_gb_linear_system(
        ops,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load
    )
    try:
        x,_,_,_=nscale.equilibrated_solve(A,-r0)
    except np.linalg.LinAlgError as exc:
        raise KLStrongFDMBirgerLinearSystemSingular(
            "SINGULAR_STRONG_FDM_LINEAR_BIRGER_SYSTEM"
        ) from exc

    U=fl.unflatten_U(x,ops)
    R,*_=frozen_birger_strong_residual(
        U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load
    )
    return KLFrozenGBInnerResult(
        state=U,
        converged=True,
        status="CONVERGED_STRONG_FDM_FROZEN_GB_LINEAR",
        residual_inf=float(np.max(np.abs(R))),
    )


def evaluate_material_state(
    U,ops,*,K0,GB,quadrature
):
    U=fl._validate_U(U,ops)
    kin=kl.field_kinematics(U,ops,geometry="LINEAR")
    bout=bm.frozen_birger_resultants_from_kinematics(
        kin,K0=K0,GB=GB,quadrature=quadrature
    )
    return bout.material_state


def make_inner_solver(
    ops,bc,*,K0,quadrature,generalized_load,counter=None
):
    if counter is None:
        counter=KLStrongFDMInnerCounter()

    def inner(state_guess,GB):
        counter.calls+=1
        return solve_frozen_gb_linear_strong_fdm(
            state_guess,ops,bc,K0=K0,GB=GB,
            quadrature=quadrature,generalized_load=generalized_load
        )

    inner.counter=counter
    return inner


def make_material_evaluator(
    ops,*,K0,quadrature
):
    def evaluate(U,GB):
        return evaluate_material_state(
            U,ops,K0=K0,GB=GB,quadrature=quadrature
        )
    return evaluate


def solve_linear_birger_strong_fdm(
    initial_U,initial_GB,
    *,
    ops,bc,E0,nu0,h,quadrature,generalized_load,
    material_diagram,birger_settings,
    previous_accepted_epsilon_i=None,
    inner_counter=None,
):
    initial_U=fl._validate_U(initial_U,ops)
    E0=float(E0); nu0=float(nu0); h=float(h)
    if E0<=0 or h<=0 or not (-1.0<nu0<0.5):
        raise ValueError("invalid E0/nu0/h")

    G0=bm.shear_modulus_from_E_nu(E0,nu0)
    K0=bm.bulk_modulus_from_E_nu(E0,nu0)

    expected=(len(quadrature.z),ops.grid.ns,ops.grid.nx)
    GB0=np.asarray(initial_GB,float)
    if GB0.shape!=expected:
        raise ValueError(f"initial_GB must have shape {expected}")
    if np.any(~np.isfinite(GB0)) or np.any(GB0<=0):
        raise ValueError("initial_GB must be finite and positive")

    inner=make_inner_solver(
        ops,bc,K0=K0,quadrature=quadrature,
        generalized_load=generalized_load,
        counter=inner_counter,
    )
    evaluator=make_material_evaluator(
        ops,K0=K0,quadrature=quadrature
    )

    result=b5.solve_birger_outer_cycle(
        initial_U,GB0,
        G0=G0,
        material_diagram=material_diagram,
        inner_solve=inner,
        evaluate_material=evaluator,
        state_vector=lambda st: np.asarray(st,float).reshape(-1),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
    )
    return result,inner.counter
