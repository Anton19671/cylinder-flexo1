#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7E: GENERAL CUF nonlinear-elastic / Birger physical driver.

Scope:
  1) Full Green-Lagrange + elastic StVK (GB=G0 fixed)
  2) Full Green-Lagrange + Birger secant material outer cycle
  3) Explicit load-step series
  4) Initial separated-profile seed validation for the accepted Y->X MVI order

No concrete higher-order CUF basis is invented. A CallableThicknessBasis is
supplied explicitly. TE1 is only a regression specialization.

CRITICAL INITIALIZATION:
The accepted current MVI order is Y halfstep first, then X. Therefore the
first nonzero residual requires nonzero admissible X partner profiles so that
T_Y = dU/dY is nonzero. Physical fields may still be exactly zero by taking
Y=0. This module never invents an X seed; it validates the supplied state and
raises INITIAL_PROFILE_SEED_REQUIRED when needed.

Adaptive bisection of a failed load increment is supported at the outer
load-series level.  The accepted physical state is never replaced by a failed
attempt; the midpoint is solved from the last accepted state and the original
target remains queued.  No SVD/pinv/lstsq/QR/Cholesky, regularization,
gauge, balancing, relaxation or line-search is introduced here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import os
import tempfile
import numpy as np
import final_progress as fprog
import final_residual_scaling as rscale
import final_v3_fd as v3fd
import final_step_predictor as fpred

import cylinder_te1_local_kernel as lk
import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7b_general_mvi_profiles as g8
import cylinder_cuf_stage7d_general_internal_mvi as g10
import cylinder_cuf_stage7f_general_linear as g12
import cylinder_cuf_stage7h_general_vk as g13
import cylinder_cuf_stage7h_general_variational as g14


class GeneralCUFDriverError(RuntimeError):
    pass

class InitialProfileSeedRequired(GeneralCUFDriverError):
    pass


@dataclass
class GeneralAcceptedStep:
    step_index: int
    load_factor: float
    state: g8.ProfileStateCUF
    solver_status: str
    solver_iterations: int
    GB: Optional[np.ndarray]=None
    epsilon_i: Optional[np.ndarray]=None
    sigma_i: Optional[np.ndarray]=None
    detF: Optional[np.ndarray]=None
    max_epsilon_i: Optional[float]=None
    max_abs_J_minus_1: Optional[float]=None
    unloading_mask: Optional[np.ndarray]=None
    critical_epsilon_i_index: Optional[tuple]=None
    critical_J_index: Optional[tuple]=None
    unloading_point_fraction: Optional[float]=None
    solver_history: Optional[list]=None
    residual_inf_by_block: Optional[dict]=None
    # Sequential accepted-state number used for STEP_XXXX files.
    # requested_step_index identifies the original user load step; substeps
    # inserted by automatic bisection keep that original target index.
    requested_step_index: Optional[int]=None
    subdivision_depth: int=0
    is_substep: bool=False


@dataclass
class GeneralLoadSeriesResult:
    mode: str
    converged: bool
    status: str
    requested_steps: int
    accepted_steps: list
    failed_step_index: Optional[int]
    failed_step_status: Optional[str]
    output_dir: Optional[str]
    failed_step_code: Optional[str]=None
    failed_step_diagnostic_ru: Optional[str]=None
    failed_step_recommendation_ru: Optional[str]=None

    @property
    def last_accepted(self):
        return self.accepted_steps[-1] if self.accepted_steps else None


MODES=(
    "CUF_LINEAR_ELASTIC","CUF_LINEAR_BIRGER",
    "CUF_VK_ELASTIC","CUF_VK_BIRGER",
    "CUF_NL_ELASTIC","CUF_BIRGER",
)


def _failure_diagnostics(status):
    text="" if status is None else str(status)
    if "BIRGER_EPSILON_I_LIMIT_EXCEEDED" in text:
        return {
            "code":"BIRGER_EPSILON_I_LIMIT_EXCEEDED",
            "diagnostic_ru":"Превышен заданный предел интенсивности деформаций Биргера.",
            "recommendation_ru":"Шаг не принят. Уменьшите приращение нагрузки или конечную нагрузку.",
        }
    if "BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED" in text:
        return {
            "code":"BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED",
            "diagnostic_ru":"Превышен заданный предел |J-1| для модели Биргера.",
            "recommendation_ru":"Шаг не принят. Уменьшите приращение нагрузки; не ослабляйте предел без физического обоснования.",
        }
    if "BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED" in text:
        return {
            "code":"BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED",
            "diagnostic_ru":"Обнаружена локальная разгрузка, для которой применяемая деформационная теория Биргера неприменима.",
            "recommendation_ru":"Шаг не принят. Проверьте монотонность нагружения и уменьшите приращение нагрузки.",
        }
    if "DET_F_STEP_TOO_LARGE" in text:
        return {
            "code":"DET_F_STEP_TOO_LARGE",
            "diagnostic_ru":"Приращение нагрузки слишком велико: несколько последовательных MVI-коррекций упёрлись в ограничение detF.",
            "recommendation_ru":"Текущий шаг отклонён. Драйвер должен уменьшить приращение и повторить расчёт от последнего принятого состояния.",
        }
    if "INVALID_DEFORMATION_MAPPING" in text or "InvalidDeformationMapping" in text:
        return {
            "code":"INVALID_DEFORMATION_MAPPING",
            "diagnostic_ru":"Следующий шаг нагрузки дал detF <= 0: деформационное отображение локально обращается.",
            "recommendation_ru":"Не отключайте проверку detF. Уменьшите шаг нагрузки или конечную нагрузку; принятые шаги до остановки являются последней допустимой веткой.",
        }
    if "MAX_HALFSTEP_ITERATIONS" in text or "HalfstepDidNotConverge" in text:
        return {
            "code":"MVI_HALFSTEP_NOT_CONVERGED",
            "diagnostic_ru":"Внутренний MVI halfstep не достиг projected_tolerance за заданное число итераций.",
            "recommendation_ru":"Уменьшите шаг нагрузки. В Expert можно увеличить max_halfstep_iterations, но расчет станет дольше.",
        }
    if "INITIAL_PROFILE_SEED_REQUIRED" in text or "InitialProfileSeedRequired" in text:
        return {
            "code":"INITIAL_PROFILE_SEED_REQUIRED",
            "diagnostic_ru":"Для MVI не задана допустимая начальная пара X/Y профилей.",
            "recommendation_ru":"Используйте автоматические начальные профили из GUI или корректный NPZ с X/Y.",
        }
    if text:
        return {
            "code":"STEP_FAILED",
            "diagnostic_ru":"Шаг нагрузки остановлен с ошибкой решателя.",
            "recommendation_ru":"Проверьте failed_step_status и сохраненные принятые шаги; для продолжения обычно нужен меньший шаг нагрузки.",
        }
    return {"code":None,"diagnostic_ru":None,"recommendation_ru":None}


def _full_shape(state,ops):
    return (3,state.m,ops.grid.ns,ops.grid.nx)


def _as_load(load,state,ops):
    a=np.asarray(load,float)
    expected=_full_shape(state,ops)
    if a.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if not np.all(np.isfinite(a)):
        raise ValueError("generalized_load contains non-finite values")
    return a


def _v4_measure_multipliers(ops,basis,bc,*,E0,quadrature):
    """Reviewer V-4 residual multipliers for CUF MVI stopping measures only."""
    h=v3fd.thickness_from_quadrature(quadrature)
    return rscale.cuf_residual_measure_multipliers(
        basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,bc=bc,E0=E0,h=h,
    ).reshape(-1)


def _initial_y_tangent_has_direction(state,grid):
    T,_=g8.build_direction_tangent(state,grid,"Y")
    if T.shape[1]==0:
        return False
    return bool(np.any(np.linalg.norm(T,axis=0)>1e-14))


def require_seed_if_nonzero_residual(state,grid,residual_fn):
    """
    Accepted current order starts with Y-halfstep.
    If R != 0 but T_Y has no active nonzero direction, first physical
    correction is impossible. Do not invent a seed.
    """
    R=np.asarray(residual_fn(state),float).reshape(-1)
    if not np.all(np.isfinite(R)):
        raise GeneralCUFDriverError("NONFINITE_INITIAL_RESIDUAL")
    if float(np.max(np.abs(R)))==0.0:
        return
    if not _initial_y_tangent_has_direction(state,grid):
        raise InitialProfileSeedRequired(
            "INITIAL_PROFILE_SEED_REQUIRED: first nonzero CUF load state has "
            "no active Y-halfstep direction because the supplied X partner "
            "profiles are zero/degenerate. Supply admissible nonzero X seeds; "
            "Y may remain zero so the physical initial field is exactly zero."
        )


def make_nonlinear_elastic_residual_function(
    ops,*,E0,nu0,quadrature,basis,generalized_load,bc,formulation="STRONG"
):
    ec=lk.ElasticConstants(float(E0),float(nu0))
    form=str(formulation).upper()
    if form=="STRONG":
        # MVI projects the Green-Lagrange flux residual and its exact tangent
        # with the same surface weights.  The full-grid FDM direct-D2 linear
        # correction is a different discrete operator: inserting it here
        # changes the MVI equations (including the clamped boundary layer).
        return g10.make_physical_residual_function(
            ops,K0=ec.K0,GB=ec.G0,quadrature=quadrature,basis=basis,
            generalized_load=generalized_load,bc=bc
        )
    if form=="VARIATIONAL":
        return g14.make_variational_residual_function(
            ops,K0=ec.K0,GB=ec.G0,quadrature=quadrature,basis=basis,
            generalized_load=generalized_load,bc=bc
        )
    raise ValueError("MVI formulation must be STRONG or VARIATIONAL")


def solve_general_nonlinear_elastic_load_step(
    initial_state,ops,*,E0,nu0,quadrature,basis,generalized_load,
    bc,inner_settings,validate_initial_seed=True
):
    if initial_state.m!=basis.m or bc.m!=basis.m:
        raise ValueError("state/basis/BC m mismatch")
    p=_as_load(generalized_load,initial_state,ops)
    fn=make_nonlinear_elastic_residual_function(
        ops,E0=E0,nu0=nu0,quadrature=quadrature,basis=basis,
        generalized_load=p,bc=bc,formulation=inner_settings.formulation
    )
    if validate_initial_seed:
        require_seed_if_nonzero_residual(initial_state,ops.grid,fn)
    projection_weights=(
        np.ones(initial_state.n_fields*ops.grid.n_surface_nodes,dtype=float)
        if str(inner_settings.formulation).upper()=="VARIATIONAL"
        else None
    )
    return g10.solve_internal_mvi(
        initial_state,ops.grid,bc,fn,inner_settings,
        weights=projection_weights,
        measure_multipliers=_v4_measure_multipliers(
            ops,basis,bc,E0=E0,quadrature=quadrature
        ),
        quadrature=quadrature
    )


def evaluate_general_material_points(
    state,ops,*,K0,GB,quadrature,basis
):
    """K2b batched material-point evaluation for nonlinear CUF MVI."""
    q,qx,qs=g8.profile_fields_and_first_derivatives(state,ops)
    surf=(ops.grid.ns,ops.grid.nx)
    nz=quadrature.z.size
    eps=np.empty((nz,)+surf,float)
    sig=np.empty_like(eps)
    det=np.empty_like(eps)
    GBa,mode=g7._gb_mode(GB,nz,surf)
    R=float(ops.grid.radius); K0=float(K0)

    for g,z in enumerate(quadrature.z):
        if mode=="scalar": GBg=float(GBa)
        elif mode=="surface": GBg=GBa
        else: GBg=GBa[g]
        _H,_Ft,_Fp,abc=g7._batched_general_gradient_quantities(q,qx,qs,z,R,basis)
        F=abc.copy(); F[0,0]+=1.0; F[1,1]+=1.0; F[2,2]+=1.0
        J=g7._batched_det3(F)
        g7._raise_if_invalid_batched_F(J,z,surf)
        E=g7._batched_green_lagrange(F)
        S=g7._batched_birger_second_piola(E,K0,GBg)
        eps[g]=g7._batched_strain_intensity(E)
        sig[g]=g7._batched_stress_intensity(S)
        det[g]=J
    return b5.MaterialPointState(eps,sig,det)


def solve_general_birger_load_step(
    initial_state,ops,*,E0,nu0,quadrature,basis,generalized_load,
    bc,inner_settings,birger_settings,material_diagram,
    initial_GB=None,previous_accepted_epsilon_i=None,
    validate_initial_seed=True
):
    if initial_state.m!=basis.m or bc.m!=basis.m:
        raise ValueError("state/basis/BC m mismatch")
    ec=lk.ElasticConstants(float(E0),float(nu0))
    p=_as_load(generalized_load,initial_state,ops)
    gbshape=(quadrature.z.size,ops.grid.ns,ops.grid.nx)

    if initial_GB is None:
        GB0=np.full(gbshape,ec.G0,float)
    else:
        GB0=np.broadcast_to(
            np.asarray(initial_GB,float),gbshape
        ).astype(float,copy=True)

    def inner(st,GBfield):
        fn=g10.make_physical_residual_function(
            ops,K0=ec.K0,GB=GBfield,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc
        )
        if validate_initial_seed:
            require_seed_if_nonzero_residual(st,ops.grid,fn)
        return g10.solve_internal_mvi(
            st,ops.grid,bc,fn,inner_settings,
            measure_multipliers=_v4_measure_multipliers(
                ops,basis,bc,E0=E0,quadrature=quadrature
            ),
            quadrature=quadrature
        )

    def evaluator(st,GBfield):
        return evaluate_general_material_points(
            st,ops,K0=ec.K0,GB=GBfield,quadrature=quadrature,basis=basis
        )

    return b5.solve_birger_outer_cycle(
        initial_state,GB0,G0=ec.G0,
        material_diagram=material_diagram,
        inner_solve=inner,
        evaluate_material=evaluator,
        state_vector=lambda st: g8.reconstruct_vector(st,ops.grid),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
    )


def _save_step(step,output_dir,*,K0=None):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    data={
        "X":step.state.X,
        "Y":step.state.Y,
        "step_index":np.asarray(step.step_index,int),
        "load_factor":np.asarray(step.load_factor,float),
    }
    for key in ("GB","epsilon_i","sigma_i","detF","unloading_mask"):
        val=getattr(step,key)
        if val is not None:
            data[key]=np.asarray(val)
    # Publish the accepted state atomically.  A hard GUI STOP can arrive at
    # any moment, so readers must never observe a half-written ZIP/NPZ file.
    final_npz=output_dir/f"STEP_{step.step_index:04d}.npz"
    fd,tmp_name=tempfile.mkstemp(
        prefix=f".STEP_{step.step_index:04d}_",suffix=".npz",dir=str(output_dir)
    )
    os.close(fd)
    tmp_npz=Path(tmp_name)
    try:
        with tmp_npz.open("wb") as f:
            np.savez_compressed(f,**data)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_npz,final_npz)
    finally:
        try:
            if tmp_npz.exists():tmp_npz.unlink()
        except OSError:
            pass
    meta={
        "step_index":step.step_index,
        "load_factor":step.load_factor,
        "solver_status":step.solver_status,
        "solver_iterations":step.solver_iterations,
        "max_epsilon_i":step.max_epsilon_i,
        "max_abs_J_minus_1":step.max_abs_J_minus_1,
        "critical_epsilon_i_index":step.critical_epsilon_i_index,
        "critical_J_index":step.critical_J_index,
        "unloading_point_fraction":step.unloading_point_fraction,
        "has_GB":step.GB is not None,
        "has_unloading_mask":step.unloading_mask is not None,
        "solver_history":step.solver_history,
        "residual_inf_by_block":step.residual_inf_by_block,
        "requested_step_index":step.requested_step_index,
        "subdivision_depth":int(step.subdivision_depth),
        "is_substep":bool(step.is_substep),
    }
    if step.GB is not None and K0 is not None:
        gb=np.asarray(step.GB,float)
        k0=float(K0)
        # Reviewer J-15: output-only effective Poisson-ratio diagnostic for
        # constant K0 and the already accepted variable GB field.
        nu_eff=(3.0*k0-2.0*gb)/(2.0*(3.0*k0+gb))
        meta["birger_nu_effective_min"]=float(np.min(nu_eff))
        meta["birger_nu_effective_max"]=float(np.max(nu_eff))
    final_json=output_dir/f"STEP_{step.step_index:04d}.json"
    tmp_json=output_dir/f".STEP_{step.step_index:04d}.json.tmp"
    try:
        with tmp_json.open("w",encoding="utf-8") as f:
            f.write(json.dumps(meta,indent=2,ensure_ascii=False))
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_json,final_json)
    finally:
        try:
            if tmp_json.exists():tmp_json.unlink()
        except OSError:
            pass


def _accepted_step_v4_block_norms(
    mode,step,ops,*,E0,nu0,quadrature,basis,bc,generalized_load,
    formulation="STRONG"
):
    """Output-only block norms using the same residual formulation as the accepted MVI step."""
    formulation=str(formulation).upper()
    p=np.asarray(generalized_load,float)
    if mode=="CUF_LINEAR_ELASTIC":
        fn=g12.make_linear_elastic_residual_function(
            ops,E0=E0,nu0=nu0,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc,formulation=formulation
        )
    elif mode=="CUF_LINEAR_BIRGER":
        ec=lk.ElasticConstants(float(E0),float(nu0))
        fn=g12.make_linear_residual_function(
            ops,K0=ec.K0,GB=step.GB,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc
        )
    elif mode=="CUF_VK_ELASTIC":
        fn=g13.make_vk_hooke_residual_function(
            ops,E0=E0,nu0=nu0,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc,formulation=formulation
        )
    elif mode=="CUF_NL_ELASTIC":
        fn=make_nonlinear_elastic_residual_function(
            ops,E0=E0,nu0=nu0,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc,formulation=formulation
        )
    elif mode=="CUF_VK_BIRGER":
        ec=lk.ElasticConstants(float(E0),float(nu0))
        fn=g13.make_vk_residual_function(
            ops,K0=ec.K0,GB=step.GB,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc
        )
    else:
        ec=lk.ElasticConstants(float(E0),float(nu0))
        fn=g10.make_physical_residual_function(
            ops,K0=ec.K0,GB=step.GB,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc
        )
    R=np.asarray(fn(step.state),float).reshape(-1)
    mm=_v4_measure_multipliers(ops,basis,bc,E0=E0,quadrature=quadrature)
    Rm=mm*R
    return rscale.cuf_residual_inf_by_block(
        Rm,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
    )


def _is_subdividable_failure(status):
    """Return True only for numerical step failures that a smaller increment can fix."""
    text="" if status is None else str(status)
    upper=text.upper()
    # Input/seed/configuration errors must remain loud; subdivision cannot fix them.
    nonretry=(
        "INITIAL_PROFILE_SEED_REQUIRED","MISSING_REQUIRED_INPUT","UNSPECIFIED_REQUIRED_INPUT",
        "SHAPE MISMATCH","SHAPE=","MUST HAVE SHAPE","NON-FINITE","NONFINITE_INITIAL_RESIDUAL",
        "VALUEERROR:","GENERALCASEINPUTERROR",
    )
    if any(tok in upper for tok in nonretry):
        return False
    # Known numerical failures from MVI, deformation mapping and Birger cycles.
    numerical=(
        "MAX_HALFSTEP_ITERATIONS","HALFSTEPDIDNOTCONVERGE","DET_F","DETF",
        "INVALID_DEFORMATION_MAPPING","INVALIDDEFORMATIONMAPPING",
        "NOT_CONVERGED","NOT_CONVERGE","MAX_XY","MAX_OUTER",
        "MAX_BIRGER_OUTER_ITERATIONS","DIVERG",
        "GUARDED_CORRECTION_FAILED","INNER_MVI",
        "BIRGER_EPSILON_I_LIMIT_EXCEEDED",
        "BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED",
        "BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED",
    )
    return any(tok in upper for tok in numerical)


def _make_accepted_step_for_mode(
    mode,step_initial,ops,*,E0,nu0,quadrature,basis,load,bc,inner_settings,
    birger_settings,material_diagram,current_GB,previous_eps
):
    """Solve exactly one target load; no retry/subdivision is performed here."""
    if mode=="CUF_LINEAR_ELASTIC":
        sol=g12.solve_general_linear_elastic_load_step(
            step_initial,ops,E0=E0,nu0=nu0,quadrature=quadrature,
            basis=basis,generalized_load=load,bc=bc,
            inner_settings=inner_settings,validate_initial_seed=True
        )
        if not sol.converged:
            raise GeneralCUFDriverError(sol.status)
        return GeneralAcceptedStep(0,0.0,sol.state.copy(),sol.status,sol.cycles,
                                   solver_history=list(sol.history))

    if mode=="CUF_LINEAR_BIRGER":
        sol=g12.solve_general_linear_birger_load_step(
            step_initial,ops,E0=E0,nu0=nu0,quadrature=quadrature,
            basis=basis,generalized_load=load,bc=bc,
            inner_settings=inner_settings,birger_settings=birger_settings,
            material_diagram=material_diagram,initial_GB=current_GB,
            previous_accepted_epsilon_i=previous_eps,validate_initial_seed=True
        )
    elif mode=="CUF_VK_ELASTIC":
        sol=g13.solve_general_vk_hooke_load_step(
            step_initial,ops,E0=E0,nu0=nu0,quadrature=quadrature,
            basis=basis,generalized_load=load,bc=bc,
            inner_settings=inner_settings,validate_initial_seed=True
        )
        if not sol.converged:
            raise GeneralCUFDriverError(sol.status)
        return GeneralAcceptedStep(0,0.0,sol.state.copy(),sol.status,sol.cycles,
                                   solver_history=list(sol.history))
    elif mode=="CUF_VK_BIRGER":
        sol=g13.solve_general_vk_birger_load_step(
            step_initial,ops,E0=E0,nu0=nu0,quadrature=quadrature,
            basis=basis,generalized_load=load,bc=bc,
            inner_settings=inner_settings,birger_settings=birger_settings,
            material_diagram=material_diagram,initial_GB=current_GB,
            previous_accepted_epsilon_i=previous_eps,validate_initial_seed=True
        )
    elif mode=="CUF_NL_ELASTIC":
        sol=solve_general_nonlinear_elastic_load_step(
            step_initial,ops,E0=E0,nu0=nu0,quadrature=quadrature,
            basis=basis,generalized_load=load,bc=bc,
            inner_settings=inner_settings,validate_initial_seed=True
        )
        if not sol.converged:
            raise GeneralCUFDriverError(sol.status)
        return GeneralAcceptedStep(0,0.0,sol.state.copy(),sol.status,sol.cycles,
                                   solver_history=list(sol.history))
    else:
        sol=solve_general_birger_load_step(
            step_initial,ops,E0=E0,nu0=nu0,quadrature=quadrature,
            basis=basis,generalized_load=load,bc=bc,
            inner_settings=inner_settings,birger_settings=birger_settings,
            material_diagram=material_diagram,initial_GB=current_GB,
            previous_accepted_epsilon_i=previous_eps,validate_initial_seed=True
        )

    if not sol.converged:
        raise GeneralCUFDriverError(sol.status)
    return GeneralAcceptedStep(
        0,0.0,sol.state.copy(),sol.status,sol.outer_iterations,
        GB=np.asarray(sol.GB,float).copy(),
        epsilon_i=np.asarray(sol.material_state.epsilon_i,float).copy(),
        sigma_i=np.asarray(sol.material_state.sigma_i,float).copy(),
        detF=np.asarray(sol.material_state.detF,float).copy(),
        max_epsilon_i=float(sol.max_epsilon_i),
        max_abs_J_minus_1=float(sol.max_abs_J_minus_1),
        unloading_mask=(None if sol.unloading_mask is None else
                        np.asarray(sol.unloading_mask,bool).copy()),
        critical_epsilon_i_index=sol.critical_epsilon_i_index,
        critical_J_index=sol.critical_J_index,
        unloading_point_fraction=float(sol.unloading_point_fraction),
        solver_history=list(sol.history),
    )


def solve_general_explicit_load_series(
    mode,initial_state,ops,*,E0,nu0,quadrature,basis,bc,
    load_steps,inner_settings,load_factors=None,
    birger_settings=None,material_diagram=None,initial_GB=None,
    output_dir=None,step_predictor_mode="ERROR",
    automatic_step_subdivision=True,max_subdivision_depth=8,
    proactive_step_adaptation=True,proactive_outer_soft_limit=8,
    proactive_outer_hard_limit=20,proactive_max_refinement_depth=3
):
    mode=str(mode).upper()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    loads=[_as_load(x,initial_state,ops).copy() for x in load_steps]
    if not loads:
        raise ValueError("at least one explicit load step required")
    factors=(
        [float(i+1) for i in range(len(loads))]
        if load_factors is None else [float(x) for x in load_factors]
    )
    if len(factors)!=len(loads):
        raise ValueError("load_factors length mismatch")
    if int(max_subdivision_depth)<0:
        raise ValueError("max_subdivision_depth must be >=0")
    if int(proactive_outer_soft_limit)<1:
        raise ValueError("proactive_outer_soft_limit must be >=1")
    if int(proactive_outer_hard_limit)<=int(proactive_outer_soft_limit):
        raise ValueError("proactive_outer_hard_limit must be > proactive_outer_soft_limit")
    if int(proactive_max_refinement_depth)<0:
        raise ValueError("proactive_max_refinement_depth must be >=0")

    if mode in ("CUF_LINEAR_BIRGER","CUF_VK_BIRGER","CUF_BIRGER") and (
        birger_settings is None or material_diagram is None
    ):
        raise ValueError("Birger settings and external material diagram required")

    predictor_mode=fpred.normalize_mode(step_predictor_mode)
    current=initial_state.copy()
    previous_state=None
    previous_factor=None
    current_factor=0.0
    current_load=np.zeros_like(loads[0])
    current_GB=None if initial_GB is None else np.asarray(initial_GB,float).copy()
    previous_eps=None
    accepted=[]
    failed_index=None
    failed_status=None
    completed_requested=0
    subdivisions_used=0
    proactive_insertions=0
    out=Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)

    # Queue entries retain the original requested index.  If a target fails,
    # a midpoint is inserted immediately before it; the original target stays
    # queued and is retried only after the midpoint has been accepted.
    queue=[{
        "load":load.copy(),"factor":float(factor),"requested_index":int(k),
        "depth":0,"proactive_depth":0,"is_substep":False,
    } for k,(load,factor) in enumerate(zip(loads,factors),start=1)]
    pos=0
    while pos < len(queue):
        job=queue[pos]
        load=np.asarray(job["load"],float)
        factor=float(job["factor"])
        requested_index=int(job["requested_index"])
        depth=int(job["depth"])
        proactive_depth=int(job.get("proactive_depth",0))
        is_substep=bool(job["is_substep"])
        accepted_index=len(accepted)+1

        fprog.set_context(
            step_index=int(accepted_index),
            accepted_step_index=int(accepted_index),
            requested_step_index=int(requested_index),
            subdivision_depth=int(depth),
            is_substep=bool(is_substep),
            load_factor=float(factor),
        )
        fprog.emit("step_start",stage="LOAD_STEP")
        step_initial=fpred.predict_state(
            previous_state,current,factor,current_factor,previous_factor,
            mode=predictor_mode
        )
        try:
            step=_make_accepted_step_for_mode(
                mode,step_initial,ops,E0=E0,nu0=nu0,quadrature=quadrature,
                basis=basis,load=load,bc=bc,inner_settings=inner_settings,
                birger_settings=birger_settings,material_diagram=material_diagram,
                current_GB=current_GB,previous_eps=previous_eps,
            )
        except Exception as exc:
            status=f"{type(exc).__name__}: {exc}"
            can_split=(
                bool(automatic_step_subdivision)
                and depth < int(max_subdivision_depth)
                and _is_subdividable_failure(status)
            )
            if can_split:
                mid_factor=0.5*(float(current_factor)+factor)
                mid_load=0.5*(np.asarray(current_load,float)+load)
                # Numerical guard: subdivision must create a genuinely new target.
                if mid_factor!=factor and not np.array_equal(mid_load,load):
                    # The failed target itself moves one level deeper as well.
                    # Otherwise, after an accepted midpoint the target would be
                    # retried again at depth=0 and max_subdivision_depth would
                    # never bound successive halvings toward the same target.
                    job["depth"]=depth+1
                    queue.insert(pos,{
                        "load":mid_load,"factor":mid_factor,
                        "requested_index":requested_index,"depth":depth+1,
                        "proactive_depth":proactive_depth,"is_substep":True,
                    })
                    subdivisions_used+=1
                    fprog.emit(
                        "load_step_subdivided",stage="LOAD_STEP_SUBDIVISION",
                        requested_step_index=requested_index,
                        failed_target_factor=factor,mid_factor=mid_factor,
                        subdivision_depth=depth+1,failure_status=status,
                    )
                    continue
            failed_index=requested_index
            failed_status=status
            break

        step.step_index=accepted_index
        step.load_factor=factor
        step.requested_step_index=requested_index
        step.subdivision_depth=depth
        step.is_substep=is_substep
        step.residual_inf_by_block=_accepted_step_v4_block_norms(
            mode,step,ops,E0=E0,nu0=nu0,quadrature=quadrature,
            basis=basis,bc=bc,generalized_load=load,
            formulation=inner_settings.formulation
        )
        accepted.append(step)
        previous_state=current.copy()
        previous_factor=current_factor
        current=step.state.copy()
        current_factor=factor
        current_load=load.copy()
        if mode in ("CUF_LINEAR_BIRGER","CUF_VK_BIRGER","CUF_BIRGER"):
            current_GB=step.GB.copy()
            previous_eps=step.epsilon_i.copy()
        if not is_substep:
            completed_requested+=1
        if out is not None:
            K0_out=(float(E0)/(3.0*(1.0-2.0*float(nu0)))) if step.GB is not None else None
            _save_step(step,out,K0=K0_out)
        # Proactive load-step refinement for Birger branches.  A converged but
        # expensive outer fixed-point iteration is a warning that the *next*
        # load interval is too large.  Refine only that upcoming interval; do
        # not change the already accepted state or the requested target loads.
        # This is deliberately separate from failure-driven subdivision above.
        if (
            bool(proactive_step_adaptation)
            and mode in ("CUF_LINEAR_BIRGER","CUF_VK_BIRGER","CUF_BIRGER")
            and pos+1 < len(queue)
        ):
            outer_count=int(step.solver_iterations)
            split_levels=(
                2 if outer_count > int(proactive_outer_hard_limit) else
                1 if outer_count > int(proactive_outer_soft_limit) else 0
            )
            if split_levels>0:
                next_job=queue[pos+1]
                next_factor=float(next_job["factor"])
                next_load=np.asarray(next_job["load"],float)
                base_depth=int(next_job.get("depth",0))
                base_proactive_depth=int(next_job.get("proactive_depth",0))
                # Proactive refinement has an independent shallow depth budget.
                # It is deliberately separate from failure-driven subdivision,
                # so proactive refinement cannot generate vanishing microsteps
                # and cannot consume the emergency retry depth budget.
                allowed=max(0,int(proactive_max_refinement_depth)-base_proactive_depth)
                split_levels=min(split_levels,allowed)
                if split_levels>0 and next_factor!=float(current_factor):
                    nseg=2**split_levels
                    new_proactive_depth=base_proactive_depth+split_levels
                    # The original next target belongs to the same proactively
                    # refined interval, so advance its proactive depth too.
                    next_job["proactive_depth"]=new_proactive_depth
                    new_jobs=[]
                    for j in range(1,nseg):
                        a=float(j)/float(nseg)
                        sf=(1.0-a)*float(current_factor)+a*next_factor
                        sl=(1.0-a)*np.asarray(current_load,float)+a*next_load
                        new_jobs.append({
                            "load":sl,"factor":sf,
                            "requested_index":int(next_job["requested_index"]),
                            "depth":base_depth,
                            "proactive_depth":new_proactive_depth,
                            "is_substep":True,
                        })
                    if new_jobs:
                        queue[pos+1:pos+1]=new_jobs
                        proactive_insertions+=len(new_jobs)
                        fprog.emit(
                            "load_step_proactive_refinement",stage="LOAD_STEP_ADAPTATION",
                            outer_iterations=outer_count,
                            soft_limit=int(proactive_outer_soft_limit),
                            hard_limit=int(proactive_outer_hard_limit),
                            proactive_max_refinement_depth=int(proactive_max_refinement_depth),
                            proactive_depth=new_proactive_depth,
                            split_levels=split_levels,inserted_substeps=len(new_jobs),
                            from_factor=float(current_factor),to_factor=next_factor,
                            requested_step_index=int(next_job["requested_index"]),
                        )

        try:
            q=np.einsum("cmrx,cmrs->cmsx",np.asarray(step.state.X,float),np.asarray(step.state.Y,float))
            w=q[2,0]; js,ix=np.unravel_index(int(np.argmax(np.abs(w))),w.shape)
            fprog.emit(
                "step_complete",stage="STEP_COMPLETE",solver_status=step.solver_status,
                converged=True,solver_iterations=int(step.solver_iterations),
                requested_step_index=requested_index,subdivision_depth=depth,
                is_substep=is_substep,w_abs_max=float(np.max(np.abs(w))),
                w_at_abs_max=float(w[js,ix]),wmax_is=int(js),wmax_ix=int(ix),
                max_epsilon_i=step.max_epsilon_i,max_abs_J_minus_1=step.max_abs_J_minus_1,
            )
        except Exception:
            fprog.emit(
                "step_complete",stage="STEP_COMPLETE",solver_status=step.solver_status,
                converged=True,solver_iterations=int(step.solver_iterations),
                requested_step_index=requested_index,subdivision_depth=depth,is_substep=is_substep,
            )
        pos+=1

    ok=(failed_status is None and completed_requested==len(loads))
    failure_diag=_failure_diagnostics(failed_status)
    result=GeneralLoadSeriesResult(
        mode=mode,converged=ok,
        status="CONVERGED_LOAD_SERIES" if ok else "LOAD_SERIES_STOPPED",
        requested_steps=len(loads),accepted_steps=accepted,
        failed_step_index=failed_index,failed_step_status=failed_status,
        output_dir=None if out is None else str(out),
        failed_step_code=failure_diag["code"],
        failed_step_diagnostic_ru=failure_diag["diagnostic_ru"],
        failed_step_recommendation_ru=failure_diag["recommendation_ru"],
    )

    if out is not None:
        summary={
            "mode":mode,"status":result.status,"converged":result.converged,
            "requested_steps":len(loads),"completed_requested_steps":completed_requested,
            "accepted_steps":len(accepted),"inserted_substeps":sum(bool(s.is_substep) for s in accepted),
            "subdivision_attempts":subdivisions_used,
            "proactive_inserted_substeps":proactive_insertions,
            "failed_step_index":failed_index,"failed_step_status":failed_status,
            "failed_step_code":failure_diag["code"],
            "failed_step_diagnostic_ru":failure_diag["diagnostic_ru"],
            "failed_step_recommendation_ru":failure_diag["recommendation_ru"],
            "first_halfstep":"Y",
            "initial_seed_policy":"X/Y profiles loaded from dispatcher-prepared NPZ",
            "automatic_step_subdivision":bool(automatic_step_subdivision),
            "automatic_retry":bool(automatic_step_subdivision),
            "max_subdivision_depth":int(max_subdivision_depth),
            "proactive_step_adaptation":bool(proactive_step_adaptation),
            "proactive_outer_soft_limit":int(proactive_outer_soft_limit),
            "proactive_outer_hard_limit":int(proactive_outer_hard_limit),
            "proactive_max_refinement_depth":int(proactive_max_refinement_depth),
            "detf_guarded_correction":bool(getattr(inner_settings,"detf_guarded_correction",False)),
            "max_correction_halvings":int(getattr(inner_settings,"max_correction_halvings",0)),
        }
        (out/"SERIES_SUMMARY.json").write_text(
            json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
        )
    return result


def load_initial_profiles_npz(path,*,m,rank,nx,ns):
    """
    Explicit launcher input. Required arrays:
      X shape (3,m,rank,nx)
      Y shape (3,m,rank,ns)
    """
    path=Path(path)
    try:
        d=np.load(path)
    except Exception as exc:
        raise GeneralCUFDriverError(f"cannot load initial profiles: {path}: {exc}") from exc
    if "X" not in d.files or "Y" not in d.files:
        raise GeneralCUFDriverError("initial profile NPZ must contain X and Y")
    X=np.asarray(d["X"],float)
    Y=np.asarray(d["Y"],float)
    if X.shape!=(3,m,rank,nx):
        raise GeneralCUFDriverError(
            f"X initial profiles must have shape {(3,m,rank,nx)}, got {X.shape}"
        )
    if Y.shape!=(3,m,rank,ns):
        raise GeneralCUFDriverError(
            f"Y initial profiles must have shape {(3,m,rank,ns)}, got {Y.shape}"
        )
    return g8.ProfileStateCUF(X,Y)
