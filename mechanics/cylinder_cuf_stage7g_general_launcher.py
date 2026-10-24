#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7G: strict CASE launcher for the GENERAL CUF branch.

The launcher does NOT define any concrete higher-order CUF model.
The case must explicitly supply a Python basis provider returning
CallableThicknessBasis with the accepted F_tau(z), F'_tau(z).

Required explicit inputs:
- mode among four general CUF modes;
- rank;
- L,R,h,E0,nu0;
- nx,ns,stencil;
- thickness quadrature;
- all 6m end boundary conditions (3m at each end);
- MVI tolerances/limits;
- explicit initial X/Y profiles NPZ;
- explicit generalized load series NPZ;
- output directory;
- for Birger modes: outer settings and external material diagram callable.

No physical/numerical defaults.
No automatic seed construction.
Automatic failed-step bisection is enabled by default and may be configured in load_step_adaptation.
"""
from __future__ import annotations
from pathlib import Path
import importlib
import json
import sys
import numpy as np

import cylinder_te1_mvi_core as t1
import cylinder_te1_local_kernel as lk
import cylinder_te1_stage2b as s2
import cylinder_te1_stage5a_birger as b5
import cylinder_te1_stage3a_connector as c3
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7c_general_boundary as g9
import cylinder_cuf_stage7d_general_internal_mvi as g10
import cylinder_cuf_stage7e_general_driver as g11


class GeneralCaseInputError(RuntimeError):
    pass


COMPONENT_NAMES=("u","v","w")


def _req(d,key,where):
    if key not in d:
        raise GeneralCaseInputError(f"MISSING_REQUIRED_INPUT: {where}.{key}")
    if d[key] is None:
        raise GeneralCaseInputError(f"UNSPECIFIED_REQUIRED_INPUT: {where}.{key}")
    return d[key]


def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0:
        raise GeneralCaseInputError(f"{name} must be finite and >0")
    return v


def _nonneg(x,name):
    v=float(x)
    if not np.isfinite(v) or v<0:
        raise GeneralCaseInputError(f"{name} must be finite and >=0")
    return v


def _intmin(x,name,mn):
    v=int(x)
    if v<mn:
        raise GeneralCaseInputError(f"{name} must be >= {mn}")
    return v


def _load_json(path):
    p=Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneralCaseInputError(f"CASE_FILE_NOT_FOUND: {p}") from exc
    except json.JSONDecodeError as exc:
        raise GeneralCaseInputError(f"INVALID_CASE_JSON: {exc}") from exc


def _resolve_module_callable(spec,case_dir,what):
    module_name=str(_req(spec,"module",what))
    callable_name=str(_req(spec,"callable",what))
    extra=spec.get("python_path")
    if extra is not None:
        p=Path(extra)
        if not p.is_absolute(): p=case_dir/p
        if str(p.resolve()) not in sys.path:
            sys.path.insert(0,str(p.resolve()))
    try:
        mod=importlib.import_module(module_name)
    except Exception as exc:
        raise GeneralCaseInputError(
            f"cannot import {what} module '{module_name}': {exc}"
        ) from exc
    fn=getattr(mod,callable_name,None)
    if fn is None or not callable(fn):
        raise GeneralCaseInputError(
            f"{what} callable '{callable_name}' not found in '{module_name}'"
        )
    return fn


def _load_basis(spec,case_dir):
    provider=_resolve_module_callable(spec,case_dir,"cuf_basis")
    try:
        basis=provider()
    except Exception as exc:
        raise GeneralCaseInputError(f"CUF basis provider failed: {exc}") from exc
    if not isinstance(basis,g7.CallableThicknessBasis):
        raise GeneralCaseInputError(
            "CUF basis provider must return CallableThicknessBasis"
        )
    # evaluate once at z=0 to catch broken providers
    basis.values(0.0); basis.derivatives(0.0)
    return basis


def _bc_data(value,ns,case_dir):
    if isinstance(value,dict):
        if set(value)!={"npy"}:
            raise GeneralCaseInputError("BC data dict supports only {'npy': path}")
        p=Path(value["npy"])
        if not p.is_absolute(): p=case_dir/p
        try: a=np.asarray(np.load(p),float)
        except Exception as exc:
            raise GeneralCaseInputError(f"cannot load BC npy {p}: {exc}") from exc
        if a.shape!=(ns,):
            raise GeneralCaseInputError(f"BC npy must have shape ({ns},)")
        return a
    a=np.asarray(value,float)
    if a.ndim==0: return float(a)
    if a.shape==(ns,): return a
    raise GeneralCaseInputError(f"BC data must be scalar or shape ({ns},)")


def _parse_end(cfg,m,ns,case_dir,side):
    rows=[]
    for comp in COMPONENT_NAMES:
        comp_cfg=_req(cfg,comp,f"boundary_conditions.{side}")
        if not isinstance(comp_cfg,list) or len(comp_cfg)!=m:
            raise GeneralCaseInputError(
                f"boundary_conditions.{side}.{comp} must be list of length m={m}"
            )
        row=[]
        for tau,item in enumerate(comp_cfg):
            if not isinstance(item,dict):
                raise GeneralCaseInputError(
                    f"boundary_conditions.{side}.{comp}[{tau}] must be object"
                )
            context=f"boundary_conditions.{side}.{comp}[{tau}]"
            kind=str(_req(item,"kind",context)).upper()
            data=_req(item,"data",context)
            if kind=="ELASTIC":
                if "stiffness" not in item:
                    raise GeneralCaseInputError(f"{context}.stiffness is required for ELASTIC")
                row.append(g9.GeneralFieldEndCondition(
                    kind,
                    g9.ScalarOrCircumferentialData(_bc_data(data,ns,case_dir)),
                    g9.ScalarOrCircumferentialData(_bc_data(item["stiffness"],ns,case_dir)),
                    g9.ScalarOrCircumferentialData(_bc_data(item.get("reference",0.0),ns,case_dir)),
                ))
            else:
                row.append(g9.GeneralFieldEndCondition(
                    kind,g9.ScalarOrCircumferentialData(
                        _bc_data(data,ns,case_dir)
                    )
                ))
        rows.append(tuple(row))
    return g9.GeneralEndBoundarySet(tuple(rows))


def _parse_bc(cfg,m,ns,case_dir):
    left=_parse_end(_req(cfg,"left","boundary_conditions"),m,ns,case_dir,"left")
    right=_parse_end(_req(cfg,"right","boundary_conditions"),m,ns,case_dir,"right")
    bc=g9.GeneralCylinderEndBC(left,right)
    bc.check_rigid_mode_requirement()
    return bc


def _load_general_loads(path,m,ns,nx):
    p=Path(path)
    try: d=np.load(p)
    except Exception as exc:
        raise GeneralCaseInputError(f"cannot load loads NPZ {p}: {exc}") from exc
    if "loads" not in d.files:
        raise GeneralCaseInputError("loads NPZ must contain 'loads'")
    loads=np.asarray(d["loads"],float)
    expected_tail=(3,m,ns,nx)
    if loads.ndim!=5 or loads.shape[1:]!=expected_tail:
        raise GeneralCaseInputError(
            f"loads must have shape (n_steps,3,{m},{ns},{nx}), got {loads.shape}"
        )
    if loads.shape[0]<1 or not np.all(np.isfinite(loads)):
        raise GeneralCaseInputError("loads empty or non-finite")
    factors=None
    if "load_factors" in d.files:
        factors=np.asarray(d["load_factors"],float)
        if factors.shape!=(loads.shape[0],):
            raise GeneralCaseInputError("load_factors shape mismatch")
        factors=factors.tolist()
    return [loads[k].copy() for k in range(loads.shape[0])],factors


def build_general_case(case_json):
    case_json=Path(case_json).resolve()
    case_dir=case_json.parent
    cfg=_load_json(case_json)

    mode=str(_req(cfg,"mode","case")).upper()
    if mode not in g11.MODES:
        raise GeneralCaseInputError(f"mode must be one of {g11.MODES}")

    rank=_intmin(_req(cfg,"rank","case"),"rank",1)
    basis=_load_basis(_req(cfg,"cuf_basis","case"),case_dir)
    m=basis.m

    geom=_req(cfg,"geometry","case")
    L=_pos(_req(geom,"L","geometry"),"geometry.L")
    R=_pos(_req(geom,"R","geometry"),"geometry.R")
    h=_pos(_req(geom,"h","geometry"),"geometry.h")
    if 1-h/(2*R)<=0:
        raise GeneralCaseInputError("invalid geometry: H<=0 inside thickness")

    mat=_req(cfg,"material","case")
    E0=_pos(_req(mat,"E0","material"),"material.E0")
    nu0=float(_req(mat,"nu0","material"))
    lk.ElasticConstants(E0,nu0)

    gr=_req(cfg,"grid","case")
    nx=_intmin(_req(gr,"nx","grid"),"grid.nx",7)
    ns=_intmin(_req(gr,"ns","grid"),"grid.ns",7)
    if ns % 2 == 0:
        raise GeneralCaseInputError(
            "grid.ns должно быть нечётным: при чётном ns центральный "
            "периодический D1s обнуляет пилообразную моду (-1)^j"
        )
    stencil=_intmin(_req(gr,"stencil","grid"),"grid.stencil",3)
    if stencil%2!=1 or nx<stencil or ns<stencil:
        raise GeneralCaseInputError("stencil must be odd and <= nx,ns")

    grid=t1.CylinderGrid(L,R,nx,ns)
    ops=c3.CylinderSpatialOperators.build(grid,stencil)

    qc=_req(cfg,"quadrature","case")
    qtype=str(_req(qc,"type","quadrature")).upper()
    if qtype=="GAUSS":
        quadrature=s2.gauss_thickness_quadrature(
            h,_intmin(_req(qc,"n_gauss","quadrature"),"quadrature.n_gauss",1)
        )
    elif qtype in ("LAYERED","SUBLAYERS"):
        bounds=qc.get("layer_bounds",qc.get("bounds"))
        if bounds is None:
            raise GeneralCaseInputError("LAYERED quadrature requires layer_bounds")
        orders=qc.get("n_gauss_per_layer",qc.get("n_gauss"))
        if orders is None:
            raise GeneralCaseInputError("LAYERED quadrature requires n_gauss or n_gauss_per_layer")
        try:
            quadrature=s2.layered_thickness_quadrature(h,bounds,orders)
        except Exception as exc:
            raise GeneralCaseInputError(f"invalid LAYERED quadrature: {exc}") from exc
    else:
        raise GeneralCaseInputError("quadrature.type must be GAUSS or LAYERED")

    bc=_parse_bc(_req(cfg,"boundary_conditions","case"),m,ns,case_dir)

    im=_req(cfg,"inner_mvi","case")
    inner=g10.GeneralInternalMVISettings(
        projected_tolerance=_pos(_req(im,"projected_tolerance","inner_mvi"),"inner_mvi.projected_tolerance"),
        increment_tolerance=_pos(_req(im,"increment_tolerance","inner_mvi"),"inner_mvi.increment_tolerance"),
        tangent_fd_step=_pos(_req(im,"tangent_fd_step","inner_mvi"),"inner_mvi.tangent_fd_step"),
        max_halfstep_iterations=_intmin(_req(im,"max_halfstep_iterations","inner_mvi"),"inner_mvi.max_halfstep_iterations",1),
        max_xy_cycles=_intmin(_req(im,"max_xy_cycles","inner_mvi"),"inner_mvi.max_xy_cycles",1),
        detf_guarded_correction=bool(im.get("detf_guarded_correction",True)),
        max_correction_halvings=_intmin(im.get("max_correction_halvings",24),"inner_mvi.max_correction_halvings",0),
        full_residual_tolerance=_pos(
            im.get("full_residual_tolerance",im.get("projected_tolerance")),
            "inner_mvi.full_residual_tolerance"
        ),
        # Normal TE1--TE4 CASE runs now use the true frozen-partner Newton
        # halfstep.  Set PLATE_STYLE_ONE_CORRECTION explicitly to reproduce
        # the pre-strict legacy algorithm.
        algorithm=str(im.get("algorithm","STRICT_HALFSTEP")).upper(),
        formulation=str(im.get("formulation","STRONG")).upper(),
    )
    if str(inner.formulation).upper()=="VARIATIONAL":
        if mode not in ("CUF_LINEAR_ELASTIC","CUF_VK_ELASTIC","CUF_NL_ELASTIC"):
            raise GeneralCaseInputError(
                "VARIATIONAL MVI is currently verified for Hooke CUF with "
                "LINEAR, VON_KARMAN or GREEN_LAGRANGE kinematics"
            )
        if not (bc.left.all_geometric() and bc.right.all_geometric()):
            raise GeneralCaseInputError(
                "VARIATIONAL MVI requires geometric conditions on both cylinder ends"
            )

    prof=Path(_req(cfg,"initial_profiles_npz","case"))
    if not prof.is_absolute(): prof=case_dir/prof
    initial=g11.load_initial_profiles_npz(
        prof,m=m,rank=rank,nx=nx,ns=ns
    )
    # apply/recheck only homogeneous geometric data; never rewrite nonzero data
    try:
        g9.enforce_homogeneous_geometric_x_profiles(initial,grid,bc)
    except Exception as exc:
        raise GeneralCaseInputError(f"initial profiles violate BC policy: {exc}") from exc

    lp=Path(_req(cfg,"loads_npz","case"))
    if not lp.is_absolute(): lp=case_dir/lp
    loads,factors=_load_general_loads(lp,m,ns,nx)

    birger_settings=None
    diagram=None
    if mode in ("CUF_LINEAR_BIRGER","CUF_VK_BIRGER","CUF_BIRGER"):
        bg=_req(cfg,"birger","case")
        birger_settings=b5.BirgerSettings(
            epsilon_tol=_nonneg(_req(bg,"epsilon_tol","birger"),"birger.epsilon_tol"),
            omega=_pos(_req(bg,"omega","birger"),"birger.omega"),
            G_ref=_pos(_req(bg,"G_ref","birger"),"birger.G_ref"),
            sigma_ref=_pos(_req(bg,"sigma_ref","birger"),"birger.sigma_ref"),
            tol_G=_pos(_req(bg,"tol_G","birger"),"birger.tol_G"),
            tol_sigma=_pos(_req(bg,"tol_sigma","birger"),"birger.tol_sigma"),
            tol_u=_pos(_req(bg,"tol_u","birger"),"birger.tol_u"),
            max_outer_iterations=_intmin(_req(bg,"max_outer_iterations","birger"),"birger.max_outer_iterations",1),
            epsilon_i_limit=_pos(_req(bg,"epsilon_i_limit","birger"),"birger.epsilon_i_limit"),
            delta_J_limit=_pos(_req(bg,"delta_J_limit","birger"),"birger.delta_J_limit"),
            initial_slope_rtol=_pos(bg.get("initial_slope_rtol",0.05),"birger.initial_slope_rtol"),
            reject_local_unloading=bg.get("reject_local_unloading",True),
            unloading_tolerance=_nonneg(
                bg.get("unloading_tolerance",1.0e-12),
                "birger.unloading_tolerance"
            ),
        )
        diagram=_resolve_module_callable(
            _req(bg,"material_diagram","birger"),case_dir,"material_diagram"
        )

    out=Path(_req(cfg,"output_dir","case"))
    if not out.is_absolute(): out=case_dir/out

    adapt=(cfg.get("load_step_adaptation",{}) or {})
    automatic_step_subdivision=bool(adapt.get("enabled",True))
    max_subdivision_depth=_intmin(
        adapt.get("max_subdivision_depth",8),
        "load_step_adaptation.max_subdivision_depth",0
    )
    proactive_step_adaptation=bool(adapt.get("proactive_enabled",True))
    proactive_outer_soft_limit=_intmin(
        adapt.get("outer_soft_limit",8),
        "load_step_adaptation.outer_soft_limit",1
    )
    proactive_outer_hard_limit=_intmin(
        adapt.get("outer_hard_limit",20),
        "load_step_adaptation.outer_hard_limit",2
    )
    if proactive_outer_hard_limit<=proactive_outer_soft_limit:
        raise ValueError("load_step_adaptation.outer_hard_limit must be > outer_soft_limit")

    return dict(
        mode=mode,rank=rank,basis=basis,m=m,
        E0=E0,nu0=nu0,grid=grid,ops=ops,quadrature=quadrature,
        bc=bc,inner_settings=inner,initial_state=initial,
        load_steps=loads,load_factors=factors,
        birger_settings=birger_settings,material_diagram=diagram,
        output_dir=out,case_json=case_json,
        step_predictor_mode=(cfg.get("step_predictor",{}) or {}).get("mode","ERROR"),
        mvi_algorithm=str(inner.algorithm).upper(),
        mvi_formulation=str(inner.formulation).upper(),
        automatic_step_subdivision=automatic_step_subdivision,
        max_subdivision_depth=max_subdivision_depth,
        proactive_step_adaptation=proactive_step_adaptation,
        proactive_outer_soft_limit=proactive_outer_soft_limit,
        proactive_outer_hard_limit=proactive_outer_hard_limit,
    )


def run_general_case(case_json):
    c=build_general_case(case_json)
    res=g11.solve_general_explicit_load_series(
        c["mode"],c["initial_state"],c["ops"],
        E0=c["E0"],nu0=c["nu0"],quadrature=c["quadrature"],
        basis=c["basis"],bc=c["bc"],
        load_steps=c["load_steps"],load_factors=c["load_factors"],
        inner_settings=c["inner_settings"],
        birger_settings=c["birger_settings"],
        material_diagram=c["material_diagram"],
        step_predictor_mode=c["step_predictor_mode"],
        automatic_step_subdivision=c["automatic_step_subdivision"],
        max_subdivision_depth=c["max_subdivision_depth"],
        proactive_step_adaptation=c["proactive_step_adaptation"],
        proactive_outer_soft_limit=c["proactive_outer_soft_limit"],
        proactive_outer_hard_limit=c["proactive_outer_hard_limit"],
        output_dir=c["output_dir"],
    )
    c["output_dir"].mkdir(parents=True,exist_ok=True)

    # Output-only publication of accepted CUF/MVI states.  The general load
    # driver historically defined _save_step() but did not call it, so a run
    # could report N accepted steps while the RUN folder contained no STEP
    # files.  Online plots, STOP/FULL recovery and Excel then had no history.
    # Persist the already-computed accepted states here; no MVI mechanics or
    # constitutive calculations are changed.
    K0_out=float(c["E0"])/(3.0*(1.0-2.0*float(c["nu0"])))
    for step in res.accepted_steps:
        step_npz=c["output_dir"]/f"STEP_{int(step.step_index):04d}.npz"
        step_json=c["output_dir"]/f"STEP_{int(step.step_index):04d}.json"
        if not (step_npz.exists() and step_json.exists()):
            g11._save_step(step,c["output_dir"],K0=K0_out if step.GB is not None else None)
    last_step=(res.accepted_steps[-1] if res.accepted_steps else None)
    last_history=(
        (last_step.solver_history[-1] if last_step is not None and last_step.solver_history else None)
    )
    (c["output_dir"]/ "RUN_RESULT.json").write_text(
        json.dumps({
            "mode":res.mode,
            "cuf_m":c["m"],
            "status":res.status,
            "converged":res.converged,
            "requested_steps":res.requested_steps,
            "accepted_steps":len(res.accepted_steps),
            "failed_step_index":res.failed_step_index,
            "failed_step_status":res.failed_step_status,
            "failed_step_code":getattr(res,"failed_step_code",None),
            "failed_step_diagnostic_ru":getattr(res,"failed_step_diagnostic_ru",None),
            "failed_step_recommendation_ru":getattr(res,"failed_step_recommendation_ru",None),
            "basis_source":"external provider",
            "initial_profiles_source":"external NPZ",
            "mvi_algorithm":c["mvi_algorithm"],
            "mvi_formulation":c["mvi_formulation"],
            "final_mvi_status":(None if last_step is None else str(last_step.solver_status)),
            "final_projected_relative":(
                None if last_history is None else last_history.get("projected_relative")
            ),
            "final_x_projected_relative":(
                None if last_history is None else last_history.get("X_projected_relative_after_cycle")
            ),
            "final_y_projected_relative":(
                None if last_history is None else last_history.get("Y_projected_relative_after_cycle")
            ),
            "final_full_residual_relative_to_load":(
                None if last_history is None else last_history.get("full_residual_relative_to_load")
            ),
            "full_equilibrium_satisfied":(
                None if last_history is None else last_history.get("full_equilibrium_satisfied")
            ),
            "automatic_step_subdivision":c["automatic_step_subdivision"],
            "max_subdivision_depth":c["max_subdivision_depth"],
        },indent=2,ensure_ascii=False),encoding="utf-8"
    )
    return res
