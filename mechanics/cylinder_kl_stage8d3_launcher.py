#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8D-3: strict CASE launcher for the six-mode KL cylinder solver.

Usage:
    python RUN_KL_CASE.py CASE_KL.json

No physical defaults are embedded.
All required physical/numerical data must be explicit in CASE_KL.json
and referenced NPZ/Python files.

Supported modes:
    KL_LINEAR_ELASTIC
    KL_LINEAR_BIRGER
    KL_VK_ELASTIC
    KL_VK_BIRGER
    KL_GL_ELASTIC
    KL_GL_BIRGER

Initial profiles NPZ:
    X shape (3,rank,nx)
    Y shape (3,rank,ns)

Loads NPZ:
    loads shape (nsteps,3,ns,nx)
    optional load_factors shape (nsteps,)

Birger material diagram:
    external Python callable Phi(epsilon_i)
"""
from __future__ import annotations
from pathlib import Path
import importlib
import importlib.util
import json
import sys
import numpy as np

import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage2b as s2
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage5a_birger as b5

import cylinder_kl_stage8b1_boundary as kb
import cylinder_kl_stage8b2_profiles as kp
import cylinder_kl_stage8b3_essential_bc as ke
import cylinder_kl_stage8b5_strict_mvi as km
import cylinder_kl_stage8d1_four_mode_driver as d1
import cylinder_kl_stage8d2_load_series as d2


class KLCaseError(RuntimeError):
    pass


def _read_json(path):
    p=Path(path)
    try:
        data=json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise KLCaseError(f"cannot read CASE JSON: {p}") from exc
    if not isinstance(data,dict):
        raise KLCaseError("CASE root must be a JSON object")
    return data,p.resolve()


def _require(obj,key,context):
    if key not in obj or obj[key] is None:
        raise KLCaseError(f"missing required field: {context}.{key}")
    return obj[key]


def _require_dict(obj,key,context):
    x=_require(obj,key,context)
    if not isinstance(x,dict):
        raise KLCaseError(f"{context}.{key} must be an object")
    return x


def _resolve(case_path,value):
    p=Path(str(value))
    if not p.is_absolute():
        p=case_path.parent/p
    return p.resolve()


def _positive_float(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0:
        raise KLCaseError(f"{name} must be finite and positive")
    return v


def _finite_float(x,name):
    v=float(x)
    if not np.isfinite(v):
        raise KLCaseError(f"{name} must be finite")
    return v


def _positive_int(x,name,min_value=1):
    try:
        v=int(x)
    except Exception as exc:
        raise KLCaseError(f"{name} must be integer") from exc
    if v<min_value:
        raise KLCaseError(f"{name} must be >= {min_value}")
    if float(x)!=float(v):
        raise KLCaseError(f"{name} must be integer-valued")
    return v


def _load_external_callable(spec,case_path,context):
    if not isinstance(spec,dict):
        raise KLCaseError(f"{context} must be an object")
    module_name=_require(spec,"module",context)
    callable_name=_require(spec,"callable",context)
    python_path=spec.get("python_path",None)

    if python_path is not None:
        pp=_resolve(case_path,python_path)
        if not pp.exists():
            raise KLCaseError(f"{context}.python_path does not exist: {pp}")
        if str(pp) not in sys.path:
            sys.path.insert(0,str(pp))

    try:
        mod=importlib.import_module(str(module_name))
    except Exception as exc:
        raise KLCaseError(
            f"cannot import module {module_name!r} for {context}"
        ) from exc

    if not hasattr(mod,str(callable_name)):
        raise KLCaseError(
            f"module {module_name!r} has no callable {callable_name!r}"
        )
    fn=getattr(mod,str(callable_name))
    if not callable(fn):
        raise KLCaseError(
            f"{context}.{callable_name} is not callable"
        )
    return fn


def _parse_bc_condition(obj,context):
    if not isinstance(obj,dict):
        raise KLCaseError(f"{context} must be an object")
    kind=str(_require(obj,"kind",context)).upper()
    if kind not in ("GEOMETRIC","NATURAL","ELASTIC"):
        raise KLCaseError(
            f"{context}.kind must be GEOMETRIC, NATURAL or ELASTIC"
        )
    value=_require(obj,"value",context)
    if kind=="ELASTIC":
        if "stiffness" not in obj:
            raise KLCaseError(f"{context}.stiffness is required for ELASTIC")
        return kb.KLBoundaryCondition(
            kind,kb.KLBoundaryData(value),
            kb.KLBoundaryData(obj["stiffness"]),
            kb.KLBoundaryData(obj.get("reference",0.0)),
        )
    return kb.KLBoundaryCondition(kind,kb.KLBoundaryData(value))


def _parse_end_bc(obj,context):
    if not isinstance(obj,dict):
        raise KLCaseError(f"{context} must be an object")
    return kb.KLEndBoundarySet(
        u=_parse_bc_condition(_require(obj,"u",context),f"{context}.u"),
        v=_parse_bc_condition(_require(obj,"v",context),f"{context}.v"),
        w=_parse_bc_condition(_require(obj,"w",context),f"{context}.w"),
        slope=_parse_bc_condition(
            _require(obj,"slope",context),f"{context}.slope"
        ),
    )


def _parse_boundary_conditions(case):
    bcj=_require_dict(case,"boundary_conditions","CASE")
    left=_parse_end_bc(_require(bcj,"left","CASE.boundary_conditions"),
                       "CASE.boundary_conditions.left")
    right=_parse_end_bc(_require(bcj,"right","CASE.boundary_conditions"),
                        "CASE.boundary_conditions.right")
    return kb.KLCylinderEndBC(left,right)


def _load_profiles(path,rank,grid):
    p=Path(path)
    if not p.exists():
        raise KLCaseError(f"initial_profiles_npz does not exist: {p}")
    try:
        with np.load(p,allow_pickle=False) as z:
            if "X" not in z.files or "Y" not in z.files:
                raise KLCaseError("initial profiles NPZ must contain X and Y")
            X=np.asarray(z["X"],float)
            Y=np.asarray(z["Y"],float)
    except KLCaseError:
        raise
    except Exception as exc:
        raise KLCaseError(f"cannot read initial profiles NPZ: {p}") from exc

    expected_X=(3,rank,grid.nx)
    expected_Y=(3,rank,grid.ns)
    if X.shape!=expected_X:
        raise KLCaseError(
            f"X shape {X.shape}, expected {expected_X}"
        )
    if Y.shape!=expected_Y:
        raise KLCaseError(
            f"Y shape {Y.shape}, expected {expected_Y}"
        )
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise KLCaseError("initial X/Y contain non-finite values")
    return kp.ProfileStateKL(X.copy(),Y.copy())


def _load_loads(path,grid):
    p=Path(path)
    if not p.exists():
        raise KLCaseError(f"loads_npz does not exist: {p}")
    try:
        with np.load(p,allow_pickle=False) as z:
            if "loads" not in z.files:
                raise KLCaseError("loads NPZ must contain 'loads'")
            loads=np.asarray(z["loads"],float)
            factors=(
                None if "load_factors" not in z.files
                else np.asarray(z["load_factors"],float)
            )
    except KLCaseError:
        raise
    except Exception as exc:
        raise KLCaseError(f"cannot read loads NPZ: {p}") from exc

    expected_tail=(3,grid.ns,grid.nx)
    if loads.ndim!=4 or tuple(loads.shape[1:])!=expected_tail:
        raise KLCaseError(
            f"loads shape {loads.shape}, expected (nsteps,{expected_tail[0]},"
            f"{expected_tail[1]},{expected_tail[2]})"
        )
    if loads.shape[0]<1:
        raise KLCaseError("loads must contain at least one step")
    if not np.all(np.isfinite(loads)):
        raise KLCaseError("loads contain non-finite values")

    if factors is not None:
        if factors.shape!=(loads.shape[0],):
            raise KLCaseError(
                f"load_factors shape {factors.shape}, expected ({loads.shape[0]},)"
            )
        if not np.all(np.isfinite(factors)):
            raise KLCaseError("load_factors contain non-finite values")
        factors=[float(x) for x in factors]
    return [loads[k].copy() for k in range(loads.shape[0])],factors


def _build_quadrature(qj,h):
    qtype=str(_require(qj,"type","CASE.quadrature")).upper()
    if qtype=="GAUSS":
        ng=_positive_int(
            _require(qj,"n_gauss","CASE.quadrature"),
            "CASE.quadrature.n_gauss",
            min_value=1,
        )
        return s2.gauss_thickness_quadrature(h,ng)
    if qtype in ("LAYERED","SUBLAYERS"):
        bounds=qj.get("layer_bounds",qj.get("bounds"))
        if bounds is None:
            raise KLCaseError("LAYERED quadrature requires layer_bounds")
        orders=qj.get("n_gauss_per_layer",qj.get("n_gauss"))
        if orders is None:
            raise KLCaseError("LAYERED quadrature requires n_gauss or n_gauss_per_layer")
        try:
            return s2.layered_thickness_quadrature(h,bounds,orders)
        except Exception as exc:
            raise KLCaseError(f"invalid LAYERED quadrature: {exc}") from exc
    raise KLCaseError("quadrature.type must be GAUSS or LAYERED")


def _build_mvi_settings(j):
    return km.KLInternalMVISettings(
        projected_tolerance=_positive_float(
            _require(j,"projected_tolerance","CASE.inner_mvi"),
            "CASE.inner_mvi.projected_tolerance"
        ),
        increment_tolerance=_positive_float(
            _require(j,"increment_tolerance","CASE.inner_mvi"),
            "CASE.inner_mvi.increment_tolerance"
        ),
        tangent_fd_step=_positive_float(
            _require(j,"tangent_fd_step","CASE.inner_mvi"),
            "CASE.inner_mvi.tangent_fd_step"
        ),
        max_halfstep_iterations=_positive_int(
            _require(j,"max_halfstep_iterations","CASE.inner_mvi"),
            "CASE.inner_mvi.max_halfstep_iterations"
        ),
        max_xy_cycles=_positive_int(
            _require(j,"max_xy_cycles","CASE.inner_mvi"),
            "CASE.inner_mvi.max_xy_cycles"
        ),
    )


def _build_birger(case,case_path,mode,G0):
    if not d1.is_birger_mode(mode):
        return None,None
    bj=_require_dict(case,"birger","CASE")
    diagram_spec=_require_dict(bj,"material_diagram","CASE.birger")
    diagram=_load_external_callable(
        diagram_spec,case_path,"CASE.birger.material_diagram"
    )
    settings=b5.BirgerSettings(
        epsilon_tol=_finite_float(
            _require(bj,"epsilon_tol","CASE.birger"),
            "CASE.birger.epsilon_tol"
        ),
        omega=_positive_float(
            _require(bj,"omega","CASE.birger"),
            "CASE.birger.omega"
        ),
        G_ref=_positive_float(
            _require(bj,"G_ref","CASE.birger"),
            "CASE.birger.G_ref"
        ),
        sigma_ref=_positive_float(
            _require(bj,"sigma_ref","CASE.birger"),
            "CASE.birger.sigma_ref"
        ),
        tol_G=_positive_float(
            _require(bj,"tol_G","CASE.birger"),
            "CASE.birger.tol_G"
        ),
        tol_sigma=_positive_float(
            _require(bj,"tol_sigma","CASE.birger"),
            "CASE.birger.tol_sigma"
        ),
        tol_u=_positive_float(
            _require(bj,"tol_u","CASE.birger"),
            "CASE.birger.tol_u"
        ),
        max_outer_iterations=_positive_int(
            _require(bj,"max_outer_iterations","CASE.birger"),
            "CASE.birger.max_outer_iterations"
        ),
        epsilon_i_limit=_positive_float(
            _require(bj,"epsilon_i_limit","CASE.birger"),
            "CASE.birger.epsilon_i_limit"
        ),
        delta_J_limit=_positive_float(
            _require(bj,"delta_J_limit","CASE.birger"),
            "CASE.birger.delta_J_limit"
        ),
        initial_slope_rtol=_positive_float(
            bj.get("initial_slope_rtol",0.05),
            "CASE.birger.initial_slope_rtol"
        ),
        reject_local_unloading=bj.get("reject_local_unloading",True),
        unloading_tolerance=_finite_float(
            bj.get("unloading_tolerance",1.0e-12),
            "CASE.birger.unloading_tolerance"
        ),
    )
    return diagram,settings


def run_kl_case(case_json):
    case,case_path=_read_json(case_json)

    mode=d1.validate_mode(_require(case,"mode","CASE"))
    rank=_positive_int(_require(case,"rank","CASE"),"CASE.rank")

    gj=_require_dict(case,"geometry","CASE")
    L=_positive_float(_require(gj,"L","CASE.geometry"),"CASE.geometry.L")
    R=_positive_float(_require(gj,"R","CASE.geometry"),"CASE.geometry.R")
    h=_positive_float(_require(gj,"h","CASE.geometry"),"CASE.geometry.h")

    mj=_require_dict(case,"material","CASE")
    E0=_positive_float(_require(mj,"E0","CASE.material"),"CASE.material.E0")
    nu0=_finite_float(_require(mj,"nu0","CASE.material"),"CASE.material.nu0")
    if not (-1.0<nu0<0.5):
        raise KLCaseError("CASE.material.nu0 must satisfy -1 < nu0 < 0.5")

    grj=_require_dict(case,"grid","CASE")
    nx=_positive_int(_require(grj,"nx","CASE.grid"),"CASE.grid.nx",min_value=7)
    ns=_positive_int(_require(grj,"ns","CASE.grid"),"CASE.grid.ns",min_value=7)
    if ns % 2 == 0:
        raise KLCaseError(
            "CASE.grid.ns должно быть нечётным: при чётном ns центральный "
            "периодический D1s обнуляет пилообразную моду (-1)^j"
        )
    stencil=_positive_int(
        _require(grj,"stencil","CASE.grid"),
        "CASE.grid.stencil",
        min_value=3,
    )
    if stencil!=7:
        raise KLCaseError("current verified KL launcher requires grid.stencil=7")

    grid=mvi.CylinderGrid(L,R,nx,ns)
    ops=c3.CylinderSpatialOperators.build(grid,stencil)

    qj=_require_dict(case,"quadrature","CASE")
    quadrature=_build_quadrature(qj,h)

    bc=_parse_boundary_conditions(case)
    # Explicitly reject fully natural/free-free before any solve.
    bc.check_rigid_mode_requirement()

    maps=ke.build_x_constraint_maps(ops,bc)

    imj=_require_dict(case,"inner_mvi","CASE")
    mvi_settings=_build_mvi_settings(imj)

    profile_path=_resolve(
        case_path,_require(case,"initial_profiles_npz","CASE")
    )
    state=_load_profiles(profile_path,rank,grid)

    # Do NOT silently project an incompatible profile state.
    ke.reduce_constrained_state_X(state,maps)

    loads_path=_resolve(case_path,_require(case,"loads_npz","CASE"))
    loads,load_factors=_load_loads(loads_path,grid)

    output_dir=_resolve(case_path,_require(case,"output_dir","CASE"))

    G0=E0/(2.0*(1.0+nu0))
    material_diagram,birger_settings=_build_birger(
        case,case_path,mode,G0
    )

    result=d2.run_kl_load_series(
        mode,
        state,
        loads,
        ops=ops,maps=maps,bc=bc,
        E0=E0,nu0=nu0,h=h,
        quadrature=quadrature,
        mvi_settings=mvi_settings,
        load_factors=load_factors,
        material_diagram=material_diagram,
        birger_settings=birger_settings,
        step_predictor_mode=case.get("step_predictor",{}).get("mode","ERROR"),
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True,exist_ok=True)
    summary={
        "mode":result.mode,
        "converged":result.converged,
        "status":result.status,
        "requested_steps":result.requested_steps,
        "accepted_steps":len(result.accepted_steps),
        "failed_step_index":result.failed_step_index,
        "failed_step_status":result.failed_step_status,
        "rank":rank,
        "nx":nx,
        "ns":ns,
        "L":L,
        "R":R,
        "h":h,
        "E0":E0,
        "nu0":nu0,
        "output_dir":str(output_dir),
    }
    (output_dir/"CASE_RUN_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),
        encoding="utf-8"
    )
    return result
