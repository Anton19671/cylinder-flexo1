#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUF-TE1 Stage 6A: strict input layer and launcher for an explicit physical case.

NO numerical defaults are supplied for the physical problem.
All required physical/numerical values must be explicitly present in the case JSON.

Load arrays are read from an external NPZ file:
    loads : shape (n_steps, 6, ns, nx)
Optional:
    load_factors : shape (n_steps,)

For BIRGER, the material diagram is NOT embedded or interpolated here.
The JSON must specify a Python module + callable name that implements:
    Phi(epsilon_i) -> sigma_i
using the user's accepted material diagram and conversion/interpolation rules.
"""
from __future__ import annotations

from pathlib import Path
import importlib
import json
import numpy as np

import cylinder_te1_mvi_core as mvi
import cylinder_te1_local_kernel as nk
import cylinder_te1_stage2b as s2
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage3b_boundary as b3
import cylinder_te1_stage3c_internal_mvi as c4
import cylinder_te1_stage5a_birger as b5
import cylinder_te1_stage5b_load_driver as d5


class CaseInputError(RuntimeError):
    pass


REQUIRED_TOP = (
    "mode",
    "geometry",
    "material",
    "grid",
    "quadrature",
    "boundary_conditions",
    "inner_mvi",
    "loads_npz",
    "output_dir",
)


def _required(mapping, key, where):
    if key not in mapping:
        raise CaseInputError(f"MISSING_REQUIRED_INPUT: {where}.{key}")
    value = mapping[key]
    if value is None:
        raise CaseInputError(f"UNSPECIFIED_REQUIRED_INPUT: {where}.{key}")
    return value


def _positive(value, name):
    x=float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise CaseInputError(f"{name} must be finite and >0")
    return x


def _nonnegative(value, name):
    x=float(value)
    if not np.isfinite(x) or x < 0.0:
        raise CaseInputError(f"{name} must be finite and >=0")
    return x


def _integer_min(value, name, minimum):
    x=int(value)
    if x < minimum:
        raise CaseInputError(f"{name} must be >= {minimum}")
    return x


def _load_json(path):
    path=Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CaseInputError(f"CASE_FILE_NOT_FOUND: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CaseInputError(f"INVALID_CASE_JSON: {exc}") from exc


def _parse_data(value, ns, case_dir):
    """
    BC data can be:
      - scalar number;
      - explicit list of ns numbers;
      - {"npy": "relative/or/absolute/path.npy"} with shape (ns,).
    """
    if isinstance(value, dict):
        if set(value.keys()) != {"npy"}:
            raise CaseInputError("BC data dict must contain only key 'npy'")
        p=Path(value["npy"])
        if not p.is_absolute():
            p=case_dir/p
        try:
            arr=np.load(p)
        except Exception as exc:
            raise CaseInputError(f"cannot read BC npy: {p}: {exc}") from exc
        arr=np.asarray(arr,float)
        if arr.shape != (ns,):
            raise CaseInputError(f"BC npy {p} must have shape ({ns},)")
        return arr
    arr=np.asarray(value,float)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape == (ns,):
        return arr
    raise CaseInputError(f"BC data must be scalar or shape ({ns},)")


def _parse_end(end_cfg, ns, case_dir, side):
    if not isinstance(end_cfg, dict):
        raise CaseInputError(f"{side} boundary_conditions must be an object")
    mapping={}
    for name in b3.FIELD_NAMES:
        item=_required(end_cfg,name,f"boundary_conditions.{side}")
        if not isinstance(item,dict):
            raise CaseInputError(f"boundary_conditions.{side}.{name} must be an object")
        kind=str(_required(item,"kind",f"boundary_conditions.{side}.{name}")).upper()
        data=_required(item,"data",f"boundary_conditions.{side}.{name}")
        mapping[name]=b3.FieldEndCondition(
            kind,
            b3.ScalarOrCircumferentialData(_parse_data(data,ns,case_dir))
        )
    extra=set(end_cfg)-set(b3.FIELD_NAMES)
    if extra:
        raise CaseInputError(f"unknown fields at {side} end: {sorted(extra)}")
    return b3.EndBoundarySet.from_mapping(mapping)


def _parse_bc(cfg, ns, case_dir):
    left=_parse_end(_required(cfg,"left","boundary_conditions"),ns,case_dir,"left")
    right=_parse_end(_required(cfg,"right","boundary_conditions"),ns,case_dir,"right")
    bc=b3.CylinderEndBC(left,right)
    # Do not invent fully-free rigid-mode constraints.
    bc.check_rigid_mode_requirement()
    return bc


def _load_steps(path, expected_shape):
    path=Path(path)
    try:
        data=np.load(path)
    except Exception as exc:
        raise CaseInputError(f"cannot load load-step NPZ {path}: {exc}") from exc
    if "loads" not in data.files:
        raise CaseInputError("load NPZ must contain array 'loads'")
    loads=np.asarray(data["loads"],float)
    if loads.ndim != 4 or loads.shape[1:] != expected_shape:
        raise CaseInputError(
            f"loads must have shape (n_steps,{expected_shape[0]},{expected_shape[1]},{expected_shape[2]}), "
            f"got {loads.shape}"
        )
    if loads.shape[0] < 1:
        raise CaseInputError("loads must contain at least one step")
    if not np.all(np.isfinite(loads)):
        raise CaseInputError("loads contain non-finite values")
    factors=None
    if "load_factors" in data.files:
        factors=np.asarray(data["load_factors"],float)
        if factors.shape != (loads.shape[0],):
            raise CaseInputError("load_factors shape mismatch")
        if not np.all(np.isfinite(factors)):
            raise CaseInputError("load_factors contain non-finite values")
    return [loads[k].copy() for k in range(loads.shape[0])], (
        None if factors is None else factors.tolist()
    )


def _material_diagram(cfg, case_dir):
    module_name=str(_required(cfg,"module","birger.material_diagram"))
    callable_name=str(_required(cfg,"callable","birger.material_diagram"))
    extra_path=cfg.get("python_path")
    if extra_path is not None:
        import sys
        p=Path(extra_path)
        if not p.is_absolute():
            p=case_dir/p
        sys.path.insert(0,str(p))
    try:
        mod=importlib.import_module(module_name)
    except Exception as exc:
        raise CaseInputError(f"cannot import material diagram module '{module_name}': {exc}") from exc
    fn=getattr(mod,callable_name,None)
    if fn is None or not callable(fn):
        raise CaseInputError(
            f"material diagram callable '{callable_name}' not found in module '{module_name}'"
        )
    return fn


def build_case(case_json):
    case_json=Path(case_json).resolve()
    case_dir=case_json.parent
    cfg=_load_json(case_json)

    for key in REQUIRED_TOP:
        _required(cfg,key,"case")

    mode=str(cfg["mode"]).upper()
    if mode not in d5.MODES:
        raise CaseInputError(f"mode must be one of {d5.MODES}")

    g=cfg["geometry"]
    L=_positive(_required(g,"L","geometry"),"geometry.L")
    R=_positive(_required(g,"R","geometry"),"geometry.R")
    h=_positive(_required(g,"h","geometry"),"geometry.h")
    # H=1+z/R must stay positive throughout thickness.
    if 1.0 - h/(2.0*R) <= 0.0:
        raise CaseInputError("invalid geometry: H=1+z/R is non-positive inside thickness")

    mat=cfg["material"]
    E0=_positive(_required(mat,"E0","material"),"material.E0")
    nu0=float(_required(mat,"nu0","material"))
    # rely on ElasticConstants for its exact admissibility checks too
    nk.ElasticConstants(E0,nu0)

    gr=cfg["grid"]
    nx=_integer_min(_required(gr,"nx","grid"),"grid.nx",7)
    ns=_integer_min(_required(gr,"ns","grid"),"grid.ns",7)
    if ns % 2 == 0:
        raise CaseInputError(
            "grid.ns должно быть нечётным: при чётном ns центральный "
            "периодический D1s обнуляет пилообразную моду (-1)^j"
        )
    stencil=_integer_min(_required(gr,"stencil","grid"),"grid.stencil",3)
    if stencil % 2 != 1:
        raise CaseInputError("grid.stencil must be odd")
    if nx < stencil or ns < stencil:
        raise CaseInputError("grid nx/ns must be >= stencil")

    grid=mvi.CylinderGrid(L,R,nx,ns)
    ops=c3.CylinderSpatialOperators.build(grid,stencil)

    qc=cfg["quadrature"]
    qtype=str(_required(qc,"type","quadrature")).upper()
    if qtype == "GAUSS":
        ng=_integer_min(_required(qc,"n_gauss","quadrature"),"quadrature.n_gauss",1)
        quadrature=s2.gauss_thickness_quadrature(h,ng)
    elif qtype == "SUBLAYERS":
        bounds=_required(qc,"bounds","quadrature")
        orders=_required(qc,"n_gauss_per_layer","quadrature")
        quadrature=s2.layered_thickness_quadrature(h,bounds,orders)
    else:
        raise CaseInputError("quadrature.type must be GAUSS or SUBLAYERS")

    bc=_parse_bc(cfg["boundary_conditions"],ns,case_dir)

    im=cfg["inner_mvi"]
    inner_settings=c4.InternalMVISettings(
        projected_tolerance=_positive(_required(im,"projected_tolerance","inner_mvi"),"inner_mvi.projected_tolerance"),
        increment_tolerance=_positive(_required(im,"increment_tolerance","inner_mvi"),"inner_mvi.increment_tolerance"),
        tangent_fd_step=_positive(_required(im,"tangent_fd_step","inner_mvi"),"inner_mvi.tangent_fd_step"),
        max_halfstep_iterations=_integer_min(_required(im,"max_halfstep_iterations","inner_mvi"),"inner_mvi.max_halfstep_iterations",1),
        max_xy_cycles=_integer_min(_required(im,"max_xy_cycles","inner_mvi"),"inner_mvi.max_xy_cycles",1),
    )

    loads_path=Path(cfg["loads_npz"])
    if not loads_path.is_absolute():
        loads_path=case_dir/loads_path
    load_steps,load_factors=_load_steps(
        loads_path,(6,grid.ns,grid.nx)
    )

    rank=_integer_min(_required(cfg,"rank","case"),"case.rank",1)
    # Never start both MVI factors at zero: the frozen-partner tangent
    # then has no directions and cannot respond to a nonzero load.
    # A user-supplied profile NPZ remains the authoritative initial state.
    if rank>min(grid.nx,grid.ns):
        raise CaseInputError("rank exceeds grid dimensions")
    profile_source=cfg.get("initial_profiles_npz")
    if profile_source is not None:
        profile_path=Path(profile_source)
        if not profile_path.is_absolute():
            profile_path=case_dir/profile_path
        try:
            with np.load(profile_path,allow_pickle=False) as saved:
                X=np.asarray(saved["X"],float).copy()
                Y=np.asarray(saved["Y"],float).copy()
        except (OSError,KeyError,ValueError) as exc:
            raise CaseInputError(
                "initial_profiles_npz must contain finite X/Y profiles") from exc
        if X.shape!=(6,rank,grid.nx) or Y.shape!=(6,rank,grid.ns):
            raise CaseInputError("initial_profiles_npz X/Y shape mismatch")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise CaseInputError("initial_profiles_npz contains nonfinite profiles")
        initial_state=mvi.ProfileStateTE1(X,Y)
    else:
        # Deterministic rank-fixed cold start: zero reconstructed displacement,
        # but independent nonzero circumferential partners.
        Y=np.broadcast_to(
            np.cos(np.pi*np.arange(rank)[:,None]*
                   (np.arange(grid.ns)[None,:]+0.5)/grid.ns),
            (6,rank,grid.ns)).copy()
        Y=np.broadcast_to(
            np.linalg.qr(Y[0].T,mode="reduced")[0].T,
            (6,rank,grid.ns)).copy()
        initial_state=mvi.ProfileStateTE1(
            np.zeros((6,rank,grid.nx),float),Y)

    birger_settings=None
    diagram=None
    if mode in ("LINEAR_BIRGER", "BIRGER"):
        bg=_required(cfg,"birger","case")
        birger_settings=b5.BirgerSettings(
            epsilon_tol=_nonnegative(_required(bg,"epsilon_tol","birger"),"birger.epsilon_tol"),
            omega=_positive(_required(bg,"omega","birger"),"birger.omega"),
            G_ref=_positive(_required(bg,"G_ref","birger"),"birger.G_ref"),
            sigma_ref=_positive(_required(bg,"sigma_ref","birger"),"birger.sigma_ref"),
            tol_G=_positive(_required(bg,"tol_G","birger"),"birger.tol_G"),
            tol_sigma=_positive(_required(bg,"tol_sigma","birger"),"birger.tol_sigma"),
            tol_u=_positive(_required(bg,"tol_u","birger"),"birger.tol_u"),
            max_outer_iterations=_integer_min(_required(bg,"max_outer_iterations","birger"),"birger.max_outer_iterations",1),
            epsilon_i_limit=_positive(_required(bg,"epsilon_i_limit","birger"),"birger.epsilon_i_limit"),
            delta_J_limit=_positive(_required(bg,"delta_J_limit","birger"),"birger.delta_J_limit"),
            initial_slope_rtol=_positive(bg.get("initial_slope_rtol",0.05),"birger.initial_slope_rtol"),
        )
        if birger_settings.omega > 1.0:
            raise CaseInputError("birger.omega must satisfy 0<omega<=1")
        diagram=_material_diagram(
            _required(bg,"material_diagram","birger"),case_dir
        )

    out=Path(cfg["output_dir"])
    if not out.is_absolute():
        out=case_dir/out

    return {
        "mode":mode,
        "E0":E0,
        "nu0":nu0,
        "grid":grid,
        "ops":ops,
        "quadrature":quadrature,
        "bc":bc,
        "inner_settings":inner_settings,
        "load_steps":load_steps,
        "load_factors":load_factors,
        "birger_settings":birger_settings,
        "material_diagram":diagram,
        "initial_state":initial_state,
        "output_dir":out,
        "case_json":case_json,
    }


def run_case(case_json):
    c=build_case(case_json)
    result=d5.solve_explicit_load_series(
        c["mode"],
        c["initial_state"],
        c["ops"],
        E0=c["E0"],
        nu0=c["nu0"],
        quadrature=c["quadrature"],
        bc=c["bc"],
        load_steps=c["load_steps"],
        load_factors=c["load_factors"],
        inner_settings=c["inner_settings"],
        birger_settings=c["birger_settings"],
        material_diagram=c["material_diagram"],
        output_dir=c["output_dir"],
    )
    run_meta={
        "case_json":str(c["case_json"]),
        "mode":result.mode,
        "status":result.status,
        "converged":result.converged,
        "requested_steps":result.requested_steps,
        "accepted_steps":len(result.accepted_steps),
        "failed_step_index":result.failed_step_index,
        "failed_step_status":result.failed_step_status,
        "failed_step_code":getattr(result,"failed_step_code",None),
        "failed_step_diagnostic_ru":getattr(result,"failed_step_diagnostic_ru",None),
        "failed_step_recommendation_ru":getattr(result,"failed_step_recommendation_ru",None),
    }
    c["output_dir"].mkdir(parents=True,exist_ok=True)
    (c["output_dir"]/ "RUN_RESULT.json").write_text(
        json.dumps(run_meta,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    return result
