#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage11A unified dispatcher for the verified cylinder solver branches.

This module contains routing/input/output logic only.
It does NOT contain shell mechanics and does NOT modify any verified solver.

Unified choices:
  theory: CUF | KL
  cuf_model: TE1 | TE2 | TE3 | TE4
  method: MVI | FDM
  mode:
    LINEAR_ELASTIC
    LINEAR_BIRGER
    LINEAR_J2
    NONLINEAR_ELASTIC
    NONLINEAR_BIRGER
    NONLINEAR_J2

MVI is delegated to the verified Stage7G (CUF) / Stage8D3 (KL) launchers.
FDM is delegated to:
  CUF: Stage10A / 10B / 10C / 10D
  KL : Stage9B / 9C / 9D1 / 9D2 / 9E (GL+Hooke) / 9F (GL+Birger)
"""
from __future__ import annotations

from pathlib import Path
import os
import importlib
import importlib.util
import json
import hashlib
import sys
import tempfile
import numpy as np
import final_progress as fprog

import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage2b as s2
import cylinder_te1_stage3a_connector as c3
import cylinder_mkr_fd2_operators as mkrfd2
import cylinder_te1_stage5a_birger as b5

import cylinder_cuf_stage7c_general_boundary as cbc
import cylinder_cuf_stage7g_general_launcher as cuf_mvi_launcher
import cylinder_cuf_stage7d_general_internal_mvi as cuf_mvi_core
import cylinder_cuf_stage7e_general_driver as cuf_mvi_driver
import cylinder_cuf_stage7k_j2_mvi as cuf_j2_mvi
import cylinder_cuf_stage10a_strong_fdm_linear_elastic as c10a
import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as c10b
import cylinder_cuf_stage10c_strong_fdm_linear_birger as c10c
import cylinder_cuf_stage10d_strong_fdm_full_gl_birger as c10d
import cylinder_cuf_stage10e_strong_fdm_vk as c10e
import cylinder_cuf_stage10f_strong_fdm_j2 as c10f

import cylinder_kl_stage8b1_boundary as kbc
import cylinder_kl_stage8b3_essential_bc as kess
import cylinder_kl_stage8d3_launcher as kl_mvi_launcher
import cylinder_kl_stage8b2_profiles as kprofiles
import cylinder_kl_stage8b5_strict_mvi as kl_mvi_core
import cylinder_kl_stage8m_j2_mvi as kl_j2_mvi
import cylinder_kl_stage9b_strong_fdm_linear as k9b
import cylinder_kl_stage9c_strong_fdm_vk_elastic as k9c
import cylinder_kl_stage9e_strong_fdm_gl_elastic as k9e
import cylinder_kl_stage9d1_strong_fdm_linear_birger as k9d1
import cylinder_kl_stage9d2_strong_fdm_vk_birger as k9d2
import cylinder_kl_stage9f_strong_fdm_gl_birger as k9f
import cylinder_kl_stage9j_strong_fdm_linear_j2 as k9j
import cylinder_kl_stage9k_strong_fdm_vk_j2 as k9k
import cylinder_kl_stage9l_strong_fdm_gl_j2 as k9l

import final_te_basis_provider as tebasis
import final_load_builder as loadbuilder
import final_birger_curve as birgercurve
import final_j2_case as j2case
import final_j2_history_io as j2history
import final_units as funits
import final_step_predictor as fpred
import final_boundary_registry as bcreg
import final_capability_registry as capreg
import flexo_scale_policy as flexoscale




def _atomic_write_json(path, data):
    """Crash-resistant JSON publish: write+fsync temporary file, then replace."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    payload=json.dumps(data,indent=2,ensure_ascii=False)
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def _atomic_savez(path, **arrays):
    """Crash-resistant NPZ publish. A previous accepted checkpoint is never truncated."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def _recovery_checkpoint(cfg,c,theory,mode,load_meta,loads,steps,all_converged,ops):
    """Self-contained summary rewritten after every attempted load step."""
    return {
        "format":"CYLINDER_RECOVERY_CHECKPOINT_V1",
        "checkpoint":True,
        "theory":theory,
        "cuf_model":c.get("cuf_model"),
        "method":"FDM",
        "mode":mode,
        "geometry_model":c["geometry_model"],
        "capability":c["capability"].to_dict(),
        "material_model":c["material_model"],
        "units":c["units"],
        "converged":False,
        "status":"IN_PROGRESS_OR_INTERRUPTED",
        "requested_steps":int(len(loads)),
        "accepted_steps":int(sum(1 for x in steps if x.get("converged"))),
        "attempted_steps":int(len(steps)),
        "geometry":{"L":c["L"],"R":c["R"],"h":c["h"]},
        "material":{"E0":c["E0"],"nu0":c["nu0"]},
        "grid":{"nx":c["nx"],"ns":c["ns"],"stencil":c["stencil"]},
        "spatial_discretization":_fdm_spatial_metadata(c),
        "quadrature":dict(c["quadrature_spec"]),
        "validation_warnings":list(c["quadrature_warnings"])+list(c["unit_validation"].get("warnings",[]))+list(c["boundary_layer_warnings"])+list(c["capability_warnings"]),
        "boundary_conditions":cfg.get("boundary_conditions"),
        "load_definition":load_meta,
        "postprocess":cfg.get("postprocess",{}),
        "birger_definition":cfg.get("birger") if any(x.get("birger_iterations") is not None for x in steps) else None,
        "steps":steps,
        "last_saved_step":int(len(steps)),
    }

THEORIES=capreg.THEORIES
CUF_MODELS=tebasis.SUPPORTED_TE_MODELS
if tuple(CUF_MODELS)!=tuple(capreg.CUF_MODELS):
    raise RuntimeError("CUF capability registry and basis provider disagree")
LEGACY_THEORY_ALIAS="CUF_TE1"
METHODS=capreg.METHODS
# Only elastic cases can be dispatched by cylinder-flexo1.
MODES=("LINEAR_ELASTIC","NONLINEAR_ELASTIC")

# FDM/MKR operator policy is explicit.  Keep the historical matched
# high-order discretisation as the backward-compatible production default;
# INDEPENDENT_FD2 is the verification path that deliberately does not share
# the MVI differentiation matrices.
FDM_OPERATOR_SCHEMES=("MATCHED_HIGH_ORDER","INDEPENDENT_FD2")

UNIT_SYSTEMS={
    "N_MM_MPA": {
        "length":"mm", "stress":"MPa", "pressure":"N/mm^2",
        "line_load":"N/mm", "force":"N", "displacement":"mm",
    },
    "SI_N_M_PA": {
        "length":"m", "stress":"Pa", "pressure":"Pa",
        "line_load":"N/m", "force":"N", "displacement":"m",
    },
}

CUF_MVI_MODE={
    "LINEAR_ELASTIC":"CUF_LINEAR_ELASTIC",
    "LINEAR_BIRGER":"CUF_LINEAR_BIRGER",
    "NONLINEAR_ELASTIC":"CUF_NL_ELASTIC",
    "NONLINEAR_BIRGER":"CUF_BIRGER",
}
KL_MVI_MODE={
    "LINEAR_ELASTIC":"KL_LINEAR_ELASTIC",
    "LINEAR_BIRGER":"KL_LINEAR_BIRGER",
    "NONLINEAR_ELASTIC":"KL_VK_ELASTIC",
    "NONLINEAR_BIRGER":"KL_VK_BIRGER",
}


def _cuf_child_mode(c):
    geometry=c["geometry_model"]
    birger=c["material_model"]=="BIRGER"
    if geometry=="LINEAR":
        return "CUF_LINEAR_BIRGER" if birger else "CUF_LINEAR_ELASTIC"
    if geometry=="VON_KARMAN":
        return "CUF_VK_BIRGER" if birger else "CUF_VK_ELASTIC"
    if geometry=="GREEN_LAGRANGE":
        return "CUF_BIRGER" if birger else "CUF_NL_ELASTIC"
    raise FinalCaseError(f"unsupported CUF geometry_model: {geometry}")


def _kl_child_mode(c):
    geometry=c["geometry_model"]
    birger=c["material_model"]=="BIRGER"
    if geometry=="LINEAR":
        return "KL_LINEAR_BIRGER" if birger else "KL_LINEAR_ELASTIC"
    if geometry=="VON_KARMAN":
        return "KL_VK_BIRGER" if birger else "KL_VK_ELASTIC"
    if geometry=="GREEN_LAGRANGE":
        return "KL_GL_BIRGER" if birger else "KL_GL_ELASTIC"
    raise FinalCaseError(f"unsupported Kirchhoff-Love geometry_model: {geometry}")


class FinalCaseError(RuntimeError):
    pass


def _read_json(path):
    p=Path(path).resolve()
    try:
        cfg=json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FinalCaseError(f"cannot read FINAL_CASE JSON: {p}: {exc}") from exc
    if not isinstance(cfg,dict):
        raise FinalCaseError("FINAL_CASE root must be an object")
    return cfg,p


def _req(d,key,where):
    if key not in d or d[key] is None:
        raise FinalCaseError(f"missing required field {where}.{key}")
    return d[key]


def _dict(d,key,where):
    v=_req(d,key,where)
    if not isinstance(v,dict):
        raise FinalCaseError(f"{where}.{key} must be an object")
    return v


def _pos(v,name):
    x=float(v)
    if not np.isfinite(x) or x<=0:
        raise FinalCaseError(f"{name} must be finite and >0")
    return x


def _nonneg(v,name):
    x=float(v)
    if not np.isfinite(x) or x<0:
        raise FinalCaseError(f"{name} must be finite and >=0")
    return x


def _int(v,name,minv=1):
    try:
        x=int(v)
    except Exception as exc:
        raise FinalCaseError(f"{name} must be integer") from exc
    if x<minv or float(v)!=float(x):
        raise FinalCaseError(f"{name} must be integer >= {minv}")
    return x


def _float_list(v,name,minlen=1):
    if isinstance(v,str):
        raw=[x.strip() for x in v.replace(";",",").split(",") if x.strip()]
    elif isinstance(v,(list,tuple,np.ndarray)):
        raw=list(v)
    else:
        raise FinalCaseError(f"{name} must be a numeric list")
    if len(raw)<minlen:
        raise FinalCaseError(f"{name} must contain at least {minlen} values")
    try: out=[float(x) for x in raw]
    except Exception as exc: raise FinalCaseError(f"{name} must contain only numbers") from exc
    if not np.all(np.isfinite(out)):
        raise FinalCaseError(f"{name} contains non-finite values")
    return out


def _fdm_operator_scheme(cfg):
    scheme=str(cfg.get("fdm_operator_scheme","MATCHED_HIGH_ORDER")).upper().strip()
    if scheme not in FDM_OPERATOR_SCHEMES:
        raise FinalCaseError(
            f"fdm_operator_scheme must be one of {FDM_OPERATOR_SCHEMES}"
        )
    return scheme


def _build_fdm_spatial_operators(c,grid):
    scheme=str(c["fdm_operator_scheme"]).upper()
    if scheme=="MATCHED_HIGH_ORDER":
        return c3.CylinderSpatialOperators.build(grid,c["stencil"])
    if scheme=="INDEPENDENT_FD2":
        return mkrfd2.MKRSpatialOperators.build(grid)
    raise FinalCaseError(f"unsupported fdm_operator_scheme: {scheme}")


def _fdm_spatial_metadata(c):
    scheme=str(c["fdm_operator_scheme"]).upper()
    if scheme=="INDEPENDENT_FD2":
        return {
            "scheme":"INDEPENDENT_FD2",
            "operator_class":"MKRSpatialOperators",
            "formal_order":2,
            "matched_to_mvi":False,
            "independent_from_mvi":True,
            "note":"MKR/FDM uses the dedicated second-order FD operators; MVI keeps its own profile operators.",
        }
    return {
        "scheme":"MATCHED_HIGH_ORDER",
        "operator_class":"CylinderSpatialOperators",
        "derivative_stencil_points":int(c["stencil"]),
        "matched_to_mvi":True,
        "independent_from_mvi":False,
        "note":"Backward-compatible production mode: full-grid FDM and MVI use the same spatial differentiation matrices for direct comparison.",
    }


def _quadrature_config(qj,h,mode):
    """V6: build and validate GAUSS or LAYERED thickness quadrature."""
    qtype=str(_req(qj,"type","quadrature")).upper()
    warnings=[]
    if qtype=="GAUSS":
        ng=_int(_req(qj,"n_gauss","quadrature"),"quadrature.n_gauss",1)
        quad=s2.gauss_thickness_quadrature(h,ng)
        spec={"type":"GAUSS","n_gauss":ng}
        if mode in ("LINEAR_BIRGER","NONLINEAR_BIRGER") and ng<7:
            warnings.append(
                f"V6: n_gauss={ng} при расчёте Биргера может недостаточно разрешать "
                "упругопластическую границу по толщине; аудит рекомендует >=7 либо LAYERED."
            )
    elif qtype in ("LAYERED","SUBLAYERS"):
        bounds=_float_list(
            qj.get("layer_bounds",qj.get("bounds")),
            "quadrature.layer_bounds",minlen=2
        )
        raw_orders=qj.get("n_gauss_per_layer",qj.get("n_gauss"))
        if raw_orders is None:
            raise FinalCaseError(
                "missing required field quadrature.n_gauss or quadrature.n_gauss_per_layer"
            )
        if isinstance(raw_orders,(list,tuple,np.ndarray)):
            orders=[_int(x,"quadrature.n_gauss_per_layer",1) for x in raw_orders]
            orders_for_builder=orders
        else:
            orders_for_builder=_int(raw_orders,"quadrature.n_gauss",1)
            orders=[orders_for_builder]*(len(bounds)-1)
        try:
            quad=s2.layered_thickness_quadrature(h,bounds,orders_for_builder)
        except Exception as exc:
            raise FinalCaseError(f"invalid LAYERED quadrature: {exc}") from exc
        spec={
            "type":"LAYERED",
            "layer_bounds":[float(x) for x in bounds],
            "n_gauss_per_layer":[int(x) for x in orders],
        }
    else:
        raise FinalCaseError("quadrature.type must be GAUSS or LAYERED")
    return spec,quad,warnings


def _resolve(case_path,value):
    p=Path(str(value))
    if not p.is_absolute():
        p=case_path.parent/p
    return p.resolve()


def _load_callable(spec,case_path):
    if not isinstance(spec,dict):
        raise FinalCaseError("birger.material_diagram must be an object")
    module_name=_req(spec,"module","birger.material_diagram")
    callable_name=_req(spec,"callable","birger.material_diagram")
    python_path=spec.get("python_path")

    old_path=list(sys.path)
    try:
        if python_path:
            pp=_resolve(case_path,python_path)
            sys.path.insert(0,str(pp))
        mod=importlib.import_module(str(module_name))
        fn=getattr(mod,str(callable_name))
    except Exception as exc:
        raise FinalCaseError(
            f"cannot load material diagram {module_name}.{callable_name}: {exc}"
        ) from exc
    finally:
        sys.path[:]=old_path

    if not callable(fn):
        raise FinalCaseError("material diagram target is not callable")
    return fn


def _common(cfg):
    theory_raw=str(_req(cfg,"theory","CASE")).upper()
    legacy_te1=(theory_raw==LEGACY_THEORY_ALIAS)
    theory="CUF" if legacy_te1 else theory_raw
    method=str(_req(cfg,"method","CASE")).upper()
    mode=str(_req(cfg,"mode","CASE")).upper()
    # DYNAMIC is never a synonym for static continuation/load increments.
    # Validate its time/mass/method contract, then block until the actual
    # coupled FDM or MVI spatial residual and BC assembly is integrated.
    from cylinder_newmark_mvi import calculation_kind, validate_dynamic_case
    try:
        requested_kind=calculation_kind(cfg)
        if requested_kind=="DYNAMIC":
            time_contract=validate_dynamic_case(cfg)
            raise FinalCaseError(
                "DYNAMIC_SPATIAL_PDE_NOT_CONNECTED: "
                f"{time_contract['method']} выбран как пространственный метод; "
                "шаг Ньюмарка и требование полного остатка реализованы, "
                "но физические уравнения цилиндра с ГУ, массой, "
                "G3D/флексо и X/Y-МВИ пока не собраны. "
                "Запуск статического маршрута вместо динамики запрещён."
            )
    except ValueError as exc:
        raise FinalCaseError(f"Неверный CASE Статика/Динамика: {exc}") from exc
    # A micro CASE must never fall through the old classical CUF/KL routes.
    try:
        scale_model=flexoscale.scale_from_case(cfg)
    except ValueError as exc:
        raise FinalCaseError(f"Неверный выбор макро/микро: {exc}") from exc
    if scale_model=="MICRO":
        try:
            flexoscale.micro_material_from_case(cfg)
        except (ValueError, KeyError, TypeError) as exc:
            raise FinalCaseError(f"MICRO G3D: неверный паспорт материала: {exc}") from exc
        raise FinalCaseError(
            "MICRO G3D: материальный закон проверен, но связанный градиентно-"
            "флексоэлектрический решатель с высшими ГУ и временными циклами МВИ "
            "ещё не реализован. Запуск классического MACRO-кода запрещён."
        )
    if theory in capreg.PLANNED_THEORIES:
        raise FinalCaseError(
            f"{theory}: теория зарегистрирована, но её вариационные уравнения "
            "и граничные условия ещё не реализованы. Нельзя использовать CUF/KL "
            "расчёт под названием FSDT или Reddy."
        )
    if theory not in THEORIES:
        raise FinalCaseError(f"theory must be one of {THEORIES}; legacy CUF_TE1 is also accepted")
    if theory=="CUF":
        cuf_model="TE1" if legacy_te1 else str(_req(cfg,"cuf_model","CASE")).upper()
        if cuf_model not in CUF_MODELS:
            raise FinalCaseError(f"cuf_model must be one of {CUF_MODELS}")
    else:
        cuf_model=None
    if method not in METHODS:
        raise FinalCaseError(f"method must be one of {METHODS}")
    if mode not in MODES:
        raise FinalCaseError(f"mode must be one of {MODES}")

    # V14 separates geometry and material explicitly.  Legacy CASE files are
    # interpreted exactly as before: CUF NONLINEAR meant full GL/StVK, while
    # KL NONLINEAR meant von Karman.
    geometry_model=str(cfg.get("geometry_model") or "").upper().strip()
    if not geometry_model:
        if mode.startswith("LINEAR"):
            geometry_model="LINEAR"
        elif theory=="CUF":
            geometry_model="GREEN_LAGRANGE"
        else:
            geometry_model="VON_KARMAN"
    material_model=str(cfg.get("material_model") or "").upper().strip()
    if not material_model:
        material_model="BIRGER" if mode.endswith("BIRGER") else ("J2_INCREMENTAL" if mode.endswith("J2") else "HOOKE")
    allowed_geometry=capreg.GEOMETRY_MODELS
    if geometry_model not in allowed_geometry:
        raise FinalCaseError(
            f"geometry_model for {theory} must be one of {allowed_geometry}"
        )
    if material_model!="HOOKE":
        raise FinalCaseError(
            "cylinder-flexo1 допускает только упругий материал HOOKE; "
            "Биргер и J2-пластичность отключены."
        )
    material_suffix="_ELASTIC"
    expected_mode=("LINEAR" if geometry_model=="LINEAR" else "NONLINEAR") + material_suffix
    if mode!=expected_mode:
        raise FinalCaseError(
            f"CASE mode={mode} conflicts with geometry_model={geometry_model} "
            f"and material_model={material_model}; expected mode={expected_mode}"
        )

    # Compatibility metadata field; the incremental J2 path is unavailable.
    j2_parameters=None

    capability=capreg.case_capability(cfg)
    if capability.status==capreg.UNAVAILABLE:
        raise FinalCaseError(
            f"UNAVAILABLE_COMBINATION [{capability.code}]: {capability.message_ru}"
        )
    capability_warnings=(
        [f"LIMITED_COMBINATION [{capability.code}]: {capability.message_ru}"]
        if capability.status==capreg.LIMITED else []
    )

    gj=_dict(cfg,"geometry","CASE")
    L=_pos(_req(gj,"L","geometry"),"geometry.L")
    R=_pos(_req(gj,"R","geometry"),"geometry.R")
    h=_pos(_req(gj,"h","geometry"),"geometry.h")
    if theory=="CUF" and 1.0-h/(2.0*R)<=0.0:
        raise FinalCaseError("CUF geometry invalid: H<=0 inside thickness")

    mj=_dict(cfg,"material","CASE")
    E0=_pos(_req(mj,"E0","material"),"material.E0")
    nu0=float(_req(mj,"nu0","material"))
    if not np.isfinite(nu0) or not (-1.0<nu0<0.5):
        raise FinalCaseError("material.nu0 must satisfy -1 < nu0 < 0.5")

    # Stage12 interface fix: independent spatial grids for FDM and MVI.
    # Legacy CASE files with a single "grid" object remain accepted as fallback.
    legacy_gr=cfg.get("grid")
    if legacy_gr is not None and not isinstance(legacy_gr,dict):
        raise FinalCaseError("CASE.grid must be an object when supplied")

    def _parse_grid_obj(raw,name):
        if raw is None:
            raise FinalCaseError(
                f"CASE.{name} is required (legacy CASE.grid is accepted as fallback)"
            )
        if not isinstance(raw,dict):
            raise FinalCaseError(f"CASE.{name} must be an object")
        nx0=_int(_req(raw,"nx",name),f"{name}.nx",7)
        ns0=_int(_req(raw,"ns",name),f"{name}.ns",7)
        if ns0 % 2 == 0:
            raise FinalCaseError(
                f"{name}.ns должно быть нечётным. При чётном ns центральный "
                "периодический оператор D1s обнуляет пилообразную окружную "
                "моду (-1)^j, создавая скрытую нулевую моду."
            )
        st0=_int(_req(raw,"stencil",name),f"{name}.stencil",3)
        if st0%2!=1 or st0>nx0 or st0>ns0:
            raise FinalCaseError(f"{name}.stencil must be odd and <= nx,ns")
        return {"nx":nx0,"ns":ns0,"stencil":st0}

    raw_fdm=cfg.get("fdm_grid",legacy_gr)
    raw_mvi=cfg.get("mvi_grid",legacy_gr)
    if raw_fdm is None and raw_mvi is not None:
        raw_fdm=raw_mvi
    if raw_mvi is None and raw_fdm is not None:
        raw_mvi=raw_fdm
    fdm_grid=_parse_grid_obj(raw_fdm,"fdm_grid")
    mvi_grid=_parse_grid_obj(raw_mvi,"mvi_grid")

    active_grid=fdm_grid if method=="FDM" else mvi_grid
    nx=active_grid["nx"]
    ns=active_grid["ns"]
    stencil=active_grid["stencil"]

    qj=_dict(cfg,"quadrature","CASE")
    quadrature_spec,quadrature,quadrature_warnings=_quadrature_config(qj,h,mode)
    # n_gauss is retained internally as the total number of through-thickness
    # points for compatibility with shape checks; the canonical description is
    # quadrature_spec.
    n_gauss=int(quadrature.z.size)

    # V5: the unit system is mandatory and operational at the dispatcher
    # boundary.  Kernels remain coherent-unit mechanics, but declared-unit
    # scalars are checked after conversion to SI so an unmistakable Pa/MPa or
    # m/mm scale slip cannot pass merely with a different report label.
    uj=cfg.get("units")
    if not isinstance(uj,dict):
        raise FinalCaseError("V5: CASE.units with explicit units.system is required")
    unit_system=str(uj.get("system","")).upper()
    if unit_system not in UNIT_SYSTEMS:
        raise FinalCaseError(f"units.system must be one of {tuple(UNIT_SYSTEMS)}")
    units={"system":unit_system, **UNIT_SYSTEMS[unit_system]}
    try:
        unit_validation=funits.validate_case_scalars(cfg,unit_system)
    except funits.UnitPolicyError as exc:
        raise FinalCaseError(str(exc)) from exc

    # Reviewer V-9: warn when the axial mesh does not resolve the classical
    # cylindrical-shell boundary-layer decay length.  This is diagnostics
    # only; the calculation is never blocked or the mesh altered automatically.
    beta=(3.0*(1.0-nu0**2)/(R**2*h**2))**0.25
    decay=1.0/beta
    dx=L/(nx-1)
    dx_required=0.5*decay
    boundary_layer_warnings=[]
    if dx>dx_required:
        nx_rec=int(np.ceil(L*2.0*beta))+1
        boundary_layer_warnings.append(
            f"BOUNDARY_LAYER_UNRESOLVED: dx={dx:.4g} > {dx_required:.4g}. "
            f"Длина затухания краевого эффекта 1/beta={decay:.4g}. "
            f"w в середине сойдётся, wmax у заделки — нет. "
            f"Рекомендуемое nx >= {nx_rec}."
        )

    output_dir=Path(str(_req(cfg,"output_dir","CASE")))
    fdm_operator_scheme=_fdm_operator_scheme(cfg)

    return dict(
        theory=theory,cuf_model=cuf_model,method=method,mode=mode,scale_model=scale_model,
        geometry_model=geometry_model,material_model=material_model,j2_parameters=j2_parameters,units=units,
        capability=capability,capability_warnings=capability_warnings,
        unit_validation=unit_validation,
        L=L,R=R,h=h,E0=E0,nu0=nu0,
        nx=nx,ns=ns,stencil=stencil,n_gauss=n_gauss,
        quadrature=quadrature,quadrature_spec=quadrature_spec,
        quadrature_warnings=quadrature_warnings,
        boundary_layer_warnings=boundary_layer_warnings,
        fdm_grid=fdm_grid,mvi_grid=mvi_grid,
        fdm_operator_scheme=fdm_operator_scheme,
        active_grid=("fdm_grid" if method=="FDM" else "mvi_grid"),
        output_dir=output_dir,
    )


def _validate_bc_model_tag(cfg,c):
    bj=_dict(cfg,"boundary_conditions","CASE")
    saved=bj.get("model")
    actual=bcreg.model_tag(c["theory"],c.get("cuf_model"))
    if saved not in (None,"") and str(saved).upper()!=actual:
        raise FinalCaseError(
            f"BOUNDARY_MODEL_MISMATCH: CASE boundary_conditions.model={saved!r}, "
            f"but active kinematic model is {actual}. Re-select or rebuild boundary conditions."
        )
    return actual


def _simple_bc_pair(cfg,theory):
    bj=_dict(cfg,"boundary_conditions","CASE")
    if bj.get("advanced") is not None:
        return None
    left=str(_req(bj,"left","boundary_conditions")).upper()
    right=str(_req(bj,"right","boundary_conditions")).upper()
    allowed=bcreg.STANDARD_PRESETS+bcreg.LEGACY_PRESETS
    if left not in allowed or right not in allowed:
        raise FinalCaseError(f"simple boundary left/right must be one of {allowed}")

    # Existing CASE files used CUF SIMPLY_SUPPORTED as the old movable S4
    # support.  New GUI/template CASE files explicitly store V2 and use the
    # physically matching immovable S3 support.
    version=str(bj.get("registry_version","BC_REGISTRY_V1")).upper()
    if version not in ("BC_REGISTRY_V1",bcreg.BC_REGISTRY_VERSION):
        raise FinalCaseError(
            f"unsupported boundary registry version: {version!r}; "
            f"use BC_REGISTRY_V1 or {bcreg.BC_REGISTRY_VERSION}"
        )
    if str(theory).upper()=="CUF" and version=="BC_REGISTRY_V1":
        left="PINNED_S4" if left=="SIMPLY_SUPPORTED" else left
        right="PINNED_S4" if right=="SIMPLY_SUPPORTED" else right
    return left,right




def _advanced_bc_cfg(cfg):
    bj=_dict(cfg,"boundary_conditions","CASE")
    adv=bj.get("advanced")
    if adv is None:
        return None
    if not isinstance(adv,dict):
        raise FinalCaseError("boundary_conditions.advanced must be an object")
    return adv


def _engineering_bc_cfg(cfg):
    bj=_dict(cfg,"boundary_conditions","CASE")
    eng=bj.get("engineering")
    if eng is None:
        return {}
    if not isinstance(eng,dict):
        raise FinalCaseError("boundary_conditions.engineering must be an object")
    return eng


def _normalize_relative_npy_refs(obj,base_dir):
    """Normalize only explicit {'npy': path} BC data references for child-case relocation."""
    if isinstance(obj,dict):
        if set(obj)=={"npy"}:
            pp=Path(str(obj["npy"]))
            if not pp.is_absolute():
                pp=(base_dir/pp).resolve()
            return {"npy":str(pp)}
        return {k:_normalize_relative_npy_refs(v,base_dir) for k,v in obj.items()}
    if isinstance(obj,list):
        return [_normalize_relative_npy_refs(v,base_dir) for v in obj]
    return obj


def _make_advanced_bc(cfg,case_path,theory,m,ns):
    adv=_advanced_bc_cfg(cfg)
    if adv is None:
        return None
    adv=_normalize_relative_npy_refs(adv,case_path.parent)
    if theory=="CUF":
        # Reuse the verified Stage7G parser exactly.
        return cuf_mvi_launcher._parse_bc(adv,m,ns,case_path.parent)
    # Reuse the verified Stage8D3 parser exactly.
    bc=kl_mvi_launcher._parse_boundary_conditions({"boundary_conditions":adv})
    bc.check_rigid_mode_requirement()
    return bc


def _eng_item(eng,side,field,tau=None):
    try:
        item=eng[side][field] if tau is None else eng[side][field][int(tau)]
    except Exception as exc:
        suffix=f"[{tau}]" if tau is not None else ""
        raise FinalCaseError(
            f"ELASTIC_ENGINEERING requires boundary_conditions.engineering.{side}.{field}{suffix}"
        ) from exc
    if not isinstance(item,dict):
        raise FinalCaseError("engineering BC entry must be an object")
    if "stiffness" not in item:
        raise FinalCaseError("engineering BC entry requires stiffness")
    try:
        k=float(item["stiffness"])
        q0=float(item.get("reference",0.0))
        tb=float(item.get("traction",item.get("value",0.0)))
    except Exception as exc:
        raise FinalCaseError("engineering BC stiffness/reference/traction must be scalar numbers") from exc
    if not np.isfinite(k) or k<0.0 or not np.isfinite(q0) or not np.isfinite(tb):
        raise FinalCaseError("engineering BC values must be finite and stiffness nonnegative")
    return {"stiffness":k,"reference":q0,"traction":tb}


def _cuf_simple_end(kind,m,engineering=None,side=None):
    rows=[]
    for comp_index,comp in enumerate(("u","v","w")):
        row=[]
        for tau in range(int(m)):
            bc_kind=bcreg.cuf_kinds_tau(kind,tau)[comp_index]
            if bc_kind=="ELASTIC":
                e=_eng_item(engineering or {},side,comp,tau)
                row.append(cbc.GeneralFieldEndCondition(
                    "ELASTIC",
                    cbc.ScalarOrCircumferentialData(e["traction"]),
                    cbc.ScalarOrCircumferentialData(e["stiffness"]),
                    cbc.ScalarOrCircumferentialData(e["reference"]),
                ))
            else:
                row.append(cbc.GeneralFieldEndCondition(
                    bc_kind,cbc.ScalarOrCircumferentialData(0.0)
                ))
        rows.append(tuple(row))
    return cbc.GeneralEndBoundarySet(tuple(rows))


def _cuf_simple_bc_json(kind,m,engineering=None,side=None):
    out={}
    for comp_index,comp in enumerate(("u","v","w")):
        vals=[]
        for tau in range(int(m)):
            bc_kind=bcreg.cuf_kinds_tau(kind,tau)[comp_index]
            if bc_kind=="ELASTIC":
                e=_eng_item(engineering or {},side,comp,tau)
                vals.append({
                    "kind":"ELASTIC","data":e["traction"],
                    "stiffness":e["stiffness"],"reference":e["reference"],
                })
            else:
                vals.append({"kind":bc_kind,"data":0.0})
        out[comp]=vals
    return out


def _make_cuf_simple_bc(pair,m,engineering=None):
    left,right=pair
    for kind in (left,right):
        # normalize/validate through the registry
        bcreg.cuf_kinds_tau(kind,0)
    bc=cbc.GeneralCylinderEndBC(
        _cuf_simple_end(left,m,engineering,"left"),
        _cuf_simple_end(right,m,engineering,"right"),
    )
    bc.check_rigid_mode_requirement()
    return bc


def _kl_simple_end(kind,engineering=None,side=None):
    kinds=bcreg.kl_kinds(kind)
    fields=("u","v","w","slope")
    conds=[]
    for field,bc_kind in zip(fields,kinds):
        if bc_kind=="ELASTIC":
            e=_eng_item(engineering or {},side,field)
            conds.append(kbc.KLBoundaryCondition(
                "ELASTIC",kbc.KLBoundaryData(e["traction"]),
                kbc.KLBoundaryData(e["stiffness"]),
                kbc.KLBoundaryData(e["reference"]),
            ))
        else:
            conds.append(kbc.KLBoundaryCondition(bc_kind,kbc.KLBoundaryData(0.0)))
    return kbc.KLEndBoundarySet(*conds)


def _kl_simple_bc_json(kind,engineering=None,side=None):
    kinds=bcreg.kl_kinds(kind)
    out={}
    for field,bc_kind in zip(("u","v","w","slope"),kinds):
        if bc_kind=="ELASTIC":
            e=_eng_item(engineering or {},side,field)
            out[field]={
                "kind":"ELASTIC","value":e["traction"],
                "stiffness":e["stiffness"],"reference":e["reference"],
            }
        else:
            out[field]={"kind":bc_kind,"value":0.0}
    return out


def _make_kl_simple_bc(pair,engineering=None):
    left,right=pair
    for kind in (left,right):
        bcreg.kl_kinds(kind)
    bc=kbc.KLCylinderEndBC(
        _kl_simple_end(left,engineering,"left"),
        _kl_simple_end(right,engineering,"right"),
    )
    bc.check_rigid_mode_requirement()
    return bc


def _make_case_bc(cfg,case_path,c):
    """Single source of truth for BC construction used by validation/FDM/MVI."""
    _validate_bc_model_tag(cfg,c)
    theory=c["theory"]
    m=tebasis.get_basis(c["cuf_model"]).m if theory=="CUF" else None
    pair=_simple_bc_pair(cfg,theory)
    if pair is None:
        bc=_make_advanced_bc(cfg,case_path,theory,m,c["ns"])
        return bc,None
    eng=_engineering_bc_cfg(cfg)
    if theory=="CUF":
        bc=_make_cuf_simple_bc(pair,m,eng)
    else:
        bc=_make_kl_simple_bc(pair,eng)
    return bc,pair

def _validate_mvi_homogeneous_geometric_bc(bc,theory,ns,geometry=None):
    """MVI essential-profile parameterizations currently support qbar=0 only.

    FDM collocation supports nonzero prescribed geometric end data.  The MVI
    branches enforce essential data through homogeneous X-profile constraints,
    so a nonzero GEOMETRIC value must be rejected explicitly at CASE preflight
    rather than failing later inside a half-step.
    """
    t=str(theory).upper().strip()
    nss=int(ns)
    if t=="KL":
        if str(geometry or "").upper()=="GREEN_LAGRANGE":
            for side,end in (("left",bc.left),("right",bc.right)):
                if end.slope.kind=="GEOMETRIC" and not all(
                    getattr(end,field).kind=="GEOMETRIC"
                    for field in ("u","v","w")
                ):
                    raise FinalCaseError(
                        f"KL_FULL_GL_MVI_ROTATION_BC_REQUIRES_FIXED_EDGE: {side}; "
                        "an exact prescribed director rotation in the current MVI "
                        "profile space requires u, v and w to be GEOMETRIC on the "
                        "same edge. Use a standard clamped edge, a natural/elastic "
                        "moment condition, or FDM for this custom mixed support."
                    )
        for side,end in (("left",bc.left),("right",bc.right)):
            for field in ("u","v","w","slope"):
                cond=getattr(end,field)
                if cond.kind=="GEOMETRIC" and np.max(np.abs(cond.data.array(nss)))>0.0:
                    raise FinalCaseError(
                        f"MVI_NONZERO_GEOMETRIC_BC_UNSUPPORTED: {side}.{field}; "
                        "current MVI essential-profile constraints require homogeneous qbar=0. "
                        "Use FDM for nonzero prescribed end displacement/rotation or reformulate with a lifting field."
                    )
        return
    if t=="CUF":
        for side,end in (("left",bc.left),("right",bc.right)):
            for a,comp in enumerate(("u","v","w")):
                for tau,cond in enumerate(end.conditions[a]):
                    if cond.kind=="GEOMETRIC" and np.max(np.abs(cond.data.array(nss)))>0.0:
                        raise FinalCaseError(
                            f"MVI_NONZERO_GEOMETRIC_BC_UNSUPPORTED: {side}.{comp}[{tau}]; "
                            "current CUF MVI X-profile constraints require homogeneous qbar=0. "
                            "Use FDM for nonzero prescribed generalized end displacement or add an explicit lifting field."
                        )
        return
    raise FinalCaseError(f"unsupported theory for MVI BC validation: {theory}")


def _load_loads_npz(path,theory,m,ns,nx):
    try:
        with np.load(path,allow_pickle=False) as z:
            if "loads" not in z.files:
                raise FinalCaseError("loads NPZ must contain 'loads'")
            loads=np.asarray(z["loads"],float)
            factors=None if "load_factors" not in z.files else np.asarray(z["load_factors"],float)
    except FinalCaseError:
        raise
    except Exception as exc:
        raise FinalCaseError(f"cannot read loads NPZ {path}: {exc}") from exc

    tail=(3,m,ns,nx) if theory=="CUF" else (3,ns,nx)
    expected_ndim=5 if theory=="CUF" else 4
    if loads.ndim!=expected_ndim or tuple(loads.shape[1:])!=tail:
        raise FinalCaseError(
            f"loads shape {loads.shape}; expected (nsteps,{','.join(map(str,tail))})"
        )
    if loads.shape[0]<1 or not np.all(np.isfinite(loads)):
        raise FinalCaseError("loads must contain finite data and at least one step")
    if factors is not None:
        if factors.shape!=(loads.shape[0],) or not np.all(np.isfinite(factors)):
            raise FinalCaseError("load_factors shape/data invalid")
    return loads,factors


def _load_series_for_case(cfg,case_path,c,basis=None,generated_dir=None):
    """Return explicit load array/factors from legacy NPZ or new load_definition."""
    spec=cfg.get("load_definition")
    if spec is None:
        p=_resolve(case_path,_req(cfg,"loads_npz","CASE"))
        m=basis.m if basis is not None else None
        loads,factors=_load_loads_npz(p,c["theory"],m,c["ns"],c["nx"])
        return loads,factors,{
            "type":"EXTERNAL_NPZ","path":str(p),
            "load_kinematics":"DEAD_REFERENCE",
            "load_direction_update":"FIXED_IN_REFERENCE_CONFIGURATION",
        },p,None
    if not isinstance(spec,dict):
        raise FinalCaseError("CASE.load_definition must be an object")
    typ=str(_req(spec,"type","load_definition")).upper()
    if typ=="EXTERNAL_NPZ":
        try:
            kin=loadbuilder.normalize_load_kinematics(spec)
        except Exception as exc:
            raise FinalCaseError(f"invalid load_definition: {exc}") from exc
        value=spec.get("npz",cfg.get("loads_npz"))
        if value is None:
            raise FinalCaseError("load_definition EXTERNAL_NPZ requires npz")
        p=_resolve(case_path,value)
        m=basis.m if basis is not None else None
        loads,factors=_load_loads_npz(p,c["theory"],m,c["ns"],c["nx"])
        meta=json.loads(json.dumps(spec))
        meta["load_kinematics"]=kin
        meta["load_direction_update"]="FIXED_IN_REFERENCE_CONFIGURATION"
        return loads,factors,meta,p,None
    try:
        loads,factors,meta,radial,masks=loadbuilder.build_generated_loads(
            spec,theory=c["theory"],L=c["L"],R=c["R"],h=c["h"],
            nx=c["nx"],ns=c["ns"],stencil=c["stencil"],basis=basis,return_masks=True
        )
    except Exception as exc:
        raise FinalCaseError(f"invalid load_definition: {exc}") from exc
    p=None
    if generated_dir is not None:
        gd=Path(generated_dir); gd.mkdir(parents=True,exist_ok=True)
        p=gd/"GENERATED_LOADS.npz"
        np.savez_compressed(
            p,loads=loads,load_factors=factors,radial_surface_field=radial,
            load_front_surface_mask=np.asarray(masks["front_surface_mask"],bool),
            load_support_surface_mask=np.asarray(masks["support_surface_mask"],bool)
        )
        (gd/"LOAD_DEFINITION.json").write_text(
            json.dumps({"definition":spec,"generated_meta":meta},indent=2,ensure_ascii=False),encoding="utf-8")
    front_for_newton=(np.asarray(masks["front_surface_mask"],bool) if typ=="PATCH_RADIAL_FACE" else None)
    return loads,factors,{"definition":json.loads(json.dumps(spec)),"generated_meta":meta},p,front_for_newton


def _linear_residual_tolerance(cfg):
    lj=_dict(cfg,"linear","CASE")
    return _pos(
        _req(lj,"residual_tolerance_relative","linear"),
        "linear.residual_tolerance_relative"
    )


def _newton_settings(cfg,theory):
    nj=_dict(cfg,"newton","CASE")
    if "residual_tolerance" in nj:
        raise FinalCaseError(
            "V2: newton.residual_tolerance is an obsolete ABSOLUTE tolerance; "
            "use dimensionless newton.residual_tolerance_relative (recommended 1e-9)"
        )
    kw=dict(
        residual_tolerance_relative=_pos(
            _req(nj,"residual_tolerance_relative","newton"),
            "newton.residual_tolerance_relative"
        ),
        increment_tolerance=_pos(_req(nj,"increment_tolerance","newton"),"newton.increment_tolerance"),
        fd_step=_pos(_req(nj,"fd_step","newton"),"newton.fd_step"),
        max_iterations=_int(_req(nj,"max_iterations","newton"),"newton.max_iterations",1),
    )
    return c10b.CUFStrongFDMNewtonSettings(**kw) if theory=="CUF" else k9c.KLStrongFDMNewtonSettings(**kw)


def _birger_settings(cfg,case_path):
    bj=_dict(cfg,"birger","CASE")
    settings=b5.BirgerSettings(
        epsilon_tol=_nonneg(_req(bj,"epsilon_tol","birger"),"birger.epsilon_tol"),
        omega=_pos(_req(bj,"omega","birger"),"birger.omega"),
        G_ref=_pos(_req(bj,"G_ref","birger"),"birger.G_ref"),
        sigma_ref=_pos(_req(bj,"sigma_ref","birger"),"birger.sigma_ref"),
        tol_G=_pos(_req(bj,"tol_G","birger"),"birger.tol_G"),
        tol_sigma=_pos(_req(bj,"tol_sigma","birger"),"birger.tol_sigma"),
        tol_u=_pos(_req(bj,"tol_u","birger"),"birger.tol_u"),
        max_outer_iterations=_int(_req(bj,"max_outer_iterations","birger"),"birger.max_outer_iterations",1),
        epsilon_i_limit=_pos(_req(bj,"epsilon_i_limit","birger"),"birger.epsilon_i_limit"),
        delta_J_limit=_pos(_req(bj,"delta_J_limit","birger"),"birger.delta_J_limit"),
        limit_policy=str(bj.get("limit_policy","REJECT")),
        initial_slope_rtol=_pos(bj.get("initial_slope_rtol",0.05),"birger.initial_slope_rtol"),
        reject_local_unloading=bj.get("reject_local_unloading",True),
        unloading_tolerance=_nonneg(
            bj.get("unloading_tolerance",1.0e-12),
            "birger.unloading_tolerance"
        ),
    )
    curve=bj.get("curve")
    if curve is not None:
        if not isinstance(curve,dict):
            raise FinalCaseError("birger.curve must be an object")
        cmode=str(_req(curve,"mode","birger.curve")).upper()
        if cmode=="EXTERNAL_PYTHON":
            diagram=_load_callable(_req(curve,"material_diagram","birger.curve"),case_path)
        else:
            try:
                diagram=birgercurve.build_curve(curve,base_dir=case_path.parent)
                # deterministic sanity evaluation at zero; curve range policy remains explicit.
                diagram(np.array([0.0]))
            except Exception as exc:
                raise FinalCaseError(f"invalid birger.curve: {exc}") from exc
    else:
        diagram=_load_callable(_req(bj,"material_diagram","birger"),case_path)

    # K5/K6 fail-fast validation for both FDM and MVI inputs.  The same check
    # is repeated inside the generic Birger outer cycle so direct launcher use
    # is protected as well.
    mj=_dict(cfg,"material","CASE")
    E0=_pos(_req(mj,"E0","material"),"material.E0")
    nu0=float(_req(mj,"nu0","material"))
    if not np.isfinite(nu0) or not (-1.0 < nu0 < 0.5):
        raise FinalCaseError("material.nu0 must satisfy -1 < nu0 < 0.5")
    G0=E0/(2.0*(1.0+nu0))
    try:
        b5.validate_material_diagram_near_origin(
            diagram,G0=G0,epsilon_tol=settings.epsilon_tol,
            initial_slope_rtol=settings.initial_slope_rtol,
        )
        b5.validate_material_diagram_monotonic(
            diagram,epsilon_i_limit=settings.epsilon_i_limit
        )
    except Exception as exc:
        raise FinalCaseError(f"invalid Birger intensity curve: {exc}") from exc
    return settings,diagram


def _initial_state(cfg,case_path,theory,ns,nx,m):
    sj=_dict(cfg,"initial_state","CASE")
    source=str(_req(sj,"source","initial_state")).upper()
    shape=(3,m,ns,nx) if theory=="CUF" else (3,ns,nx)
    if source=="ZERO":
        return np.zeros(shape,float)
    if source!="NPZ":
        raise FinalCaseError("initial_state.source must be ZERO or NPZ")
    p=_resolve(case_path,_req(sj,"npz","initial_state"))
    key="q" if theory=="CUF" else "U"
    try:
        with np.load(p,allow_pickle=False) as z:
            if key not in z.files:
                raise FinalCaseError(f"initial state NPZ must contain '{key}'")
            a=np.asarray(z[key],float)
    except FinalCaseError:
        raise
    except Exception as exc:
        raise FinalCaseError(f"cannot read initial state NPZ {p}: {exc}") from exc
    if a.shape!=shape or not np.all(np.isfinite(a)):
        raise FinalCaseError(f"initial {key} shape must be {shape} and finite")
    return a


def _initial_GB(cfg,case_path,G0,quadrature,h,ns,nx):
    """Z4: load GB only when its through-thickness quadrature is identical."""
    gj=_dict(cfg,"initial_GB","CASE")
    source=str(_req(gj,"source","initial_GB")).upper()
    qz=np.asarray(quadrature.z,float)
    qw=np.asarray(quadrature.w,float)
    shape=(qz.size,ns,nx)
    if source=="ELASTIC_G0":
        return np.full(shape,float(G0))
    if source!="NPZ":
        raise FinalCaseError("initial_GB.source must be ELASTIC_G0 or NPZ")
    p=_resolve(case_path,_req(gj,"npz","initial_GB"))
    try:
        with np.load(p,allow_pickle=False) as z:
            if "GB" not in z.files:
                raise FinalCaseError("initial GB NPZ must contain 'GB'")
            required=("quadrature_z","quadrature_w","thickness_h")
            missing=[k for k in required if k not in z.files]
            if missing:
                raise FinalCaseError(
                    "Z4: initial GB NPZ lacks quadrature metadata " + str(missing) +
                    "; legacy GB state cannot be safely reused. Re-export a state with the current solver."
                )
            a=np.asarray(z["GB"],float)
            z_saved=np.asarray(z["quadrature_z"],float).reshape(-1)
            w_saved=np.asarray(z["quadrature_w"],float).reshape(-1)
            h_saved=float(np.asarray(z["thickness_h"],float).reshape(()))
    except FinalCaseError:
        raise
    except Exception as exc:
        raise FinalCaseError(f"cannot read initial GB NPZ {p}: {exc}") from exc
    if a.shape!=shape or not np.all(np.isfinite(a)) or np.any(a<=0):
        raise FinalCaseError(f"initial GB shape must be {shape}, finite and positive")
    htol=128.0*np.finfo(float).eps*max(1.0,abs(float(h)),abs(h_saved))
    if abs(h_saved-float(h))>htol:
        raise FinalCaseError(f"Z4: initial GB thickness_h={h_saved:.17g} does not match CASE h={float(h):.17g}")
    if z_saved.shape!=qz.shape or w_saved.shape!=qw.shape:
        raise FinalCaseError("Z4: initial GB quadrature metadata shape does not match current quadrature")
    ztol=256.0*np.finfo(float).eps*max(1.0,float(np.max(np.abs(qz))) if qz.size else 1.0)
    wtol=256.0*np.finfo(float).eps*max(1.0,float(np.max(np.abs(qw))) if qw.size else 1.0)
    if not np.allclose(z_saved,qz,rtol=0.0,atol=ztol) or not np.allclose(w_saved,qw,rtol=0.0,atol=wtol):
        raise FinalCaseError(
            "Z4: initial GB quadrature points/weights do not match current thickness quadrature; "
            "GB is defined at integration points and cannot be silently remapped"
        )
    return a


def _output_path(case_path,c,*,require_clean=True):
    out=c["output_dir"]
    if not out.is_absolute():
        out=case_path.parent/out
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if require_clean:
        stale=sorted(p.name for p in out.glob("STEP_[0-9][0-9][0-9][0-9]*"))
        if (out/"FINAL_RUN_SUMMARY.json").exists():
            stale.append("FINAL_RUN_SUMMARY.json")
        if stale:
            raise FinalCaseError(
                f"OUTPUT_DIR_NOT_EMPTY: {out} already contains results "
                f"({len(stale)} files, e.g. {stale[0]}). "
                "Choose a new output_dir; results of different runs are never mixed."
            )
    return out




def _unloading_fraction(unloading_mask):
    """Output-only Birger unloading diagnostic requested by reviewer J-13."""
    if unloading_mask is None:
        return None
    mask=np.asarray(unloading_mask,bool)
    if mask.size == 0 or not np.any(mask):
        return None
    return float(np.mean(mask))


def _append_unloading_warning(summary, frac):
    """Make a non-empty unloading mask visible without changing acceptance/mechanics."""
    if frac is None:
        return
    summary["birger_unloading_point_fraction"]=float(frac)
    summary.setdefault("validation_warnings",[]).append(
        f"РАЗГРУЗКА: Delta eps_i<0 в {float(frac):.1%} точек квадратуры. "
        "Деформационная теория Биргера на разгрузке неприменима; "
        "результат этого шага не является решением задачи пластичности."
    )


def _serialize_step(index,factor,result,state,GB=None):
    d={
        "step_index":int(index),
        "load_factor":None if factor is None else float(factor),
        "status":str(getattr(result,"status","UNKNOWN")),
        "converged":bool(getattr(result,"converged",False)),
    }
    for name in (
        "iterations","residual_inf","residual_relative","residual_scale",
        "matrix_condition_1",
        "residual_inf_interior","residual_inf_boundary","residual_inf_load_front",
        "residual_convergence_inf","residual_convergence_relative",
        "increment_inf","outer_iterations","eta_u",
        "max_epsilon_i","max_abs_J_minus_1",
        "epsilon_limit_exceeded","delta_J_limit_exceeded",
        "critical_epsilon_i_index","critical_J_index",
        "unloading_point_fraction",
    ):
        if hasattr(result,name):
            v=getattr(result,name)
            if isinstance(v,(np.bool_,bool)):
                d[name]=bool(v)
            elif isinstance(v,(tuple,list)):
                d[name]=[int(i) for i in v]
            elif v is not None:
                d[name]=float(v) if isinstance(v,(float,np.floating)) else int(v)
    if hasattr(result,"residual_inf_by_block") and getattr(result,"residual_inf_by_block"):
        d["residual_inf_by_block"]={
            str(k):float(v) for k,v in getattr(result,"residual_inf_by_block").items()
        }
    if hasattr(result,"warnings") and getattr(result,"warnings"):
        d["warnings"]=list(getattr(result,"warnings"))
    if hasattr(result,"history") and getattr(result,"history") is not None:
        d["history"]=getattr(result,"history")
    if hasattr(result,"inner_history") and getattr(result,"inner_history") is not None:
        d["inner_history"]=getattr(result,"inner_history")
    unloading_fraction=_unloading_fraction(getattr(result,"unloading_mask",None))
    if unloading_fraction is not None:
        d["birger_unloading_point_fraction"]=float(unloading_fraction)
    try:
        w=np.asarray(state,float)[2,0] if np.asarray(state).ndim==4 else np.asarray(state,float)[2]
        js,ix=np.unravel_index(int(np.argmax(np.abs(w))),w.shape)
        d["w_abs_max"]=float(np.max(np.abs(w)))
        d["w_at_abs_max"]=float(w[js,ix])
        d["wmax_is"]=int(js); d["wmax_ix"]=int(ix)
    except Exception:
        pass
    return d


def _birger_output_metrics(cfg,c,quad,result,GB):
    """Cheap diagnostics for an accepted Birger load step.

    This function is output-only: it reads the already accepted GB/material
    state and never feeds anything back into the constitutive update.
    """
    if GB is None:
        return {}
    try:
        E0=float(c["E0"]); nu0=float(c["nu0"]); R=float(c["R"])
        G0=E0/(2.0*(1.0+nu0))
        K0=E0/(3.0*(1.0-2.0*nu0))
        gb=np.asarray(GB,float); ratio=gb/G0
        # Reviewer J-15: K0 stays constant while GB varies in the Birger
        # variable-elasticity-parameter method.  Report the implied effective
        # Poisson-ratio range; this is output-only and does not feed back into
        # the material law or convergence.
        nu_eff=(3.0*K0-2.0*gb)/(2.0*(3.0*K0+gb))
        z=np.asarray(quad.z,float); qw=np.asarray(quad.w,float)
        H=1.0+z/R; wz=qw*H
        den=float(np.sum(wz))*gb.shape[1]*gb.shape[2]
        nonlinear=ratio < (1.0-1.0e-10)
        out={
            "GB_ratio_min":float(np.min(ratio)),
            "GB_ratio_max":float(np.max(ratio)),
            "birger_nu_effective_min":float(np.min(nu_eff)),
            "birger_nu_effective_max":float(np.max(nu_eff)),
            "birger_nonlinear_volume_fraction":float(np.sum(nonlinear*wz[:,None,None])/den),
        }
        local=np.sum(nonlinear*wz[:,None,None],axis=0)/float(np.sum(wz))
        out["birger_nonlinear_thickness_fraction_max"]=float(np.max(local))
        ms=getattr(result,"material_state",None)
        curve=((cfg.get("birger") or {}).get("curve") or {}) if isinstance(cfg,dict) else {}
        ys=curve.get("yield_stress") if isinstance(curve,dict) else None
        ye=curve.get("yield_strain") if isinstance(curve,dict) else None
        mask=None
        if ms is not None and ys not in (None,"") and getattr(ms,"sigma_i",None) is not None:
            mask=np.asarray(ms.sigma_i,float)>=float(ys)
        elif ms is not None and ye not in (None,"") and getattr(ms,"epsilon_i",None) is not None:
            mask=np.asarray(ms.epsilon_i,float)>=float(ye)
        if mask is not None and mask.shape==gb.shape:
            out["plastic_volume_fraction"]=float(np.sum(mask*wz[:,None,None])/den)
            out["plastic_thickness_fraction_max"]=float(np.max(np.sum(mask*wz[:,None,None],axis=0)/float(np.sum(wz))))
        return out
    except Exception:
        return {}


def _run_fdm(cfg,case_path,c):
    theory=c["theory"]; mode=c["mode"]
    grid=mvi.CylinderGrid(c["L"],c["R"],c["nx"],c["ns"])
    # The historical matched high-order operator remains the default so
    # existing verified runs are bit-for-bit on the same discretisation path.
    # INDEPENDENT_FD2 deliberately separates MKR/FDM differentiation from MVI
    # for an independent discretisation check.
    ops=_build_fdm_spatial_operators(c,grid)
    quad=c["quadrature"]
    basis=tebasis.get_basis(c["cuf_model"]) if theory=="CUF" else None
    m=basis.m if basis is not None else None

    bc,_pair=_make_case_bc(cfg,case_path,c)

    out=_output_path(case_path,c)
    loads,factors,load_meta,_load_file,load_front_surface_mask=_load_series_for_case(
        cfg,case_path,c,basis=basis,generated_dir=out
    )
    fprog.configure(out,theory=theory,cuf_model=c.get("cuf_model"),method="FDM",mode=mode,total_steps=int(len(loads)))
    fprog.emit("run_start",stage="START",total_steps=int(len(loads)))

    nonlinear=mode in ("NONLINEAR_ELASTIC","NONLINEAR_BIRGER")
    birger=mode in ("LINEAR_BIRGER","NONLINEAR_BIRGER")
    j2=(c["material_model"]=="J2_INCREMENTAL")

    if nonlinear or j2:
        state=_initial_state(cfg,case_path,theory,c["ns"],c["nx"],m)
    else:
        shape=(3,m,c["ns"],c["nx"]) if theory=="CUF" else (3,c["ns"],c["nx"])
        state=np.zeros(shape,float)

    newton=_newton_settings(cfg,theory) if (nonlinear or j2) else None
    linear_residual_tolerance=(
        _linear_residual_tolerance(cfg)
        if mode=="LINEAR_ELASTIC" else None
    )

    G0=c["E0"]/(2.0*(1.0+c["nu0"]))
    GB=None; previous_eps=None; birger_settings=None; diagram=None
    j2_state=None
    if j2:
        # The history point shape is identical for KL and CUF; the finite
        # GL adapter selects its own Fp-based state later.
        j2_state=j2case.initial_j2_state(
            cfg,case_path,
            finite=(c["geometry_model"]=="GREEN_LAGRANGE"),
            shape=(len(quad.z),c["ns"],c["nx"]),
            quadrature=quad,h=c["h"],
        )
    if birger:
        birger_settings,diagram=_birger_settings(cfg,case_path)
        GB=_initial_GB(cfg,case_path,G0,quad,c["h"],c["ns"],c["nx"])

    predictor_mode=fpred.normalize_mode((cfg.get("step_predictor",{}) or {}).get("mode","ERROR"))
    previous_state=None
    previous_factor=None
    current_factor=0.0
    steps=[]
    all_converged=True

    # K3: in LINEAR_ELASTIC the discrete operator depends on geometry,
    # material, grid and boundary conditions, but not on the load amplitude.
    # Assemble it once for the whole load series; only r0=R(0) is recomputed
    # for each load step.  This changes no mechanics and uses the same solver.
    linear_operator=None
    if mode=="LINEAR_ELASTIC":
        if len(loads)==0:
            raise FinalCaseError("LINEAR_ELASTIC load series is empty")
        zero_load=np.zeros_like(np.asarray(loads[0],float))
        if theory=="CUF":
            linear_operator,_=c10a.assemble_linear_operator_and_rhs(
                ops,basis,bc,E0=c["E0"],nu0=c["nu0"],
                quadrature=quad,generalized_load=zero_load
            )
        else:
            linear_operator,_=k9b.assemble_linear_strong_system(
                ops,bc,E=c["E0"],nu=c["nu0"],h=c["h"],
                generalized_load=zero_load
            )

    for k,load in enumerate(loads):
        factor=None if factors is None else factors[k]
        fprog.set_context(step_index=int(k+1),load_factor=None if factor is None else float(factor))
        fprog.emit("step_start",stage="LOAD_STEP")
        factor_value=float(k+1 if factor is None else factor)
        accepted_start_state=np.asarray(state,float).copy()
        j2_trial=None
        solve_state=state
        if mode!="LINEAR_ELASTIC":
            solve_state=fpred.predict_array(
                previous_state,state,factor_value,current_factor,previous_factor,
                mode=predictor_mode
            )

        if theory=="CUF":
            if mode=="LINEAR_ELASTIC":
                r0=c10a.linear_rhs_only(
                    ops,basis,bc,E0=c["E0"],nu0=c["nu0"],
                    quadrature=quad,generalized_load=load
                )
                res=c10a.solve_from_operator(
                    linear_operator,r0,ops,basis,bc,
                    E0=c["E0"],nu0=c["nu0"],quadrature=quad,
                    generalized_load=load,
                    linear_residual_tolerance_relative=linear_residual_tolerance
                )
                state=res.q
            elif mode=="NONLINEAR_ELASTIC":
                if c["geometry_model"]=="VON_KARMAN":
                    res=c10e.solve_vk_hooke_strong_fdm(
                        solve_state,ops,basis,bc,E0=c["E0"],nu0=c["nu0"],
                        quadrature=quad,generalized_load=load,settings=newton,
                        load_front_surface_mask=load_front_surface_mask
                    )
                else:
                    res=c10b.solve_full_gl_elastic_strong_fdm(
                        solve_state,ops,basis,bc,E0=c["E0"],nu0=c["nu0"],
                        quadrature=quad,generalized_load=load,settings=newton,
                        load_front_surface_mask=load_front_surface_mask
                    )
                state=res.q
            elif mode=="LINEAR_J2":
                res=c10f.solve_j2_strong_fdm(
                    solve_state,ops,basis,bc,geometry="LINEAR",committed=j2_state,
                    parameters=c["j2_parameters"],quadrature=quad,
                    generalized_load=load,settings=newton,E0=c["E0"],
                    load_front_surface_mask=load_front_surface_mask,
                )
                state=res.q; j2_trial=res.trial_state
            elif mode=="NONLINEAR_J2":
                if c["geometry_model"] not in ("VON_KARMAN","GREEN_LAGRANGE"):
                    raise FinalCaseError(
                        f"invalid CUF NONLINEAR_J2 geometry_model for FDM: {c['geometry_model']}"
                    )
                res=c10f.solve_j2_strong_fdm(
                    solve_state,ops,basis,bc,geometry=c["geometry_model"],
                    committed=j2_state,parameters=c["j2_parameters"],
                    quadrature=quad,generalized_load=load,settings=newton,E0=c["E0"],
                    load_front_surface_mask=load_front_surface_mask,
                )
                state=res.q; j2_trial=res.trial_state
            elif mode=="LINEAR_BIRGER":
                res,_counter=c10c.solve_linear_birger_strong_fdm(
                    solve_state,GB,ops=ops,basis=basis,bc=bc,
                    E0=c["E0"],nu0=c["nu0"],quadrature=quad,
                    generalized_load=load,material_diagram=diagram,
                    birger_settings=birger_settings,
                    previous_accepted_epsilon_i=previous_eps
                )
                state=res.state; GB=res.GB; previous_eps=res.material_state.epsilon_i
            else:
                if c["geometry_model"]=="VON_KARMAN":
                    res,_counter=c10e.solve_vk_birger_strong_fdm(
                        solve_state,GB,ops=ops,basis=basis,bc=bc,
                        E0=c["E0"],nu0=c["nu0"],quadrature=quad,
                        generalized_load=load,newton_settings=newton,
                        material_diagram=diagram,birger_settings=birger_settings,
                        previous_accepted_epsilon_i=previous_eps,
                        load_front_surface_mask=load_front_surface_mask
                    )
                else:
                    res,_counter=c10d.solve_full_gl_birger_strong_fdm(
                        solve_state,GB,ops=ops,basis=basis,bc=bc,
                        E0=c["E0"],nu0=c["nu0"],quadrature=quad,
                        generalized_load=load,newton_settings=newton,
                        material_diagram=diagram,birger_settings=birger_settings,
                        previous_accepted_epsilon_i=previous_eps,
                        load_front_surface_mask=load_front_surface_mask
                    )
                state=res.state; GB=res.GB; previous_eps=res.material_state.epsilon_i
        else:
            if mode=="LINEAR_ELASTIC":
                r0=k9b.linear_rhs_only(
                    ops,bc,E=c["E0"],nu=c["nu0"],h=c["h"],
                    generalized_load=load
                )
                res=k9b.solve_from_operator(
                    linear_operator,r0,ops,bc,
                    E=c["E0"],nu=c["nu0"],h=c["h"],
                    generalized_load=load,
                    linear_residual_tolerance_relative=linear_residual_tolerance
                )
                state=res.U
            elif mode=="NONLINEAR_ELASTIC":
                if c["geometry_model"]=="VON_KARMAN":
                    res=k9c.solve_vk_elastic_strong_fdm(
                        solve_state,ops,bc,E=c["E0"],nu=c["nu0"],h=c["h"],
                        generalized_load=load,settings=newton,
                        load_front_surface_mask=load_front_surface_mask
                    )
                elif c["geometry_model"]=="GREEN_LAGRANGE":
                    res=k9e.solve_gl_elastic_strong_fdm(
                        solve_state,ops,bc,E=c["E0"],nu=c["nu0"],h=c["h"],
                        generalized_load=load,settings=newton,
                        load_front_surface_mask=load_front_surface_mask
                    )
                else:
                    raise FinalCaseError(
                        f"invalid KL NONLINEAR_ELASTIC geometry_model for FDM: {c['geometry_model']}"
                    )
                state=res.U
            elif mode=="LINEAR_J2":
                if c["geometry_model"]!="LINEAR":
                    raise FinalCaseError("LINEAR_J2 requires geometry_model=LINEAR")
                res=k9j.solve_linear_j2_strong_fdm(
                    solve_state,ops,bc,committed=j2_state,
                    parameters=c["j2_parameters"],quadrature=quad,
                    generalized_load=load,settings=newton,
                    load_front_surface_mask=load_front_surface_mask,
                )
                state=res.U; j2_trial=res.trial_state
            elif mode=="NONLINEAR_J2":
                if c["geometry_model"]=="VON_KARMAN":
                    res=k9k.solve_vk_j2_strong_fdm(
                        solve_state,ops,bc,committed=j2_state,
                        parameters=c["j2_parameters"],quadrature=quad,
                        generalized_load=load,settings=newton,
                        load_front_surface_mask=load_front_surface_mask,
                    )
                elif c["geometry_model"]=="GREEN_LAGRANGE":
                    res=k9l.solve_gl_j2_strong_fdm(
                        solve_state,ops,bc,committed=j2_state,
                        parameters=c["j2_parameters"],quadrature=quad,
                        generalized_load=load,settings=newton,
                        load_front_surface_mask=load_front_surface_mask,
                    )
                else:
                    raise FinalCaseError(
                        f"invalid KL NONLINEAR_J2 geometry_model for FDM: {c['geometry_model']}"
                    )
                state=res.U; j2_trial=res.trial_state
            elif mode=="LINEAR_BIRGER":
                res,_counter=k9d1.solve_linear_birger_strong_fdm(
                    solve_state,GB,ops=ops,bc=bc,E0=c["E0"],nu0=c["nu0"],
                    h=c["h"],quadrature=quad,generalized_load=load,
                    material_diagram=diagram,birger_settings=birger_settings,
                    previous_accepted_epsilon_i=previous_eps
                )
                state=res.state; GB=res.GB; previous_eps=res.material_state.epsilon_i
            else:
                if c["geometry_model"]=="VON_KARMAN":
                    res,_counter=k9d2.solve_vk_birger_strong_fdm(
                        solve_state,GB,ops=ops,bc=bc,E0=c["E0"],nu0=c["nu0"],
                        h=c["h"],quadrature=quad,generalized_load=load,
                        newton_settings=newton,material_diagram=diagram,
                        birger_settings=birger_settings,
                        previous_accepted_epsilon_i=previous_eps,
                        load_front_surface_mask=load_front_surface_mask
                    )
                elif c["geometry_model"]=="GREEN_LAGRANGE":
                    res,_counter=k9f.solve_gl_birger_strong_fdm(
                        solve_state,GB,ops=ops,bc=bc,E0=c["E0"],nu0=c["nu0"],
                        h=c["h"],quadrature=quad,generalized_load=load,
                        newton_settings=newton,material_diagram=diagram,
                        birger_settings=birger_settings,
                        previous_accepted_epsilon_i=previous_eps,
                        load_front_surface_mask=load_front_surface_mask
                    )
                else:
                    raise FinalCaseError(
                        f"invalid KL NONLINEAR_BIRGER geometry_model for FDM: {c['geometry_model']}"
                    )
                state=res.state; GB=res.GB; previous_eps=res.material_state.epsilon_i

        converged=bool(getattr(res,"converged",False))
        # Commit incremental plastic history only for an accepted global step.
        if j2 and converged:
            if j2_trial is None:
                raise FinalCaseError("J2 solver converged without a trial material state")
            j2_state=j2_trial
        steps.append(_serialize_step(k,factor,res,state,GB))
        if birger:
            steps[-1].update(_birger_output_metrics(cfg,c,quad,res,GB))
        state_key="q" if theory=="CUF" else "U"
        save_kw={state_key:np.asarray(state,float)}
        if j2 and converged and j2_state is not None:
            ms=getattr(res,"material_state",None)
            save_kw["j2_history_format"]=np.asarray(j2history.FORMAT)
            if hasattr(j2_state,"plastic_strain"):
                save_kw["j2_history_kind"]=np.asarray(j2history.SMALL)
                save_kw["j2_plastic_strain"]=np.asarray(j2_state.plastic_strain,float)
                save_kw["j2_equivalent_stress"]=np.asarray(j2_state.equivalent_stress,float)
            else:
                save_kw["j2_history_kind"]=np.asarray(j2history.FINITE)
                save_kw["j2_Fp"]=np.asarray(j2_state.Fp,float)
                save_kw["j2_equivalent_kirchhoff_stress"]=np.asarray(
                    j2_state.equivalent_kirchhoff_stress,float
                )
            save_kw["j2_kappa"]=np.asarray(j2_state.kappa,float)
            save_kw["quadrature_z"]=np.asarray(quad.z,float)
            save_kw["quadrature_w"]=np.asarray(quad.w,float)
            save_kw["thickness_h"]=np.asarray(float(c["h"]))
            if ms is not None:
                for name in (
                    "yield_stress","plastic_multiplier","plastic_mask",
                    "unloading_mask","reloading_mask","stress","total_strain",
                    "second_piola","cauchy","green_strain",
                    "deformation_gradient","plastic_deformation_gradient",
                ):
                    value=getattr(ms,name,None)
                    if value is not None:
                        save_kw["j2_"+name]=np.asarray(value)
            _j2_resultants_output(save_kw,getattr(res,"resultants",None))
        if GB is not None:
            save_kw["GB"]=np.asarray(GB,float)
            # Z4 provenance: GB values live at these exact thickness integration points.
            save_kw["quadrature_z"]=np.asarray(quad.z,float)
            save_kw["quadrature_w"]=np.asarray(quad.w,float)
            save_kw["thickness_h"]=np.asarray(float(c["h"]))

            # Output-only enrichment: preserve the material-point state already
            # computed by Birger.  No constitutive calculation is repeated here.
            ms=getattr(res,"material_state",None)
            if ms is not None:
                for name in ("epsilon_i","sigma_i","detF"):
                    value=getattr(ms,name,None)
                    if value is not None:
                        save_kw[name]=np.asarray(value)
            unloading=getattr(res,"unloading_mask",None)
            if unloading is not None:
                save_kw["unloading_mask"]=np.asarray(unloading,bool)
        # A step that did not converge is kept for diagnostics but must never share
        # the file-naming of an accepted step: postprocess globs STEP_*_STATE.npz.
        suffix="" if converged else "_NOT_CONVERGED"
        _atomic_savez(out/f"STEP_{k+1:04d}_STATE{suffix}.npz",**save_kw)
        _atomic_write_json(out/f"STEP_{k+1:04d}{suffix}.json",steps[-1])
        # Recovery checkpoint is published immediately after every step file.
        # If Python/Windows/the postprocessor stops later, all previously accepted
        # steps and their load history remain readable without rerunning mechanics.
        _atomic_write_json(
            out/"RECOVERY_CHECKPOINT.json",
            _recovery_checkpoint(cfg,c,theory,mode,load_meta,loads,steps,all_converged,ops)
        )
        ev=dict(steps[-1]); ev.pop("history",None); ev.pop("inner_history",None)
        # Live GUI uses one-based step numbers; keep the established STEP JSON
        # schema untouched (its step_index remains legacy zero-based).
        ev.pop("step_index",None)
        fprog.emit("step_complete",stage="STEP_COMPLETE",step_index=int(k+1),accepted=converged,**ev)
        if not converged:
            all_converged=False
            break
        previous_state=accepted_start_state
        previous_factor=current_factor
        current_factor=factor_value

    summary={
        "theory":theory,
        "cuf_model":c.get("cuf_model"),
        "method":"FDM",
        "mode":mode,
        "geometry_model":c["geometry_model"],
        "capability":c["capability"].to_dict(),
        "material_model":c["material_model"],
        "units":c["units"],
        "converged":all_converged and len(steps)==len(loads),
        "status":"CONVERGED_FINAL_FDM_SERIES" if all_converged and len(steps)==len(loads)
                 else "FAILED_FINAL_FDM_SERIES",
        "requested_steps":int(len(loads)),
        "accepted_steps":int(sum(1 for s in steps if s.get("converged"))),
        "attempted_steps":int(len(steps)),
        "geometry":{"L":c["L"],"R":c["R"],"h":c["h"]},
        "material":{"E0":c["E0"],"nu0":c["nu0"]},
        "grid":{"nx":c["nx"],"ns":c["ns"],"stencil":c["stencil"]},
        "spatial_discretisation":_fdm_spatial_metadata(c),
        "quadrature":dict(c["quadrature_spec"]),
        "validation_warnings":list(c["quadrature_warnings"])+list(c["unit_validation"].get("warnings",[]))+list(c["boundary_layer_warnings"])+list(c["capability_warnings"]),
        "boundary_conditions":cfg.get("boundary_conditions"),
        "load_definition":load_meta,
        "postprocess":cfg.get("postprocess",{}),
        "birger_definition":cfg.get("birger") if birger else None,
        "steps":steps,
    }
    failed_step=next((s for s in steps if not s.get("converged",False)),None)
    if failed_step is not None:
        failed_status=str(failed_step.get("status","UNKNOWN"))
        summary["failed_step_status"]=failed_status
        if "BIRGER_EPSILON_I_LIMIT_EXCEEDED" in failed_status:
            summary["failed_step_code"]="BIRGER_EPSILON_I_LIMIT_EXCEEDED"
            summary["failed_step_diagnostic_ru"]="Превышен заданный предел интенсивности деформаций Биргера."
            summary["failed_step_recommendation_ru"]="Шаг не принят. Уменьшите приращение нагрузки или конечную нагрузку."
        elif "BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED" in failed_status:
            summary["failed_step_code"]="BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED"
            summary["failed_step_diagnostic_ru"]="Превышен заданный предел |J-1| для модели Биргера."
            summary["failed_step_recommendation_ru"]="Шаг не принят. Уменьшите приращение нагрузки; не ослабляйте предел без физического обоснования."
        elif "BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED" in failed_status:
            summary["failed_step_code"]="BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED"
            summary["failed_step_diagnostic_ru"]="Обнаружена локальная разгрузка, недопустимая для применяемой деформационной теории Биргера."
            summary["failed_step_recommendation_ru"]="Шаг не принят. Проверьте монотонность и уменьшите приращение нагрузки."
    first_unloading=next((
        s.get("birger_unloading_point_fraction") for s in steps
        if s.get("converged") and s.get("birger_unloading_point_fraction") is not None
    ),None)
    _append_unloading_warning(summary,first_unloading)
    _atomic_write_json(out/"FINAL_RUN_SUMMARY.json",summary)
    # Final summary supersedes the recovery snapshot, but keep the latter as an
    # explicit record that every step was checkpointed during the run.
    _atomic_write_json(out/"RECOVERY_CHECKPOINT.json",{**summary,"format":"CYLINDER_RECOVERY_CHECKPOINT_V1","checkpoint":True,"last_saved_step":int(len(steps))})
    fprog.emit("run_done",stage="DONE",converged=bool(summary["converged"]),status=summary["status"],completed_steps=int(summary["accepted_steps"]))
    return summary


def _mvi_child_birger(cfg,case_path,out):
    """Translate new GUI Birger curve specification to verified launcher callable input."""
    bj=json.loads(json.dumps(_dict(cfg,"birger","CASE")))
    curve=bj.get("curve")
    if curve is None:
        spec=bj.get("material_diagram")
        if isinstance(spec,dict) and spec.get("python_path"):
            spec["python_path"]=str(_resolve(case_path,spec["python_path"]))
        return bj
    mode=str(curve.get("mode","")).upper()
    if mode=="EXTERNAL_PYTHON":
        spec=curve.get("material_diagram")
        if not isinstance(spec,dict):
            raise FinalCaseError("birger.curve EXTERNAL_PYTHON requires material_diagram")
        if spec.get("python_path"):
            spec["python_path"]=str(_resolve(case_path,spec["python_path"]))
        bj["material_diagram"]=spec
        bj.pop("curve",None)
        return bj
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    module_path=out/"generated_birger_curve_provider.py"
    curve_abs=json.loads(json.dumps(curve))
    if mode=="CSV" and curve_abs.get("path"):
        curve_abs["path"]=str(_resolve(case_path,curve_abs["path"]))
    code=(
        "import numpy as np\n"
        "from final_birger_curve import build_curve\n"
        f"SPEC={repr(curve_abs)}\n"
        f"BASE_DIR={repr(str(case_path.parent.resolve()))}\n"
        "_CURVE=build_curve(SPEC,base_dir=BASE_DIR)\n"
        "def Phi(eps):\n    return _CURVE(np.asarray(eps,float))\n"
    )
    module_path.write_text(code,encoding="utf-8")
    bj["material_diagram"]={
        "module":"generated_birger_curve_provider",
        "callable":"Phi",
        "python_path":str(out),
    }
    bj.pop("curve",None)
    return bj


def _build_auto_mvi_profiles(c,rank,bc=None):
    """Build a deterministic, unit-consistent initial MVI profile bank.

    This is launcher/input preparation only; it does not change the MVI
    equations, tangent, half-step order or convergence logic.  The axial
    envelope is zero together with its analytical first derivative at both
    ends, so it is admissible as a starting direction for every homogeneous
    simple KL/CUF end condition supported by the dispatcher.  Expert-supplied
    NPZ profiles still take precedence.
    """
    rank=_int(rank,"rank",1)
    if rank>8:
        raise FinalCaseError("MVI rank must be 1..8")

    nx=int(c["nx"]); ns=int(c["ns"])
    xi=np.linspace(0.0,1.0,nx)
    theta=2.0*np.pi*np.arange(ns,dtype=float)/float(ns)
    envelope=xi*xi*(1.0-xi)*(1.0-xi)
    # X carries displacement units and Y is dimensionless.  Scaling with h
    # makes the same physical seed in m-N-Pa and mm-N-MPa unit systems.
    amplitude=1.0e-5*float(c["h"])

    def periodic_seed(r,phase):
        """V11 independent periodic bank retained for the KL solver."""
        if r==0:
            return (
                1.0
                +0.05*np.cos(theta+phase)
                +0.025*np.sin(2.0*theta+phase)
            )
        return (
            np.cos(float(r+1)*theta+phase)
            +0.5*np.sin(float(r+2)*theta+phase)
        )

    if c["theory"]=="KL":
        X=np.empty((3,rank,nx),float)
        Y=np.empty((3,rank,ns),float)
        for component in range(3):
            phase=0.2*component
            for r in range(rank):
                X[component,r]=(
                    amplitude*(1.0+0.1*component)/(r+1.0)
                    *envelope*np.cos(float(r)*np.pi*xi)
                )
                Y[component,r]=periodic_seed(r,phase)
        # Project the sampled seed onto the *active* homogeneous essential
        # KL constraints.  Older builds always used CLAMPED-CLAMPED here,
        # which silently made the automatic seed more restrictive than the
        # selected support.
        if bc is None:
            raise FinalCaseError("active KL boundary set is required for automatic MVI profiles")
        grid=mvi.CylinderGrid(float(c["L"]),float(c["R"]),nx,ns)
        ops=c3.CylinderSpatialOperators.build(grid,int(c["stencil"]))
        maps=kess.build_x_constraint_maps(ops,bc)
        for component,cmap in enumerate(maps.maps):
            free=np.asarray(cmap.free_columns,int)
            for r in range(rank):
                X[component,r]=cmap.expand(X[component,r,free])
    else:
        m=tebasis.get_basis(c["cuf_model"]).m
        X=np.empty((3,m,rank,nx),float)
        Y=np.empty((3,m,rank,ns),float)
        z_scale=max(0.5*abs(float(c["h"])),1.0e-30)
        for component in range(3):
            for tau in range(m):
                # For F_tau=z^tau, q_tau has units length^(1-tau).
                # Keep the physical contribution q_tau*z^tau at O(amplitude)
                # instead of giving high-order TE terms displacement-sized
                # coefficients and blowing up the initial deformation map.
                tau_scale=z_scale**tau
                for r in range(rank):
                    # Avoid nodal zeros of cos(r*pi*xi) on very coarse
                    # clamped grids.  With nx=7 the only strongly active
                    # point can be xi=0.5, where the old r=1 seed vanished
                    # and made MVI-2 look rank-degenerate before the first
                    # physical cycle.  A deterministic phase shift keeps the
                    # same admissible envelope but gives every requested rank
                    # a nonzero independent X partner.
                    shape=(
                        np.cos(float(r)*np.pi*xi+0.37*float(r+1))
                        +0.25*np.sin(float(r+1)*np.pi*xi+0.19*float(component+tau+1))
                    )
                    axial_seed=envelope*shape
                    norm=float(np.linalg.norm(axial_seed))
                    if norm<=1.0e-30:
                        raise FinalCaseError("automatic CUF X seed is zero")
                    # Normalize only the axial shape.  Normalizing the final
                    # X profile would cancel 1/z_scale**tau and artificially
                    # amplify high-order TE terms in the first MVI tangent.
                    X[component,tau,r]=(
                        amplitude*(1.0+0.1*component+0.05*tau)/(r+1.0)/tau_scale
                        *axial_seed/norm
                    )
                    Y[component,tau,r]=0.0

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise FinalCaseError("automatic MVI profiles contain non-finite values")
    return X,Y


def _write_auto_mvi_profiles(out,c,rank,bc=None,filename="AUTO_INITIAL_PROFILES_MVI.npz"):
    """Persist an automatically generated profile bank inside the run."""
    X,Y=_build_auto_mvi_profiles(c,rank,bc=bc)
    target=Path(out)/"INPUT"/filename
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        target,
        X=X,
        Y=Y,
        source=np.asarray(
            "AUTO_DETERMINISTIC_V2_INDEPENDENT_HARMONICS"
            if c["theory"]=="KL"
            else "AUTO_DETERMINISTIC_V3_PLATE_STYLE_X_YZERO"
        ),
        theory=np.asarray(c["theory"]),
        cuf_model=np.asarray(c.get("cuf_model") or ""),
        rank=np.asarray(int(rank)),
        nx=np.asarray(int(c["nx"])),
        ns=np.asarray(int(c["ns"])),
        thickness_h=np.asarray(float(c["h"])),
    )
    return target.resolve()



def _j2_mvi_settings(cfg,theory):
    """Parse the existing strict MVI controls without changing their meaning."""
    im=_dict(cfg,"inner_mvi","CASE")
    common=dict(
        projected_tolerance=_pos(_req(im,"projected_tolerance","inner_mvi"),"inner_mvi.projected_tolerance"),
        increment_tolerance=_pos(_req(im,"increment_tolerance","inner_mvi"),"inner_mvi.increment_tolerance"),
        tangent_fd_step=_pos(_req(im,"tangent_fd_step","inner_mvi"),"inner_mvi.tangent_fd_step"),
        max_halfstep_iterations=_int(_req(im,"max_halfstep_iterations","inner_mvi"),"inner_mvi.max_halfstep_iterations",1),
        max_xy_cycles=_int(_req(im,"max_xy_cycles","inner_mvi"),"inner_mvi.max_xy_cycles",1),
    )
    max_correction_halvings=_int(
        im.get("max_correction_halvings",24),
        "inner_mvi.max_correction_halvings",0
    )
    if theory=="KL":
        return kl_mvi_core.KLInternalMVISettings(
            **common,
            max_correction_halvings=max_correction_halvings,
        )
    return cuf_mvi_core.GeneralInternalMVISettings(
        **common,
        detf_guarded_correction=bool(im.get("detf_guarded_correction",True)),
        max_correction_halvings=max_correction_halvings,
        algorithm=str(im.get("algorithm","STRICT_HALFSTEP")).upper(),
    )


def _j2_mvi_adaptation(cfg):
    aj=cfg.get("load_step_adaptation",{})
    if aj is None:
        aj={}
    if not isinstance(aj,dict):
        raise FinalCaseError("load_step_adaptation must be an object")
    return (
        bool(aj.get("enabled",True)),
        _int(
            aj.get("max_subdivision_depth",8),
            "load_step_adaptation.max_subdivision_depth",0
        ),
    )


def _load_j2_mvi_profiles(path,c,rank,basis):
    """Read exact X/Y profile data without projection, repair or rescaling."""
    p=Path(path)
    try:
        with np.load(p,allow_pickle=False) as z:
            if "X" not in z.files or "Y" not in z.files:
                raise FinalCaseError("initial profile NPZ must contain X and Y")
            X=np.asarray(z["X"],float)
            Y=np.asarray(z["Y"],float)
    except FinalCaseError:
        raise
    except Exception as exc:
        raise FinalCaseError(f"cannot read initial MVI profiles {p}: {exc}") from exc
    if c["theory"]=="KL":
        ex=(3,rank,c["nx"]); ey=(3,rank,c["ns"])
        if X.shape!=ex or Y.shape!=ey or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise FinalCaseError(f"KL MVI profile shapes must be X{ex}, Y{ey}, finite")
        return kprofiles.ProfileStateKL(X.copy(),Y.copy())
    ex=(3,basis.m,rank,c["nx"]); ey=(3,basis.m,rank,c["ns"])
    if X.shape!=ex or Y.shape!=ey or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise FinalCaseError(f"CUF MVI profile shapes must be X{ex}, Y{ey}, finite")
    return cuf_mvi_driver.load_initial_profiles_npz(
        p,m=basis.m,rank=rank,nx=c["nx"],ns=c["ns"]
    )


def _j2_resultants_output(data,resultants):
    """Persist physical, already-integrated J2 resultants without rebuilding them."""
    if resultants is None:
        raise FinalCaseError("accepted J2 step has no physical resultants")
    as_dict=getattr(resultants,"as_dict",None)
    if callable(as_dict):
        items=as_dict().items()
    else:
        names=("Nx","Ns","Nxs","Mx","Ms","Mxs")
        items=((name,getattr(resultants,name)) for name in names if hasattr(resultants,name))
    saved=0
    for name,value in items:
        array=np.asarray(value,float)
        if not np.isfinite(array).all():
            raise FinalCaseError(f"J2 resultant {name} is non-finite")
        data["j2_resultant_"+str(name)]=array
        saved+=1
    if saved==0:
        raise FinalCaseError("accepted J2 step exposes no recognized resultants")


def _j2_history_output(data,history,material_state,quad,h,resultants=None):
    data["j2_history_format"]=np.asarray(j2history.FORMAT)
    if hasattr(history,"plastic_strain"):
        data["j2_history_kind"]=np.asarray(j2history.SMALL)
        data["j2_plastic_strain"]=np.asarray(history.plastic_strain,float)
        data["j2_equivalent_stress"]=np.asarray(history.equivalent_stress,float)
    else:
        data["j2_history_kind"]=np.asarray(j2history.FINITE)
        data["j2_Fp"]=np.asarray(history.Fp,float)
        data["j2_equivalent_kirchhoff_stress"]=np.asarray(
            history.equivalent_kirchhoff_stress,float
        )
    data["j2_kappa"]=np.asarray(history.kappa,float)
    data["quadrature_z"]=np.asarray(quad.z,float)
    data["quadrature_w"]=np.asarray(quad.w,float)
    data["thickness_h"]=np.asarray(float(h))
    for name in (
        "yield_stress","plastic_multiplier","plastic_mask",
        "unloading_mask","reloading_mask","stress","total_strain",
        "second_piola","cauchy","green_strain",
        "deformation_gradient","plastic_deformation_gradient",
    ):
        value=getattr(material_state,name,None)
        if value is not None:
            data["j2_"+name]=np.asarray(value)
    _j2_resultants_output(data,resultants)


def _run_j2_mvi(cfg,case_path,c):
    """Dedicated J2 MVI route: accepted history is never delegated to Birger."""
    rank=_int(_req(cfg,"rank","CASE"),"rank",1)
    if rank>8:
        raise FinalCaseError("MVI rank must be 1..8")
    theory=c["theory"]
    basis=tebasis.get_basis(c["cuf_model"]) if theory=="CUF" else None
    grid=mvi.CylinderGrid(c["L"],c["R"],c["nx"],c["ns"])
    ops=c3.CylinderSpatialOperators.build(grid,c["stencil"])
    bc,_pair=_make_case_bc(cfg,case_path,c)
    _validate_mvi_homogeneous_geometric_bc(
        bc,theory,c["ns"],c["geometry_model"]
    )
    out=_output_path(case_path,c)
    profile_value=cfg.get("initial_profiles_npz")
    if profile_value is None or not str(profile_value).strip():
        profile_path=_write_auto_mvi_profiles(out,c,rank,bc=bc)
        profile_source="AUTO"
    else:
        profile_path=_resolve(case_path,profile_value)
        profile_source="EXTERNAL_NPZ"
    initial_state=_load_j2_mvi_profiles(profile_path,c,rank,basis)
    loads,factors,load_meta,_load_file,_front=_load_series_for_case(
        cfg,case_path,c,basis=basis,generated_dir=out
    )
    settings=_j2_mvi_settings(cfg,theory)
    adaptive,max_depth=_j2_mvi_adaptation(cfg)
    quadrature=c["quadrature"]
    history=j2case.initial_j2_state(
        cfg,case_path,
        finite=(c["geometry_model"]=="GREEN_LAGRANGE"),
        shape=(len(quadrature.z),c["ns"],c["nx"]),
        quadrature=quadrature,h=c["h"],
    )
    fprog.configure(
        out,theory=theory,cuf_model=c.get("cuf_model"),method="MVI",
        mode=c["mode"],rank=rank,total_steps=int(len(loads))
    )
    fprog.emit("run_start",stage="START",total_steps=int(len(loads)),rank=rank)

    steps=[]

    def _publish_accepted_step(step):
        """Persist one accepted MVI state before the next trial begins."""
        meta={
            "step_index":int(step.step_index),
            "requested_step_index":int(step.requested_step_index),
            "load_factor":float(step.load_factor),
            "converged":True,"status":str(step.solver_status),
            "iterations":int(step.solver_iterations),
            "convergence_meaning":"projected MVI equations",
            "subdivision_depth":int(step.subdivision_depth),
            "is_substep":bool(step.is_substep),
            "history":list(step.solver_history),
        }
        state_data={
            "X":np.asarray(step.state.X,float),
            "Y":np.asarray(step.state.Y,float),
        }
        _j2_history_output(
            state_data,step.history_state,step.material_state,quadrature,c["h"],
            resultants=step.resultants,
        )
        _atomic_savez(out/f"STEP_{int(step.step_index):04d}_STATE.npz",**state_data)
        _atomic_write_json(out/f"STEP_{int(step.step_index):04d}.json",meta)
        steps.append(meta)
        fprog.emit(
            "step_complete",stage="STEP_COMPLETE",step_index=int(step.step_index),
            requested_step_index=int(step.requested_step_index),accepted=True,
            solver_status=str(step.solver_status),
            solver_iterations=int(step.solver_iterations),
            subdivision_depth=int(step.subdivision_depth),
            is_substep=bool(step.is_substep),
        )

    args=dict(
        geometry=c["geometry_model"],initial_history=history,
        parameters=c["j2_parameters"],quadrature=quadrature,
        load_steps=loads,load_factors=factors,bc=bc,
        step_predictor_mode=(cfg.get("step_predictor",{}) or {}).get("mode","ERROR"),
        automatic_step_subdivision=adaptive,max_subdivision_depth=max_depth,
        on_step_accepted=_publish_accepted_step,
    )
    if theory=="CUF":
        result=cuf_j2_mvi.solve_j2_explicit_load_series(
            initial_state,ops,basis=basis,inner_settings=settings,E0=c["E0"],**args
        )
    else:
        result=kl_j2_mvi.solve_j2_explicit_load_series(
            initial_state,ops,settings=settings,**args
        )

    # Accepted steps are already persisted and published by the callback.


    summary={
        "theory":theory,"cuf_model":c.get("cuf_model"),"method":"MVI",
        "mode":c["mode"],"geometry_model":c["geometry_model"],
        "material_model":"J2_INCREMENTAL","capability":c["capability"].to_dict(),
        "units":c["units"],"rank":rank,
        "converged":bool(result.converged),"status":str(result.status),
        "convergence_meaning":"projected MVI equations; full strong residual is diagnostic only",
        "requested_steps":int(result.requested_steps),
        "accepted_steps":int(len(result.accepted_steps)),
        "failed_step_index":result.failed_step_index,
        "failed_step_status":result.failed_step_status,
        "failed_step_code":getattr(result,"failed_step_code",None),
        "failed_step_diagnostic_ru":getattr(result,"failed_step_diagnostic_ru",None),
        "failed_step_recommendation_ru":getattr(result,"failed_step_recommendation_ru",None),
        "automatic_step_subdivision":bool(adaptive),
        "max_subdivision_depth":int(max_depth),
        "subdivisions_used":int(getattr(result,"subdivisions_used",0)),
        "initial_profiles":{"source":profile_source,"path":str(profile_path)},
        "geometry":{"L":c["L"],"R":c["R"],"h":c["h"]},
        "material":{"E0":c["E0"],"nu0":c["nu0"]},
        "grid":{"nx":c["nx"],"ns":c["ns"],"stencil":c["stencil"]},
        "quadrature":dict(c["quadrature_spec"]),
        "validation_warnings":list(c["quadrature_warnings"])+list(c["unit_validation"].get("warnings",[]))+list(c["boundary_layer_warnings"])+list(c["capability_warnings"]),
        "boundary_conditions":cfg.get("boundary_conditions"),
        "load_definition":load_meta,"postprocess":cfg.get("postprocess",{}),
        "steps":steps,
    }
    _atomic_write_json(out/"FINAL_RUN_SUMMARY.json",summary)
    _atomic_write_json(
        out/"RECOVERY_CHECKPOINT.json",
        {**summary,"format":"CYLINDER_RECOVERY_CHECKPOINT_V1","checkpoint":True,
         "last_saved_step":int(len(steps))}
    )
    fprog.emit(
        "run_done",stage="DONE",converged=bool(result.converged),
        status=str(result.status),completed_steps=int(len(steps)),
        failed_step_code=getattr(result,"failed_step_code",None),
        failed_step_status=getattr(result,"failed_step_status",None),
    )
    return summary

def _select_cuf_mvi_formulation(cfg,c,bc_obj):
    """Resolve CUF MVI residual formulation without silently changing unsupported cases."""
    inner=dict(_dict(cfg,"inner_mvi","CASE"))
    raw=inner.get("formulation")
    explicit=raw is not None and str(raw).strip()!=""

    eligible_variational=(
        c["theory"]=="CUF"
        and c["geometry_model"] in ("LINEAR","VON_KARMAN","GREEN_LAGRANGE")
        and c["material_model"]=="HOOKE"
        and bc_obj.left.all_geometric()
        and bc_obj.right.all_geometric()
    )

    if explicit:
        formulation=str(raw).upper().strip()
        if formulation not in ("STRONG","VARIATIONAL"):
            raise FinalCaseError("inner_mvi.formulation must be STRONG or VARIATIONAL")
        if formulation=="VARIATIONAL" and not eligible_variational:
            raise FinalCaseError(
                "VARIATIONAL CUF MVI is currently verified for Hooke CUF with "
                "LINEAR, VON_KARMAN or GREEN_LAGRANGE kinematics and geometric "
                "conditions on both cylinder ends"
            )
        source="EXPLICIT"
    else:
        formulation="VARIATIONAL" if eligible_variational else "STRONG"
        source=(
            "AUTO_VARIATIONAL_HOOKE_GEOMETRIC_ENDS"
            if eligible_variational
            else "AUTO_STRONG_OUTSIDE_VERIFIED_VARIATIONAL_SCOPE"
        )

    inner["formulation"]=formulation
    return inner,source,eligible_variational


def _make_mvi_child_case(cfg,case_path,c):
    rank=_int(_req(cfg,"rank","CASE"),"rank",1)
    if not (1<=rank<=8):
        raise FinalCaseError("MVI rank must be 1..8")

    out=_output_path(case_path,c)

    # Build and validate the exact active boundary set before generating any
    # automatic profile.  Validation, MVI and FDM therefore share one BC path.
    bc_obj,pair=_make_case_bc(cfg,case_path,c)
    _validate_mvi_homogeneous_geometric_bc(bc_obj,c["theory"],c["ns"],c["geometry_model"])
    if pair is None:
        advanced=_dict(_dict(cfg,"boundary_conditions","CASE"),"advanced","boundary_conditions")
        bc_json=_normalize_relative_npy_refs(advanced,case_path.parent)
    else:
        left,right=pair
        eng=_engineering_bc_cfg(cfg)
        if c["theory"]=="CUF":
            m=tebasis.get_basis(c["cuf_model"]).m
            bc_json={
                "left":_cuf_simple_bc_json(left,m,eng,"left"),
                "right":_cuf_simple_bc_json(right,m,eng,"right"),
            }
        else:
            bc_json={
                "left":_kl_simple_bc_json(left,eng,"left"),
                "right":_kl_simple_bc_json(right,eng,"right"),
            }

    inner_mvi_cfg=dict(_dict(cfg,"inner_mvi","CASE"))
    mvi_formulation_source="NOT_APPLICABLE"
    variational_eligible=False
    if c["theory"]=="CUF":
        inner_mvi_cfg,mvi_formulation_source,variational_eligible=(
            _select_cuf_mvi_formulation(cfg,c,bc_obj)
        )

    profile_value=cfg.get("initial_profiles_npz")
    if profile_value is None or not str(profile_value).strip():
        profile=_write_auto_mvi_profiles(out,c,rank,bc=bc_obj)
        profile_source="AUTO"
    else:
        profile=_resolve(case_path,profile_value)
        profile_source="EXTERNAL_NPZ"

    basis=tebasis.get_basis(c["cuf_model"]) if c["theory"]=="CUF" else None
    _loads,_factors,_load_meta,loads,_load_front_surface_mask=_load_series_for_case(
        cfg,case_path,c,basis=basis,generated_dir=out
    )
    if loads is None:
        raise FinalCaseError("internal error: MVI generated loads were not persisted")

    child={
        "mode":(_cuf_child_mode(c) if c["theory"]=="CUF" else _kl_child_mode(c)),
        "rank":rank,
        "geometry":{"L":c["L"],"R":c["R"],"h":c["h"]},
        "material":{"E0":c["E0"],"nu0":c["nu0"]},
        "geometry_model":c["geometry_model"],
        "capability":c["capability"].to_dict(),
        "material_model":c["material_model"],
        "grid":{"nx":c["nx"],"ns":c["ns"],"stencil":c["stencil"]},
        "quadrature":dict(c["quadrature_spec"]),
        "boundary_conditions":bc_json,
        "inner_mvi":inner_mvi_cfg,
        "mvi_formulation_source":mvi_formulation_source,
        "variational_scope_eligible":bool(variational_eligible),
        "step_predictor":cfg.get("step_predictor",{"mode":"ERROR"}),
        "load_step_adaptation":cfg.get(
            "load_step_adaptation",
            {"enabled":True,"max_subdivision_depth":8,"proactive_enabled":True,"outer_soft_limit":8,"outer_hard_limit":20}
        ),
        "initial_profiles_npz":str(profile),
        "initial_profiles_source":profile_source,
        "loads_npz":str(loads),
        "output_dir":str(out),
    }
    if c["theory"]=="CUF":
        provider={
            "TE1":"get_te1_basis","TE2":"get_te2_basis",
            "TE3":"get_te3_basis","TE4":"get_te4_basis",
        }[c["cuf_model"]]
        child["cuf_basis"]={
            "module":"final_te_basis_provider",
            "callable":provider,
            "python_path":str(Path(__file__).resolve().parent.parent),
        }
        child["cuf_model"]=c["cuf_model"]
    if c["mode"] in ("LINEAR_BIRGER","NONLINEAR_BIRGER"):
        # Validate the original curve before translating it to the verified
        # child-launcher callable format.
        _birger_settings(cfg,case_path)
        child["birger"]=_mvi_child_birger(cfg,case_path,out)
    child["final_load_definition"]=_load_meta
    return child,out


def _run_mvi(cfg,case_path,c):
    if c["material_model"]=="J2_INCREMENTAL":
        return _run_j2_mvi(cfg,case_path,c)
    child,out=_make_mvi_child_case(cfg,case_path,c)
    # The child CASE stores the load series in NPZ, not as a JSON
    # ``load_factors`` field.  Read only the scalar series length for the live
    # progress header; this is diagnostics only and does not alter the loads.
    _mvi_total_steps=0
    try:
        with np.load(child["loads_npz"],allow_pickle=False) as _lz:
            if "loads" in _lz.files:
                _mvi_total_steps=int(np.asarray(_lz["loads"]).shape[0])
            elif "load_factors" in _lz.files:
                _mvi_total_steps=int(np.asarray(_lz["load_factors"]).size)
    except Exception:
        pass
    fprog.configure(out,theory=c["theory"],cuf_model=c.get("cuf_model"),method="MVI",mode=c["mode"],rank=int(child["rank"]),total_steps=int(_mvi_total_steps))
    fprog.emit("run_start",stage="START",rank=int(child["rank"]))
    child_path=out/"MVI_CHILD_CASE.json"
    child_path.write_text(json.dumps(child,indent=2,ensure_ascii=False),encoding="utf-8")

    if c["theory"]=="CUF":
        result=cuf_mvi_launcher.run_general_case(child_path)
    else:
        result=kl_mvi_launcher.run_kl_case(child_path)

    final_step=(result.accepted_steps[-1] if result.accepted_steps else None)
    final_history=(
        (final_step.solver_history[-1]
         if final_step is not None and getattr(final_step,"solver_history",None)
         else None)
    )

    summary={
        "theory":c["theory"],
        "cuf_model":c.get("cuf_model"),
        "method":"MVI",
        "mode":c["mode"],
        "geometry_model":c["geometry_model"],
        "capability":c["capability"].to_dict(),
        "material_model":c["material_model"],
        "units":c["units"],
        "rank":int(child["rank"]),
        "mvi_formulation":str((child.get("inner_mvi") or {}).get("formulation","STRONG")).upper(),
        "mvi_formulation_source":child.get("mvi_formulation_source","EXPLICIT_OR_LEGACY"),
        "variational_scope_eligible":bool(child.get("variational_scope_eligible",False)),
        "initial_profiles":{
            "source":child.get("initial_profiles_source","EXTERNAL_NPZ"),
            "path":child.get("initial_profiles_npz"),
        },
        "step_predictor":child.get("step_predictor",{"mode":"ERROR"}),
        "converged":bool(result.converged),
        "status":str(result.status),
        "final_mvi_status":(None if final_step is None else str(final_step.solver_status)),
        "final_projected_relative":(
            None if final_history is None else final_history.get("projected_relative")
        ),
        "final_x_projected_relative":(
            None if final_history is None else final_history.get("X_projected_relative_after_cycle")
        ),
        "final_y_projected_relative":(
            None if final_history is None else final_history.get("Y_projected_relative_after_cycle")
        ),
        "final_full_residual_relative_to_load":(
            None if final_history is None else final_history.get("full_residual_relative_to_load")
        ),
        "full_equilibrium_satisfied":(
            None if final_history is None else final_history.get("full_equilibrium_satisfied")
        ),
        "requested_steps":int(result.requested_steps),
        "accepted_steps":int(len(result.accepted_steps)),
        "failed_step_index":result.failed_step_index,
        "failed_step_status":result.failed_step_status,
        "failed_step_code":getattr(result,"failed_step_code",None),
        "failed_step_diagnostic_ru":getattr(result,"failed_step_diagnostic_ru",None),
        "failed_step_recommendation_ru":getattr(result,"failed_step_recommendation_ru",None),
        "child_case":str(child_path),
        "geometry":{"L":c["L"],"R":c["R"],"h":c["h"]},
        "material":{"E0":c["E0"],"nu0":c["nu0"]},
        "grid":{"nx":c["nx"],"ns":c["ns"],"stencil":c["stencil"]},
        "quadrature":dict(c["quadrature_spec"]),
        "validation_warnings":list(c["quadrature_warnings"])+list(c["unit_validation"].get("warnings",[]))+list(c["boundary_layer_warnings"])+list(c["capability_warnings"]),
        "boundary_conditions":cfg.get("boundary_conditions"),
        "load_definition":child.get("final_load_definition"),
        "postprocess":cfg.get("postprocess",{}),
        "step_predictor":cfg.get("step_predictor",{"mode":"ERROR"}),
        "load_step_adaptation":child.get(
            "load_step_adaptation",{"enabled":True,"max_subdivision_depth":8,"proactive_enabled":True,"outer_soft_limit":8,"outer_hard_limit":20}
        ),
        "birger_definition":cfg.get("birger") if c["mode"] in ("LINEAR_BIRGER","NONLINEAR_BIRGER") else None,
    }
    first_unloading=next((
        _unloading_fraction(getattr(step,"unloading_mask",None))
        for step in result.accepted_steps
        if _unloading_fraction(getattr(step,"unloading_mask",None)) is not None
    ),None)
    _append_unloading_warning(summary,first_unloading)
    _atomic_write_json(out/"FINAL_RUN_SUMMARY.json",summary)
    _atomic_write_json(out/"RECOVERY_CHECKPOINT.json",{**summary,"format":"CYLINDER_RECOVERY_CHECKPOINT_V1","checkpoint":True,"last_saved_step":int(summary.get("accepted_steps",0))})
    fprog.emit("run_done",stage="DONE",converged=bool(summary["converged"]),status=summary["status"],completed_steps=int(summary["accepted_steps"]))
    return summary


def validate_final_case(case_json,*,dry_run=True):
    cfg,path=_read_json(case_json)
    c=_common(cfg)
    if c["method"]=="MVI":
        # Validation must not populate the requested run folder.  MVI child
        # preparation generates profiles, load NPZ and possibly a curve module,
        # so perform it in an isolated temporary output directory.
        if dry_run:
            with tempfile.TemporaryDirectory(prefix="cyl_validate_") as tmp:
                c_dry={**c,"output_dir":Path(tmp)}
                _make_mvi_child_case(cfg,path,c_dry)
        else:
            _make_mvi_child_case(cfg,path,c)
    else:
        # Validate files/settings without executing.
        m=tebasis.get_basis(c["cuf_model"]).m if c["theory"]=="CUF" else None
        _make_case_bc(cfg,path,c)
        basis=tebasis.get_basis(c["cuf_model"]) if c["theory"]=="CUF" else None
        _load_series_for_case(cfg,path,c,basis=basis,generated_dir=None)
        if c["mode"]=="LINEAR_ELASTIC":
            _linear_residual_tolerance(cfg)
        if c["mode"] in ("NONLINEAR_ELASTIC","NONLINEAR_BIRGER","LINEAR_J2","NONLINEAR_J2"):
            _newton_settings(cfg,c["theory"])
            _initial_state(cfg,path,c["theory"],c["ns"],c["nx"],m)
        if c["mode"] in ("LINEAR_BIRGER","NONLINEAR_BIRGER"):
            _birger_settings(cfg,path)

    return {
        "valid":True,
        "theory":c["theory"],
        "cuf_model":c.get("cuf_model"),
        "method":c["method"],
        "mode":c["mode"],
        "geometry_model":c["geometry_model"],
        "material_model":c["material_model"],
        "capability":c["capability"].to_dict(),
        "units":c["units"],
        "unit_validation":c["unit_validation"],
        "quadrature":dict(c["quadrature_spec"]),
        "validation_warnings":list(c["unit_validation"].get("warnings",[]))+list(c["quadrature_warnings"])+list(c["boundary_layer_warnings"])+list(c["capability_warnings"]),
        "fdm_grid":dict(c["fdm_grid"]),
        "mvi_grid":dict(c["mvi_grid"]),
        "fdm_operator_scheme":c["fdm_operator_scheme"],
        "active_grid":c["active_grid"],
        "active_grid_values":{"nx":c["nx"],"ns":c["ns"],"stencil":c["stencil"]},
    }


def run_final_case(case_json):
    cfg,path=_read_json(case_json)
    c=_common(cfg)
    return _run_mvi(cfg,path,c) if c["method"]=="MVI" else _run_fdm(cfg,path,c)


def run_mvi_rank_series(case_json,ranks=range(1,9)):
    """
    Run an MVI rank series from an expert-supplied profile bank or from the
    same deterministic automatic profile policy used by a single MVI run.
    """
    cfg,path=_read_json(case_json)
    c=_common(cfg)
    if c["method"]!="MVI":
        raise FinalCaseError("run_mvi_rank_series requires method='MVI'")

    ranks=[_int(r,"rank",1) for r in ranks]
    if not ranks or any(r>8 for r in ranks):
        raise FinalCaseError("MVI rank series supports ranks 1..8")

    maxrank=max(ranks)
    base_out=_output_path(path,c)
    series_dir=base_out/"MVI_1_8_SERIES"
    if series_dir.exists() and any(series_dir.iterdir()):
        raise FinalCaseError(
            f"OUTPUT_DIR_NOT_EMPTY: {series_dir} already contains an MVI rank series. "
            "Choose a new output_dir; rank-series results are never mixed."
        )
    series_dir.mkdir(parents=True,exist_ok=True)

    # A rank comparison is meaningful only when every MVI-N receives the
    # identical discrete load history. Build/resolve it once and make every
    # child case read this one immutable NPZ instead of regenerating a copy.
    common_basis=(
        tebasis.get_basis(c["cuf_model"])
        if c["theory"]=="CUF" else None
    )
    common_load_dir=series_dir/"COMMON_LOAD"
    common_loads,common_factors,common_load_meta,common_load_path,_front=(
        _load_series_for_case(
            cfg,path,c,basis=common_basis,generated_dir=common_load_dir
        )
    )
    if common_load_path is None:
        raise FinalCaseError("rank series could not resolve a common load NPZ")
    common_loads=np.ascontiguousarray(common_loads,dtype=float)
    common_load_digest=hashlib.sha256()
    common_load_digest.update(str(common_loads.shape).encode("ascii"))
    common_load_digest.update(common_loads.tobytes(order="C"))
    if common_factors is not None:
        common_factors=np.ascontiguousarray(common_factors,dtype=float)
        common_load_digest.update(str(common_factors.shape).encode("ascii"))
        common_load_digest.update(common_factors.tobytes(order="C"))
    common_load_sha256=common_load_digest.hexdigest()
    # Use the same active boundary set as a single MVI run when automatic
    # profiles are generated.  KL automatic profiles are BC-aware.
    bc_obj,_pair=_make_case_bc(cfg,path,c)
    profile_value=cfg.get("initial_profiles_npz")
    if profile_value is None or not str(profile_value).strip():
        src=_write_auto_mvi_profiles(
            base_out,c,maxrank,bc=bc_obj,filename="AUTO_INITIAL_PROFILE_BANK_MVI_1_8.npz"
        )
        X,Y=_build_auto_mvi_profiles(c,maxrank,bc=bc_obj)
        profile_bank_source="AUTO"
    else:
        src=_resolve(path,profile_value)
        profile_bank_source="EXTERNAL_NPZ"
        try:
            with np.load(src,allow_pickle=False) as z:
                if "X" not in z.files or "Y" not in z.files:
                    raise FinalCaseError("profile bank NPZ must contain X and Y")
                X=np.asarray(z["X"],float)
                Y=np.asarray(z["Y"],float)
        except FinalCaseError:
            raise
        except Exception as exc:
            raise FinalCaseError(f"cannot read profile bank {src}: {exc}") from exc

    if c["theory"]=="KL":
        if X.ndim!=3 or Y.ndim!=3 or X.shape[0]!=3 or Y.shape[0]!=3 or X.shape[2]!=c["nx"] or Y.shape[2]!=c["ns"]:
            raise FinalCaseError("KL profile bank must have X(3,rank,nx), Y(3,rank,ns)")
        available=min(X.shape[1],Y.shape[1])
    else:
        m=tebasis.get_basis(c["cuf_model"]).m
        if X.ndim!=4 or Y.ndim!=4 or X.shape[0]!=3 or Y.shape[0]!=3 or X.shape[1]!=m or Y.shape[1]!=m or X.shape[3]!=c["nx"] or Y.shape[3]!=c["ns"]:
            raise FinalCaseError(f"CUF-{c['cuf_model']} profile bank must have X(3,{m},rank,nx), Y(3,{m},rank,ns)")
        available=min(X.shape[2],Y.shape[2])
    if available<maxrank:
        raise FinalCaseError(f"profile bank contains {available} directions, need {maxrank}")

    rows=[]

    for rank in ranks:
        rank_dir=series_dir/f"MVI_{rank}"
        rank_dir.mkdir(parents=True,exist_ok=True)
        if c["theory"]=="KL":
            Xr=X[:,:rank,:].copy(); Yr=Y[:,:rank,:].copy()
        else:
            Xr=X[:,:,:rank,:].copy(); Yr=Y[:,:,:rank,:].copy()
        prof=rank_dir/"INITIAL_PROFILES.npz"
        np.savez_compressed(prof,X=Xr,Y=Yr)

        cc=json.loads(json.dumps(cfg))
        cc["rank"]=rank
        cc["initial_profiles_npz"]=str(prof)
        cc["load_definition"]=None
        cc["loads_npz"]=str(Path(common_load_path).resolve())
        cc["output_dir"]=str((rank_dir/"OUT").resolve())
        bcj=cc.get("boundary_conditions")
        if isinstance(bcj,dict) and bcj.get("advanced") is not None:
            bcj["advanced"]=_normalize_relative_npy_refs(bcj["advanced"],path.parent)
        if c["mode"] in ("LINEAR_BIRGER","NONLINEAR_BIRGER"):
            bj=cc.get("birger")
            if isinstance(bj,dict):
                curve=bj.get("curve")
                if isinstance(curve,dict) and str(curve.get("mode","")).upper()=="CSV" and curve.get("path"):
                    curve["path"]=str(_resolve(path,curve["path"]))
                spec=(curve.get("material_diagram") if isinstance(curve,dict) and str(curve.get("mode","")).upper()=="EXTERNAL_PYTHON" else bj.get("material_diagram"))
                if isinstance(spec,dict) and spec.get("python_path"):
                    spec["python_path"]=str(_resolve(path,spec["python_path"]))
        cp=rank_dir/"FINAL_CASE.json"
        cp.write_text(json.dumps(cc,indent=2,ensure_ascii=False),encoding="utf-8")

        try:
            rr=run_final_case(cp)
            row={
                "rank":rank,
                "converged":bool(rr.get("converged",False)),
                "status":str(rr.get("status","UNKNOWN")),
                "accepted_steps":int(rr.get("accepted_steps",0)),
                "requested_steps":int(rr.get("requested_steps",0)),
                "failed_step_index":rr.get("failed_step_index"),
                "failed_step_status":rr.get("failed_step_status"),
                "common_load_sha256":common_load_sha256,
                "case":str(cp),
            }
        except Exception as exc:
            row={
                "rank":rank,"converged":False,
                "status":f"ERROR_{type(exc).__name__}: {exc}",
                "accepted_steps":0,"requested_steps":0,
                "failed_step_index":None,
                "failed_step_status":str(exc),
                "common_load_sha256":common_load_sha256,
                "case":str(cp),
            }
        rows.append(row)
        (series_dir/f"MVI_{rank}_SUMMARY.json").write_text(
            json.dumps(row,indent=2,ensure_ascii=False),encoding="utf-8")
        # Continue after a failed rank: a failure of MVI-N must not suppress
        # diagnostics for MVI-(N+1)...MVI-8. There is still no retry, load-step
        # subdivision or method change inside an individual rank run.

    summary={
        "theory":c["theory"],
        "cuf_model":c.get("cuf_model"),
        "mode":c["mode"],
        "geometry_model":c["geometry_model"],
        "capability":c["capability"].to_dict(),
        "material_model":c["material_model"],
        "method":"MVI_RANK_SERIES",
        "requested_ranks":ranks,
        "completed_ranks":[r["rank"] for r in rows],
        "all_converged":len(rows)==len(ranks) and all(r["converged"] for r in rows),
        "rows":rows,
        "common_load_npz":str(Path(common_load_path).resolve()),
        "common_load_sha256":common_load_sha256,
        "common_load_factors":(
            None if common_factors is None else common_factors.tolist()
        ),
        "common_load_definition":common_load_meta,
        "all_ranks_use_identical_load_history":all(
            r.get("common_load_sha256")==common_load_sha256 for r in rows
        ),
        "profile_bank_source":str(src),
        "profile_bank_mode":profile_bank_source,
        "seed_policy":(
            "automatic deterministic profiles saved inside RUN/INPUT"
            if profile_bank_source=="AUTO" else
            "expert-supplied profile directions"
        ),
    }
    (series_dir/"MVI_1_8_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary