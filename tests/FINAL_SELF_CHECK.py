#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Meaningful regression self-check for the final cylinder solver.

The old Stage-12 self-check used zero load, so the linear systems were never
actually solved.  This check deliberately exercises non-zero CUF-TE1 and KL
FDM calculations and also protects the K1 (algebraic equilibration) and K8
(MVI write-through) fixes.

This is a software regression check, not a literature benchmark validation.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import time
import numpy as np

import final_numeric_scaling as nscale
import final_residual_scaling as rscale
import final_k2_coloring as k2color
import final_birger_curve as birgercurve
import V3_SELF_CHECK as v3check
import V3B_SELF_CHECK as v3bcheck
import V3C_SELF_CHECK as v3ccheck
import V1_V11_SELF_CHECK as v1v11check
import V2_SELF_CHECK as v2check
import V11_2_SELF_CHECK as v112check
import V5_SELF_CHECK as v5check
import V6_SELF_CHECK as v6check
import V7_SELF_CHECK as v7check
import V8_SELF_CHECK as v8check
import V9_SELF_CHECK as v9check
import Z1_SELF_CHECK as z1check
import Z2_SELF_CHECK as z2check
import Z4_SELF_CHECK as z4check
import HARMONIC_LOAD_SELF_CHECK as harmoniccheck
import MVI_LIMIT_SELF_CHECK as mvilimitcheck
import cylinder_te1_mvi_core as te1_mvi
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage2b as s2
import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage7a_general_mechanics as cuf_mech
import cylinder_cuf_stage7f_general_linear as cuf_linear
import cylinder_kl_stage8a_elastic_mechanics as kl_mech
import cylinder_kl_stage8b1_boundary as kl_bc
import cylinder_te1_local_kernel as te1_lk
import cylinder_cuf_stage7b_general_mvi_profiles as cuf_mvi
import cylinder_cuf_stage7c_general_boundary as cuf_bc
import cylinder_cuf_stage7d_general_internal_mvi as cuf_internal
import final_te_basis_provider as tebasis
import final_load_builder as loadbuilder
import cylinder_kl_stage8b2_profiles as kl_mvi
import cylinder_cuf_stage10a_strong_fdm_linear_elastic as c10a
import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as c10b
import cylinder_kl_stage9b_strong_fdm_linear as k9b
import cylinder_kl_stage9c_strong_fdm_vk_elastic as k9c
import cylinder_kl_stage9d2_strong_fdm_vk_birger as k9d2
from cylinder_final_stage11a_dispatcher import (
    run_final_case, validate_final_case, _make_kl_simple_bc,
    _make_cuf_simple_bc, _build_auto_mvi_profiles,
)


BASE = Path(__file__).resolve().parent.parent
WORK = BASE / "SELF_CHECK_WORK"

# Reference values fixed from the verified Stage-12 + K1 + K8 build.
# Reviewer V-7 replaced composed KL D1(D1(.)) by the genuine 7-point D2
# stencil.  The KL references below were re-recorded after that discretisation
# change; the linear value changes only at round-off level, while the nonlinear
# value changes by ~3.8e-9 relative.  These remain software-regression anchors.
# NOTE: the CUF_TE1 nx=7 reference carries about 2.1% grid error in wmax
# relative to the mesh-converged value wmax -> 3.4422e-4 at nx>=101.
# These are software-regression anchors, NOT physical answers.
# Tolerances are intentionally much looser than machine precision so the test
# is robust to BLAS/platform round-off while still catching real regressions.
REFERENCE = {
    ("CUF_TE1", "LINEAR_ELASTIC"): 3.514245124035481e-4,
    ("KL",      "LINEAR_ELASTIC"): 3.3654622007506653e-4,
    ("CUF_TE1", "NONLINEAR_ELASTIC"): 3.514244933658315e-6,
    ("KL",      "NONLINEAR_ELASTIC"): 3.3654622007506663e-6,
}


def _assert_close(name, got, expected, *, rtol=2e-6, atol=1e-14):
    got = float(got)
    expected = float(expected)
    if not np.isfinite(got) or not np.isclose(got, expected, rtol=rtol, atol=atol):
        rel = abs(got - expected) / max(abs(expected), atol)
        raise AssertionError(
            f"{name}: got {got:.17g}, expected {expected:.17g}, rel={rel:.3e}"
        )


def _case_config(theory: str, mode: str):
    nonlinear = mode == "NONLINEAR_ELASTIC"
    # The nonlinear smoke case uses a 100x smaller load so it is safely in the
    # small-deformation range while still forcing a real Newton solve.
    q = 1.0e-4 if nonlinear else 1.0e-2
    cfg = {
        "theory": theory,
        "method": "FDM",
        "mode": mode,
        "geometry": {"L": 100.0, "R": 50.0, "h": 1.0},
        "material": {"E0": 70000.0, "nu0": 0.3},
        "units": {"system": "N_MM_MPA"},
        "fdm_grid": {"nx": 7, "ns": 7, "stencil": 7},
        "mvi_grid": {"nx": 7, "ns": 7, "stencil": 7},
        "quadrature": {"type": "GAUSS", "n_gauss": 3},
        "boundary_conditions": {
            "left": "CLAMPED", "right": "CLAMPED", "advanced": None
        },
        "load_definition": {
            "type": "UNIFORM_RADIAL_FACE",
            "q": q,
            "load_factors": [1.0],
        },
        "output_dir": "OUT",
    }
    if str(theory).upper().startswith("CUF"):
        cfg["load_definition"]["face"] = "OUTER"
    if mode == "LINEAR_ELASTIC":
        cfg["linear"] = {"residual_tolerance_relative": 1.0e-10}
    if nonlinear:
        cfg["initial_state"] = {"source": "ZERO", "npz": None}
        cfg["newton"] = {
            "residual_tolerance_relative": 1.0e-9,
            "increment_tolerance": 1.0e-12,
            "fd_step": 1.0e-8,
            "max_iterations": 4,
        }
    return cfg


def _run_nonzero_fdm(theory: str, mode: str):
    d = WORK / f"{theory}_{mode}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "case.json"
    p.write_text(json.dumps(_case_config(theory, mode), indent=2), encoding="utf-8")

    check = validate_final_case(p)
    if not check.get("valid", False):
        raise AssertionError(f"{theory}/{mode}: validation did not return valid=True")

    result = run_final_case(p)
    if not result.get("converged", False):
        raise AssertionError(f"{theory}/{mode}: solver did not converge: {result.get('status')}")
    if result.get("accepted_steps") != 1:
        raise AssertionError(f"{theory}/{mode}: expected exactly one completed load step")

    step = result["steps"][0]
    residual = float(step.get("residual_inf", np.inf))
    if not np.isfinite(residual):
        raise AssertionError(f"{theory}/{mode}: residual is not finite")
    residual_limit = 1.0e-10 if mode == "LINEAR_ELASTIC" else 1.0e-8
    if residual > residual_limit:
        raise AssertionError(
            f"{theory}/{mode}: residual {residual:.3e} > {residual_limit:.3e}"
        )

    with np.load(d / "OUT" / "STEP_0001_STATE.npz", allow_pickle=False) as z:
        key = "q" if theory == "CUF_TE1" else "U"
        state = np.asarray(z[key], float)
    if not np.all(np.isfinite(state)):
        raise AssertionError(f"{theory}/{mode}: non-finite state")

    w = state[2, 0] if state.ndim == 4 else state[2]
    wmax = float(np.max(np.abs(w)))
    if wmax <= 0.0:
        raise AssertionError(f"{theory}/{mode}: non-zero load produced zero displacement")
    _assert_close(f"{theory}/{mode} wmax", wmax, REFERENCE[(theory, mode)])
    return wmax, residual, step.get("iterations")



def _test_reviewer_v9_boundary_layer_warning():
    cfg=_case_config("CUF_TE1","LINEAR_ELASTIC")
    d=WORK/"REVIEWER_V9"
    d.mkdir(parents=True,exist_ok=True)
    p=d/"case.json"
    p.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    check=validate_final_case(p)
    warnings=list(check.get("validation_warnings",[]))
    hits=[w for w in warnings if str(w).startswith("BOUNDARY_LAYER_UNRESOLVED:")]
    if len(hits)!=1:
        raise AssertionError(f"reviewer V-9: expected one boundary-layer warning, got {warnings}")
    msg=hits[0]
    if "1/beta=" not in msg or "Рекомендуемое nx >=" not in msg:
        raise AssertionError(f"reviewer V-9: incomplete warning: {msg}")
    # V-9 is a warning only: a sufficiently refined grid simply removes it.
    cfg["fdm_grid"]["nx"]=101
    cfg["mvi_grid"]["nx"]=101
    p.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    check2=validate_final_case(p)
    hits2=[w for w in check2.get("validation_warnings",[]) if str(w).startswith("BOUNDARY_LAYER_UNRESOLVED:")]
    if hits2:
        raise AssertionError(f"reviewer V-9: warning persisted on refined grid: {hits2}")


def _test_reviewer_v8_kl_face_rejection():
    """Reviewer V-8: KL must not silently accept OUTER/INNER face semantics."""
    base={
        "type":"UNIFORM_RADIAL_FACE",
        "q":0.01,
        "load_factors":[1.0],
    }
    # KL midsurface load remains valid when face is absent.
    loads,_,_,_=loadbuilder.build_generated_loads(
        base,theory="KL",L=100.0,R=50.0,h=1.0,nx=7,ns=7,stencil=7
    )
    if loads.shape!=(1,3,7,7):
        raise AssertionError(f"Reviewer V-8: unexpected KL load shape {loads.shape}")
    for face in ("OUTER","INNER"):
        bad=dict(base,face=face)
        try:
            loadbuilder.build_generated_loads(
                bad,theory="KL",L=100.0,R=50.0,h=1.0,nx=7,ns=7,stencil=7
            )
        except loadbuilder.LoadDefinitionError as exc:
            msg=str(exc)
            if "KL" not in msg or "OUTER/INNER" not in msg or "Уберите поле face" not in msg:
                raise AssertionError(f"Reviewer V-8: non-actionable KL face error: {msg}")
        else:
            raise AssertionError(f"Reviewer V-8: KL silently accepted face={face}")

    # CUF keeps physical face projection.
    cuf=dict(base,face="OUTER")
    loadbuilder.build_generated_loads(
        cuf,theory="CUF",L=100.0,R=50.0,h=1.0,nx=7,ns=7,stencil=7,
        basis=tebasis.get_basis("TE1")
    )

    # GUI contract: KL omits face from saved/generated load definition and the
    # face widgets are actually removed from the grid, not merely left active.
    gui=(BASE/"interface"/"FINAL_GUI.py").read_text(encoding="utf-8")
    required=(
        'if self.vars["theory"].get()=="CUF":spec["face"]=self.vars["load_face"].get()',
        'self.vars["load_face"].set("")',
        'self.load_face_label.grid_remove()',
        'self.load_face_combo.grid_remove()',
    )
    missing=[x for x in required if x not in gui]
    if missing:
        raise AssertionError(f"Reviewer V-8: GUI face contract missing: {missing}")


def _test_radial_sign_consistency():
    """K-1: one positive radial load must give the same hoop-resultant sign."""
    signs={}
    values={}
    for theory in ("CUF_TE1","KL"):
        d=WORK/f"K1_RADIAL_SIGN_{theory}"
        d.mkdir(parents=True,exist_ok=True)
        cfg=_case_config(theory,"LINEAR_ELASTIC")
        cfg["fdm_grid"]={"nx":21,"ns":7,"stencil":7}
        p=d/"case.json"
        p.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        result=run_final_case(p)
        if not result.get("converged",False):
            raise AssertionError(
                f"K1 {theory}: sign-control calculation did not converge: "
                f"{result.get('status')}"
            )

        L=100.0; R=50.0; h=1.0; E0=70000.0; nu0=0.3
        grid=te1_mvi.CylinderGrid(L,R,21,7)
        ops=c3.CylinderSpatialOperators.build(grid,7)
        quad=s2.gauss_thickness_quadrature(h,3)
        state_path=d/"OUT"/"STEP_0001_STATE.npz"

        if theory=="CUF_TE1":
            with np.load(state_path,allow_pickle=False) as z:
                q=np.asarray(z["q"],float)
            basis=tebasis.get_basis("TE1")
            qx,qs=c10a.surface_first_derivatives(q,ops,basis)
            ec=te1_lk.ElasticConstants(E0,nu0)
            rr=cuf_linear.general_linear_thickness_resultants(
                q,qx,qs,R=R,K0=ec.K0,GB=ec.G0,
                quadrature=quad,basis=basis,
            )
            hoop=float(rr.Av_s[0,0,grid.nx//2])
        else:
            with np.load(state_path,allow_pickle=False) as z:
                U=np.asarray(z["U"],float)
            kin=kl_mech.field_kinematics(U,ops,geometry="LINEAR")
            rr=kl_mech.quadrature_elastic_resultants(
                kin.state,kl_mech.KLElasticConstants(E0,nu0,h),quad
            )
            hoop=float(rr.Ns[0,grid.nx//2])

        values[theory]=hoop
        signs[theory]=int(np.sign(hoop))

    if signs["CUF_TE1"]==0 or signs["KL"]==0 or signs["CUF_TE1"]!=signs["KL"]:
        raise AssertionError(
            "RADIAL_SIGN_CONVENTION_MISMATCH: q>0 gives N_s "
            f"{values['CUF_TE1']:+.6e} in CUF and {values['KL']:+.6e} in KL"
        )


def _test_k3_linear_operator_cache(theory: str):
    """A three-step LINEAR series must assemble its operator exactly once."""
    d = WORK / f"K3_{theory}"
    d.mkdir(parents=True, exist_ok=True)
    cfg = _case_config(theory, "LINEAR_ELASTIC")
    cfg["load_definition"]["load_factors"] = [0.25, 0.5, 1.0]
    p = d / "case.json"
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if theory == "CUF_TE1":
        module = c10a
        attr = "assemble_linear_operator_and_rhs"
        state_key = "q"
    else:
        module = k9b
        attr = "assemble_linear_strong_system"
        state_key = "U"

    original = getattr(module, attr)
    calls = {"n": 0}
    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)
    setattr(module, attr, counted)
    try:
        result = run_final_case(p)
    finally:
        setattr(module, attr, original)

    if not result.get("converged", False) or result.get("accepted_steps") != 3:
        raise AssertionError(f"K3 {theory}: three-step linear series did not converge")
    if calls["n"] != 1:
        raise AssertionError(
            f"K3 {theory}: operator assembled {calls['n']} times for 3 load steps; expected 1"
        )

    wmax = []
    for k in range(1, 4):
        with np.load(d / "OUT" / f"STEP_{k:04d}_STATE.npz", allow_pickle=False) as z:
            state = np.asarray(z[state_key], float)
        w = state[2, 0] if state.ndim == 4 else state[2]
        wmax.append(float(np.max(np.abs(w))))
    expected = REFERENCE[(theory, "LINEAR_ELASTIC")]
    _assert_close(f"K3 {theory} final wmax", wmax[-1], expected)
    if not np.allclose(
        np.asarray(wmax), expected * np.asarray([0.25, 0.5, 1.0]),
        rtol=2e-6, atol=1e-14
    ):
        raise AssertionError(f"K3 {theory}: linear load scaling failed: {wmax}")


def _test_v4_converged_state_skips_jacobian(theory: str):
    """An already-converged Newton state must require zero Jacobian builds."""
    source_dir = WORK / f"{theory}_NONLINEAR_ELASTIC"
    source_state = source_dir / "OUT" / "STEP_0001_STATE.npz"
    if not source_state.exists():
        raise AssertionError(f"V4 {theory}: converged source state is missing")

    d = WORK / f"V4_{theory}"
    d.mkdir(parents=True, exist_ok=True)
    cfg = _case_config(theory, "NONLINEAR_ELASTIC")
    cfg["initial_state"] = {"source": "NPZ", "npz": str(source_state.resolve())}
    p = d / "case.json"
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    module = c10b if theory == "CUF_TE1" else k9c
    original = module.analytic_jacobian
    calls = {"n": 0}
    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)
    module.analytic_jacobian = counted
    try:
        result = run_final_case(p)
    finally:
        module.analytic_jacobian = original

    if not result.get("converged", False) or result.get("accepted_steps") != 1:
        raise AssertionError(f"V4 {theory}: converged-state rerun did not converge")
    step = result["steps"][0]
    if int(step.get("iterations", -1)) != 0:
        raise AssertionError(
            f"V4 {theory}: converged initial state should take 0 Newton updates; "
            f"got {step.get('iterations')}"
        )
    if calls["n"] != 0:
        raise AssertionError(
            f"V4 {theory}: Jacobian assembled {calls['n']} times for an already "
            "converged Newton state; expected 0"
        )

    state_key = "q" if theory == "CUF_TE1" else "U"
    with np.load(d / "OUT" / "STEP_0001_STATE.npz", allow_pickle=False) as z:
        state = np.asarray(z[state_key], float)
    w = state[2, 0] if state.ndim == 4 else state[2]
    _assert_close(
        f"V4 {theory} converged-state wmax",
        float(np.max(np.abs(w))),
        REFERENCE[(theory, "NONLINEAR_ELASTIC")],
    )




def _test_v4_kl_nonlinear_birger_jacobian_skip():
    """V4-BIRGER: KL must not assemble J after residual has converged.

    With an exactly elastic Birger intensity curve, NONLINEAR_BIRGER must
    reproduce the KL nonlinear-elastic smoke solution.  Starting from ZERO,
    exactly one Newton Jacobian is required for this case.  Before V4-BIRGER
    the same calculation assembled three Jacobians: one useful build plus two
    builds performed only to discover that the current residual had already
    converged.
    """
    d = WORK / "V4_KL_NONLINEAR_BIRGER"
    d.mkdir(parents=True, exist_ok=True)

    E0 = 70000.0
    nu0 = 0.3
    G0 = E0 / (2.0 * (1.0 + nu0))
    cfg = _case_config("KL", "NONLINEAR_ELASTIC")
    cfg["mode"] = "NONLINEAR_BIRGER"
    cfg["initial_GB"] = {"source": "ELASTIC_G0", "npz": None}
    cfg["birger"] = {
        "epsilon_tol": 1.0e-12,
        "omega": 1.0,
        "G_ref": G0,
        "sigma_ref": 1.0,
        "tol_G": 1.0e-10,
        "tol_sigma": 1.0e-8,
        "tol_u": 1.0e-8,
        "max_outer_iterations": 2,
        "epsilon_i_limit": 1.0,
        "delta_J_limit": 1.0,
        "initial_slope_rtol": 0.05,
        "curve": {
            "mode": "FORMULA",
            "formula": "slope*eps",
            "parameters": {"slope": 3.0 * G0},
        },
    }
    cfg["output_dir"] = "OUT"
    p = d / "case.json"
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    check = validate_final_case(p)
    if not check.get("valid", False):
        raise AssertionError("V4 KL NONLINEAR_BIRGER: validation failed")

    original = k9d2.analytic_jacobian_frozen_gb
    calls = {"n": 0}
    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)
    k9d2.analytic_jacobian_frozen_gb = counted
    try:
        result = run_final_case(p)
    finally:
        k9d2.analytic_jacobian_frozen_gb = original

    if not result.get("converged", False) or result.get("accepted_steps") != 1:
        raise AssertionError("V4 KL NONLINEAR_BIRGER: calculation did not converge")
    if calls["n"] != 1:
        raise AssertionError(
            "V4 KL NONLINEAR_BIRGER: expected exactly 1 useful Jacobian build; "
            f"got {calls['n']}"
        )
    step = result["steps"][0]
    if int(step.get("outer_iterations", -1)) != 1:
        raise AssertionError(
            "V4 KL NONLINEAR_BIRGER: elastic intensity curve should converge "
            f"in one Birger outer iteration; got {step.get('outer_iterations')}"
        )

    with np.load(d / "OUT" / "STEP_0001_STATE.npz", allow_pickle=False) as z:
        U = np.asarray(z["U"], float)
        GB = np.asarray(z["GB"], float)
    _assert_close(
        "V4 KL NONLINEAR_BIRGER wmax",
        float(np.max(np.abs(U[2]))),
        REFERENCE[("KL", "NONLINEAR_ELASTIC")],
    )
    if not np.allclose(GB, G0, rtol=2e-12, atol=1e-12):
        raise AssertionError("V4 KL NONLINEAR_BIRGER: elastic Phi changed GB from G0")


def _test_k5_k6_birger_curve_guard():
    """Claude-audit guards: intensity slope and safe behaviour near zero."""
    E0=70000.0
    nu0=0.3
    G0=E0/(2.0*(1.0+nu0))
    expected=3.0*G0

    def good(eps):
        e=np.asarray(eps,float)
        return expected*e

    diag=b5.validate_material_diagram_near_origin(
        good,G0=G0,epsilon_tol=1.0e-10,initial_slope_rtol=0.05
    )
    if diag["initial_slope_relative_error"] > 1.0e-12:
        raise AssertionError("K5 good intensity curve did not reproduce slope 3G0")

    # K5: ordinary uniaxial sigma=E0*epsilon must NOT silently pass as
    # sigma_i=Phi(epsilon_i).  For nu=0.3 the mismatch is 13.333...%.
    def ordinary_uniaxial(eps):
        return E0*np.asarray(eps,float)
    try:
        b5.validate_material_diagram_near_origin(
            ordinary_uniaxial,G0=G0,epsilon_tol=1.0e-10,initial_slope_rtol=0.05
        )
    except b5.MaterialDiagramError as exc:
        if "K5" not in str(exc):
            raise AssertionError(f"K5 rejected for wrong reason: {exc}")
    else:
        raise AssertionError("K5 failed: ordinary uniaxial sigma-epsilon curve was accepted")

    # K6a: a non-zero intercept is the np.interp/HOLD failure mode from the
    # audit and would make Phi/(3 epsilon_i) blow up near zero.
    def nonzero_origin(eps):
        return 5.0 + expected*np.asarray(eps,float)
    try:
        b5.validate_material_diagram_near_origin(
            nonzero_origin,G0=G0,epsilon_tol=1.0e-10,initial_slope_rtol=0.05
        )
    except b5.MaterialDiagramError as exc:
        if "K6" not in str(exc):
            raise AssertionError(f"K6 origin rejected for wrong reason: {exc}")
    else:
        raise AssertionError("K6 failed: non-zero Phi(0) was accepted")

    # K6b: even if a user callable special-cases Phi(0) and the 1e-8 slope,
    # an excessive secant jump immediately above epsilon_tol must be caught.
    eps_tol=1.0e-6
    def hidden_jump(eps):
        e=np.asarray(eps,float)
        out=expected*e
        out=np.asarray(out,float)
        out=np.where(e > 5.0e-8, out + 100.0, out)
        return out
    try:
        b5.validate_material_diagram_near_origin(
            hidden_jump,G0=G0,epsilon_tol=eps_tol,initial_slope_rtol=0.05
        )
    except b5.MaterialDiagramError as exc:
        if "K6" not in str(exc):
            raise AssertionError(f"K6 onset rejected for wrong reason: {exc}")
    else:
        raise AssertionError("K6 failed: excessive GB jump above epsilon_tol was accepted")


def _test_v10_birger_monotonic_guard():
    """V10: hardening/plateau allowed; descending Phi branch rejected."""
    E0=70000.0; nu0=0.3; G0=E0/(2.0*(1.0+nu0)); slope=3.0*G0

    # TABLE/CSV path is protected at the exact supplied points.
    good=birgercurve.build_curve({
        "mode":"TABLE","extrapolation":"HOLD",
        "points":[[0.0,0.0],[1e-3,slope*1e-3],[2e-3,slope*1e-3],[3e-3,slope*1.2e-3]],
    })
    b5.validate_material_diagram_monotonic(good,epsilon_i_limit=3e-3)

    try:
        birgercurve.build_curve({
            "mode":"TABLE","extrapolation":"HOLD",
            "points":[[0.0,0.0],[1e-3,slope*1e-3],[2e-3,0.9*slope*1e-3]],
        })
    except birgercurve.BirgerCurveError as exc:
        if "V10" not in str(exc):
            raise AssertionError(f"V10 TABLE rejected for wrong reason: {exc}")
    else:
        raise AssertionError("V10 failed: descending TABLE branch was accepted")

    # Formula/external paths are checked numerically on the declared range.
    def descending(eps):
        e=np.asarray(eps,float)
        return slope*e*(1.0-0.75*e)
    try:
        b5.validate_material_diagram_monotonic(descending,epsilon_i_limit=1.0)
    except b5.MaterialDiagramError as exc:
        if "V10" not in str(exc):
            raise AssertionError(f"V10 callable rejected for wrong reason: {exc}")
    else:
        raise AssertionError("V10 failed: descending callable branch was accepted")

    def plateau(eps):
        e=np.asarray(eps,float)
        return np.minimum(slope*e,slope*0.2)
    b5.validate_material_diagram_monotonic(plateau,epsilon_i_limit=1.0)


def _make_k2_test_problem(*, nx, ns, te1=False):
    grid=te1_mvi.CylinderGrid(100.0,50.0,int(nx),int(ns))
    ops=c3.CylinderSpatialOperators.build(grid,7)
    quad=s2.gauss_thickness_quadrature(1.0,3)
    if te1:
        basis=tebasis.get_basis("TE1")
    else:
        basis=cuf_mech.CallableThicknessBasis(
            (lambda z:1.0,),(lambda z:0.0,),("F0=1",)
        )
    bc=cuf_bc.GeneralCylinderEndBC.clamped_clamped(basis.m)
    return ops,quad,basis,bc


def _assert_matrix_identical(name,A,B):
    A=np.asarray(A,float); B=np.asarray(B,float)
    if A.shape!=B.shape:
        raise AssertionError(f"{name}: shape mismatch {A.shape} != {B.shape}")
    if not np.array_equal(A,B):
        d=np.abs(A-B)
        raise AssertionError(
            f"{name}: matrices differ; max_abs={float(np.max(d)):.3e}, "
            f"different_entries={int(np.count_nonzero(d))}"
        )


def _test_k2_safe_coloring():
    """K2 dense-vs-colored regression including periodic-seam safety."""
    # 1) Nontrivial colored case: m=1 keeps the *physical CUF residual* while
    # making a full element-by-element dense reference affordable.
    ops,quad,basis,bc=_make_k2_test_problem(nx=11,ns=7,te1=False)
    E0=70000.0; nu0=0.3
    load=np.zeros((3,basis.m,ops.grid.ns,ops.grid.nx),float)
    load[2,0]=1.0e-4

    t0=time.perf_counter()
    Ad,rd=c10a._assemble_linear_operator_and_rhs_dense_reference(
        ops,basis,bc,E0=E0,nu0=nu0,quadrature=quad,generalized_load=load
    )
    td=time.perf_counter()-t0
    t0=time.perf_counter()
    Ak,rk=c10a.assemble_linear_operator_and_rhs(
        ops,basis,bc,E0=E0,nu0=nu0,quadrature=quad,generalized_load=load
    )
    tk=time.perf_counter()-t0
    _assert_matrix_identical("K2 CUF linear A_dense/A_K2",Ad,Ak)
    if not np.array_equal(rd,rk):
        raise AssertionError("K2 CUF linear r0 changed")
    linear_stats=dict(c10a.assemble_linear_operator_and_rhs.last_k2_stats)
    if linear_stats["perturbed_residual_evaluations"] >= linear_stats["n_columns"]:
        raise AssertionError("K2 test grid did not reduce residual evaluations")

    # Nonlinear Jacobian at a deterministic nonzero state: exact same columns.
    q=np.zeros_like(load)
    ids=np.arange(ops.grid.ns*ops.grid.nx,dtype=float).reshape(ops.grid.ns,ops.grid.nx)
    q[0,0]=1e-7*np.sin(ids/7.0)
    q[1,0]=1e-7*np.cos(ids/9.0)
    q[2,0]=2e-7*np.sin(ids/11.0)
    nl_load=np.zeros_like(load); nl_load[2,0]=1e-6
    rbase,*_=c10b.full_gl_elastic_fullgrid_residual(
        q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quad,generalized_load=nl_load
    )
    fd_step=1e-8
    t0=time.perf_counter()
    Jd,rd2=c10b._finite_difference_jacobian_dense_reference(
        q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quad,
        generalized_load=nl_load,fd_step=fd_step,base_residual=rbase
    )
    tdn=time.perf_counter()-t0
    t0=time.perf_counter()
    Jk,rk2=c10b.finite_difference_jacobian(
        q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quad,
        generalized_load=nl_load,fd_step=fd_step,base_residual=rbase
    )
    tkn=time.perf_counter()-t0
    _assert_matrix_identical("K2 CUF nonlinear J_dense/J_K2",Jd,Jk)
    if not np.array_equal(rd2,rk2):
        raise AssertionError("K2 CUF nonlinear base residual changed")
    nonlinear_stats=dict(c10b.finite_difference_jacobian.last_k2_stats)

    # 2) Explicit periodic seam guard.  Naive is mod 7 would put s=0 and
    # s=14 together on ns=15; their real two-layer supports intersect.
    ops15,_,_,_=_make_k2_test_problem(nx=7,ns=15,te1=False)
    supp=k2color.spatial_dependency_supports(ops15,layers=2)
    groups=k2color.greedy_disjoint_groups(supp,ops15.grid.ns*ops15.grid.nx)
    sid0=0*ops15.grid.nx+3
    sid14=14*ops15.grid.nx+3
    if np.intersect1d(supp[sid0],supp[sid14]).size==0:
        raise AssertionError("K2 periodic seam test setup failed: expected overlap")
    if any(sid0 in g and sid14 in g for g in groups):
        raise AssertionError("K2 cyclic coloring mixed seam-neighbour supports")

    # 3) Final TE1 linear matrix + RHS + solution/wmax against old assembly.
    opst,quadt,bt,bct=_make_k2_test_problem(nx=7,ns=7,te1=True)
    loadt=np.zeros((3,bt.m,7,7),float); loadt[2,0]=1e-4
    Atd,rtd=c10a._assemble_linear_operator_and_rhs_dense_reference(
        opst,bt,bct,E0=E0,nu0=nu0,quadrature=quadt,generalized_load=loadt
    )
    Atk,rtk=c10a.assemble_linear_operator_and_rhs(
        opst,bt,bct,E0=E0,nu0=nu0,quadrature=quadt,generalized_load=loadt
    )
    _assert_matrix_identical("K2 TE1 linear A_dense/A_K2",Atd,Atk)
    if not np.array_equal(rtd,rtk):
        raise AssertionError("K2 TE1 r0/RHS changed")
    sold=c10a.solve_from_operator(
        Atd,rtd,opst,bt,bct,E0=E0,nu0=nu0,quadrature=quadt,generalized_load=loadt,
        linear_residual_tolerance_relative=1.0e-10
    )
    snew=c10a.solve_from_operator(
        Atk,rtk,opst,bt,bct,E0=E0,nu0=nu0,quadrature=quadt,generalized_load=loadt,
        linear_residual_tolerance_relative=1.0e-10
    )
    if not np.array_equal(sold.q,snew.q):
        raise AssertionError("K2 TE1 solution q changed")
    w_old=float(np.max(np.abs(sold.q[2,0])))
    w_new=float(np.max(np.abs(snew.q[2,0])))
    if w_old!=w_new:
        raise AssertionError("K2 TE1 wmax changed")

    # Structural production-grid counter (no expensive dense build required).
    op15,_,b15,_=_make_k2_test_problem(nx=15,ns=15,te1=True)
    s15=k2color.spatial_dependency_supports(op15,layers=2)
    g15=k2color.greedy_disjoint_groups(s15,15*15)
    estimate={
        "grid":"15x15_TE1",
        "dense_perturbed_evaluations":3*b15.m*15*15,
        "k2_perturbed_evaluations":3*b15.m*len(g15),
        "reduction_factor":(15*15)/len(g15),
    }
    return {
        "linear_dense_seconds":td,"linear_k2_seconds":tk,
        "nonlinear_dense_seconds":tdn,"nonlinear_k2_seconds":tkn,
        "linear_stats":linear_stats,"nonlinear_stats":nonlinear_stats,
        "te1_wmax":w_new,"te1_15x15":estimate,
    }


def _test_reviewer_v4_residual_measure():
    """Reviewer V-4: CUF stopping norms use row/BC scaling; equations stay raw."""
    E0=70000.0; L=100.0; h=1.0
    grid=te1_mvi.CylinderGrid(L,50.0,7,7)
    ops=c3.CylinderSpatialOperators.build(grid,7)
    basis=tebasis.get_basis("TE1")
    bc=cuf_bc.GeneralCylinderEndBC.clamped_free(basis.m)
    mult=rscale.cuf_residual_measure_multipliers(
        basis_m=basis.m,ns=grid.ns,nx=grid.nx,L=L,bc=bc,E0=E0,h=h
    )
    row=rscale.cuf_row_scale(basis.m,L)
    if not np.allclose(mult[:,:, :,1],row[None,:,None],rtol=0,atol=0):
        raise AssertionError("reviewer V-4: interior L**(-tau) row scaling mismatch")
    if not np.all(mult[:,:,:,0] == E0*h/L):
        raise AssertionError("reviewer V-4: GEOMETRIC boundary scaling mismatch")
    if not np.all(mult[:,:,:,-1] == 1.0/L):
        raise AssertionError("reviewer V-4: NATURAL boundary scaling mismatch")

    # MVI projected stopping measure must use the scaled residual, while the
    # residual function itself remains unchanged.
    rank=1
    x=np.linspace(0.8,1.2,grid.nx)
    s=np.linspace(0.7,1.3,grid.ns)
    X=np.empty((3,basis.m,rank,grid.nx),float)
    Y=np.empty((3,basis.m,rank,grid.ns),float)
    for a in range(3):
        for tau in range(basis.m):
            X[a,tau,0]=(1.0+0.2*a+0.1*tau)*x
            Y[a,tau,0]=(0.9+0.1*a+0.15*tau)*s
    state=cuf_mvi.ProfileStateCUF(X,Y)
    def residual(st):
        q=cuf_mvi.reconstruct_fields(st)
        return (q + 0.01*np.arange(q.size,dtype=float).reshape(q.shape)).reshape(-1)
    weights=cuf_mvi.residual_projection_weights(state,grid)
    mm=mult.reshape(-1)
    p_scaled=cuf_internal.projected_inf_for_direction(
        state,grid,bc,"Y",residual,weights,measure_multipliers=mm
    )
    R=np.asarray(residual(state),float).reshape(-1)
    _mask,_T,_ids,_norms,_active,Tn=cuf_internal._direction_data(state,grid,bc,"Y")
    expected=float(np.max(np.abs(Tn.T@(weights*(mm*R)))))
    if not np.isclose(p_scaled,expected,rtol=2e-14,atol=2e-14):
        raise AssertionError(
            f"reviewer V-4: MVI scaled projected measure {p_scaled:.6e} != {expected:.6e}"
        )


def _test_reviewer_k2_linear_residual_guard():
    """Reviewer K-2: bad direct-solve vectors must not be accepted as CONVERGED."""
    E0=70000.0; nu0=0.3
    ops,quad,basis,bc=_make_k2_test_problem(nx=7,ns=7,te1=True)

    load=np.zeros((3,basis.m,7,7),float); load[2,0]=1.0e-4
    A,r0=c10a.assemble_linear_operator_and_rhs(
        ops,basis,bc,E0=E0,nu0=nu0,quadrature=quad,generalized_load=load
    )
    good=c10a.solve_from_operator(
        A,r0,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quad,
        generalized_load=load,linear_residual_tolerance_relative=1.0e-10
    )
    if not good.converged or good.residual_relative>1.0e-10:
        raise AssertionError("reviewer K-2 CUF normal linear solve did not satisfy residual guard")
    x=np.linalg.solve(A,-r0); j=int(np.argmax(np.abs(x))); bad=A.copy(); bad[:,j]*=1.01
    try:
        c10a.solve_from_operator(
            bad,r0,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quad,
            generalized_load=load,linear_residual_tolerance_relative=1.0e-10
        )
    except c10a.CUFStrongFDMLinearSystemSingular as exc:
        if "LINEAR_SOLVE_RESIDUAL_NOT_SATISFIED" not in str(exc):
            raise AssertionError(f"reviewer K-2 CUF rejected for wrong reason: {exc}")
    else:
        raise AssertionError("reviewer K-2 CUF accepted a deliberately inconsistent operator")

    bck=kl_bc.KLCylinderEndBC.clamped_clamped()
    loadk=np.zeros((3,7,7),float); loadk[2]=1.0e-4
    Ak,r0k=k9b.assemble_linear_strong_system(
        ops,bck,E=E0,nu=nu0,h=1.0,generalized_load=loadk
    )
    goodk=k9b.solve_from_operator(
        Ak,r0k,ops,bck,E=E0,nu=nu0,h=1.0,generalized_load=loadk,
        linear_residual_tolerance_relative=1.0e-10
    )
    if not goodk.converged or goodk.residual_relative>1.0e-10:
        raise AssertionError("reviewer K-2 KL normal linear solve did not satisfy residual guard")
    xk=np.linalg.solve(Ak,-r0k); jk=int(np.argmax(np.abs(xk))); badk=Ak.copy(); badk[:,jk]*=1.01
    try:
        k9b.solve_from_operator(
            badk,r0k,ops,bck,E=E0,nu=nu0,h=1.0,generalized_load=loadk,
            linear_residual_tolerance_relative=1.0e-10
        )
    except k9b.KLStrongFDMLinearSystemSingular as exc:
        if "LINEAR_SOLVE_RESIDUAL_NOT_SATISFIED" not in str(exc):
            raise AssertionError(f"reviewer K-2 KL rejected for wrong reason: {exc}")
    else:
        raise AssertionError("reviewer K-2 KL accepted a deliberately inconsistent operator")

    # The CASE-level parameter is mandatory for FDM LINEAR_ELASTIC.
    d=WORK/"REVIEWER_K2_MISSING_LINEAR"; d.mkdir(parents=True,exist_ok=True)
    cfg=_case_config("CUF_TE1","LINEAR_ELASTIC"); cfg.pop("linear",None)
    p=d/"case.json"; p.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    try:
        validate_final_case(p)
    except Exception as exc:
        if "CASE.linear" not in str(exc):
            raise AssertionError(f"reviewer K-2 missing CASE.linear rejected for wrong reason: {exc}")
    else:
        raise AssertionError("reviewer K-2 missing mandatory CASE.linear was accepted")


def _test_cuf_four_modes():
    """CUF TE1 four-mode smoke: Elastic/Birger x Linear/Nonlinear."""
    E0=70000.0; nu0=0.3; G0=E0/(2.0*(1.0+nu0))
    out=[]
    for mode,elastic_mode in (
        ("LINEAR_BIRGER","LINEAR_ELASTIC"),
        ("NONLINEAR_BIRGER","NONLINEAR_ELASTIC"),
    ):
        d=WORK/f"CUF_TE1_{mode}"
        d.mkdir(parents=True,exist_ok=True)
        cfg=_case_config("CUF_TE1",elastic_mode)
        cfg["mode"]=mode
        cfg["initial_GB"]={"source":"ELASTIC_G0","npz":None}
        cfg["birger"]={
            "epsilon_tol":1e-12,"omega":1.0,"G_ref":G0,"sigma_ref":1.0,
            "tol_G":1e-10,"tol_sigma":1e-8,"tol_u":1e-8,"max_outer_iterations":2,
            "epsilon_i_limit":1.0,"delta_J_limit":1.0,"initial_slope_rtol":0.05,
            "curve":{"mode":"FORMULA","formula":"slope*eps","parameters":{"slope":3.0*G0}},
        }
        p=d/"case.json"; p.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        result=run_final_case(p)
        if not result.get("converged",False) or result.get("accepted_steps")!=1:
            raise AssertionError(f"CUF four-mode {mode}: did not converge")
        with np.load(d/"OUT"/"STEP_0001_STATE.npz",allow_pickle=False) as z:
            q=np.asarray(z["q"],float)
        wmax=float(np.max(np.abs(q[2,0])))
        _assert_close(f"CUF {mode} elastic-curve wmax",wmax,REFERENCE[("CUF_TE1",elastic_mode)])
        out.append((mode,wmax,int(result["steps"][0].get("outer_iterations",-1))))
    return out



def _test_k2b_vectorized_thickness():
    """K2b: vectorized CUF thickness integration must match scalar reference."""
    basis=tebasis.get_basis("TE1")
    ns,nx=4,5
    ss=np.arange(ns,dtype=float)[:,None]
    xx=np.arange(nx,dtype=float)[None,:]
    q=np.empty((3,basis.m,ns,nx),float)
    qx=np.empty_like(q); qs=np.empty_like(q)
    for c in range(3):
        for t in range(basis.m):
            a=(c+1)*(t+1)
            q[c,t]=1e-5*a*(np.sin((ss+1)/(3+t))+0.4*np.cos((xx+1)/(4+c)))
            qx[c,t]=2e-6*a*np.cos((ss+xx+2)/(5+c+t))
            qs[c,t]=1.5e-6*a*np.sin((2*ss+xx+3)/(6+c+t))
    quad=s2.gauss_thickness_quadrature(1.0,3)
    K0=58333.333333333336; G0=26923.076923076922; R=50.0
    GB=np.stack([
        G0*(1.0+0.01*g+0.02*np.sin(ss/3.0)+0.01*np.cos(xx/4.0))
        for g in range(quad.z.size)
    ],axis=0)

    def same(tag,a,b):
        for key in cuf_mech.RESULTANT_KEYS:
            x=np.asarray(getattr(a,key),float); y=np.asarray(getattr(b,key),float)
            d=float(np.max(np.abs(x-y)))
            scale=max(float(np.max(np.abs(x))),float(np.max(np.abs(y))),1.0)
            if d > 5e-11*scale:
                raise AssertionError(
                    f"K2b {tag} {key}: scalar/vector mismatch "
                    f"max_abs={d:.3e}, scale={scale:.3e}"
                )

    a=cuf_mech.general_thickness_resultants_scalar_reference(
        q,qx,qs,R=R,K0=K0,GB=GB,quadrature=quad,basis=basis
    )
    b=cuf_mech.general_thickness_resultants(
        q,qx,qs,R=R,K0=K0,GB=GB,quadrature=quad,basis=basis
    )
    same("NONLINEAR",a,b)

    a=cuf_linear.general_linear_thickness_resultants_scalar_reference(
        q,qx,qs,R=R,K0=K0,GB=GB,quadrature=quad,basis=basis
    )
    b=cuf_linear.general_linear_thickness_resultants(
        q,qx,qs,R=R,K0=K0,GB=GB,quadrature=quad,basis=basis
    )
    same("LINEAR",a,b)

    # Mandatory nonlinear mapping guard is still executed before constitutive law.
    qb=np.zeros((3,1,3,3),float); qxb=np.zeros_like(qb); qsb=np.zeros_like(qb)
    b1=cuf_mech.CallableThicknessBasis(
        (lambda z:1.0,),(lambda z:0.0,),("1",)
    )
    qxb[0,0]=-2.0
    try:
        cuf_mech.general_thickness_resultants(
            qb,qxb,qsb,R=R,K0=K0,GB=G0,
            quadrature=quad,basis=b1
        )
    except s2.InvalidDeformationMapping:
        pass
    else:
        raise AssertionError("K2b lost detF<=0 admissibility guard")

def _test_k1_equilibrated_solve():
    # Deliberately ill-scaled but nonsingular matrix.  The recovered x must be
    # the same physical solution of Ax=b after row/column equilibration.
    A = np.array([
        [1.0, 2.0e-6, -3.0e-9],
        [4.0e7, -5.0e8, 6.0e4],
        [7.0e-5, 8.0e2, 9.0e3],
    ], float)
    x_ref = np.array([0.25, -1.5, 2.0], float)
    b = A @ x_ref
    x, As, row, col = nscale.equilibrated_solve(A, b)
    if not np.all(np.isfinite(As)) or np.any(row <= 0.0) or np.any(col <= 0.0):
        raise AssertionError("K1 equilibration returned invalid scales")
    if not np.allclose(x, x_ref, rtol=1e-11, atol=1e-11):
        raise AssertionError(f"K1 equilibration changed solution: {x} != {x_ref}")


def _noncontiguous_last_axis(shape):
    base_shape = tuple(shape[:-1]) + (2 * shape[-1],)
    base = np.zeros(base_shape, float)
    view = base[..., ::2]
    if view.flags.c_contiguous:
        raise AssertionError("test setup failed: expected non-contiguous view")
    return view


def _test_reviewer_v7_periodic_parity_and_kl_d2():
    """Reviewer V-7: even ns is rejected and KL has a genuine D2 operator."""
    # The periodic centred D1 has an exact sawtooth null mode for even ns.
    grid_even=te1_mvi.CylinderGrid(100.0,50.0,7,12)
    ops_even=c3.CylinderSpatialOperators.build(grid_even,7)
    saw=np.array([(-1.0)**j for j in range(grid_even.ns)],float)
    if float(np.max(np.abs(ops_even.D1s@saw))) > 1.0e-10:
        raise AssertionError("reviewer V-7 test setup: even-ns sawtooth is not in D1s nullspace")
    if float(np.max(np.abs(ops_even.D2s@saw))) <= 1.0e-8:
        raise AssertionError("reviewer V-7: genuine D2s still annihilates the sawtooth mode")
    if np.allclose(ops_even.D2s,ops_even.D1s@ops_even.D1s,rtol=1e-12,atol=1e-12):
        raise AssertionError("reviewer V-7: D2s is still implemented as D1s@D1s")

    # Nonperiodic D2x must reproduce the second derivative of x^2.
    x=grid_even.x
    X2=np.broadcast_to(x[None,:]**2,(grid_even.ns,grid_even.nx)).copy()
    err=float(np.max(np.abs(ops_even.Dxx(X2)-2.0)))
    if err > 1.0e-9:
        raise AssertionError(f"reviewer V-7: Dxx(x^2) error={err:.3e}")

    # Production CASE validation must reject even ns for either active grid.
    for method,key in (("FDM","fdm_grid"),("MVI","mvi_grid")):
        cfg=_case_config("CUF_TE1","LINEAR_ELASTIC")
        cfg["method"]=method
        cfg[key]=dict(cfg[key])
        cfg[key]["ns"]=12
        d=WORK/f"REVIEWER_V7_EVEN_{method}"
        d.mkdir(parents=True,exist_ok=True)
        p=d/"case.json"
        p.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        try:
            validate_final_case(p)
        except Exception as exc:
            if "нечёт" not in str(exc) and "odd" not in str(exc).lower():
                raise AssertionError(f"reviewer V-7 {method}: wrong rejection: {exc}")
        else:
            raise AssertionError(f"reviewer V-7 {method}: even ns was accepted")


def _test_k8_write_through():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ids = np.arange(values.size, dtype=int)

    X = _noncontiguous_last_axis((6, 1, 7))
    st = te1_mvi.ProfileStateTE1(X, np.zeros((6, 1, 7)))
    te1_mvi.set_active_parameters(st, "X", ids, values)
    if not np.array_equal(np.asarray(st.X.flat[ids]), values):
        raise AssertionError("K8 TE1 write-through failed on non-contiguous X")

    X = _noncontiguous_last_axis((3, 2, 1, 7))
    st = cuf_mvi.ProfileStateCUF(X, np.zeros((3, 2, 1, 7)))
    cuf_mvi.set_active_parameters(st, "X", ids, values)
    if not np.array_equal(np.asarray(st.X.flat[ids]), values):
        raise AssertionError("K8 CUF write-through failed on non-contiguous X")

    X = _noncontiguous_last_axis((3, 1, 7))
    st = kl_mvi.ProfileStateKL(X, np.zeros((3, 1, 7)))
    kl_mvi.set_active_parameters(st, "X", ids, values)
    if not np.array_equal(np.asarray(st.X.flat[ids]), values):
        raise AssertionError("K8 KL write-through failed on non-contiguous X")


def _test_kl_hinge_variants():
    expected_kl = {
        "PINNED_S1": ("NATURAL", "GEOMETRIC", "GEOMETRIC", "NATURAL"),
        "PINNED_S2": ("GEOMETRIC", "NATURAL", "GEOMETRIC", "NATURAL"),
        "SIMPLY_SUPPORTED": ("GEOMETRIC", "GEOMETRIC", "GEOMETRIC", "NATURAL"),
        "PINNED_S4": ("NATURAL", "NATURAL", "GEOMETRIC", "NATURAL"),
    }
    for name, kinds in expected_kl.items():
        bc = _make_kl_simple_bc((name, "CLAMPED"))
        got = tuple(getattr(bc.left, key).kind for key in ("u", "v", "w", "slope"))
        if got != kinds:
            raise AssertionError(("KL", name, got, kinds))

    # Movable hinges at both ends retain a rigid in-plane mode and must ask for
    # an explicit gauge rather than silently solve a singular problem.
    for name in ("PINNED_S1", "PINNED_S2", "PINNED_S4"):
        try:
            _make_kl_simple_bc((name, name))
        except kl_bc.KLRigidModeHandlingRequired:
            pass
        else:
            raise AssertionError(f"KL {name}-{name} must require rigid-mode handling")

    # CUF SIMPLY_SUPPORTED is the same physical immovable S3 hinge as KL:
    # only the midsurface coefficients are fixed; higher coefficients are
    # traction-free.  S4 remains the distinct in-plane movable support.
    bc = _make_cuf_simple_bc(("SIMPLY_SUPPORTED", "CLAMPED"), 3)
    for end_name, end in (("left", bc.left), ("right", bc.right)):
        for tau in range(3):
            got = tuple(end.conditions[a][tau].kind for a in range(3))
            expected = (
                ("GEOMETRIC", "GEOMETRIC", "GEOMETRIC")
                if end_name == "right" or tau == 0
                else ("NATURAL", "NATURAL", "NATURAL")
            )
            if got != expected:
                raise AssertionError(("CUF S3", end_name, tau, got, expected))

    # The physical S3 pair itself removes axial translation and axial rotation.
    _make_cuf_simple_bc(("SIMPLY_SUPPORTED", "SIMPLY_SUPPORTED"), 3)

    bc = _make_cuf_simple_bc(("PINNED_S4", "CLAMPED"), 3)
    for tau in range(3):
        got = tuple(bc.left.conditions[a][tau].kind for a in range(3))
        expected = ("NATURAL", "NATURAL", "GEOMETRIC")
        if got != expected:
            raise AssertionError(("CUF S4", tau, got, expected))
    try:
        _make_cuf_simple_bc(("PINNED_S4", "PINNED_S4"), 3)
    except cuf_bc.GeneralRigidModeHandlingRequired:
        pass
    else:
        raise AssertionError("CUF PINNED_S4-PINNED_S4 must require rigid-mode handling")


def _test_automatic_mvi_profiles():
    """Preserved KL seeds and plate-style CUF seeds."""
    base={
        "L":2000.0,"R":1000.0,"nx":41,"ns":41,"stencil":7,
        "h":50.0,"theory":"KL","cuf_model":None,
    }
    Xk,Yk=_build_auto_mvi_profiles(base,2)
    if Xk.shape!=(3,2,41) or Yk.shape!=(3,2,41):
        raise AssertionError(f"automatic KL profile shapes: {Xk.shape}, {Yk.shape}")
    if not np.allclose(Xk[...,0],0.0) or not np.allclose(Xk[...,-1],0.0):
        raise AssertionError("automatic KL axial profiles do not vanish at both ends")
    if not np.all(np.isfinite(Xk)) or not np.all(np.isfinite(Yk)):
        raise AssertionError("automatic KL profiles are non-finite")
    if not np.any(Yk!=0.0):
        raise AssertionError("automatic KL V11 Y seed was replaced")

    cuf={
        "L":2000.0,"R":1000.0,"nx":41,"ns":41,"stencil":7,
        "h":50.0,"theory":"CUF","cuf_model":"TE4",
    }
    Xc,Yc=_build_auto_mvi_profiles(cuf,8)
    if Xc.shape!=(3,5,8,41) or Yc.shape!=(3,5,8,41):
        raise AssertionError(f"automatic CUF profile shapes: {Xc.shape}, {Yc.shape}")
    if not np.all(np.isfinite(Xc)) or not np.all(np.isfinite(Yc)):
        raise AssertionError("automatic CUF profiles are non-finite")

    if np.any(Yc!=0.0):
        raise AssertionError("automatic plate-style CUF Y seed is not zero")
    for component in range(Xc.shape[0]):
        for tau in range(Xc.shape[1]):
            if np.linalg.matrix_rank(Xc[component,tau])!=Xc.shape[2]:
                raise AssertionError(
                    f"automatic CUF X bank is rank deficient: "
                    f"component={component}, tau={tau}"
                )
    if np.any(cuf_mvi.reconstruct_fields(cuf_mvi.ProfileStateCUF(Xc,Yc))!=0.0):
        raise AssertionError("automatic plate-style CUF physical seed is not zero")

    # The preserved KL seed represents the same displacement after mm -> m.
    si=dict(base);si["h"]=0.05
    Xsi,Ysi=_build_auto_mvi_profiles(si,2)
    if not np.allclose(Xk*1.0e-3,Xsi,rtol=1e-13,atol=1e-18):
        raise AssertionError("automatic KL MVI seed is not length-unit consistent")
    if not np.array_equal(Yk,Ysi):
        raise AssertionError("automatic MVI dimensionless Y changed with units")


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    checks = [
        ("K1 algebraic equilibration", _test_k1_equilibrated_solve),
        ("K8 MVI write-through", _test_k8_write_through),
        ("K5/K6 Birger curve guards", _test_k5_k6_birger_curve_guard),
        ("V10 Birger monotonic guard", _test_v10_birger_monotonic_guard),
        ("K2b vectorized thickness", _test_k2b_vectorized_thickness),
        ("V3 scaled central Jacobian", lambda: v3check.run_checks(verbose=False)),
        ("V3b analytic elastic tangent", lambda: v3bcheck.run_checks(verbose=False)),
        ("V3c frozen-GB Birger tangent", lambda: v3ccheck.run_checks(verbose=False)),
        ("V1/V11 physical load band", lambda: v1v11check.run_checks(verbose=False)),
        ("V2 relative Newton residual", lambda: v2check.run_checks(verbose=False)),
        ("V11.2 residual split", lambda: v112check.run_quick_checks(verbose=False)),
        ("V5 unit interpretation", lambda: v5check.run_checks(verbose=False)),
        ("V6 thickness quadrature", lambda: v6check.run_quick_checks(verbose=False)),
        ("V7 plastic volume fraction", lambda: v7check.run_checks(verbose=False)),
        ("V8 dead-reference load", lambda: v8check.run_checks(verbose=False)),
        ("V9 strict CSV parser", lambda: v9check.run_checks(verbose=False)),
        ("Z1 scaled FD weights", lambda: z1check.run_checks(verbose=False)),
        ("Z2 static hygiene", lambda: z2check.run_checks(verbose=False)),
        ("Z4 Birger compatibility", lambda: z4check.run_checks(verbose=False)),
        ("HARMONIC load generator", lambda: harmoniccheck.run_checks(verbose=False)),
        ("MVI limit controls", lambda: mvilimitcheck.run_checks(verbose=False)),
        ("CUF/KL radial sign", _test_radial_sign_consistency),
        ("Reviewer V-9 boundary warning", _test_reviewer_v9_boundary_layer_warning),
        ("Reviewer V-8 KL face contract", _test_reviewer_v8_kl_face_rejection),
        ("Reviewer V-7 odd ns and KL D2", _test_reviewer_v7_periodic_parity_and_kl_d2),
        ("Reviewer V-4 residual measure", _test_reviewer_v4_residual_measure),
        ("Reviewer K-2 residual guard", _test_reviewer_k2_linear_residual_guard),
        ("KL/CUF hinge mappings", _test_kl_hinge_variants),
        ("Automatic MVI profiles", _test_automatic_mvi_profiles),
    ]

    guarded_checks = []
    for theory in ("CUF_TE1", "KL"):
        for mode in ("LINEAR_ELASTIC", "NONLINEAR_ELASTIC"):
            guarded_checks.append(
                (
                    f"{theory} {mode} nonzero FDM",
                    lambda theory=theory, mode=mode: _run_nonzero_fdm(theory, mode),
                )
            )
    for theory in ("CUF_TE1", "KL"):
        guarded_checks.append(
            (
                f"{theory} K3 operator cache",
                lambda theory=theory: _test_k3_linear_operator_cache(theory),
            )
        )
        guarded_checks.append(
            (
                f"{theory} V4 converged state",
                lambda theory=theory: _test_v4_converged_state_skips_jacobian(theory),
            )
        )
    guarded_checks.append(
        ("KL nonlinear Birger converged state", _test_v4_kl_nonlinear_birger_jacobian_skip)
    )

    results = []

    def run_check(name, fn):
        started = time.perf_counter()
        try:
            fn()
        except Exception as exc:
            results.append(
                (
                    name,
                    "FAIL",
                    time.perf_counter() - started,
                    f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            results.append((name, "PASS", time.perf_counter() - started, ""))

    try:
        for name, fn in checks:
            run_check(name, fn)

        # The regular CUF/KL regression cases must not execute the expensive
        # SVD-based condition diagnostic.
        real_cond = np.linalg.cond

        def forbidden_cond(*args, **kwargs):
            raise AssertionError(
                "K4 regression: np.linalg.cond called on default solver path"
            )

        np.linalg.cond = forbidden_cond
        try:
            for name, fn in guarded_checks:
                run_check(name, fn)
        finally:
            np.linalg.cond = real_cond

        failures = sum(status == "FAIL" for _, status, _, _ in results)
        for name, status, elapsed, message in results:
            suffix = f" — {message}" if message else ""
            print(f"[{status}] {name} ({elapsed:.1f} s){suffix}")
        print(
            f"FINAL_SELF_CHECK: {'PASS' if failures == 0 else 'FAIL'} "
            f"({len(results) - failures}/{len(results)})"
        )
        if failures:
            raise SystemExit(1)
    finally:
        if os.environ.get("KEEP_SELF_CHECK_WORK", "0") != "1":
            shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
