#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical enforcement check for Kirchhoff-Love strong-FDM end BC.

This is deliberately stronger than a registry/GUI semantics test: it solves
one CLAMPED/CLAMPED and one S3/S3 case through the real dispatcher and then
checks the conjugate boundary quantities on the converged fields.
"""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import numpy as np

import FINAL_SELF_CHECK as base
import cylinder_te1_mvi_core as te1_mvi
import cylinder_te1_stage3a_connector as c3
import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8b1_boundary as kb
from cylinder_final_stage11a_dispatcher import run_final_case, validate_final_case

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "KL_FDM_BC_ENFORCEMENT_WORK"


def _solve(code: str):
    d = WORK / code
    d.mkdir(parents=True, exist_ok=True)
    cfg = base._case_config("KL", "LINEAR_ELASTIC")
    cfg["fdm_grid"] = {"nx": 21, "ns": 7, "stencil": 7}
    cfg["mvi_grid"] = {"nx": 21, "ns": 7, "stencil": 7}
    cfg["boundary_conditions"] = {
        "left": code,
        "right": code,
        "advanced": None,
        "model": "KL",
    }
    cfg["output_dir"] = "OUT"
    p = d / "case.json"
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    check = validate_final_case(p)
    if not check.get("valid", False):
        raise AssertionError(f"{code}: validation failed")
    result = run_final_case(p)
    if not result.get("converged", False):
        raise AssertionError(f"{code}: FDM did not converge: {result.get('status')}")

    with np.load(d / "OUT" / "STEP_0001_STATE.npz", allow_pickle=False) as z:
        U = np.asarray(z["U"], float)

    L, R, h, E, nu = 100.0, 50.0, 1.0, 70000.0, 0.3
    grid = te1_mvi.CylinderGrid(L, R, 21, 7)
    ops = c3.CylinderSpatialOperators.build(grid, 7)
    kin = kl.field_kinematics(U, ops, geometry="LINEAR")
    nm = kl.analytic_elastic_resultants(kin.state, kl.KLElasticConstants(E, nu, h))
    ql = kb.end_mechanical_quantities(kin, nm, ops, geometry="LINEAR", side="LEFT")
    qr = kb.end_mechanical_quantities(kin, nm, ops, geometry="LINEAR", side="RIGHT")

    return {
        "w_abs_max": float(np.max(np.abs(U[2]))),
        "theta_left": float(np.max(np.abs(ql.theta_n))),
        "theta_right": float(np.max(np.abs(qr.theta_n))),
        "Mx_left": float(np.max(np.abs(nm.Mx[:, 0]))),
        "Mx_right": float(np.max(np.abs(nm.Mx[:, -1]))),
        "Ttheta_left": float(np.max(np.abs(ql.T_theta))),
        "Ttheta_right": float(np.max(np.abs(qr.T_theta))),
    }


def run(verbose=True):
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    cl = _solve("CLAMPED")
    ss = _solve("SIMPLY_SUPPORTED")

    # CLAMPED must enforce zero normal rotation and develop a reaction moment.
    if max(cl["theta_left"], cl["theta_right"]) > 1e-12:
        raise AssertionError(f"CLAMPED theta_n not enforced: {cl}")
    if min(cl["Mx_left"], cl["Mx_right"]) < 1e-4:
        raise AssertionError(f"CLAMPED unexpectedly has zero edge moment: {cl}")

    # S3 must enforce Mx=0 (T_theta=-Mx=0) while allowing finite rotation.
    if max(ss["Mx_left"], ss["Mx_right"], ss["Ttheta_left"], ss["Ttheta_right"]) > 1e-10:
        raise AssertionError(f"S3 natural moment BC not enforced: {ss}")
    if min(ss["theta_left"], ss["theta_right"]) < 1e-8:
        raise AssertionError(f"S3 rotation appears spuriously clamped: {ss}")

    # The two real solutions must not collapse to the same field response.
    rel = abs(ss["w_abs_max"] - cl["w_abs_max"]) / max(cl["w_abs_max"], 1e-30)
    if rel < 1e-4:
        raise AssertionError(f"CLAMPED and S3 responses are numerically indistinguishable: rel={rel:.3e}")

    result = {"status": "PASS", "clamped": cl, "pinned_s3": ss, "w_rel_difference": rel}
    if verbose:
        print("KL_FDM_BC_ENFORCEMENT_SELF_CHECK: PASS")
        print(f"  CLAMPED theta={max(cl['theta_left'],cl['theta_right']):.3e}, Mx={max(cl['Mx_left'],cl['Mx_right']):.6e}")
        print(f"  S3      theta={max(ss['theta_left'],ss['theta_right']):.6e}, Mx={max(ss['Mx_left'],ss['Mx_right']):.3e}")
        print(f"  |dwmax| relative difference={rel:.6e}")
    return result


if __name__ == "__main__":
    run(True)
