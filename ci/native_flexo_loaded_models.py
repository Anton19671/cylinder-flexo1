"""Nonzero-loaded nonlinear native MVI flexoelectric 7 x G3D/Gz scientific campaign.

This is not a source-boundary or independent external accuracy certificate.
Each CASE is stored and run separately on GitHub Actions, never locally.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np
from cylinder_paths import activate

activate()
from independent_mvi import run_coupled_research as mvi

THEORIES = mvi.THEORIES
GRADIENTS = ("G3D", "Gz")
NX, NS = 9, 7
R, L, H = 2.0, 1.2, 0.08
PRESSURE = 100.0


def n_mech(theory):
    return (3 if theory == "KL" else
            5 if theory in ("FSDT", "REDDY") else
            3 * (int(theory[-1]) + 1))


def e_order(theory):
    return int(theory[-1]) if theory.startswith("CUF_TE") else 1


def case_and_npz(folder, theory, gradient, *, reduced_lu_rescue_limit=0):
    if theory not in THEORIES or gradient not in GRADIENTS:
        raise ValueError("INVALID_NATIVE_FLEXO_CAMPAIGN_MODEL")
    if type(reduced_lu_rescue_limit) is not int or reduced_lu_rescue_limit < 0:
        raise ValueError("INVALID_REDUCED_MVI_LU_RESCUE_LIMIT")
    nm, pe = n_mech(theory), e_order(theory)
    nf = nm + pe + 1
    theta = (2.0 * math.pi / NS) * np.arange(NS)
    forcing = np.zeros((nm, NX, NS))
    # Strict axisymmetric loading: fixed-rank one can represent this
    # symmetry class without confusing a rank restriction with BC errors.
    qface = -PRESSURE * (1.0 + H / (2.0 * R))
    if theory == "KL":
        forcing[2] = qface
    elif theory.startswith("CUF_TE"):
        k = pe + 1
        forcing[2 * k:3 * k] = qface
    else:
        forcing[2] = qface
    assert np.max(np.abs(forcing)) > 0
    # For an outer-face nominal pressure every theory has the SAME
    # curved physical face-area multiplier H(+h/2)=1+h/(2R).
    # KL has w(z)=w0, so it must not omit that multiplier.
    assert np.allclose(forcing[2] if theory=="KL" else
        forcing[2*(pe+1)] if theory.startswith("CUF_TE") else forcing[2],
        qface)
    x = np.zeros((nf, 1, NX))
    y = np.ones((nf, 1, NS))
    np.savez_compressed(folder / "seed.npz", x_profiles=x, y_profiles=y)
    np.savez_compressed(
        folder / "load.npz",
        mechanical_load=forcing,
        free_charge=np.zeros((pe + 1, NX, NS)))
    case = {
        "schema": mvi.SCHEMA, "research_opt_in": True,
        "physics": {
            "theory": theory, "geometry_model": "GREEN_LAGRANGE",
            "material": {
                "young_pa": 1.e9, "poisson": 0.3, "ell_m": 0.002,
                "mu1_c_per_m": 1.e-3, "mu2_c_per_m": -5.e-4,
                "kappa_f_per_m": 1.e-8, "gradient_model": gradient},
            "thickness_m": H, "density_kg_m3": 2500.,
            "eta_s": 0., "mass_damping_rate_per_s": 0.,
            "n_gauss": 12, "potential_order": pe},
        "grid": {"nx": NX, "ns": NS,
                 "length_m": L, "radius_m": R},
        "boundary": {"mechanical": "IMMOBILE_PARALLEL_ENDS",
                     "electric": "GROUNDED_BOTH"},
        "rank": 1,
        "solver": {
            "kind": "STATIC", "integrator": None, "dt_s": None, "steps": 0,
            "field_scales": [1.e-4] * nm + [1.0] * (pe + 1),
            "energy_scale_j": 1.0, "max_sweeps": 40,
            "max_acceleration_coordinates": 4096,
            "max_halfstep_coordinates": 4096,
            "linear_solver": "AUTO", "dense_coordinate_limit": 96,
            "krylov_relative_tolerance": 1.e-9,
            "krylov_dense_fallback_limit": reduced_lu_rescue_limit},
        "initial_state_npz": "seed.npz",
        "loads_npz": "load.npz",
        "output_dir": "RESULTS"}
    file = folder / "CASE.json"
    file.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return file, case, forcing


def run(theory, gradient, output, *, reduced_lu_rescue_limit=0):
    path = Path(output).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    path.mkdir(parents=True, exist_ok=True)
    evidence = {
        "theory": theory, "gradient_model": gradient,
        "method": "NATIVE_INDEPENDENT_MVI",
        "electrical_enabled": True, "geometry_model": "GREEN_LAGRANGE",
        "grid": [NX, NS], "rank": 1, "pressure_pa": PRESSURE,
        "pressure_face": "OUTER_PHYSICAL_Z_PLUS_H_OVER_2",
        "reduced_mvi_lu_rescue_limit": reduced_lu_rescue_limit,
        "reference_area_load_pa": -PRESSURE * (1.0 + H / (2.0 * R)),
        "git_sha": os.getenv("GITHUB_SHA", "UNKNOWN"),
        "source_higher_order_mechanical_face_certified": False,
        "independent_external_accuracy_certified": False,
        "status": "IN_PROGRESS"}
    marker = path / "PHYSICAL_EVIDENCE.json"
    def save():
        marker.write_text(json.dumps(evidence, indent=2,
                                     allow_nan=False) + "\n",
                          encoding="utf-8")
    save()
    try:
        file, case, forcing = case_and_npz(
            path, theory, gradient,
            reduced_lu_rescue_limit=reduced_lu_rescue_limit)
        print("NATIVE_FLEXO_BEGIN", theory, gradient, flush=True)
        prepared = mvi.load_case(file)
        answer = mvi.run(file, validate_only=False)
        state = path / "RESULTS" / "STATIC_ACCEPTED.npz"
        if answer.get("accepted_steps") != 1 or not state.is_file():
            raise AssertionError("NATIVE_FLEXO_STATIC_STEP_NOT_ACCEPTED")
        with np.load(state, allow_pickle=False) as f:
            if not {"mechanical", "electrical", "x_profiles",
                    "y_profiles"}.issubset(f.files):
                raise AssertionError("NATIVE_FLEXO_PHYSICAL_FIELDS_MISSING")
            q = np.asarray(f["mechanical"], float)
            potential = np.asarray(f["electrical"], float)
            x = np.asarray(f["x_profiles"], float)
            y = np.asarray(f["y_profiles"], float)
            if (q.shape != forcing.shape or
                potential.shape != (e_order(theory) + 1, NX, NS) or
                not np.isfinite(q).all() or
                not np.isfinite(potential).all()):
                raise AssertionError("NATIVE_FLEXO_BAD_PHYSICAL_FIELD")
            component = (2 * (e_order(theory) + 1) if
                         theory.startswith("CUF_TE") else 2)
            radial = q[component]
            peak = float(np.max(np.abs(radial)))
            voltage = float(np.max(np.abs(potential)))
            if peak <= 1.e-14:
                raise AssertionError("NATIVE_FLEXO_ZERO_LOADED_RADIAL_RESPONSE")
            if voltage <= 0.:
                raise AssertionError("NATIVE_FLEXO_ZERO_INDUCED_ELECTRIC_RESPONSE")
            if x.shape != (len(q)+len(potential), 1, NX):
                raise AssertionError("NATIVE_FLEXO_X_SHAPE")
            if y.shape != (len(q)+len(potential), 1, NS):
                raise AssertionError("NATIVE_FLEXO_Y_SHAPE")
        # Physical face flux is a diagnostic, NOT a solver acceptance gate:
        # report the actual local mismatch to avoid confusing generalized
        # polynomial weak equilibrium with pointwise face charge closure.
        electric_faces = answer.get("electric_faces", {})
        faces = electric_faces.get("faces", {})
        if set(faces) != {"inner", "outer"}:
            raise AssertionError("NATIVE_FLEXO_ELECTRIC_FACE_DIAGNOSTICS_MISSING")
        face_abs_max = {
            side: float(faces[side]["natural_flux_difference_abs_max_c_per_m2"])
            for side in ("inner", "outer")
        }
        if not all(math.isfinite(v) for v in face_abs_max.values()):
            raise AssertionError("NATIVE_FLEXO_ELECTRIC_FACE_DIAGNOSTICS_NONFINITE")
        evidence["electric_face_natural_flux_abs_max_c_per_m2"] = face_abs_max
        evidence["pointwise_electric_faces_certified"] = bool(
            electric_faces.get("physical_face_flux_certified", False))
        evidence["electric_face_mismatch_is_diagnostic_only"] = True
        evidence["discrete_green_audit"] = answer.get("discrete_green")
        evidence.update(
            status="ACCEPTED_RESEARCH_ONLY",
            solver_status=answer.get("status"),
            radial_max_abs_m=peak,
            potential_max_abs_v=voltage,
            nonzero_radial_response_verified=True,
            nonzero_induced_potential_verified=True,
            fixed_rank_one_physical_state_accepted=True)
        save()
        print("NATIVE_FLEXO_ACCEPTED", json.dumps(evidence), flush=True)
    except BaseException as exc:
        evidence.update(
            status="FAILED",
            failure_type=type(exc).__name__,
            failure_message=str(exc))
        save()
        (path / "TRACEBACK.txt").write_text(traceback.format_exc())
        print("NATIVE_FLEXO_FAILED", theory, gradient, str(exc), flush=True)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--theory", required=True, choices=THEORIES)
    p.add_argument("--gradient", required=True, choices=GRADIENTS)
    p.add_argument("--output", required=True)
    p.add_argument("--reduced-lu-rescue-limit", type=int, default=0)
    a = p.parse_args()
    run(a.theory, a.gradient, a.output,
        reduced_lu_rescue_limit=a.reduced_lu_rescue_limit)
