#!/usr/bin/env python3
"""Rawat C-C: controlled constitutive sensitivity, KL and CUF TE1.

This is a diagnostic, NOT a production-law replacement or an independent
validation of axial m=1. Both variants use the same grid, SBP42 operator,
geometry, density, inertial action and essential mechanical traces.

In CUF TE1 the transverse-normal strain is an independent constant-through-
thickness kinematic field. A local sigma_zz=0 Schur reduction therefore
changes the governing constitutive model and can produce a thickness-stretch
null mode. Any resulting low-frequency branch must be checked for this
degeneracy before being used as a published-mode comparison.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cylinder_paths import activate
activate()

import cuf_g3d_weak as cuf
import cylinder_modal_linear_physics as linear_physics
from cylinder_modal_reference import solve_modal_case
from modal_cammalleri_plane_stress_diagnostic import (
    plane_stress_local_tangent, reduced_plane_stress_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/verification/rawat2016/modal_linear"
REFERENCE = {2: 342., 3: 269., 4: 277., 5: 352.}
ORIGINAL_CUF_STRESS = cuf.enthalpy_derivatives
ORIGINAL_CUF_ENERGY = cuf.enthalpy_parts
ORIGINAL_KL_TANGENT = linear_physics._kl_linear_local_tangent


def _only_uncoupled_macro(material):
    if (material.ell_m != 0 or material.mu1_c_per_m != 0
            or material.mu2_c_per_m != 0):
        raise ValueError("RAWAT_PLANE_STRESS_ONLY_UNCOUPLED_MACRO")


def cuf_reduced_stress(strain6, gradient6x3, electric3, material):
    _only_uncoupled_macro(material)
    _, hyperstress, displacement = ORIGINAL_CUF_STRESS(
        strain6, gradient6x3, electric3, material)
    reduced = reduced_plane_stress_matrix(material)
    stress = reduced @ np.asarray(strain6, dtype=float)
    if abs(stress[2]) > 1.e-12:
        raise ValueError("RAWAT_PLANE_STRESS_SIGMA_ZZ_NONZERO")
    return stress, hyperstress, displacement


def cuf_reduced_energy(strain6, gradient6x3, electric3, material):
    _only_uncoupled_macro(material)
    result = dict(ORIGINAL_CUF_ENERGY(
        strain6, gradient6x3, electric3, material))
    strain = np.asarray(strain6, dtype=float)
    lam = material.lame_lambda
    normal = lam + 2. * material.shear_modulus
    eliminated_energy = .5 * normal * (
        strain[2] + lam / normal * (strain[0] + strain[1])) ** 2
    result["classical"] -= float(eliminated_energy)
    result["total"] -= float(eliminated_energy)
    return result


@contextmanager
def diagnostic_law(theory, variant):
    if variant == "production_3d":
        yield
        return
    if variant != "plane_stress_schur":
        raise ValueError("UNKNOWN_CONSTITUTIVE_VARIANT")
    if theory == "KL":
        linear_physics._kl_linear_local_tangent = plane_stress_local_tangent
    else:
        cuf.enthalpy_derivatives = cuf_reduced_stress
        cuf.enthalpy_parts = cuf_reduced_energy
    try:
        yield
    finally:
        linear_physics._kl_linear_local_tangent = ORIGINAL_KL_TANGENT
        cuf.enthalpy_derivatives = ORIGINAL_CUF_STRESS
        cuf.enthalpy_parts = ORIGINAL_CUF_ENERGY


def run(theory, grid):
    if theory not in ("KL", "CUF_TE1") or grid not in (21, 33):
        raise ValueError("EXPECTED_KL_OR_CUF_TE1_21_OR_33")
    source = BASE / f"RAWAT2016_{theory}_17x17_MODAL_FDM.json"
    case0 = json.loads(source.read_text(encoding="utf-8"))
    if case0["method"] != "FDM" or case0["calculation_kind"] != "MODAL":
        raise ValueError("NOT_A_REFERENCE_FDM_MODAL_CASE")
    report = {
        "status": "INCOMPLETE", "purpose": "Rawat transverse Poisson constraint",
        "theory": theory, "grid": [grid, grid],
        "boundary": "C-C",
        "material": case0["material"], "geometry": case0["geometry"],
        "inertia_identical_between_variants": True,
        "operator": "SBP42_FOURIER_ODD_CSR",
        "production_source_changed": False,
        "independent_axial_m1_identification": False,
        "cuf_te1_plane_stress_null_mode_screen_required": theory == "CUF_TE1",
        "published_rawat_m1_hz": REFERENCE,
        "runs": [],
    }
    output = ROOT / f"rawat_poisson_{theory}_{grid}x{grid}.json"
    for variant in ("production_3d", "plane_stress_schur"):
        case = copy.deepcopy(case0)
        case["fdm_grid"] = {
            "nx": grid, "ns": grid, "operator_scheme": "SBP42_FOURIER_ODD_CSR"}
        case["modal"]["n_modes"] = 10
        case["output_dir"] = f"OUTPUT_RAWAT_POISSON_{theory}_{variant}_{grid}"
        row = {"variant": variant, "status": "ERROR"}
        started = time.monotonic()
        print(f"[RAWAT-POISSON] START {theory} {variant} {grid}x{grid}",
              flush=True)
        try:
            with diagnostic_law(theory, variant):
                solved = solve_modal_case(case)
            families = solved.get("harmonic_family_scan")
            if not isinstance(families, list):
                raise ValueError("MISSING_PHYSICAL_FOURIER_SECTOR_SCAN")
            by_n = {}
            for n in REFERENCE:
                candidates = [r for r in families if
                              int(r["n"]) == n and
                              int(r["axial_mode_index"]) == 1]
                if len(candidates) != 1:
                    raise ValueError(f"INVALID_SECTOR_COUNT_N_{n}")
                by_n[str(n)] = {
                    "frequency_hz": float(candidates[0]["frequency_hz"]),
                    "independent_axial_mode_match": False}
            eigenvalues = np.asarray(solved["frequencies_hz"], float)
            if (len(eigenvalues) != 10 or not np.isfinite(eigenvalues).all()
                    or np.any(eigenvalues <= 0)):
                raise ValueError("NONPOSITIVE_OR_INCOMPLETE_GLOBAL_SPECTRUM")
            modes = np.asarray(solved["mechanical_modes"], float)
            artifact = f"rawat_poisson_{theory}_{variant}_{grid}x{grid}_fields.npz"
            np.savez_compressed(
                ROOT / artifact, frequencies_hz=eigenvalues,
                mechanical_modes=modes,
                circumferential_wavenumbers=np.asarray(
                    solved.get("circumferential_wavenumbers", []), int))
            row.update({
                "status": "COMPUTED_CONSTITUTIVE_DIAGNOSTIC_ONLY",
                "by_n": by_n,
                "first_ten_frequencies_hz": eigenvalues.tolist(),
                "physical_sector_scan": True,
                "global_spectrum_completeness_verified": bool(
                    solved.get("global_spectrum_completeness_verified", False)),
                "artifact": artifact})
        except Exception as exc:
            row["error"] = repr(exc)
        row["elapsed_s"] = round(time.monotonic() - started, 3)
        report["runs"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n",
                          encoding="utf-8")
        print("[RAWAT-POISSON] RESULT " + json.dumps({
            "theory": theory, "grid": [grid, grid], **row}, ensure_ascii=False),
            flush=True)
    if all(r["status"] == "COMPUTED_CONSTITUTIVE_DIAGNOSTIC_ONLY"
           for r in report["runs"]):
        original, reduced = report["runs"]
        report["comparisons"] = [{
            "n": n,
            "production_3d_hz": original["by_n"][str(n)]["frequency_hz"],
            "plane_stress_hz": reduced["by_n"][str(n)]["frequency_hz"],
            "frequency_change_percent_relative_to_3d": 100. * (
                reduced["by_n"][str(n)]["frequency_hz"] /
                original["by_n"][str(n)]["frequency_hz"] - 1.),
            "published_rawat_m1_hz": REFERENCE[n],
        } for n in REFERENCE]
        report["status"] = "COMPUTED_DIAGNOSTIC_NOT_MODE_VALIDATED"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[RAWAT-POISSON] SUMMARY " + json.dumps({
        "theory": theory, "grid": [grid, grid],
        "status": report["status"],
        "comparisons": report.get("comparisons", [])}), flush=True)
    return 0 if report["status"] == "COMPUTED_DIAGNOSTIC_NOT_MODE_VALIDATED" else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: modal_rawat_poisson_diagnostic.py KL|CUF_TE1 21|33")
    raise SystemExit(run(sys.argv[1], int(sys.argv[2])))
