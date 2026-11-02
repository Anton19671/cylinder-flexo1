#!/usr/bin/env python3
"""Independent modal literature diagnostics without changing physical solvers.

rawat: CUF TE1 C-C SBP42 FDM grid sensitivity for n=2,5.
cammalleri: locate the smallest requested modal cutoff which fails the
full Fourier eigenpair certificate on the same fixed thin-shell problem.

A successful diagnostic run is NOT a literature validation certificate.
Scientific comparisons remain explicitly UNVERIFIED until the axial mode
number and spatial/grid convergence are established independently.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from cylinder_paths import activate
activate()

from cylinder_modal_reference import solve_modal_case
from modal_mvi_rawat_te1_cc_audit import harmonic_numbers

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "verification" / "rawat2016" / "modal_linear"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _persist(name, report):
    path = ROOT / ("modal_literature_diagnostic_" + name + ".json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print("[LITERATURE-DIAG] " + name + " " +
          json.dumps(report, ensure_ascii=False), flush=True)


def _solve(case):
    result = solve_modal_case(case)
    frequencies = np.asarray(result["frequencies_hz"], dtype=float)
    fields = np.asarray(result["mechanical_modes"], dtype=float)
    numbers = result.get("circumferential_wavenumbers")
    if numbers is None:
        numbers = harmonic_numbers(fields)
    if (len(frequencies) != len(fields) or
            len(frequencies) != len(numbers) or
            not np.isfinite(frequencies).all() or
            not np.isfinite(fields).all() or
            np.any(frequencies <= 0.)):
        raise ValueError("LITERATURE_DIAGNOSTIC_INVALID_MODAL_SPECTRUM")
    return result, frequencies, fields, list(map(int, numbers))


def axial_harmonic_signature(radial_field, n):
    """Rotation-invariant axial-shape *diagnostic* for a radial Fourier mode.

    Circumferential phase is arbitrary inside a doublet. Extract the exact
    complex n-Fourier coefficient then take the dominant real direction of
    its 2x2 Gramian to estimate axial internal nodes. A lack of internal
    nodes is consistent with a first axial branch, but DOES NOT establish
    the physical shell-mode label m=1.
    """
    radial = np.asarray(radial_field, dtype=float)
    if (radial.ndim != 2 or min(radial.shape) < 5 or
            not np.isfinite(radial).all() or
            type(n) is not int or not 0 < n < radial.shape[1] // 2 + 1):
        raise ValueError("RAWAT_INVALID_AXIAL_FOURIER_FIELD")
    coefficient = np.fft.rfft(radial, axis=1)[:, n]
    phases = np.stack((coefficient.real, coefficient.imag), axis=1)
    gram = phases.T @ phases
    vals, frame = np.linalg.eigh((gram + gram.T) * 0.5)
    total = float(np.sum(vals))
    if not np.isfinite(total) or total <= 1.e-24:
        raise ValueError("RAWAT_RADIAL_HARMONIC_COMPONENT_ABSENT")
    axial = phases @ frame[:, -1]
    peak = float(np.max(np.abs(axial)))
    if peak <= 1.e-15:
        raise ValueError("RAWAT_RADIAL_AXIAL_PROFILE_ABSENT")
    interior = axial[1:-1]
    clear = interior[np.abs(interior) >= 1.e-3 * peak]
    signs = np.sign(clear)
    nodes = int(np.sum(signs[1:] * signs[:-1] < 0))
    coherence = float(np.clip(vals[-1] / total, 0.0, 1.0))
    return {
        "n": n,
        "radial_internal_axial_nodes_estimate": nodes,
        "axial_fourier_phase_coherence": coherence,
        "axial_profile": (axial / peak).tolist(),
        "axial_m1_verified": False,
        "axial_m1_candidate": bool(nodes == 0 and coherence >= 0.99),
    }


def _axial_zero_crossing_estimate(field, n):
    """Backward-compatible scalar diagnostic, never an m=1 certificate."""
    return axial_harmonic_signature(field, int(n))[
        "radial_internal_axial_nodes_estimate"]


def run_rawat():
    reference = {2: 342., 5: 352.}
    report = {
        "purpose": "Rawat2016 m=1 n=2,5 grid and axial-shape diagnostic",
        "theory": "CUF_TE1", "method": "FDM",
        "boundary": "C-C", "operator": "SBP42_FOURIER_ODD_CSR",
        "published_hz": reference, "grids": [],
        "axial_index_verified": False, "literature_validation_verified": False,
        "mesh_convergence_verified": False,
    }
    base = _read_json(CONFIG / "RAWAT2016_CUF_TE1_17x17_MODAL_FDM.json")
    issues = []
    for size in (13, 17, 21):
        case = _read_json(
            CONFIG / "fdm_13x13" /
            "RAWAT2016_CUF_TE1_13x13_MODAL_FDM.json") if size == 13 else json.loads(
                json.dumps(base))
        case["fdm_grid"] = {"nx": size, "ns": size,
                            "operator_scheme": "SBP42_FOURIER_ODD_CSR"}
        case["modal"]["n_modes"] = 12
        case["output_dir"] = "OUTPUT_RAWAT2016_GRID_CHECK_" + str(size)
        row = {"grid": [size, size], "status": "ERROR"}
        started = time.monotonic()
        try:
            result, frequencies, modes, numbers = _solve(case)
            row["frequencies_hz"] = [float(v) for v in frequencies]
            row["circumferential_wavenumbers"] = numbers
            row["global_spectrum_completeness_verified"] = bool(
                result.get("global_spectrum_completeness_verified", False))
            row["matched_harmonics"] = {}
            for n, published in reference.items():
                indices = [i for i, nn in enumerate(numbers) if nn == n]
                if not indices:
                    raise ValueError("LITERATURE_DIAGNOSTIC_MISSING_N" + str(n))
                index = indices[0]
                value = float(frequencies[index])
                w0 = modes[index, 4]  # CUF TE1 w_0 is the fifth component
                row["matched_harmonics"][str(n)] = {
                    "mode_index": index + 1,
                    "frequency_hz": value,
                    "published_m1_hz": published,
                    "difference_percent": 100. * (value / published - 1.),
                    "radial_component_axial_zero_crossings_estimate":
                        _axial_zero_crossing_estimate(w0, n),
                    "axial_m_verified": False,
                }
            row["status"] = "SOLVED_AXIAL_INDEX_UNVERIFIED"
        except Exception as exc:
            row["error"] = repr(exc)
            issues.append("GRID_" + str(size) + ":" + type(exc).__name__)
        finally:
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            report["grids"].append(row)
            report["issues"] = issues
            _persist("rawat_grid", report)

    if len(report["grids"]) == 3 and not issues:
        rows = report["grids"]
        report["frequency_changes_percent"] = {
            str(n): [
                100. * (rows[i]["matched_harmonics"][str(n)]["frequency_hz"] /
                        rows[i-1]["matched_harmonics"][str(n)]["frequency_hz"] - 1.)
                for i in (1, 2)
            ]
            for n in reference
        }
        report["diagnostic_completed"] = True
    else:
        report["diagnostic_completed"] = False
    _persist("rawat_grid", report)
    return 0 if report["diagnostic_completed"] else 1


def run_cammalleri():
    case = _read_json(CONFIG / "fdm_13x13" /
                     "RAWAT2016_KL_13x13_MODAL_FDM.json")
    case["geometry"] = {"L": .305, "R": .076, "h": .000254}
    case["material"] = {"E0": 207e9, "nu0": .3}
    case["modal"]["density_kg_m3"] = 7833
    case["fdm_grid"] = {
        "nx": 13, "ns": 21, "operator_scheme": "SBP42_FOURIER_ODD_CSR"}
    report = {
        "purpose": "Cammalleri2023 first failing Fourier eigenpair cutoff",
        "theory": "KL", "method": "FDM", "grid": [13, 21],
        "model": "physical FULL_FOURIER_BLOCK_GENERALIZED_EIGEN",
        "results": [], "all_cutoffs_verified": False,
        "literature_validation_verified": False,
    }
    for modes in (2, 4, 8, 12):
        case["modal"]["n_modes"] = modes
        case["output_dir"] = "OUTPUT_CAMMALLERI_CUTOFF_" + str(modes)
        row = {"requested_modes": modes, "status": "ERROR"}
        started = time.monotonic()
        try:
            result, frequencies, _, numbers = _solve(case)
            row.update(
                status="EIGENPAIRS_VERIFIED",
                frequencies_hz=[float(v) for v in frequencies],
                circumferential_wavenumbers=numbers,
                global_spectrum_completeness_verified=bool(
                    result.get("global_spectrum_completeness_verified", False)))
        except Exception as exc:
            # The original physical-eigenpair gate remains strict. The
            # enhanced error includes its physical / projected indicators.
            row["error"] = repr(exc)
        finally:
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            report["results"].append(row)
            _persist("cammalleri_cutoff", report)
    report["all_cutoffs_verified"] = all(
        item["status"] == "EIGENPAIRS_VERIFIED" for item in report["results"])
    report["diagnostic_completed"] = True
    _persist("cammalleri_cutoff", report)
    # Diagnosis itself is always complete if all four cutoffs were attempted.
    # Spectral validation MUST inspect all_cutoffs_verified separately.
    return 0


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("rawat", "cammalleri"):
        raise ValueError("Expected rawat or cammalleri")
    return run_rawat() if sys.argv[1] == "rawat" else run_cammalleri()


if __name__ == "__main__":
    raise SystemExit(main())
