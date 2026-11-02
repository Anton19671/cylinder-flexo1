#!/usr/bin/env python3
"""Eight lowest physical cylinder frequencies: full FDM vs MVI-1..8.

Runs actual KL shell operators, identical axial C-S traces, same SBP42 grid.
Records failures and partial converged modal frequencies, never fabricates
remaining modes. The FDM solver is the independent full-space DISCRETE
reference, not an external FEM / literature validation.
"""
from __future__ import annotations

import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cylinder_paths import activate
activate()
from cylinder_modal_reference import (
    _modal_spec, _component_separation_ranks, solve_modal_case)
from cylinder_modal_mvi import _SeparatedModalProblem
from modal_mvi_physical_smoke import reference_case

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "modal_mvi_spectrum_audit_result.json"
N = 8


def physical_case():
    case = reference_case()
    case["modal"]["n_modes"] = N
    case["reference_mvi"]["max_sweeps"] = 32
    return case


def make_case(base, method, rank=None):
    case = copy.deepcopy(base)
    case["method"] = method
    if method == "FDM":
        case["fdm_grid"] = case.pop("mvi_grid")
        case.pop("rank", None)
        case.pop("reference_mvi", None)
        case["modal"].pop("mode_tolerance", None)
    else:
        case["rank"] = rank
    return case


def harmonic_numbers(fields):
    a = np.asarray(fields, float)
    spectra = np.abs(np.fft.rfft(a, axis=-1))**2
    return np.argmax(np.sum(spectra, axis=(1, 2)), axis=-1).astype(int).tolist()


def mass_actions(fields, f, density):
    z = np.zeros_like(fields[0])
    return np.stack([f.mass_action(q, density_kg_m3=density,
                                  mechanical=z, velocity=z)
                     for q in fields])


def overlap_matrices(reference, trial, f, density):
    mref = mass_actions(reference, f, density)
    mtrial = mass_actions(trial, f, density)
    qref = reference.reshape(len(reference), -1)
    qtrial = trial.reshape(len(trial), -1)
    gram = qref @ mtrial.reshape(len(trial), -1).T
    norms_ref = np.einsum("ij,ij->i", qref, mref.reshape(len(reference), -1))
    norms_trial = np.einsum("ij,ij->i", qtrial, mtrial.reshape(len(trial), -1))
    mac = gram**2 / (norms_ref[:, None] * norms_trial[None, :])
    self_gram = qtrial @ mtrial.reshape(len(trial), -1).T
    self_mac = self_gram**2 / (norms_trial[:, None] * norms_trial[None, :])
    return mac, self_mac

def degenerate_subspace_mass_mac(reference, trial, f, density):
    """Squared mass-metric principal overlaps of two equally sized subspaces.

    A member of a degenerate pair may rotate within its exact eigenspace, so
    individual-vector MAC is not a valid pass/fail measure for that pair.
    """
    mref = mass_actions(reference, f, density)
    mtrial = mass_actions(trial, f, density)
    qref = reference.reshape(len(reference), -1)
    qtrial = trial.reshape(len(trial), -1)
    norms_ref = np.einsum("ij,ij->i", qref, mref.reshape(len(reference), -1))
    norms_trial = np.einsum("ij,ij->i", qtrial, mtrial.reshape(len(trial), -1))
    gram = qref @ mtrial.reshape(len(trial), -1).T
    normalized_gram = gram / np.sqrt(norms_ref[:, None] * norms_trial[None, :])
    # The eigenvalues of GᵀG are squared cosines of principal angles.
    values = np.linalg.eigvalsh(normalized_gram.T @ normalized_gram)
    return np.clip(values[::-1], 0.0, 1.0)


def check_spectrum_acceptance(report, *,
                              max_relative_frequency_difference=2.e-5,
                              min_principal_mass_mac=0.999,
                              max_mass_orthogonality_mac=1.e-6,
                              max_stiffness_operator_difference=1.e-6):
    """Require physical eight-mode FDM agreement, not merely completed jobs.

    A repeated eigenvalue has an arbitrary orthogonal rotation of its
    cosine/sine pair. Only the two-dimensional mass-metric eigenspace is
    invariant: pairwise vector MAC is diagnostic, never the pass/fail rule.
    This verifier inspects EXISTING audit outputs; it cannot change K, M,
    MVI iteration, eigenvalues, modes, ranks, or acceptance tolerances.
    """
    issues = []
    all_names = ["MKR_FDM"] + [f"MVI_{rank}" for rank in range(1, 9)]
    for name in all_names:
        if report.get("runs", {}).get(name, {}).get("status") != "SOLVED":
            issues.append(name + ":NOT_SOLVED")
    fdm_orth = report.get("fdm_max_offdiag_mass_mac")
    if (fdm_orth is None or not np.isfinite(fdm_orth)
            or fdm_orth > max_mass_orthogonality_mac):
        issues.append("FDM:MASS_ORTHOGONALITY_NOT_VERIFIED")
    for name in all_names[1:]:
        comparison = report.get("comparisons", {}).get(name)
        if comparison is None:
            issues.append(name + ":COMPARISON_MISSING")
            continue
        rows = comparison.get("by_frequency_index", [])
        if len(rows) != 8:
            issues.append(name + ":EIGHT_FREQUENCIES_NOT_COMPARED")
        else:
            for row in rows:
                percent = row.get("difference_percent")
                if (percent is None or not np.isfinite(percent)
                        or percent > 100. * max_relative_frequency_difference):
                    issues.append(name + ":FREQUENCY_MISMATCH_MODE_" +
                                  str(row.get("mode")))
        orthogonality = comparison.get("max_offdiag_mass_mac")
        if (orthogonality is None or not np.isfinite(orthogonality)
                or orthogonality > max_mass_orthogonality_mac):
            issues.append(name + ":MASS_ORTHOGONALITY_NOT_VERIFIED")
        pairs = comparison.get("degenerate_pair_subspaces", {})
        if set(pairs) != {"1-2", "3-4", "5-6", "7-8"}:
            issues.append(name + ":DOUBLET_SUBSPACE_MISSING")
            continue
        for key, pair in pairs.items():
            values = np.asarray(pair.get("principal_mass_mac", []), float)
            if (values.shape != (2,) or not np.isfinite(values).all()
                    or np.min(values) < min_principal_mass_mac):
                issues.append(name + ":DOUBLET_SUBSPACE_FAILED_" + key)
    probe = report.get("stiffness_operator_probe", {}).get(
        "relative_norm_difference")
    if (probe is None or not np.isfinite(probe)
            or probe > max_stiffness_operator_difference):
        issues.append("K_OPERATOR_NOT_VERIFIED")
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "max_relative_frequency_difference": max_relative_frequency_difference,
        "minimum_required_principal_mass_mac": min_principal_mass_mac,
        "max_mass_orthogonality_mac": max_mass_orthogonality_mac,
        "max_stiffness_operator_difference": max_stiffness_operator_difference,
        "scope": "same_discretization_FDM_vs_fixed_rank_MVI",
    }


def main():
    start = time.monotonic()
    base = physical_case()
    report = {
        "scope": "KL_C_S_9x9_SBP42_FIRST_EIGHT_PHYSICAL_MODES",
        "validation": "same_discretization_not_external_validation",
        "n_requested": N,
        "grid": base["mvi_grid"],
        "geometry": base["geometry"],
        "material": base["material"],
        "boundary_conditions": base["boundary_conditions"],
        "density_kg_m3": base["modal"]["density_kg_m3"],
        "runs": {},
        "comparisons": {},
        "all_nine_solved": False,
        "all_spectra_verified": False,
        "spectrum_acceptance": {"status": "NOT_RUN", "issues": []},
    }
    physical = {}
    result = {}
    try:
        full_case = make_case(base, "FDM")
        print("[SPECTRUM] START MKR_FDM full discrete eigensystem", flush=True)
        t = time.monotonic()
        try:
            r = solve_modal_case(full_case)
            fq = np.asarray(r["frequencies_hz"], dtype=float)
            q = np.asarray(r["mechanical_modes"], dtype=float)
            if len(fq) != N or len(q) != N:
                raise ValueError(f"FDM returned {len(fq)} instead of {N} modes")
            result["MKR_FDM"] = (fq, q)
            report["runs"]["MKR_FDM"] = {
                "status": "SOLVED", "elapsed_s": round(time.monotonic()-t, 3),
                "frequencies_hz": fq.tolist(),
                "wavenumbers": harmonic_numbers(q),
                "component_shape_ranks": _component_separation_ranks(q)}
            print("[SPECTRUM] RESULT MKR_FDM " +
                  json.dumps(report["runs"]["MKR_FDM"]), flush=True)
        except Exception as exc:
            report["runs"]["MKR_FDM"] = {
                "status": "FAILED", "error": repr(exc),
                "traceback": traceback.format_exc(limit=6)}
            print("[SPECTRUM] FAILED MKR_FDM: " + repr(exc), flush=True)

        # All rank configurations are independent solves on the same physics.
        for rank in range(1, 9):
            name = f"MVI_{rank}"
            mode_events = []
            branch_events = []
            t = time.monotonic()
            print("[SPECTRUM] START " + name, flush=True)

            def details(**event):
                stage = event.get("stage")
                if stage == "MVI_MODE_DONE":
                    data = {k: event.get(k) for k in (
                        "mode", "frequency_hz", "sweep", "rank")}
                    mode_events.append(data)
                    print("[SPECTRUM] MODE " + name + " " +
                          json.dumps(data), flush=True)
                if (int(event.get("mode", 0)) >= 7 and stage in (
                        "MVI_CANDIDATE_INITIAL_STATE",
                        "MVI_CANDIDATE_INITIAL_X_DONE",
                        "MVI_HALFSTEP_HARMONIC_AUDIT",
                        "MVI_CANDIDATE_REJECTED",
                        "MVI_CANDIDATE_SELECTED")):
                    branch_events.append({k: event.get(k) for k in (
                        "stage", "mode", "attempt", "seed", "sweep", "axis",
                        "frequency_hz", "circumferential_harmonic",
                        "circumferential_harmonic_energy_fraction",
                        "max_previous_mass_overlap", "reason")})
            try:
                r = solve_modal_case(
                    make_case(base, "MVI", rank),
                    detail_callback=details)
                fq = np.asarray(r["frequencies_hz"], dtype=float)
                q = np.asarray(r["mechanical_modes"], dtype=float)
                if len(fq) != N or len(q) != N:
                    raise ValueError(f"{name} returned {len(fq)} instead of {N} modes")
                result[name] = (fq, q)
                report["runs"][name] = {
                    "status": "SOLVED",
                    "elapsed_s": round(time.monotonic()-t, 3),
                    "frequencies_hz": fq.tolist(),
                    "wavenumbers": harmonic_numbers(q),
                    "component_shape_ranks": _component_separation_ranks(q),
                    "global_mode_ranks": r.get("global_mode_ranks"),
                    "mvi_sweeps": r.get("mvi_sweeps"),
                    "xy_directions": r.get("mvi_final_xy_trial_dimensions"),
                    "mvi_multistart_audit": r.get("mvi_multistart_audit"),
                    "modes_7_8_branch_audit": branch_events,
                    "global_spectrum_completeness_verified":
                        bool(r.get("global_spectrum_completeness_verified", False))}
                print("[SPECTRUM] RESULT " + name + " " +
                      json.dumps(report["runs"][name]), flush=True)
            except Exception as exc:
                report["runs"][name] = {
                    "status": "FAILED", "elapsed_s": round(time.monotonic()-t, 3),
                    "completed_modes_before_failure": mode_events,
                    "error": repr(exc), "traceback": traceback.format_exc(limit=8)}
                print("[SPECTRUM] FAILED " + name + " " +
                      json.dumps(report["runs"][name]), flush=True)
            # Save after every rank for diagnosis if a CI job is interrupted.
            OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")

        if "MKR_FDM" in result:
            from cylinder_modal_reference import _modal_spec
            prepared, density, _ = _modal_spec(make_case(base, "FDM"))
            f = prepared.problem.functional
            ref_freq, ref_shapes = result["MKR_FDM"]
            _, self_m = overlap_matrices(ref_shapes, ref_shapes, f, density)
            report["fdm_max_offdiag_mass_mac"] = float(np.max(
                np.abs(self_m - np.diag(np.diag(self_m)))))
            for rank in range(1, 9):
                name = f"MVI_{rank}"
                if name not in result:
                    continue
                freq, shapes = result[name]
                mac, self_mac = overlap_matrices(ref_shapes, shapes, f, density)
                rows, cols = linear_sum_assignment(-mac)
                matches = sorted([{
                    "fdm_mode": int(i + 1), "mvi_mode": int(j + 1),
                    "fdm_hz": float(ref_freq[i]),
                    "mvi_hz": float(freq[j]),
                    "difference_percent": float(
                        100. * abs(ref_freq[i]-freq[j]) / ref_freq[i]),
                    "mass_mac": float(mac[i, j])}
                    for i, j in zip(rows, cols)],
                    key=lambda e: e["fdm_mode"])
                degenerate_pairs = {}
                for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
                    principal_mac = degenerate_subspace_mass_mac(
                        ref_shapes[[left, right]], shapes[[left, right]], f, density)
                    degenerate_pairs[f"{left + 1}-{right + 1}"] = {
                        "principal_mass_mac": principal_mac.tolist(),
                        "minimum_principal_mass_mac": float(np.min(principal_mac)),
                        "subspace_verified": bool(np.min(principal_mac) >= 0.999)}
                report["comparisons"][name] = {
                    "by_frequency_index": [
                        {"mode": i+1, "fdm_hz": float(ref_freq[i]),
                         "mvi_hz": float(freq[i]),
                         "difference_percent": float(
                             100.*abs(ref_freq[i]-freq[i])/ref_freq[i]),
                         "mass_mac_same_index": float(mac[i,i])}
                        for i in range(N)],
                    "optimal_mass_mac_assignment": matches,
                    "degenerate_pair_subspaces": degenerate_pairs,
                    "max_offdiag_mass_mac": float(np.max(
                        np.abs(self_mac-np.diag(np.diag(self_mac))))),
                    "mac_matrix": mac.tolist()}
                print("[SPECTRUM] COMPARISON " + name + " " +
                      json.dumps({
                          "frequency_difference_percent": [
                              x["difference_percent"] for x in
                              report["comparisons"][name]["by_frequency_index"]],
                          "optimal_matches": matches,
                          "max_self_mac": report["comparisons"][name][
                              "max_offdiag_mass_mac"]}), flush=True)

            # Compare a nonsingle-harmonic admissible field, probing the
            # exact weak residual vs the local-tangent MVI stiffness action.
            if len(ref_shapes) >= 3:
                field = (ref_shapes[0] + .47*ref_shapes[1]
                         + .21*ref_shapes[2])
                mvi_case = make_case(base, "MVI", 1)
                mprep, mdensity, _ = _modal_spec(mvi_case)
                op = _SeparatedModalProblem(mprep, mdensity, 1)
                direction = field.ravel()
                linstep = op.step
                fullplus = mprep.problem.evaluate_effective_residual(
                    linstep*field, None)[1]
                fullminus = mprep.problem.evaluate_effective_residual(
                    -linstep*field, None)[1]
                kref = (fullplus-fullminus)/(2*linstep)
                kmvi = op.stiffness(direction)
                report["stiffness_operator_probe"] = {
                    "relative_norm_difference": float(
                        np.linalg.norm(kref-kmvi) /
                        max(np.linalg.norm(kref), 1.e-30)),
                    "probe_is_mixture_of_first_three_fdm_modes": True}
                print("[SPECTRUM] K_OPERATOR_PROBE " +
                      json.dumps(report["stiffness_operator_probe"]), flush=True)
        report["all_nine_solved"] = all(
            report["runs"].get(name, {}).get("status") == "SOLVED"
            for name in ["MKR_FDM"] + [f"MVI_{rank}" for rank in range(1,9)])
        report["spectrum_acceptance"] = check_spectrum_acceptance(report)
        report["all_spectra_verified"] = (
            report["spectrum_acceptance"]["status"] == "PASS")
    except Exception as exc:
        report["harness_error"] = traceback.format_exc(limit=12)
        print("[SPECTRUM] AUDIT ERROR " + repr(exc), flush=True)
    finally:
        report["elapsed_s"] = round(time.monotonic()-start, 3)
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print("[SPECTRUM] SUMMARY " + json.dumps({
            "statuses": {n:v["status"] for n,v in report["runs"].items()},
            "all_nine_solved": report["all_nine_solved"],
            "all_spectra_verified": report["all_spectra_verified"],
            "acceptance_issues": report["spectrum_acceptance"]["issues"],
            "elapsed_s": report["elapsed_s"]}), flush=True)
    return 0 if report["all_nine_solved"] and report["all_spectra_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
