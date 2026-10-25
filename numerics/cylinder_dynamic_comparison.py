#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent, fail-closed comparison of *calculated* Newmark MVI/FDM runs.

A comparison is not proof of mesh/time convergence. It refuses mismatched
physics/time grids and incomplete or nonconverged runs, and never invents
field values in the absence of result NPZ files.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import csv
import math
import numpy as np

FIELDS = ("mechanical", "radial_displacement", "electrical", "velocity",
          "acceleration", "damping_force", "full_residual", "constraint_reaction")


def _finite_number(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(label + ": finite numeric value required")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(label + ": finite numeric value required") from exc
    if not math.isfinite(number):
        raise ValueError(label + ": finite numeric value required")
    return number


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected JSON object")
    return result


def _grid(case, method):
    name = "mvi_grid" if method == "MVI" else "fdm_grid"
    grid = case.get(name) or case.get("grid")
    if not isinstance(grid, dict):
        raise ValueError(f"missing {name}/grid in {method} physical CASE")
    # Only the spatial grid is physical. Differing legacy FD stencils are
    # rejected by the reference solver before a run is produced.
    return {"nx": int(grid["nx"]), "ns": int(grid["ns"])}


def _physics(case, method):
    """Strip *only* solver/search/output fields, retaining all physics."""
    omitted = {"method", "rank", "reference_mvi", "reference_newton", "output_dir",
               "mvi_grid", "fdm_grid", "grid"}
    physics = {key: deepcopy(value) for key, value in case.items()
               if key not in omitted}
    physics["_comparison_grid"] = _grid(case, method)
    return physics


def _read_completed_run(folder, method):
    folder = Path(folder).expanduser().resolve()
    summary = _read_json(folder / "FINAL_RUN_SUMMARY.json")
    if summary.get("method") != method or summary.get("calculation_kind") != "DYNAMIC":
        raise ValueError(f"{folder}: expected converged DYNAMIC {method} run")
    n = summary.get("requested_steps")
    accepted = summary.get("accepted_steps")
    steps = summary.get("steps")
    if (type(n) is not int or n < 1 or type(accepted) is not int or
        accepted != n or not isinstance(steps, list) or len(steps) != n or
        summary.get("converged") is not True or summary.get("status") != "CONVERGED"):
        raise ValueError(f"{method}: full Newmark run is incomplete or not converged")
    case = _read_json(folder / "REFERENCE_CASE.json")
    if str(case.get("method", "")).upper() != method:
        raise ValueError(f"{method}: CASE/summary method mismatch")
    frames = []
    required = {"mechanical", "velocity", "acceleration", "time_s", "load_factor",
                "full_residual", "constraint_reaction"}
    last_time = -math.inf
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict) or step.get("index") != idx or             step.get("converged") is not True or             step.get("full_equilibrium_satisfied") is not True:
            raise ValueError(f"{method} step {idx}: equilibrium or indexing invalid")
        for key in ("admissible_residual_scaled", "residual_relative_to_initial",
                    "constraint_defect_scaled"):
            if _finite_number(step.get(key), f"{method} step {idx} {key}") < 0:
                raise ValueError(f"{method} step {idx}: negative {key}")
        filename = step.get("state_file")
        if (not isinstance(filename, str) or filename != Path(filename).name or
            filename != f"REFERENCE_DYNAMIC_STEP_{idx:04d}.npz"):
            raise ValueError(f"{method} step {idx}: invalid accepted-state file name")
        with np.load(folder / filename, allow_pickle=False) as saved:
            if not required.issubset(saved.files):
                raise ValueError(f"{method} step {idx}: missing mandatory computed fields")
            frame = {key: np.array(saved[key], dtype=float).copy()
                     for key in saved.files if key in required | {"electrical", "damping_force"}}
        if any(not np.isfinite(v).all() for v in frame.values()):
            raise ValueError(f"{method} step {idx}: nonfinite field values")
        time = _finite_number(step.get("time_s"), "step time")
        factor = _finite_number(step.get("load_factor"), "step load factor")
        if time <= last_time:
            raise ValueError(f"{method} step {idx}: nonmonotonic time history")
        last_time = time
        if (frame["time_s"].shape != () or frame["load_factor"].shape != () or
            not math.isclose(time, float(frame["time_s"]), rel_tol=1.e-12, abs_tol=1.e-13) or
            not math.isclose(factor, float(frame["load_factor"]), rel_tol=1.e-12, abs_tol=1.e-13)):
            raise ValueError(f"{method} step {idx}: NPZ and summary disagree")
        if frame["mechanical"].ndim != 3 or frame["mechanical"].size == 0:
            raise ValueError(f"{method} step {idx}: invalid mechanical field shape")
        for key in ("velocity", "acceleration", "damping_force"):
            if key in frame and frame[key].shape != frame["mechanical"].shape:
                raise ValueError(f"{method} step {idx}: {key} field shape mismatch")
        if bool(summary.get("electrical_enabled")) != ("electrical" in frame):
            raise ValueError(f"{method} step {idx}: electric state missing/inconsistent")
        if "electrical" in frame and (frame["electrical"].ndim != 3 or
                                    frame["electrical"].shape[1:] != frame["mechanical"].shape[1:]):
            raise ValueError(f"{method} step {idx}: electric spatial shape mismatch")
        state_size = frame["mechanical"].size + (frame["electrical"].size if "electrical" in frame else 0)
        for key in ("full_residual", "constraint_reaction"):
            if frame[key].shape != (state_size,):
                raise ValueError(f"{method} step {idx}: {key} vector shape mismatch")
        frames.append(frame)
    return summary, case, frames


def _errors(candidate, reference):
    if candidate.shape != reference.shape:
        raise ValueError(f"computed field shapes differ: {candidate.shape} vs {reference.shape}")
    d = candidate - reference
    absolute = float(np.max(np.abs(d), initial=0))
    baseline = float(np.max(np.abs(reference), initial=0))
    norm_baseline = float(np.linalg.norm(reference.ravel()))
    return {"max_abs_difference": absolute,
            "l2_difference": float(np.linalg.norm(d.ravel())),
            "fdm_peak_abs": baseline,
            "max_rel_to_fdm_peak": None if baseline == 0 else absolute / baseline,
            "l2_rel_to_fdm": None if norm_baseline == 0 else
                             float(np.linalg.norm(d.ravel())) / norm_baseline}


def compare_dynamic_runs(mvi_folder, fdm_folder):
    """Return measured differences only if physics/mesh/time/steps all match."""
    mvi, c_mvi, states_mvi = _read_completed_run(mvi_folder, "MVI")
    fdm, c_fdm, states_fdm = _read_completed_run(fdm_folder, "FDM")
    if _physics(c_mvi, "MVI") != _physics(c_fdm, "FDM"):
        raise ValueError("MVI/FDM physics, boundary conditions, load or spatial grids differ")
    passport = ("theory", "scale_model", "geometry_model", "material_model",
                "field_order", "thickness_basis", "electrical_enabled",
                "density_kg_m3", "newmark", "damping", "requested_steps")
    for item in passport:
        if mvi.get(item) != fdm.get(item):
            raise ValueError(f"MVI/FDM summary passport differs: {item}")
    radial = 2*(int(mvi["theory"][-1])+1) if str(mvi["theory"]).startswith("CUF_") else 2
    result = {"comparison": "calculated dynamic MVI versus FDM (FDM reference)",
              "physical_case_equal": True, "spatial_grid": _grid(c_mvi, "MVI"),
              "requested_steps": mvi["requested_steps"],
              "mvi_rank": mvi.get("rank"),
              "time_step_s": _finite_number(mvi["newmark"]["dt_s"], "dt_s"),
              "convergence_study_verified": False, "steps": []}
    for index, (record_mvi, record_fdm, a, b) in enumerate(
        zip(mvi["steps"], fdm["steps"], states_mvi, states_fdm), 1):
        t_mvi, t_fdm = float(a["time_s"]), float(b["time_s"])
        if not math.isclose(t_mvi, t_fdm, rel_tol=1.e-12, abs_tol=1.e-13):
            raise ValueError(f"step {index}: physical time mismatch")
        if not math.isclose(float(a["load_factor"]), float(b["load_factor"]),
                            rel_tol=1.e-12, abs_tol=1.e-13):
            raise ValueError(f"step {index}: applied load history mismatch")
        metrics = {}
        for key in ("mechanical", "electrical", "velocity", "acceleration",
                    "damping_force", "full_residual", "constraint_reaction"):
            if (key in a) != (key in b):
                raise ValueError(f"step {index}: missing {key} in one method")
            if key in a:
                metrics[key] = _errors(a[key], b[key])
        if a["mechanical"].shape[0] <= radial:
            raise ValueError("radial displacement component not present")
        metrics["radial_displacement"] = _errors(a["mechanical"][radial],
                                                  b["mechanical"][radial])
        result["steps"].append({
            "index": index, "time_s": t_mvi, "load_factor": float(a["load_factor"]),
            "field_errors": metrics,
            "mvi_admissible_residual_scaled": record_mvi["admissible_residual_scaled"],
            "fdm_admissible_residual_scaled": record_fdm["admissible_residual_scaled"],
            "mvi_constraint_defect_scaled": record_mvi["constraint_defect_scaled"],
            "fdm_constraint_defect_scaled": record_fdm["constraint_defect_scaled"],
            "mvi_projected_residual_scaled": record_mvi.get("projected_residual_scaled"),
        })
    return result


def export_comparison(result, json_path, csv_path=None):
    """Persist numerical errors with explicit nullable relative metrics."""
    output = Path(json_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("step", "time_s", "field", "max_abs_difference",
                             "fdm_peak_abs", "max_rel_to_fdm_peak",
                             "l2_difference", "l2_rel_to_fdm"))
            for step in result["steps"]:
                for name, errors in step["field_errors"].items():
                    writer.writerow((step["index"], step["time_s"], name,
                                     errors["max_abs_difference"], errors["fdm_peak_abs"],
                                     errors["max_rel_to_fdm_peak"], errors["l2_difference"],
                                     errors["l2_rel_to_fdm"]))
    return output


def run_dynamic_pair(case_path, output_root, *, rank=None):
    """Execute matched physical CASE twice using actual reference solvers.

    Explicitly invokes MVI and FDM, never substitutes a solution from the
    other method, and will not overwrite any previous output or CASE file.
    """
    from cylinder_reference_case import prepare_reference_case, run_reference_dynamic_case

    case_path, output_root = Path(case_path).resolve(), Path(output_root).resolve()
    original = _read_json(case_path)
    if str(original.get("calculation_kind", "")).upper() != "DYNAMIC":
        raise ValueError("matched validation requires a DYNAMIC CASE")
    original_method = str(original.get("method", "")).upper()
    if original_method not in ("MVI", "FDM"):
        raise ValueError("original CASE method must be MVI/FDM")
    # Derived CASE files live in output_root, not alongside the original.
    # Resolve relative CSV time-law inputs before relocation, or the same
    # physical experiment would silently open a different input file.
    def resolve_table(law):
        if isinstance(law, dict) and "csv_path" in law:
            raw = Path(str(law["csv_path"])).expanduser()
            law["csv_path"] = str((case_path.parent / raw).resolve()
                                  if not raw.is_absolute() else raw.resolve())
    resolve_table((original.get("dynamics") or {}).get("time_law"))
    for family in ("mechanical", "electrical"):
        sites = (((original.get("boundary_conditions") or {}).get("variational") or {})
                 .get(family, {}))
        for site in sites.values():
            if isinstance(site, dict):
                resolve_table(site.get("time_law"))
    base_grid = _grid(original, original_method)
    mvi_rank = rank if rank is not None else original.get("rank", 1)
    if type(mvi_rank) is not int or not 1 <= mvi_rank <= 8:
        raise ValueError("MVI rank must be integer 1..8")
    candidates = {}
    for method in ("MVI", "FDM"):
        c = deepcopy(original)
        for key in ("rank", "reference_mvi", "grid", "mvi_grid", "fdm_grid"):
            c.pop(key, None)
        c["method"] = method
        c["mvi_grid" if method == "MVI" else "fdm_grid"] = base_grid
        if method == "MVI":
            c["rank"] = mvi_rank
            c["reference_mvi"] = deepcopy(original.get("reference_mvi", {"max_sweeps": 30}))
        c["output_dir"] = str(output_root / method)
        # Preflight both before making output directories.
        prepare_reference_case(c, base_dir=case_path.parent)
        candidates[method] = c
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("comparison output root must be empty (no overwrites)")
    output_root.mkdir(parents=True, exist_ok=True)
    for method, c in candidates.items():
        target = output_root / ("CASE_" + method + ".json")
        target.write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for method, c in candidates.items():
        case_file = output_root / ("CASE_" + method + ".json")
        summary = run_reference_dynamic_case(c, case_file)
        if summary.get("converged") is not True:
            raise RuntimeError(f"{method} did not converge: {summary.get('status')}; "
                               f"results retained in {output_root / method}")
    result = compare_dynamic_runs(output_root / "MVI", output_root / "FDM")
    export_comparison(result, output_root / "COMPARISON.json",
                      output_root / "COMPARISON.csv")
    return result


__all__ = ("compare_dynamic_runs", "export_comparison", "run_dynamic_pair")
