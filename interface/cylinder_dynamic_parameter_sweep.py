#!/usr/bin/env python3
"""Reproducible independent dynamic parameter sweeps for cylindrical shells.

Every calculation invokes the existing nonlinear elastic Newmark engine.
The sweep never substitutes a reduced oscillator, changes the MVI rank, or
claims branch-following continuation. Every candidate is preflighted before
output creation, and incomplete histories never enter a bifurcation diagram.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

# Only one scalar physical input can vary. Theory, grid, MVI rank, boundary
# topology, time step and integration settings are held fixed per series.
ALLOWED_PARAMETERS = frozenset({
    "load_definition.q",
    "dynamics.time_law.amplitude",
    "dynamics.time_law.frequency_hz",
    "dynamics.time_law.phase_rad",
    "material.E0",
    "material.nu0",
    "micro.ell_m",
    "micro.mu1_c_per_m",
    "micro.mu2_c_per_m",
    "micro.kappa_f_per_m",
    "dynamics.density_kg_m3",
})
_SPEC_KEYS = frozenset({
    "base_case", "parameter_path", "values", "output_dir", "parameter_name",
    "probe_name", "transient_fraction", "forcing_hz", "export_each_run",
})


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: real JSON number required")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}: number must be finite")
    return number


def _read_object(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON object required")
    return value


def _atomic_json(path, content):
    target = Path(path)
    temp = target.with_name(target.name + ".tmp")
    with temp.open("w", encoding="utf-8") as fp:
        json.dump(content, fp, ensure_ascii=False, indent=2, allow_nan=False)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temp, target)


def _set_scalar(case, path, value):
    obj = case
    keys = path.split(".")
    for key in keys[:-1]:
        if not isinstance(obj, dict) or key not in obj:
            raise ValueError(f"parameter field {path!r} does not exist")
        obj = obj[key]
    if not isinstance(obj, dict) or keys[-1] not in obj:
        raise ValueError(f"parameter field {path!r} does not exist")
    _finite_number(obj[keys[-1]], path + " initial value")
    obj[keys[-1]] = float(value)


def _absolutize_csv_inputs(node, origin):
    """Keep tabulated loads and moving boundary conditions valid on relocation."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "csv_path" and isinstance(value, str):
                path = Path(value).expanduser()
                node[key] = str((origin / path).resolve() if not path.is_absolute()
                                else path.resolve())
            else:
                _absolutize_csv_inputs(value, origin)
    elif isinstance(node, list):
        for item in node:
            _absolutize_csv_inputs(item, origin)


def plan_parameter_sweep(spec_file):
    """Build all candidate cases without side effects or solving equations."""
    spec_path = Path(spec_file).expanduser().resolve()
    spec = _read_object(spec_path)
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ValueError(f"unsupported sweep specification keys: {sorted(unknown)}")
    base_ref = spec.get("base_case")
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError("base_case must point to a JSON CASE file")
    base_path = Path(base_ref).expanduser()
    if not base_path.is_absolute():
        base_path = spec_path.parent / base_path
    base_path = base_path.resolve()
    base = _read_object(base_path)
    if str(base.get("calculation_kind", "")).upper() != "DYNAMIC":
        raise ValueError("bifurcation sweep requires calculation_kind=DYNAMIC")
    if str(base.get("solver_backend", "")).upper() != "VARIATIONAL_REFERENCE":
        raise ValueError("sweep requires VARIATIONAL_REFERENCE backend")
    if str(base.get("method", "")).upper() not in ("FDM", "MVI"):
        raise ValueError("sweep requires explicit FDM or MVI")
    key = spec.get("parameter_path")
    if key not in ALLOWED_PARAMETERS:
        raise ValueError(f"unsupported parameter_path {key!r}; allowed: {sorted(ALLOWED_PARAMETERS)}")
    values = spec.get("values")
    if not isinstance(values, list) or not 2 <= len(values) <= 256:
        raise ValueError("values must contain 2..256 independent parameter values")
    values = [_finite_number(v, f"values[{i}]") for i, v in enumerate(values)]
    if len(set(values)) != len(values):
        raise ValueError("parameter values must be distinct")
    output_ref = spec.get("output_dir")
    if not isinstance(output_ref, str) or not output_ref.strip():
        raise ValueError("explicit output_dir required for parameter study")
    root = Path(output_ref).expanduser()
    if not root.is_absolute():
        root = spec_path.parent / root
    root = root.resolve()
    if root == base_path or root == spec_path:
        raise ValueError("sweep output must not overwrite its input")
    name = spec.get("parameter_name", key.split(".")[-1])
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", name):
        raise ValueError("parameter_name requires 1..48 ASCII letters/digits/underscores")
    probe = spec.get("probe_name", "P01_CENTER_0")
    if not isinstance(probe, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,28}", probe):
        raise ValueError("invalid probe_name")
    transient = _finite_number(spec.get("transient_fraction", .5), "transient_fraction")
    if not 0 <= transient < .9:
        raise ValueError("transient_fraction must lie in [0,0.9)")
    forcing = spec.get("forcing_hz")
    if forcing is not None:
        forcing = _finite_number(forcing, "forcing_hz")
        if forcing <= 0:
            raise ValueError("forcing_hz must be positive")
        if key == "dynamics.time_law.frequency_hz":
            raise ValueError("do not override forcing_hz when sweeping load frequency")
    each = spec.get("export_each_run", True)
    if not isinstance(each, bool):
        raise ValueError("export_each_run must be boolean")
    jobs = []
    for index, value in enumerate(values, start=1):
        case = copy.deepcopy(base)
        _set_scalar(case, key, value)
        _absolutize_csv_inputs(case, base_path.parent)
        run_dir = root / "RUNS" / f"RUN_{index:04d}"
        case["output_dir"] = str(run_dir)
        if not each:
            post = case.setdefault("postprocess", {})
            if not isinstance(post, dict):
                raise ValueError("postprocess must be an object")
            post["dynamic_analytics"] = False
        jobs.append({"index": index, "value": value, "case": case,
                     "case_file": root / "CASES" / f"CASE_{index:04d}.json",
                     "run_dir": run_dir})
    return {"spec_path": spec_path, "spec": spec, "base_path": base_path,
            "base_case": base, "output_dir": root, "parameter_path": key,
            "parameter_name": name, "probe_name": probe,
            "transient_fraction": transient, "forcing_hz": forcing,
            "export_each_run": each, "jobs": jobs}


def run_parameter_sweep(spec_file, *, preflight_only=False,
                        prepare_case=None, solve_case=None, export_study=None):
    """Execute full solver runs in sequence; never reuse or overwrite output.

    Optional injected callbacks exist only for isolated tests. A normal call
    invokes the existing cylinder solver and its bifurcation exporter.
    """
    plan = plan_parameter_sweep(spec_file)
    if prepare_case is None:
        from cylinder_reference_case import prepare_reference_case
        prepare_case = prepare_reference_case
    from cylinder_dynamic_analytics import PROBE_LOCATIONS, load_dynamic_run
    if plan["probe_name"] not in {name for name, _, _ in PROBE_LOCATIONS}:
        raise ValueError("probe_name must match one of seven standard monitor points")
    root = plan["output_dir"]
    if root.exists() and any(root.iterdir()):
        raise ValueError("OUTPUT_DIR_NOT_EMPTY: parameter sweep requires a fresh directory")

    # Validate the WHOLE study before creating any files or running any steps.
    for job in plan["jobs"]:
        validation = prepare_case(job["case"], base_dir=plan["base_path"].parent)
        if getattr(validation, "calculation_kind", None) != "DYNAMIC":
            raise ValueError(f"job {job['index']}: preflight did not return DYNAMIC")
    description = {
        "study_type": "INDEPENDENT_COLD_START_PARAMETER_SWEEP",
        "parameter_path": plan["parameter_path"],
        "parameter_name": plan["parameter_name"],
        "values": [j["value"] for j in plan["jobs"]],
        "base_case": str(plan["base_path"]),
        "base_case_sha256": hashlib.sha256(plan["base_path"].read_bytes()).hexdigest(),
        "theory": plan["base_case"].get("theory"),
        "cuf_model": plan["base_case"].get("cuf_model"),
        "method": plan["base_case"].get("method"),
        "rank": plan["base_case"].get("rank"),
        "probe_name": plan["probe_name"],
        "transient_fraction": plan["transient_fraction"],
        "forcing_hz_override": plan["forcing_hz"],
        "per_run_analytics": plan["export_each_run"],
        "output_dir": str(root),
        "preflighted_cases": len(plan["jobs"]),
        "status": "PREFLIGHT_VALID" if preflight_only else "RUNNING",
        "completed_runs": 0,
        "runs": [],
    }
    if preflight_only:
        return description
    if solve_case is None:
        from cylinder_reference_case import run_reference_case
        solve_case = run_reference_case
    if export_study is None:
        from cylinder_dynamic_analytics import export_bifurcation_study
        export_study = export_bifurcation_study
    root.mkdir(parents=True, exist_ok=True)
    (root / "CASES").mkdir()
    (root / "RUNS").mkdir()
    report_path = root / "SWEEP_STATUS.json"
    for job in plan["jobs"]:
        _atomic_json(job["case_file"], job["case"])
        description["runs"].append({
            "index": job["index"], "parameter_value": job["value"],
            "case_file": str(job["case_file"]), "run_dir": str(job["run_dir"]),
            "status": "NOT_STARTED"})
    _atomic_json(report_path, description)
    completed = []
    for job, record in zip(plan["jobs"], description["runs"]):
        record["status"] = "RUNNING"
        _atomic_json(report_path, description)
        try:
            # The written CASE file is the exact input consumed by the solver.
            result = solve_case(job["case"], job["case_file"])
            if not isinstance(result, dict) or result.get("converged") is not True:
                raise RuntimeError("dynamic solver did not finish all time steps")
            trajectory = load_dynamic_run(job["run_dir"])
            if not trajectory.complete:
                raise RuntimeError("accepted Newmark trajectory is incomplete")
            record["accepted_steps"] = result.get("accepted_steps")
            record["status"] = "COMPLETE"
            completed.append(job)
            description["completed_runs"] = len(completed)
        except Exception as exc:
            record["status"] = "FAILED"
            record["error_type"] = type(exc).__name__
            record["message"] = str(exc)
            description["status"] = "INCOMPLETE_NO_BIFURCATION"
            _atomic_json(report_path, description)
            raise RuntimeError(
                f"parameter run {job['index']} failed; previous runs retained in {root}"
            ) from exc
        _atomic_json(report_path, description)
    manifest = root / "BIFURCATION_RUNS.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(("parameter_value", "run_dir"))
        for job in completed:
            writer.writerow((repr(job["value"]), str(job["run_dir"])))
    try:
        study = export_study(
            manifest, output_dir=root / "BIFURCATION_ANALYTICS",
            parameter_name=plan["parameter_name"], probe_name=plan["probe_name"],
            transient_fraction=plan["transient_fraction"],
            forcing_hz=plan["forcing_hz"])
    except Exception as exc:
        description["status"] = "POSTPROCESS_FAILED_PHYSICAL_RUNS_RETAINED"
        description["postprocess_error"] = {
            "type": type(exc).__name__, "message": str(exc)}
        _atomic_json(report_path, description)
        raise RuntimeError(
            "all parameter runs completed but bifurcation export failed"
        ) from exc
    description["status"] = "COMPLETED"
    description["bifurcation"] = study
    description["manifest_csv"] = str(manifest)
    _atomic_json(report_path, description)
    return description


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Full cylindrical-shell FDM/MVI independent parameter study")
    parser.add_argument("spec_json", help="sweep JSON: base_case, parameter_path, values, output_dir")
    parser.add_argument("--preflight", action="store_true",
                        help="check all cases without creating output")
    args = parser.parse_args(argv)
    result = run_parameter_sweep(args.spec_json, preflight_only=args.preflight)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
