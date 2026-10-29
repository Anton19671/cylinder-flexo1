"""Archive stopped/failed Cylinder-Flexo1 runs without making up solver results.

Called only AFTER the solver process has exited. The solver owns all accepted
NPZ states and FINAL_RUN_SUMMARY.json; this module never edits those files.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil


def archive_terminated_run(output_dir, *, status, case_path=None, journal_path=None,
                           calculation_kind=None, method=None, rank=None,
                           expected_steps=None, run_kind=None,
                           last_modal_event=None, exit_code=None,
                           error_type=None, error_detail=None, traceback_text=None):
    if status not in ("STOPPED_BY_USER", "FAILED"):
        raise ValueError("status must be STOPPED_BY_USER or FAILED")
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Result destination is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    # Only immutable accepted states in the solver's summary may be counted.
    accepted = []
    summary_file = root / "FINAL_RUN_SUMMARY.json"
    if summary_file.is_file():
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if summary.get("solver_backend") == "VARIATIONAL_REFERENCE":
                from cylinder_reference_progress_adapter import accepted_reference_steps
                accepted = accepted_reference_steps(root)
        except (OSError, TypeError, ValueError, KeyError):
            pass

    files = {}
    if case_path is not None:
        source = Path(case_path).expanduser().resolve()
        if source.is_file():
            destdir = root / "INPUT"
            destdir.mkdir(exist_ok=True)
            dest = destdir / (("STOPPED_CASE" if status == "STOPPED_BY_USER" else "FAILED_CASE") + (source.suffix.lower() or ".json"))
            if source != dest:
                shutil.copy2(source, dest)
            files["case"] = str(dest.relative_to(root))

    if journal_path is not None:
        source = Path(journal_path).expanduser().resolve()
        if source.is_file():
            destdir = root / "JOURNALS"
            destdir.mkdir(exist_ok=True)
            dest = destdir / source.name
            if source != dest:
                shutil.copy2(source, dest)
            files["journal"] = str(dest.relative_to(root))

    # A modal event may report a frequency before any mode field has been
    # committed to disk. Never call that mode accepted.
    event = last_modal_event if isinstance(last_modal_event, dict) else {}
    partial_modes = 0
    partial_file = root / "MODAL_PARTIAL_MODES.npz"
    if str(calculation_kind).upper() == "MODAL" and partial_file.is_file():
        try:
            import numpy as np
            with np.load(partial_file, allow_pickle=False) as saved:
                frequencies = saved["frequencies_hz"]
                fields = saved["mechanical_modes"]
                if (frequencies.ndim == 1 and np.isfinite(frequencies).all() and
                        fields.ndim == 4 and fields.shape[0] == len(frequencies) and
                        np.isfinite(fields).all()):
                    partial_modes = len(frequencies)
                    files["modal_partial_modes"] = partial_file.name
        except (OSError, ValueError, KeyError):
            pass
    last_stage = {
        key: event[key] for key in
        ("stage", "mode", "n_modes", "rank", "sweep", "axis",
         "column", "total_columns", "frequency_hz", "elapsed_s")
        if key in event and isinstance(event[key], (str, int, float, type(None)))
    }
    record = {
        "status": status,
        "completed": False,
        "saved_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "output_dir": str(root),
        "calculation_kind": calculation_kind,
        "method": method,
        "rank": rank,
        "requested_steps_or_modes": expected_steps,
        "run_kind": run_kind,
        "process_exit_code": exit_code,
        "error_type": error_type,
        "error_detail": error_detail,
        "accepted_steps_verified": partial_modes if str(calculation_kind).upper() == "MODAL" else len(accepted),
        "accepted_state_files": ([partial_file.name] if partial_modes else
                                 [row["state_file"] for row in accepted]),
        "last_modal_event": last_stage,
        "files": files,
        "note": ("An unfinished eigenmode/iteration is not a physical result. "
                 "Any accepted STATIC/DYNAMIC states remain in their original files. "
                 "An interrupted MODAL solve may have no stored modal fields."),
    }
    filename = "STOPPED_RUN.json" if status == "STOPPED_BY_USER" else "FAILED_RUN.json"
    target = root / filename
    tmp = root / (filename + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    if traceback_text:
        trace_target = root / "FAILED_TRACEBACK.txt"
        trace_tmp = root / "FAILED_TRACEBACK.txt.tmp"
        trace_tmp.write_text(str(traceback_text), encoding="utf-8")
        os.replace(trace_tmp, trace_target)
    return record


def archive_stopped_run(output_dir, **kwargs):
    return archive_terminated_run(output_dir, status="STOPPED_BY_USER", **kwargs)


def archive_failed_run(output_dir, **kwargs):
    return archive_terminated_run(output_dir, status="FAILED", **kwargs)


__all__ = ("archive_stopped_run", "archive_failed_run", "archive_terminated_run")
