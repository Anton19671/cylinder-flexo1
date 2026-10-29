"""Archive a stopped Cylinder-Flexo1 run without making up solver results.

Called only AFTER the solver process has exited. The solver owns all accepted
NPZ states and FINAL_RUN_SUMMARY.json; this module never edits those files.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil


def archive_stopped_run(output_dir, *, case_path=None, journal_path=None,
                        calculation_kind=None, method=None, rank=None,
                        expected_steps=None, run_kind=None,
                        last_modal_event=None, exit_code=None):
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
            dest = destdir / ("STOPPED_CASE" + (source.suffix.lower() or ".json"))
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
    last_stage = {
        key: event[key] for key in
        ("stage", "mode", "n_modes", "rank", "sweep", "axis",
         "column", "total_columns", "frequency_hz", "elapsed_s")
        if key in event and isinstance(event[key], (str, int, float, type(None)))
    }
    record = {
        "status": "STOPPED_BY_USER",
        "completed": False,
        "saved_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "output_dir": str(root),
        "calculation_kind": calculation_kind,
        "method": method,
        "rank": rank,
        "requested_steps_or_modes": expected_steps,
        "run_kind": run_kind,
        "process_exit_code": exit_code,
        "accepted_steps_verified": len(accepted),
        "accepted_state_files": [row["state_file"] for row in accepted],
        "last_modal_event": last_stage,
        "files": files,
        "note": ("An unfinished eigenmode/iteration is not a physical result. "
                 "Any accepted STATIC/DYNAMIC states remain in their original files. "
                 "An interrupted MODAL solve may have no stored modal fields."),
    }
    target = root / "STOPPED_RUN.json"
    tmp = root / "STOPPED_RUN.json.tmp"
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return record


__all__ = ("archive_stopped_run",)
