"""Crash-safe restart contracts for native independent coupled MVI.

The RUN_SUMMARY.json accepted_steps field is the single commit marker.
A new time step is accepted only when its complete NPZ, history row,
and updated summary have been written. An uncommitted step is discarded
on restart; no numerical solution is ever fabricated or recalculated
under the name of a previously accepted step.

Uses process locks that are released automatically on normal exit,
exception, or process termination (flock on POSIX, msvcrt on Windows).
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from independent_mvi.coupled_newmark import CoupledMVIDynamicState
from independent_mvi.coupled_surface import CoupledMVIFactors
from independent_mvi.coupled_physical_recovery import recover_cuf_displacements

_STATE_FILE=re.compile(r"^STEP_([0-9]+)\.npz$")
_F64=np.finfo(float)


def _hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            digest.update(part)
    return digest.hexdigest()


def input_manifest(case_path,initial_path,loads_path):
    """Hash the exact CASE and both NPZ inputs, without interpreting them."""
    return dict(
        case_sha256=_hash_file(case_path),
        initial_npz_sha256=_hash_file(initial_path),
        loads_npz_sha256=_hash_file(loads_path),
    )


@contextmanager
def exclusive_run_lock(folder):
    """Prevent concurrent writers; OS releases lock after a hard crash."""
    path=Path(folder)/"RUN_LOCK"
    with path.open("a+b") as handle:
        handle.seek(0)
        if os.name=="nt":
            import msvcrt
            handle.write(b"1")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:
                raise RuntimeError("COUPLED_MVI_OUTPUT_ALREADY_RUNNING") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("COUPLED_MVI_OUTPUT_ALREADY_RUNNING") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def _check_stored_field(actual,expected,label):
    if actual.shape!=expected.shape or not np.isfinite(actual).all():
        raise ValueError("COUPLED_MVI_CHECKPOINT_"+label+"_SHAPE_OR_FINITE")
    discrepancy=float(np.linalg.norm((actual-expected).ravel()))
    magnitude=float(np.linalg.norm(expected.ravel()))
    if not math.isfinite(discrepancy) or (
            discrepancy>1.e-11*max(magnitude,_F64.tiny)):
        raise ValueError("COUPLED_MVI_CHECKPOINT_"+label+"_MISMATCH")


def _time_close(actual,expected,dt):
    return (math.isfinite(actual) and
            abs(actual-expected)<=max(
                128*_F64.eps*max(abs(actual),abs(expected),_F64.tiny),
                1.e-11*abs(dt)))


def load_committed_state(folder,step,surface,configuration,dt_s):
    """Read and validate the exact accepted X/Y physical state; no solve."""
    path=Path(folder)/f"STEP_{step:04d}.npz"
    try:
        with np.load(path,allow_pickle=False) as archive:
            required={"x_profiles","y_profiles","mechanical","electrical",
                      "velocity","acceleration","time_s"}
            if not required<=set(archive.files):
                raise ValueError("COUPLED_MVI_CHECKPOINT_MISSING_FIELDS")
            data={key:np.asarray(archive[key],dtype=float).copy()
                  for key in required}
            if surface.theory.startswith("CUF"):
                for key in ("physical_z_stations_m",
                            "physical_uvw_stations_m"):
                    if key not in archive.files:
                        raise ValueError("COUPLED_MVI_CHECKPOINT_MISSING_CUF_FACES")
                    data[key]=np.asarray(archive[key],dtype=float).copy()
    except (OSError,EOFError,KeyError) as exc:
        raise ValueError("COUPLED_MVI_CHECKPOINT_UNREADABLE") from exc

    if data["time_s"].shape!=():
        raise ValueError("COUPLED_MVI_CHECKPOINT_TIME_INVALID")
    t=float(data["time_s"])
    if not _time_close(t,step*dt_s,dt_s):
        raise ValueError("COUPLED_MVI_CHECKPOINT_TIME_MISMATCH")
    factors=CoupledMVIFactors(data["x_profiles"],data["y_profiles"])
    surface._check_factors(factors)
    fields=surface.reconstruct(factors)
    _check_stored_field(data["mechanical"],
                        fields[:surface.n_mechanical],"MECHANICAL")
    _check_stored_field(data["electrical"],
                        fields[surface.n_mechanical:],"ELECTRIC")
    recovered=recover_cuf_displacements(
        surface,fields[:surface.n_mechanical])
    if recovered is not None:
        _check_stored_field(data["physical_z_stations_m"],
                            recovered.z_stations_m,"STATIONS")
        _check_stored_field(data["physical_uvw_stations_m"],
                            recovered.displacement_stations_m,"PHYSICAL_FACES")
    state=CoupledMVIDynamicState(
        t,factors,data["velocity"],data["acceleration"])
    configuration._validate_state(state)
    return state


def _read_committed_history(path,accepted,dt_s):
    if not Path(path).is_file():
        raise ValueError("COUPLED_MVI_CHECKPOINT_HISTORY_MISSING")
    with Path(path).open("r",encoding="utf-8",newline="") as stream:
        reader=csv.DictReader(stream)
        fieldnames=reader.fieldnames
        rows=list(reader)
    if (not fieldnames or "step" not in fieldnames or
            "time_s" not in fieldnames or len(rows)<=accepted):
        raise ValueError("COUPLED_MVI_CHECKPOINT_HISTORY_INCOMPLETE")
    committed=rows[:accepted+1]
    for i,row in enumerate(committed):
        try:
            stored_step=int(row["step"])
            stored_time=float(row["time_s"])
        except (ValueError,TypeError,KeyError) as exc:
            raise ValueError("COUPLED_MVI_CHECKPOINT_HISTORY_INVALID") from exc
        if stored_step!=i or not _time_close(stored_time,i*dt_s,dt_s):
            raise ValueError("COUPLED_MVI_CHECKPOINT_HISTORY_INVALID")
    return fieldnames,committed,len(rows)>accepted+1


def _atomic_history_trim(path,columns,rows):
    temporary=Path(path).with_name(Path(path).name+".pending")
    try:
        with temporary.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_resume(output,case,manifest,*,surface,configuration,steps,dt_s,
                   input_rank):
    """Validate the last committed checkpoint and trim unaccepted work.

    Called only while the output process lock is held. A failed check
    leaves previously accepted state and summary unchanged.
    """
    output=Path(output)
    try:
        original=json.loads((output/"CASE.json").read_text(encoding="utf-8"))
        summary=json.loads((output/"RUN_SUMMARY.json").read_text(
            encoding="utf-8"))
    except (OSError,ValueError) as exc:
        raise ValueError("COUPLED_MVI_CHECKPOINT_METADATA_MISSING") from exc
    if original!=case or summary.get("input_manifest")!=manifest:
        raise ValueError("COUPLED_MVI_CHECKPOINT_INPUT_CHANGED")
    accepted=summary.get("accepted_steps")
    if (type(accepted) is not int or accepted<0 or accepted>=steps or
            summary.get("calculation_kind")!="DYNAMIC" or
            summary.get("rank")!=input_rank or
            summary.get("steps")!=steps):
        raise ValueError("COUPLED_MVI_CHECKPOINT_STEP_OR_RANK_INVALID")
    if summary.get("status") not in ("IN_PROGRESS","FAILED"):
        raise ValueError("COUPLED_MVI_CHECKPOINT_NOT_RESUMABLE")
    if not _time_close(float(summary.get("time_s",-1.)),
                       accepted*dt_s,dt_s):
        raise ValueError("COUPLED_MVI_CHECKPOINT_SUMMARY_TIME_MISMATCH")
    state=load_committed_state(
        output,accepted,surface,configuration,dt_s)
    if state.factors.rank!=input_rank:
        raise ValueError("COUPLED_MVI_CHECKPOINT_RANK_CHANGED")
    history_path=output/"PHYSICAL_RADIAL_HISTORY.csv"
    fields,rows,excess=_read_committed_history(
        history_path,accepted,dt_s)

    # Completed NPZ files beyond the summary marker were never committed.
    # Remove only exact method-owned state names, never unrelated files.
    for candidate in output.iterdir():
        found=_STATE_FILE.fullmatch(candidate.name)
        if found and int(found.group(1))>accepted:
            candidate.unlink()
    if excess:
        _atomic_history_trim(history_path,fields,rows)
    return state,summary


def append_history_row(path,row):
    """O(1) per step: never rewrite an entire long history each step."""
    path=Path(path)
    needs_header=not path.exists()
    with path.open("a",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


__all__=("input_manifest","exclusive_run_lock","load_committed_state",
         "prepare_resume","append_history_row")
