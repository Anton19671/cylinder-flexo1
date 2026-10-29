"""Explicit, lossless FDM/MVI selection for a saved variational CASE.

A selector click is only a GUI draft. At launch create a NEW CASE next to the
source, keeping relative external file references anchored to the same folder.
Never overwrite the input CASE or resize a grid to make a requested rank fit.
"""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from cylinder_case_io import save_case_file


def selected_case(original, method, rank=None):
    if not isinstance(original, dict) or original.get("solver_backend") != "VARIATIONAL_REFERENCE":
        raise ValueError("Select a saved VARIATIONAL_REFERENCE CASE")
    method = str(method).upper()
    if method not in ("FDM", "MVI"):
        raise ValueError("Choose FDM (MKR) or MVI-1..8")
    if method == "MVI":
        if type(rank) is not int or not 1 <= rank <= 8:
            raise ValueError("MVI rank must be an integer from 1 through 8")
    elif rank is not None:
        raise ValueError("FDM has no MVI rank")
    old_method = str(original.get("method", "")).upper()
    if old_method not in ("FDM", "MVI"):
        raise ValueError("The source CASE has an unknown spatial method")
    old_key = "mvi_grid" if old_method == "MVI" else "fdm_grid"
    new_key = "mvi_grid" if method == "MVI" else "fdm_grid"
    grid = original.get(old_key, original.get("grid"))
    if not isinstance(grid, dict) or not grid:
        raise ValueError("The source CASE has no valid spatial grid")
    result = copy.deepcopy(original)
    old_backend = str(original.get("fdm_backend", "DENSE_REFERENCE")).upper()
    scheme = str(grid.get("operator_scheme", "LEGACY_REFERENCE")).upper()
    if old_method != method and (old_backend == "MATRIX_FREE_GMRES" or
                                  scheme == "SBP21_PERIODIC_ODD_CSR"):
        raise ValueError(
            "Cannot switch a matrix-free/CSR CASE silently. "
            "Create a separate CASE with an explicitly selected compatible operator scheme.")
    result.pop("fdm_grid", None)
    result.pop("mvi_grid", None)
    result.pop("grid", None)
    result[new_key] = copy.deepcopy(grid)
    result["method"] = method
    if method == "MVI":
        nx, ns = int(grid.get("nx", 0)), int(grid.get("ns", 0))
        if rank > min(nx, ns):
            raise ValueError(
                f"MVI-{rank} requires nx and ns >= {rank}; current grid is {nx}x{ns}. "
                "Increase the grid explicitly in the CASE editor; no automatic resizing.")
        result["rank"] = rank
        if old_method != "MVI":
            result["reference_mvi"] = {"max_sweeps": 30}
    else:
        result.pop("rank", None)
        result.pop("reference_mvi", None)
    return result


def create_launch_case(source_path, selected):
    """Persist an auto-CASE only upon launch. Preserve all source file references."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not isinstance(selected, dict):
        raise TypeError("selected CASE must be a mapping")
    raw = selected.get("output_dir")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("The source CASE has no result output_dir")
    base = Path(raw).expanduser()
    if not base.is_absolute():
        base = source.parent / base
    base = base.resolve()
    method = str(selected["method"]).upper()
    token = method + (str(selected["rank"]) if method == "MVI" else "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(20):
        unique = uuid4().hex[:10]
        basename = f"{token}_{stamp}_{unique}"
        target_case = source.with_name(f"{source.stem}__AUTO_{basename}{source.suffix}")
        target_result = base.with_name(base.name + "__" + basename)
        if target_case.exists() or target_result.exists():
            continue
        candidate = copy.deepcopy(selected)
        candidate["output_dir"] = str(target_result)
        # Exclusive creation also protects the original CASE on concurrent launches.
        with target_case.open("x", encoding="utf-8"):
            pass
        try:
            save_case_file(target_case, candidate)
        except BaseException:
            target_case.unlink(missing_ok=True)
            raise
        return target_case, candidate
    raise FileExistsError("Unable to allocate unique auto-CASE and output directory")


__all__ = ("selected_case", "create_launch_case")
