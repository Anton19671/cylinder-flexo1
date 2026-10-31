"""Explicit, lossless FDM/MVI selection for a saved variational CASE.

A selector click is only a GUI draft. At launch create a NEW CASE next to the
source, keeping relative external file references anchored to the same folder.
Never overwrite the input CASE or resize a grid to make a requested rank fit.
"""
from __future__ import annotations

import copy
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from cylinder_case_io import save_case_file


def selected_case(original, method, rank=None):
    if not isinstance(original, dict) or original.get("solver_backend") != "VARIATIONAL_REFERENCE":
        raise ValueError("Select a saved VARIATIONAL_REFERENCE CASE")
    method = str(method).upper()
    if method == "MKR":
        method = "FDM"
    if method not in ("FDM", "MVI"):
        raise ValueError("Choose FDM (MKR) or MVI-1..8")
    if method == "MVI":
        if type(rank) is not int or not 1 <= rank <= 8:
            raise ValueError("MVI rank must be an integer from 1 through 8")
    elif rank is not None:
        raise ValueError("FDM has no MVI rank")
    old_method = str(original.get("method", "")).upper()
    if old_method == "MKR":
        old_method = "FDM"
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
    kind = str(original.get("calculation_kind", "")).upper()
    scale = str(original.get("scale_model", "")).upper()
    nx, ns = int(grid.get("nx", 0)), int(grid.get("ns", 0))
    # Dense and CSR SBP42/Fourier operators represent the SAME spatial
    # differentiation. For a homogeneous MACRO MODAL problem we can switch
    # their storage/backends without modifying physical input parameters.
    compatible_fourier = (kind == "MODAL" and scale == "MACRO" and
        scheme in ("SBP42_FOURIER_ODD", "SBP42_FOURIER_ODD_CSR") and
        ((scheme == "SBP42_FOURIER_ODD" and old_backend == "DENSE_REFERENCE") or
         (scheme == "SBP42_FOURIER_ODD_CSR" and old_backend == "MATRIX_FREE_GMRES")))
    switching_sparse = (old_method != method and
        (old_backend == "MATRIX_FREE_GMRES" or
         scheme in ("SBP21_PERIODIC_ODD_CSR", "SBP42_FOURIER_ODD_CSR")))
    if switching_sparse and not compatible_fourier:
        raise ValueError(
            "Cannot switch an incompatible matrix-free/CSR CASE. "
            "Save a separate CASE with an explicit compatible spatial operator.")
    # Repair a legacy invalid dense modal FDM draft, but only when its exact
    # SBP42/Fourier CSR counterpart is available and the grid is too large
    # for the dense reference FDM route.
    upgrade_fdm = (method == "FDM" and compatible_fourier and
        (old_method == "MVI" or
         (old_backend == "DENSE_REFERENCE" and nx * ns > 256)))
    downgrade_mvi = (method == "MVI" and old_method == "FDM" and
                     compatible_fourier and scheme == "SBP42_FOURIER_ODD_CSR")
    if downgrade_mvi and (nx > 21 or ns > 63 or ns % 2 != 1):
        raise ValueError(
            "MODAL Fourier MVI requires nx <= 21 and odd ns <= 63.")
    translated_grid = copy.deepcopy(grid)
    result.pop("fdm_grid", None)
    result.pop("mvi_grid", None)
    result.pop("grid", None)
    result[new_key] = translated_grid
    result["method"] = method
    if kind == "MODAL" and method == "FDM" and isinstance(result.get("modal"), dict):
        # These options apply only to the fixed-rank MVI spectral search.
        # Leaving them on a converted FDM case causes modal preflight to
        # reject an otherwise valid MKR/FDM selection before launching.
        for field in ("mvi_spectrum_method", "mode_tolerance",
                      "axial_modes_per_harmonic"):
            result["modal"].pop(field, None)
    if upgrade_fdm:
        translated_grid["operator_scheme"] = "SBP42_FOURIER_ODD_CSR"
        result["fdm_backend"] = "MATRIX_FREE_GMRES"
        options = copy.deepcopy(result.get("reference_newton") or {})
        options.pop("max_free_dofs", None)
        options.pop("min_reciprocal_condition", None)
        options.setdefault("max_total_dofs", 100000)
        options.setdefault("max_surface_nodes", 8192)
        options.setdefault("max_constraint_rows", 20000)
        result["reference_newton"] = options
    elif downgrade_mvi:
        translated_grid["operator_scheme"] = "SBP42_FOURIER_ODD"
        result["fdm_backend"] = "DENSE_REFERENCE"
        options = copy.deepcopy(result.get("reference_newton") or {})
        for field in ("max_total_dofs", "max_surface_nodes", "max_constraint_rows",
                      "krylov_rtol", "krylov_restart", "krylov_max_matvecs",
                      "krylov_preconditioner", "ilu_drop_tolerance",
                      "ilu_fill_factor", "ilu_max_surface_nodes"):
            options.pop(field, None)
        options.setdefault("max_free_dofs", 512)
        result["reference_newton"] = options
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


def _safe_run_label(text):
    """A user-chosen Windows-safe directory label, not a filesystem path."""
    if text is None:
        return "RUN"
    if not isinstance(text, str):
        raise ValueError("Calculation name must be text")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    value = re.sub(r"\s+", "_", value).strip(" ._")
    if not value:
        return "RUN"
    value = value[:48].rstrip(" ._")
    if value.split(".")[0].upper() in (
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
        "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2",
        "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"):
        value = "RUN_" + value
    return value or "RUN"


def create_launch_case(source_path, selected, *, run_label=""):
    """Write a unique run CASE and choose an empty child of the result root.

    User's saved physical CASE and previous output folders stay untouched.
    The new CASE lives beside its source, preserving relative input paths;
    only its output_dir is redirected to the new per-run child folder.
    """
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
    if base.exists() and not base.is_dir():
        raise NotADirectoryError(f"Result root is a file, not a folder: {base}")
    method = str(selected.get("method", "")).upper()
    if method == "MKR":
        method = "FDM"
        selected = copy.deepcopy(selected)
        selected["method"] = "FDM"
    if method not in ("FDM", "MVI"):
        raise ValueError("Unknown spatial method")
    if method == "MVI":
        token = f"MVI{int(selected['rank'])}"
    else:
        token = "FDM"
    model = str(selected.get("theory", "MODEL")).upper()
    if model == "CUF":
        model += "_" + str(selected.get("cuf_model", "")).upper()
    kind = str(selected.get("calculation_kind", "STATIC")).upper()
    grid = selected.get("mvi_grid" if method == "MVI" else "fdm_grid") or selected.get("grid") or {}
    nx, ns = grid.get("nx", ""), grid.get("ns", "")
    descriptor = f"{kind}_{model}_{token}_g{nx}x{ns}"
    label = _safe_run_label(run_label)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(30):
        unique = uuid4().hex[:8]
        basename = f"{label}__{descriptor}__{stamp}_{unique}"
        target_case = source.with_name(
            f"{source.stem[:65]}__AUTO_{stamp}_{unique}{source.suffix}")
        target_result = base / basename
        if target_case.exists() or target_result.exists():
            continue
        candidate = copy.deepcopy(selected)
        candidate["output_dir"] = str(target_result)
        # Exclusive file allocation; never modify the saved original CASE.
        with target_case.open("x", encoding="utf-8"):
            pass
        try:
            save_case_file(target_case, candidate)
        except BaseException:
            target_case.unlink(missing_ok=True)
            raise
        return target_case, candidate
    raise FileExistsError("Unable to allocate unique run CASE and output directory")


__all__ = ("selected_case", "create_launch_case")
