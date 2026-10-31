"""Truthful per-harmonic modal output for MVI Fourier scans and retained FDM modes.

This module never assembles operators or solves eigenproblems. All frequencies
come exclusively from saved physical solver results. A Fourier amplitude of
an eigenvector is NOT a new eigenfrequency in that sector.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CATALOG_FILE = "HARMONIC_FAMILY_CATALOG.csv"
CATALOG_COLUMNS = (
    "method", "rank", "circumferential_n", "axial_candidate_index",
    "frequency_hz", "rotational_degeneracy", "saved_mode_numbers",
    "frequency_source", "status", "axial_index_verified", "scope",
)


def harmonic_family_catalog(result):
    """Return (rows, coverage) including *every resolvable circumferential n*.

    When a physical Fourier family scan has been performed, the catalogue
    includes all candidates from the scan, not just globally retained modes.
    Without that scan, only the dominant n of the retained eigenvectors can
    be identified. Unseen n must have a blank frequency and an explicit status.
    """
    fields = np.asarray(result["mechanical_modes"], dtype=float)
    frequencies = np.asarray(result["frequencies_hz"], dtype=float)
    if (fields.ndim != 4 or fields.shape[0] != len(frequencies)
            or fields.shape[-1] < 3 or not np.isfinite(fields).all()
            or not np.isfinite(frequencies).all()):
        raise ValueError("MODAL_HARMONIC_CATALOG_INVALID_EIGENMODES")
    method = str(result.get("method", "UNKNOWN"))
    rank = int(result["rank"]) if result.get("rank") is not None else ""
    nmax = fields.shape[-1] // 2
    family = result.get("harmonic_family_scan")
    rows = []
    base = {"method": method, "rank": rank, "axial_candidate_index": "",
            "frequency_hz": "", "rotational_degeneracy": "",
            "saved_mode_numbers": "", "axial_index_verified": False}
    if family is not None:
        if not isinstance(family, list):
            raise ValueError("MODAL_HARMONIC_CATALOG_BAD_FAMILY_SCAN")
        grouped = {}
        for entry in family:
            n, axial = int(entry["n"]), int(entry["axial_mode_index"])
            if not 0 <= n <= nmax or axial < 1:
                raise ValueError("MODAL_HARMONIC_CATALOG_BAD_FAMILY_INDEX")
            grouped.setdefault(n, []).append(entry)
        picked = {}
        for mode, (n, axial) in enumerate(zip(result["circumferential_wavenumbers"],
                                             result["axial_mode_indices"]), start=1):
            picked.setdefault((int(n), int(axial)), []).append(mode)
        scope = "FOURIER_SECTORS_AXIAL_CANDIDATES_NOT_GLOBAL_SPECTRUM"
        for n in range(nmax + 1):
            if n not in grouped:
                rows.append({**base, "circumferential_n": n,
                             "frequency_source": "NONE", "status": "SECTOR_NOT_COMPUTED",
                             "scope": scope})
                continue
            for entry in sorted(grouped[n], key=lambda x: (int(x["axial_mode_index"]),
                                                       float(x["frequency_hz"]))):
                axial = int(entry["axial_mode_index"])
                frequency = float(entry["frequency_hz"])
                if not np.isfinite(frequency) or frequency < 0.:
                    raise ValueError("MODAL_HARMONIC_CATALOG_INVALID_FREQUENCY")
                saved = picked.get((n, axial), [])
                rows.append({**base, "circumferential_n": n,
                             "axial_candidate_index": axial, "frequency_hz": frequency,
                             "rotational_degeneracy": int(entry["degeneracy"]),
                             "saved_mode_numbers": ",".join(map(str, saved)),
                             "frequency_source": "FOURIER_SECTOR_SOLVE",
                             "status": "COMPUTED_SAVED" if saved else "COMPUTED_NOT_SAVED",
                             "axial_index_verified": bool(entry.get("axial_mode_index_verified", False)),
                             "scope": scope})
    else:
        from cylinder_modal_diagnostics import inspect_modal_shapes
        diagnostics = inspect_modal_shapes(fields, frequencies, result["theory"])
        grouped = {}
        for info in diagnostics["modes"]:
            n = int(info["harmonic"])
            grouped.setdefault(n, []).append(info)
        scope = "RETAINED_MODES_DOMINANT_FFT_ONLY_NOT_SECTOR_SPECTRUM"
        for n in range(nmax + 1):
            if n not in grouped:
                rows.append({**base, "circumferential_n": n,
                             "frequency_source": "NONE", "status": "NO_RETAINED_MODE",
                             "scope": scope})
            else:
                for info in grouped[n]:
                    rows.append({**base, "circumferential_n": n,
                                 "frequency_hz": float(info["frequency_hz"]),
                                 "saved_mode_numbers": str(info["mode"]),
                                 "frequency_source": "SAVED_EIGENVECTOR_DOMINANT_FFT",
                                 "status": "SAVED_MODE_NOT_INDEPENDENT_SECTOR_SCAN",
                                 "scope": scope})
    return rows, scope


def write_harmonic_family_catalog(path, rows):
    """Write all sector coverage rows with explicit blank unavailable values."""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CATALOG_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for entry in rows:
            writer.writerow(entry)


__all__ = ("harmonic_family_catalog", "write_harmonic_family_catalog",
           "CATALOG_FILE", "CATALOG_COLUMNS")
