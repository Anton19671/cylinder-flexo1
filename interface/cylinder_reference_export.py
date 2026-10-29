#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only export of VARIATIONAL_REFERENCE results.

STATIC/DYNAMIC/MODAL are handled independently. This module never calls a
numerical solver, reinterprets CUF thickness coefficients, or changes a STEP.
All plots are built exclusively from saved physical result arrays.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# This file is launched directly by the Tk GUI, not as a package module.
# In that process sys.path[0] is interface/, so mechanics/ and numerics/
# must be activated explicitly before importing postprocessing modules.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from cylinder_paths import activate as _activate_paths
_activate_paths()

import numpy as np


def _summary(root):
    path = root / "FINAL_RUN_SUMMARY.json"
    if not path.is_file():
        raise ValueError("FINAL_RUN_SUMMARY.json not found: finish the calculation first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("solver_backend") != "VARIATIONAL_REFERENCE":
        raise ValueError("This exporter accepts VARIATIONAL_REFERENCE results only")
    kind = str(data.get("calculation_kind", "")).upper()
    if kind not in ("STATIC", "DYNAMIC", "MODAL"):
        raise ValueError("Unknown reference calculation_kind")
    return data, kind


def _write_csv(path, headings, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)
        writer.writerows(rows)


def _write_info(directory, summary, kind):
    lines = [
        "Cylinder-Flexo1: export of saved results; the solver is not rerun.",
        f"Calculation: {kind}",
        f"Theory: {summary.get('theory', '-')}",
        f"Method: {summary.get('method', '-')}",
        f"Rank: {summary.get('rank', '-')}",
        f"Status: {summary.get('status', '-')}",
        f"Requested modes: {summary.get('n_modes', '-')}",
        f"Accepted steps: {summary.get('accepted_steps', '-')}",
        "Only completed, physically saved quantities are exported.",
    ]
    (directory / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _modal(root, summary, dest, mode):
    filename = root / "REFERENCE_MODAL_MODES.npz"
    if not filename.is_file():
        raise ValueError("REFERENCE_MODAL_MODES.npz not found; MODAL output incomplete")
    with np.load(filename, allow_pickle=False) as file:
        frequency = np.asarray(file["frequencies_hz"], dtype=float)
        omega = np.asarray(file["omega_rad_s"], dtype=float)
        eigen = np.asarray(file["omega_squared_rad2_s2"], dtype=float)
        fields = np.asarray(file["mechanical_modes"], dtype=float)
        x = np.asarray(file["x_m"], dtype=float)
        s = np.asarray(file["s_m"], dtype=float)
        electric = (np.asarray(file["electrical_modes"], dtype=float)
                    if "electrical_modes" in file else None)
    if (frequency.ndim != 1 or len(frequency) == 0 or
            not np.isfinite(frequency).all() or fields.ndim != 4 or
            fields.shape[0] != len(frequency) or
            fields.shape[2:] != (len(x), len(s))):
        raise ValueError("Saved MODAL frequency/mode arrays have incompatible shapes")
    _write_csv(dest / "FREQUENCIES.csv",
               ("mode", "frequency_hz", "omega_rad_s", "omega_squared_rad2_s2"),
               ((i+1, frequency[i], omega[i], eigen[i]) for i in range(len(frequency))))
    # The original frequency workbook belongs to the result root only.
    # Older runs may still have 00_MODAL_RESULTS.xlsx. Do not duplicate it.
    # Independent shape inspection is read-only: no modal solve or rerun.
    from cylinder_modal_diagnostics import (
        inspect_modal_shapes, recover_modal_displacements)
    scheme = str(summary.get("operator_scheme", "UNKNOWN"))
    if scheme == "UNKNOWN":
        case_file = root / "REFERENCE_CASE.json"
        if case_file.is_file():
            source_case = json.loads(case_file.read_text(encoding="utf-8"))
            grid = source_case.get("mvi_grid" if summary.get("method") == "MVI"
                                   else "fdm_grid", {})
            scheme = str(grid.get("operator_scheme", "UNKNOWN"))
    diagnostic = inspect_modal_shapes(fields, frequency, summary.get("theory"),
                                      operator_scheme=scheme)
    _write_csv(dest / "MODE_CLASSIFICATION.csv",
               ("mode", "frequency_hz", "harmonic", "angular_phase_deg",
                "dominant_component", "mode_type", "u_energy_fraction",
                "v_energy_fraction", "w_energy_fraction"),
               ([item.get(key, "") for key in
                 ("mode", "frequency_hz", "harmonic", "angular_phase_deg",
                  "dominant_component", "mode_type", "u_energy_fraction",
                  "v_energy_fraction", "w_energy_fraction")]
                for item in diagnostic["modes"]))
    _write_csv(dest / "FREQUENCY_DOUBLETS.csv",
               ("mode_i", "mode_j", "frequency_gap_hz", "euclidean_mac",
                "phase_difference_deg", "classification"),
               ([item.get(key, "") for key in
                 ("mode_i", "mode_j", "frequency_gap_hz", "euclidean_mac",
                  "phase_difference_deg", "classification")]
                for item in diagnostic["pairs"]))
    _write_csv(dest / "MODE_MAC_EUCLIDEAN.csv",
               ("mode", *(f"mode_{i+1}" for i in range(len(frequency)))),
               ((i+1, *map(float, diagnostic["euclidean_mac"][i]))
                for i in range(len(frequency))))
    if mode == "FULL":
        case_file = root / "REFERENCE_CASE.json"
        if not case_file.is_file():
            raise ValueError("FULL MODAL export needs saved REFERENCE_CASE.json")
        case_data = json.loads(case_file.read_text(encoding="utf-8"))
        z, displacement = recover_modal_displacements(
            fields, summary["theory"], float(case_data["geometry"]["h"]))
        _write_csv(dest / "MODE_DISPLACEMENTS_THROUGH_THICKNESS.csv",
                   ("mode", "ix", "js", "x_m", "s_m", "theta_rad",
                    "z_m", "u_shape", "v_shape", "w_shape"),
                   ((k+1, ix, js, float(x[ix]), float(s[js]),
                     float(s[js]/float(case_data["geometry"]["R"])),
                     float(z[iz]), *map(float, displacement[k, iz, :, ix, js]))
                    for k in range(len(frequency))
                    for iz in range(len(z))
                    for ix in range(len(x)) for js in range(len(s))))
    if mode == "FULL":
        from cylinder_modal_full_export import export_modal_full
        export_modal_full(root, dest, summary, frequency, fields, electric,
                          x, s, diagnostic)
    if mode == "FAST":
        return
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(frequency)+1), frequency, "o-")
    ax.set(xlabel="Mode number", ylabel="Natural frequency [Hz]",
           title="Cylinder: natural frequencies of the saved MODAL result")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest / "FREQUENCIES.png", dpi=170)
    plt.close(fig)
    theory = str(summary.get("theory", "")).upper()
    if theory.startswith("CUF_TE") and theory[-1] in "1234":
        labels = [f"{axis}{i}" for axis in ("u", "v", "w")
                  for i in range(int(theory[-1])+1)]
        radial = 2 * (int(theory[-1])+1)
    else:
        labels = (["u", "v", "w"] if theory == "KL" else
                  ["u", "v", "w", "theta_x", "theta_s"])
        radial = 2
    if len(labels) != fields.shape[1] or radial >= fields.shape[1]:
        raise ValueError("Saved MODAL mechanical components disagree with theory")
    # A mode is mass-normalized; its plot is a mode SHAPE, not a measured deflection.
    components = range(fields.shape[1]) if mode == "FULL" else (radial,)
    for i in range(len(frequency)):
        for c in components:
            fig, ax = plt.subplots(figsize=(8, 5))
            data = fields[i, c].T
            extent = (float(x[0]), float(x[-1]), float(s[0]),
                      float(s[-1]) if len(s) > 1 else 0.0)
            image = ax.imshow(data, origin="lower", aspect="auto", extent=extent)
            fig.colorbar(image, ax=ax, label="Mass-normalized modal coefficient")
            ax.set(xlabel="x [m]", ylabel="s [m]",
                   title=f"Mode {i+1}, f={frequency[i]:.6g} Hz: {labels[c]}")
            fig.tight_layout()
            fig.savefig(dest / f"MODE_{i+1:02d}_{labels[c]}.png", dpi=160)
            plt.close(fig)


def _physical(root, summary, kind, dest, mode):
    # Reference step catalog checks summary acceptance AND the actual saved NPZ.
    from cylinder_reference_progress_adapter import accepted_reference_steps
    from cylinder_reference_results import reference_step_data
    steps = accepted_reference_steps(root)
    if not steps:
        raise ValueError("No accepted reference STEP NPZ files to export")
    headings = ("step", "state_file", "status")
    _write_csv(dest / "ACCEPTED_STEPS.csv", headings,
               ((row["index"], row["state_file"], "ACCEPTED") for row in steps))
    if mode == "FAST":
        return
    from matplotlib import pyplot as plt
    for row in steps:
        step = int(row["index"])
        record = reference_step_data(root, step)
        field_names = (("w(z)",) if mode == "STANDARD" else
                       tuple(record["volumes"].keys()))
        z = np.asarray(record["z"], float)
        iz = int(np.argmin(np.abs(z)))
        for field in field_names:
            data = np.asarray(record["volumes"][field], float)
            if data.ndim != 3 or not np.isfinite(data).all():
                raise ValueError(f"Invalid saved physical field: {field}")
            plane = data[iz]
            fig, ax = plt.subplots(figsize=(8, 5))
            image = ax.imshow(plane, origin="lower", aspect="auto")
            fig.colorbar(image, ax=ax)
            ax.set(xlabel="x node", ylabel="s node",
                   title=f"{kind} STEP {step}, {field}, z={z[iz]:.6g} m")
            fig.tight_layout()
            safe_name = field.replace("/", "_").replace("(", "").replace(")", "")
            fig.savefig(dest / f"STEP_{step:04d}_{safe_name}.png", dpi=160)
            plt.close(fig)


def _quick(root, summary, kind):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    report = root / "REPORT"
    report.mkdir(exist_ok=True)
    pdf = report / "QUICK_SUMMARY_RU_EN.pdf"
    rows = [
        f"Calculation: {kind}", f"Theory: {summary.get('theory', '-')}",
        f"Method: {summary.get('method', '-')}",
        f"Rank: {summary.get('rank', '-')}",
        f"Status: {summary.get('status', '-')}",
    ]
    if kind == "MODAL":
        f = summary.get("frequencies_hz", [])
        if not isinstance(f, list) or not f:
            raise ValueError("Final summary contains no saved natural frequencies")
        rows.extend(f"Mode {i}: {float(value):.8g} Hz"
                    for i, value in enumerate(f, 1))
    else:
        from cylinder_reference_progress_adapter import accepted_reference_steps
        steps = accepted_reference_steps(root)
        if not steps:
            raise ValueError("No saved accepted steps for quick summary")
        rows.append(f"Accepted saved steps: {len(steps)}")
        rows.extend(f"Step {int(row['index'])}: {row['state_file']}"
                    for row in steps[:25])
    # PDF includes exact saved numbers only; no unstored physical results.
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.05, 0.96, "Cylinder-Flexo1 / Quick Summary",
            fontsize=17, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.91, "\n".join(rows), fontsize=11,
            va="top", family="monospace", transform=ax.transAxes,
            linespacing=1.6)
    with PdfPages(pdf) as document:
        document.savefig(fig)
    return pdf


def export(root, mode):
    root = Path(root).expanduser().resolve()
    summary, kind = _summary(root)
    mode = str(mode).upper()
    if mode == "QUICK":
        target = _quick(root, summary, kind)
    elif mode in ("FAST", "STANDARD", "FULL"):
        target = root / "EXPORT" / mode
        target.mkdir(parents=True, exist_ok=True)
        _write_info(target, summary, kind)
        if kind == "MODAL":
            _modal(root, summary, target, mode)
        else:
            _physical(root, summary, kind, target, mode)
    else:
        raise ValueError(f"Unknown export mode: {mode}")
    # Never announce success unless the promised files actually exist.
    if kind == "MODAL" and mode == "FULL":
        from cylinder_modal_full_export import (
            CHARACTERISTICS_XLSX, FIELDS_XLSX, PLOTS_DIR)
        required = (root / CHARACTERISTICS_XLSX,
                    root / FIELDS_XLSX,
                    target / PLOTS_DIR / "ФОРМА_01" / "01_3D_ФОРМА.png")
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise RuntimeError("MODAL_FULL_EXPORT_INCOMPLETE: " + "; ".join(missing))
    print(f"REFERENCE EXPORT {kind} {mode} SAVED: {target}", flush=True)
    return target


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise ValueError("Usage: cylinder_reference_export.py OUTPUT_DIR FAST|STANDARD|FULL|QUICK")
    export(argv[0], argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
