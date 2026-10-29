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

import numpy as np

# The GUI launches this file as a script; make all solver module folders
# available just as in the main runner, independently of the launch cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cylinder_paths import activate
activate()


def _summary(root):
    path = root / "FINAL_RUN_SUMMARY.json"
    if not path.is_file():
        archive = next((root / name for name in ("STOPPED_RUN.json", "FAILED_RUN.json")
                        if (root / name).is_file()), None)
        if archive is None:
            raise ValueError("No final, stopped or failed run record found")
        data = json.loads(archive.read_text(encoding="utf-8"))
        data["solver_backend"] = "VARIATIONAL_REFERENCE"
        data["accepted_steps"] = data.get("accepted_steps_verified", 0)
        data["n_modes"] = data.get("requested_steps_or_modes")
        case_path = root / "REFERENCE_CASE.json"
        if case_path.is_file():
            case = json.loads(case_path.read_text(encoding="utf-8"))
            theory = str(case.get("theory", ""))
            data["theory"] = ("CUF_" + str(case.get("cuf_model", ""))
                              if theory.upper() == "CUF" else theory)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        for name in ("STOPPED_RUN.json", "FAILED_RUN.json"):
            if (root / name).is_file():
                data["status"] = json.loads((root / name).read_text(encoding="utf-8"))["status"]
                break
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


def _modal(root, summary, dest, mode, *, partial=False):
    filename = root / ("MODAL_PARTIAL_MODES.npz" if partial else "REFERENCE_MODAL_MODES.npz")
    if not filename.is_file():
        raise ValueError("REFERENCE_MODAL_MODES.npz not found; MODAL output incomplete")
    with np.load(filename, allow_pickle=False) as file:
        frequency = np.asarray(file["frequencies_hz"], dtype=float)
        omega = np.asarray(file["omega_rad_s"], dtype=float)
        eigen = np.asarray(file["omega_squared_rad2_s2"], dtype=float)
        fields = np.asarray(file["mechanical_modes"], dtype=float)
        x = np.asarray(file["x_m"], dtype=float)
        s = np.asarray(file["s_m"], dtype=float)
        electrical = (np.asarray(file["electrical_modes"], dtype=float)
                      if "electrical_modes" in file else None)
    if (frequency.ndim != 1 or len(frequency) == 0 or
            not np.isfinite(frequency).all() or fields.ndim != 4 or
            fields.shape[0] != len(frequency) or
            fields.shape[2:] != (len(x), len(s))):
        raise ValueError("Saved MODAL frequency/mode arrays have incompatible shapes")
    _write_csv(dest / "FREQUENCIES.csv",
               ("mode", "frequency_hz", "omega_rad_s", "omega_squared_rad2_s2"),
               ((i+1, frequency[i], omega[i], eigen[i]) for i in range(len(frequency))))
    workbook = root / "00_MODAL_RESULTS.xlsx"
    if workbook.is_file():
        # The solver's original, self-describing Excel workbook is authoritative.
        import shutil
        shutil.copy2(workbook, dest / "00_MODAL_RESULTS.xlsx")
    if partial:
        from cylinder_excel_tables import LabeledWorkbook, write_labeled_table
        theory = str(summary.get("theory", "")).upper()
        if theory.startswith("CUF_TE") and theory[-1] in "1234":
            partial_labels = [f"{axis}{i}" for axis in ("u", "v", "w")
                              for i in range(int(theory[-1])+1)]
        else:
            partial_labels = (["u", "v", "w"] if theory == "KL" else
                              ["u", "v", "w", "theta_x", "theta_s"])
        if len(partial_labels) != fields.shape[1]:
            raise ValueError("Saved MODAL mechanical components disagree with theory")
        with LabeledWorkbook(str(dest / "MODAL_PARTIAL.xlsx")) as wb:
            write_labeled_table(wb, "STATUS", ("code", "value"),
                (("status", summary.get("status", "PARTIAL")),
                 ("mode_order", "discovery"), ("saved_modes", len(frequency))))
            write_labeled_table(wb, "FREQUENCIES",
                ("discovery_order", "frequency_hz", "omega_rad_s", "omega_squared_rad2_s2"),
                ((i+1, float(frequency[i]), float(omega[i]), float(eigen[i]))
                 for i in range(len(frequency))))
            write_labeled_table(wb, "MODE_FIELDS",
                ("discovery_order", "component", "ix", "js", "x_m", "s_m", "value"),
                ((i+1, partial_labels[c], ix, js, float(x[ix]), float(s[js]), float(fields[i,c,ix,js]))
                 for i in range(len(frequency)) for c in range(fields.shape[1])
                 for ix in range(len(x)) for js in range(len(s))))
            if electrical is not None:
                if electrical.ndim != 4 or electrical.shape[0] != len(frequency) or electrical.shape[2:] != (len(x), len(s)):
                    raise ValueError("Saved MODAL electric arrays have incompatible shapes")
                write_labeled_table(wb, "ELECTRIC_MODES",
                    ("discovery_order", "component", "ix", "js", "value"),
                    ((i+1, f"phi{c}", ix, js, float(electrical[i,c,ix,js]))
                     for i in range(len(frequency)) for c in range(electrical.shape[1])
                     for ix in range(len(x)) for js in range(len(s))))
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
        if mode == "FULL":
            import shutil
            for source in root.rglob("*"):
                relative = source.relative_to(root)
                if (not source.is_file() or source.is_symlink() or
                    relative.parts[0] == "EXPORT" or
                    source.suffix.lower() not in (".xlsx", ".xls", ".csv", ".json", ".npz", ".txt", ".log")):
                    continue
                destination = target / "FILES" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            from cylinder_excel_tables import LabeledWorkbook, write_labeled_table
            with LabeledWorkbook(str(target / "RUN_STATUS.xlsx")) as wb:
                write_labeled_table(wb, "STATUS", ("code", "value"),
                    ((key, summary.get(key, "")) for key in
                     ("calculation_kind", "status", "method", "rank", "theory",
                      "n_modes", "accepted_steps", "error_type", "error_detail")))
        if kind == "MODAL":
            complete = (root / "REFERENCE_MODAL_MODES.npz").is_file()
            partial = (root / "MODAL_PARTIAL_MODES.npz").is_file()
            if complete or partial:
                _modal(root, summary, target, mode, partial=not complete)
            elif summary.get("status") not in ("STOPPED_BY_USER", "FAILED"):
                raise ValueError("No saved MODAL forms to export")
        else:
            from cylinder_reference_progress_adapter import accepted_reference_steps
            if accepted_reference_steps(root):
                _physical(root, summary, kind, target, mode)
            elif summary.get("status") not in ("STOPPED_BY_USER", "FAILED"):
                raise ValueError("No accepted reference STEP NPZ files to export")
    else:
        raise ValueError(f"Unknown export mode: {mode}")
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
