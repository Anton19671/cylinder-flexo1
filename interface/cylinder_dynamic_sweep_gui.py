#!/usr/bin/env python3
"""Tk launcher for independent dynamic bifurcation studies of the full shell.

Only the already-saved DYNAMIC CASE is used as the base for each study.
This module is UI orchestration, not a new physical solver or continuation method.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cylinder_dynamic_parameter_sweep import ALLOWED_PARAMETERS, plan_parameter_sweep
from cylinder_dynamic_analytics import PROBE_LOCATIONS

# Each value is the actual physical CASE key, not a freely evaluated string.
PARAMETER_LABELS = (
    ("load_definition.q", "q: \u043d\u0430\u0433\u0440\u0443\u0437\u043a\u0430"),
    ("dynamics.time_law.amplitude", "F(t): \u0430\u043c\u043f\u043b\u0438\u0442\u0443\u0434\u0430"),
    ("dynamics.time_law.frequency_hz", "F(t): \u0447\u0430\u0441\u0442\u043e\u0442\u0430, Hz"),
    ("dynamics.time_law.phase_rad", "F(t): \u0444\u0430\u0437\u0430, rad"),
    ("material.E0", "E: \u043c\u043e\u0434\u0443\u043b\u044c, Pa"),
    ("material.nu0", "nu: \u043f\u0443\u0430\u0441\u0441\u043e\u043d"),
    ("micro.ell_m", "ell: \u0434\u043b\u0438\u043d\u0430, m"),
    ("micro.mu1_c_per_m", "mu1: C/m"),
    ("micro.mu2_c_per_m", "mu2: C/m"),
    ("micro.kappa_f_per_m", "kappa: F/m"),
    ("dynamics.density_kg_m3", "rho: kg/m3"),
)


def _scalar_at_path(case, path):
    value = case
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def available_sweep_parameters(case):
    """Offer only allowed physical inputs actually present in this saved CASE."""
    if not isinstance(case, dict) or str(case.get("calculation_kind", "")).upper() != "DYNAMIC":
        return ()
    if str(case.get("solver_backend", "")).upper() != "VARIATIONAL_REFERENCE":
        return ()
    return tuple((path, label, _scalar_at_path(case, path))
                 for path, label in PARAMETER_LABELS
                 if path in ALLOWED_PARAMETERS and _scalar_at_path(case, path) is not None)


def build_sweep_spec(base_case, parameter_path, lower, upper, count, output_dir,
                     *, probe_name="P01_CENTER_0", transient_fraction=.5,
                     export_each_run=True):
    """Pure validated UI-to-JSON conversion, with inclusive linear endpoints."""
    base_path = Path(base_case).expanduser().resolve()
    try:
        case = json.loads(base_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read saved base CASE: {exc}") from exc
    paths = {row[0] for row in available_sweep_parameters(case)}
    if parameter_path not in paths:
        raise ValueError("Choose an existing permitted scalar from a saved DYNAMIC CASE")
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 256:
        raise ValueError("Number of parameter calculations must be 2..256")

    def number(value, title):
        if isinstance(value, bool):
            raise ValueError(f"{title} must be a real number")
        try:
            result = float(str(value).strip().replace(",", "."))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{title} must be a real number") from exc
        if not math.isfinite(result):
            raise ValueError(f"{title} must be finite")
        return result

    low, high = number(lower, "First value"), number(upper, "Last value")
    if low == high:
        raise ValueError("Parameter endpoints must be distinct")
    fraction = number(transient_fraction, "Transient fraction")
    if not 0 <= fraction < .9:
        raise ValueError("Transient fraction must be in [0, 0.9)")
    if not isinstance(export_each_run, bool):
        raise ValueError("Per-run Excel output flag must be boolean")
    if probe_name not in {p[0] for p in PROBE_LOCATIONS}:
        raise ValueError("Choose one of the seven standard monitor probes")
    folder = str(output_dir).strip()
    if not folder:
        raise ValueError("Select the output folder for this study")
    target = Path(folder).expanduser().resolve()
    if target == base_path or target == base_path.parent:
        raise ValueError("Output directory must be separate from the source CASE")
    # Compute linear endpoints in a way that preserves the explicitly entered
    # final value. A duplicate after rounding invalidates the physical sweep.
    values = [low + (high - low) * (i / (count - 1)) for i in range(count)]
    values[-1] = high
    if not all(math.isfinite(x) for x in values) or len(set(values)) != count:
        raise ValueError("Range/step generate duplicate or nonfinite values")
    # Labels are ASCII because the exporter uses them as workbook identifiers.
    short_name = {"load_definition.q": "pressure_q"}.get(
        parameter_path, parameter_path.split(".")[-1])
    return {
        "base_case": str(base_path),
        "parameter_path": parameter_path,
        "values": values,
        "output_dir": str(target),
        "parameter_name": short_name,
        "probe_name": probe_name,
        "transient_fraction": fraction,
        "export_each_run": export_each_run,
    }


def save_sweep_spec(spec_file, spec):
    """Store a study separately from its results, then run pure sweep planning."""
    path = Path(spec_file).expanduser().resolve()
    output = Path(spec["output_dir"]).resolve()
    if path == output or output in path.parents:
        raise ValueError("Save the study JSON outside its output directory")
    if not path.parent.is_dir():
        raise ValueError("Study JSON parent directory must already exist")
    if path == Path(spec["base_case"]).resolve():
        raise ValueError("Never overwrite the source CASE with a sweep specification")
    if path.exists():
        raise ValueError("Study JSON already exists; use a new filename")
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    try:
        plan_parameter_sweep(path)
    except Exception:
        # This is our newly written file, not a pre-existing user document.
        path.unlink()
        raise
    return path


class DynamicSweepDialog(tk.Toplevel):
    """Parameter sweep controls. All solver work runs in the root GUI thread pool."""
    def __init__(self, parent, case_path):
        super().__init__(parent)
        self.parent = parent
        self.base_path = Path(case_path).resolve()
        self.base = json.loads(self.base_path.read_text(encoding="utf-8-sig"))
        self.parameters = available_sweep_parameters(self.base)
        if not self.parameters:
            raise ValueError("Save and select a supported DYNAMIC CASE first")
        self.title("\u0414\u0438\u043d\u0430\u043c\u0438\u043a\u0430: \u0431\u0438\u0444\u0443\u0440\u043a\u0430\u0446\u0438\u0438")
        self.geometry("740x520")
        self.transient(parent)
        self.resizable(True, True)
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        self.labels = {path: f"{label}  [{path}]" for path, label, _ in self.parameters}
        self.paths_by_label = {v: k for k, v in self.labels.items()}
        self.parameter = tk.StringVar(value=self.labels[self.parameters[0][0]])
        initial = self.parameters[0][2]
        self.low = tk.StringVar(value=f"{initial:.9g}")
        self.high = tk.StringVar(value=f"{initial*1.5 if initial else .01:.9g}")
        self.count = tk.StringVar(value="10")
        self.transient_fraction = tk.StringVar(value="0.5")
        self.probe = tk.StringVar(value=PROBE_LOCATIONS[0][0])
        self.export_each = tk.BooleanVar(value=True)
        default_output = self.base_path.parent / "OUTPUT_DYNAMIC_BIFURCATION"
        self.output_dir = tk.StringVar(value=str(default_output))

        header = f"CASE: {self.base_path.name} | {self.base.get('method', '')}"
        if str(self.base.get("method", "")).upper() == "MVI":
            header += f" | rank={self.base.get('rank')}"
        ttk.Label(frame, text=header, wraplength=670).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Label(frame, text="\u0418\u0437\u043c\u0435\u043d\u044f\u0435\u043c\u044b\u0439 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440").grid(
            row=1, column=0, sticky="w", pady=5)
        selection = ttk.Combobox(frame, textvariable=self.parameter,
                                 values=tuple(self.labels.values()),
                                 state="readonly", width=60)
        selection.grid(row=1, column=1, columnspan=2, sticky="ew", pady=5)
        selection.bind("<<ComboboxSelected>>", self._on_parameter)

        def input_row(number, label, variable):
            ttk.Label(frame, text=label).grid(row=number, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=variable).grid(
                row=number, column=1, columnspan=2, sticky="ew", pady=6)

        input_row(2, "\u041d\u0430\u0447\u0430\u043b\u043e \u0434\u0438\u0430\u043f\u0430\u0437\u043e\u043d\u0430", self.low)
        input_row(3, "\u041a\u043e\u043d\u0435\u0446 \u0434\u0438\u0430\u043f\u0430\u0437\u043e\u043d\u0430", self.high)
        input_row(4, "\u0422\u043e\u0447\u0435\u043a \u0432 \u0441\u0435\u0440\u0438\u0438 (2-256)", self.count)
        ttk.Label(frame, text="\u041a\u043e\u043d\u0442\u0440\u043e\u043b\u044c\u043d\u0430\u044f \u0442\u043e\u0447\u043a\u0430").grid(
            row=5, column=0, sticky="w", pady=6)
        ttk.Combobox(frame, textvariable=self.probe,
                     values=tuple(x[0] for x in PROBE_LOCATIONS), state="readonly",
                     width=36).grid(row=5, column=1, columnspan=2, sticky="ew", pady=6)
        input_row(6, "\u041e\u0442\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u043f\u0435\u0440\u0435\u0445\u043e\u0434\u043d\u043e\u0439 \u043f\u0440\u043e\u0446\u0435\u0441\u0441", self.transient_fraction)
        ttk.Label(frame, text="\u041f\u0430\u043f\u043a\u0430 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432").grid(
            row=7, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.output_dir).grid(
            row=7, column=1, sticky="ew", pady=6)
        ttk.Button(frame, text="...", width=5, command=self._browse_output).grid(
            row=7, column=2, padx=(6, 0))
        ttk.Checkbutton(frame,
                        text="Excel \u0434\u043b\u044f \u043a\u0430\u0436\u0434\u043e\u0433\u043e \u0440\u0430\u0441\u0447\u0451\u0442\u0430",
                        variable=self.export_each).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=6)
        steps = self.base.get("dynamics", {}).get("n_steps", "?")
        info = ("\u0421\u0435\u0440\u0438\u044f \u043d\u0435 \u043c\u0435\u043d\u044f\u0435\u0442 \u0442\u0435\u043e\u0440\u0438\u044e, \u0441\u0435\u0442\u043a\u0443, \u0440\u0430\u043d\u0433 \u041c\u0412\u0418 \u0438 \u0448\u0430\u0433 \u043f\u043e \u0432\u0440\u0435\u043c\u0435\u043d\u0438. "
                f"n_steps={steps}. "
                "\u0414\u043b\u044f \u0431\u0438\u0444\u0443\u0440\u043a\u0430\u0446\u0438\u0439 \u043d\u0443\u0436\u0435\u043d \u0434\u043b\u0438\u043d\u043d\u044b\u0439 \u0443\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0432\u0448\u0438\u0439\u0441\u044f \u0440\u044f\u0434.")
        ttk.Label(frame, text=info, wraplength=665, foreground="#835000").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(8, 14))
        buttons = ttk.Frame(frame)
        buttons.grid(row=10, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Button(buttons, text="\u041f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c \u0441\u0435\u0440\u0438\u044e",
                   command=lambda: self._submit(True)).pack(side="left", padx=4)
        ttk.Button(buttons, text="\u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0441\u0435\u0440\u0438\u044e",
                   command=lambda: self._submit(False)).pack(side="left", padx=4)
        ttk.Button(buttons, text="\u0417\u0430\u043a\u0440\u044b\u0442\u044c",
                   command=self.destroy).pack(side="right", padx=4)
        frame.columnconfigure(1, weight=1)

    def _on_parameter(self, _event=None):
        key = self.paths_by_label[self.parameter.get()]
        value = dict((p, initial) for p, _, initial in self.parameters)[key]
        self.low.set(f"{value:.9g}")
        self.high.set(f"{value*1.5 if value else .01:.9g}")

    def _browse_output(self):
        parent = filedialog.askdirectory(parent=self, title="\u041f\u0430\u043f\u043a\u0430 \u0434\u043b\u044f \u0441\u0435\u0440\u0438\u0438")
        if parent:
            self.output_dir.set(str(Path(parent) / "OUTPUT_DYNAMIC_BIFURCATION"))

    def _submit(self, preflight_only):
        if self.parent.running:
            messagebox.showwarning("\u0420\u0430\u0441\u0447\u0451\u0442", "\u0423\u0436\u0435 \u0438\u0434\u0451\u0442 \u0440\u0430\u0441\u0447\u0451\u0442", parent=self)
            return
        try:
            key = self.paths_by_label[self.parameter.get()]
            raw_count = self.count.get().strip()
            if not raw_count.isdecimal():
                raise ValueError("Number of parameter calculations must be an integer")
            spec = build_sweep_spec(
                self.base_path, key, self.low.get(), self.high.get(), int(raw_count),
                self.output_dir.get(), probe_name=self.probe.get(),
                transient_fraction=self.transient_fraction.get(),
                export_each_run=self.export_each.get())
            destination = filedialog.asksaveasfilename(
                parent=self, title="\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c JSON \u0441\u0435\u0440\u0438\u0438",
                initialdir=str(self.base_path.parent),
                initialfile="SWEEP_" + spec["parameter_name"] + ".json",
                defaultextension=".json", filetypes=(("JSON", "*.json"),))
            if not destination:
                return
            path = save_sweep_spec(destination, spec)
            self.parent.start_parameter_sweep(path, preflight_only=preflight_only)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("\u0421\u0435\u0440\u0438\u044f", str(exc), parent=self)


__all__ = ("PARAMETER_LABELS", "available_sweep_parameters",
           "build_sweep_spec", "save_sweep_spec", "DynamicSweepDialog")
