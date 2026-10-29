#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theory-aware variational boundary GUI: edits the real physical CASE.

The displayed mechanical components come from the same ordering as the
boundary assembler. G3D/Gz and electrode controls are taken from this CASE,
not guessed from the legacy GUI state. Unrecognized JSON keys are retained.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from cylinder_variational_boundary import validate_boundary_passport
from cylinder_boundary_ui_model import boundary_ui_model, theory_from_case


def merge_boundary_controls(spec, mechanical, electrical, previous=None):
    """Merge *changed* widgets into JSON without erasing independent expert data."""
    if not isinstance(spec, dict):
        raise ValueError("variational BC must be a JSON object")
    data = copy.deepcopy(spec)
    old_m, old_e = previous if previous is not None else ({}, {})
    for side, selection in mechanical.items():
        item = data.setdefault("mechanical", {}).setdefault(side, {})
        for key, value in selection.items():
            if previous is None or value != old_m.get(side, {}).get(key):
                item[key] = copy.deepcopy(value)
        if item.get("support", "FREE").upper() != "CUSTOM":
            item.pop("fixed_components", None)
    for side, selection in electrical.items():
        before = old_e.get(side, {})
        mode_changed = previous is None or selection["mode"] != before.get("mode")
        value_changed = previous is None or selection["value"] != before.get("value")
        if not mode_changed and not value_changed:
            continue
        item = data.setdefault("electrical", {}).setdefault(side, {})
        if mode_changed:
            item["mode"] = selection["mode"]
        if item.get("mode", "INSULATED") == "INSULATED":
            item.pop("value", None)
            item.pop("time_law", None)
        elif selection["value"].strip():
            raw = selection["value"].strip().replace(",", ".")
            item["value"] = float(raw)
        elif value_changed or "value" not in item:
            raise ValueError(side + ": введите напряжение или заряд")
    return data


def _adapt_zero_vectors(spec, component_count):
    """Remove incompatible all-zero defaults; never remap nonzero physical BC."""
    result = copy.deepcopy(spec)
    mechanical = result.get("mechanical", {})
    for side in ("left", "right"):
        site = mechanical.get(side, {})
        for key in ("prescribed", "traction", "double_traction", "triple_traction",
                    "normal_double_traction", "normal_triple_traction",
                    "spring", "spring_reference"):
            value = site.get(key)
            if not isinstance(value, list) or len(value) == component_count:
                continue
            if all(float(v) == 0.0 for v in value):
                site.pop(key)
            else:
                raise ValueError(
                    f"{side}.{key}: было {len(value)} компонент, выбранная теория "
                    f"требует {component_count}; сопоставьте физические компоненты вручную")
    return result


def open_boundary_editor(gui, *, reference_case, reference_path=None, on_apply=None):
    """Saved CASE -> atomic write; CASE editor -> in-memory callback (no disk write)."""
    if not isinstance(reference_case, dict):
        raise ValueError("Параметры вариационного CASE не переданы")
    if reference_path is None and on_apply is None:
        raise ValueError("Не указан путь CASE или обработчик изменения")
    model = boundary_ui_model(reference_case)
    spec = _adapt_zero_vectors(
        (reference_case.get("boundary_conditions") or {}).get("variational") or {},
        model["n_components"])
    path = Path(reference_path) if reference_path is not None else None

    win = tk.Toplevel(gui)
    win.title(f"Краевые условия: {model['theory']} · {model['gradient_caption']}")
    win.geometry("970x760")
    win.transient(gui)
    win.columnconfigure(0, weight=1)
    win.rowconfigure(1, weight=1)

    header = ttk.LabelFrame(win, text="Математическая модель для КУ", padding=8)
    header.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
    ttk.Label(header, text=f"{model['theory']}: {model['n_components']} механических компонент").grid(
        row=0, column=0, sticky="w", padx=4)
    ttk.Label(header, text="Градиент:").grid(row=0, column=1, sticky="w", padx=8)
    grad = tk.StringVar(value=model["gradient_model"])
    gradcombo = ttk.Combobox(header, textvariable=grad, width=13,
                             values=("G3D", "Gz"), state=("readonly" if
                             model["gradient_model"] != "OFF" else "disabled"))
    gradcombo.grid(row=0, column=2, sticky="w")
    caption = ttk.Label(header, text=model["gradient_caption"], wraplength=480)
    caption.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 2))
    ttk.Label(header, text=(
        f"Электрическая модель: {'ВКЛ' if model['electric_enabled'] else 'ВЫКЛ'}; "
        f"потенциал: {', '.join(model['electric_components']) or 'не решается'}"
    )).grid(row=2, column=0, columnspan=3, sticky="w")
    ttk.Label(header, text=(
        "Смена KL / FSDT / Reddy / Carrera TE1–TE4 и флексосвязи выполняется "
        "в редакторе CASE. Здесь набор КУ автоматически соответствует выбранной теории."
    ), wraplength=890).grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))

    def gradient_changed(*_args):
        name = grad.get()
        prefix = ("Gz — усечённая, 6 компонент" if name == "Gz"
                  else "G3D — полная, 18 компонент")
        caption.configure(text=prefix + ". Краевые операторы не заменяются автоматически.")
        win.title(f"Краевые условия: {model['theory']} · {name}")
    grad.trace_add("write", gradient_changed)

    book = ttk.Notebook(win)
    book.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
    mech_tab = ttk.Frame(book, padding=7)
    electric_tab = ttk.Frame(book, padding=7)
    full_tab = ttk.Frame(book, padding=7)
    book.add(mech_tab, text=f"Механика {model['theory']} · {model['n_components']} КУ")
    book.add(electric_tab, text=("Электрика" if model["electric_enabled"] else "Электрика (ВЫКЛ)"))
    book.add(full_tab, text="Полный JSON / время / нагрузки")

    mech_tab.columnconfigure(0, weight=1)
    mech_tab.rowconfigure(0, weight=1)
    canvas = tk.Canvas(mech_tab, highlightthickness=0)
    scroll = ttk.Scrollbar(mech_tab, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scroll.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    scroll.grid(row=0, column=1, sticky="ns")
    mech = ttk.Frame(canvas, padding=5)
    cw = canvas.create_window((0, 0), window=mech, anchor="nw")
    mech.bind("<Configure>", lambda _event: canvas.configure(
        scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(cw, width=event.width))
    mech.columnconfigure(0, weight=1)

    mechanical_vars = {}
    for row_index, side in enumerate(("left", "right")):
        item = (spec.get("mechanical") or {}).get(side) or {}
        frame = ttk.LabelFrame(mech, text="x=0" if side == "left" else "x=L", padding=7)
        frame.grid(row=row_index, column=0, sticky="ew", pady=6)
        frame.columnconfigure(1, weight=1)
        support = tk.StringVar(value=str(item.get("support", "FREE")).upper())
        higher = tk.StringVar(value=str(item.get("higher", "NATURAL")).upper())
        ttk.Label(frame, text="Закрепление").grid(row=0, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=support, values=model["support_choices"],
                     state="readonly", width=27).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(frame, text="Высшие нормальные следы").grid(row=1, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=higher, values=model["higher_choices"],
                     state="readonly", width=34).grid(row=1, column=1, sticky="w", padx=5)
        fixed_label = ttk.Label(frame, text="", wraplength=760)
        fixed_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        custom_frame = ttk.LabelFrame(frame, text="CUSTOM: фиксированные компоненты", padding=5)
        custom_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        custom = []
        selected = set(item.get("fixed_components", ()))
        for i, name in enumerate(model["components"]):
            var = tk.BooleanVar(value=i in selected)
            ttk.Checkbutton(custom_frame, text=f"{i}: {name}", variable=var).grid(
                row=i // 5, column=i % 5, sticky="w", padx=6, pady=2)
            custom.append(var)

        prescribed_box = ttk.LabelFrame(frame, text=(
            "Предписанные значения компонент (для закреплённых неизвестных)"), padding=5)
        prescribed_box.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        existing = item.get("prescribed")
        if existing is None:
            existing = [0.] * model["n_components"]
        if len(existing) != model["n_components"]:
            raise ValueError(f"{side}: несовместимый вектор prescribed из другой теории")
        prescribed = []
        for i, name in enumerate(model["components"]):
            ttk.Label(prescribed_box, text=name).grid(
                row=i // 5 * 2, column=i % 5, sticky="w", padx=5)
            var = tk.StringVar(value=str(existing[i]))
            ttk.Entry(prescribed_box, textvariable=var, width=13).grid(
                row=i // 5 * 2 + 1, column=i % 5, sticky="w", padx=5, pady=1)
            prescribed.append(var)

        def update_support(*_args, support=support, label=fixed_label,
                           custom_frame=custom_frame):
            name = support.get()
            if name == "CUSTOM":
                fixed = "CUSTOM: задайте флаги ниже"
                custom_frame.grid()
            else:
                fixed = "Фиксируются: " + (", ".join(model["support_components"].get(name, ()))
                                            or "нет (свободный торец)")
                custom_frame.grid_remove()
            label.configure(text=fixed)
        support.trace_add("write", update_support)
        update_support()
        mechanical_vars[side] = (support, higher, custom, prescribed)

    ttk.Label(mech, text=(
        "Пружины, торцевые нагрузки, double/triple traction и time_law — "
        "во вкладке «Полный JSON». Невозможные сочетания отклоняет решатель."
    ), wraplength=850).grid(row=2, column=0, sticky="w", pady=9)

    electric_tab.columnconfigure(1, weight=1)
    electric_vars = {}
    for index, side in enumerate(("left", "right", "outer", "inner")):
        item = (spec.get("electrical") or {}).get(side) or {}
        mode = tk.StringVar(value=str(item.get("mode", "INSULATED")).upper())
        value = tk.StringVar(value=str(item.get("value", "")))
        ttk.Label(electric_tab, text=side).grid(row=index, column=0, sticky="w", padx=7, pady=9)
        choices = model["electric_modes"] if model["electric_enabled"] else ("INSULATED",)
        combo = ttk.Combobox(electric_tab, textvariable=mode, values=choices,
                             state=("readonly" if model["electric_enabled"] else "disabled"),
                             width=22)
        combo.grid(row=index, column=1, sticky="w", padx=8)
        entry = ttk.Entry(electric_tab, textvariable=value, width=24)
        entry.grid(row=index, column=2, sticky="w", padx=8)
        def update_electric(*_args, mode=mode, entry=entry):
            entry.configure(state=("normal" if model["electric_enabled"]
                                   and mode.get() != "INSULATED" else "disabled"))
        mode.trace_add("write", update_electric)
        update_electric()
        electric_vars[side] = (mode, value)
    ttk.Label(electric_tab, text=(
        "POTENTIAL: потенциал [В]; CHARGE: свободный заряд электрода [Кл/м²]; "
        "INSULATED: без заданного потока. Условия относятся к реальной "
        "электрической модели CASE, а не к числу механических коэффициентов."
    ), wraplength=770).grid(row=5, column=0, columnspan=3, sticky="w", pady=13)

    full_tab.columnconfigure(0, weight=1)
    full_tab.rowconfigure(0, weight=1)
    editor = tk.Text(full_tab, wrap="none", undo=True, font=("Consolas", 10))
    editor.grid(row=0, column=0, sticky="nsew")
    bar = ttk.Scrollbar(full_tab, command=editor.yview)
    bar.grid(row=0, column=1, sticky="ns")
    editor.configure(yscrollcommand=bar.set)
    editor.insert("1.0", json.dumps(spec, ensure_ascii=False, indent=2))
    ttk.Label(full_tab, text=(
        "Полный JSON сохраняет индивидуальные условия: fixed_components, "
        "traction, double_traction, spring, time_law и электрические значения."
    ), wraplength=800).grid(row=1, column=0, sticky="w", pady=5)

    def selections():
        ms = {}
        for side, (support, higher, custom, prescribed) in mechanical_vars.items():
            ms[side] = {"support": support.get(), "higher": higher.get(),
                        "prescribed": [float(v.get().replace(",", ".")) for v in prescribed]}
            if support.get() == "CUSTOM":
                ms[side]["fixed_components"] = [
                    i for i, value in enumerate(custom) if value.get()]
        es = {}
        if model["electric_enabled"]:
            for side, (mode, value) in electric_vars.items():
                es[side] = {"mode": mode.get(), "value": value.get()}
        return ms, es

    previous = selections()

    def merged():
        ms, es = selections()
        data = json.loads(editor.get("1.0", "end"))
        data = merge_boundary_controls(data, ms, es, previous=previous)
        if not model["electric_enabled"]:
            data["electrical"] = {}
        return data

    def sync():
        nonlocal previous
        try:
            data = merged()
            editor.delete("1.0", "end")
            editor.insert("1.0", json.dumps(data, ensure_ascii=False, indent=2))
            previous = selections()
        except Exception as exc:
            messagebox.showerror("Краевые условия", str(exc), parent=win)

    def apply():
        try:
            data = merged()
            updated = copy.deepcopy(reference_case)
            if model["gradient_model"] != "OFF":
                updated.setdefault("micro", {})["gradient_model"] = grad.get()
            validate_boundary_passport(
                data, electrical_enabled=model["electric_enabled"],
                calculation_kind=updated["calculation_kind"],
                theory=theory_from_case(updated))
            updated.setdefault("boundary_conditions", {})["variational"] = copy.deepcopy(data)
            if on_apply is not None:
                on_apply(data, grad.get())
            else:
                from cylinder_reference_case import prepare_reference_case
                prepare_reference_case(updated, base_dir=path.parent)
                temp = path.with_name(path.name + ".bc.tmp")
                try:
                    temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
                    os.replace(temp, path)
                finally:
                    if temp.exists():
                        temp.unlink()
                gui.adopt(updated, path)
            win.destroy()
        except Exception as exc:
            messagebox.showerror("Некорректные КУ", str(exc), parent=win)

    buttons = ttk.Frame(win, padding=8)
    buttons.grid(row=2, column=0, sticky="ew")
    ttk.Button(buttons, text="Перенести выбор в JSON", command=sync).pack(side="left", padx=4)
    ttk.Button(buttons, text="Применить к CASE", command=apply).pack(side="left", padx=4)
    ttk.Label(buttons, text=(
        "Изменения попадут в расчётный CASE" if path is not None
        else "Изменения попадут в открытый редактор CASE"
    )).pack(side="left", padx=8)
    ttk.Button(buttons, text="Отмена", command=win.destroy).pack(side="right")


__all__ = ("merge_boundary_controls", "open_boundary_editor")
