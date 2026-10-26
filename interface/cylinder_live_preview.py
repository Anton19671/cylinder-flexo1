# -*- coding: utf-8 -*-
"""Input-driven interactive cylinder schematic for the Tk GUI.

This is a visualization of the entered geometry, end supports and load intent,
not a displacement/stress solution and not an exact contour of q(x, theta).
"""
from __future__ import annotations

from math import pi, sin
import tkinter as tk


_LOAD_NAMES = {
    "UNIFORM_RADIAL_FACE": "Равномерное радиальное давление",
    "HARMONIC_RADIAL_FACE": "Гармоническое давление",
    "PATCH_RADIAL_FACE": "Давление на участке",
    "LINE_RADIAL_GENERATOR": "Линейная / полосовая нагрузка",
    "POINT_RADIAL_FORCE": "Сосредоточенная сила",
    "EXTERNAL_NPZ": "Поле нагрузки из NPZ",
}


def bind_cylinder_view(gui, canvas):
    """Rotate and zoom a drawing, never changing calculation parameters."""
    canvas._cyl_drag_x = None

    def start(event):
        canvas._cyl_drag_x = event.x

    def move(event):
        old = canvas._cyl_drag_x
        canvas._cyl_drag_x = event.x
        if old is None:
            return
        gui._cylinder_view_angle = (
            getattr(gui, "_cylinder_view_angle", 0.0) +
            (event.x - old) * 0.016
        ) % (2 * pi)
        gui._schedule_problem_scheme_redraw(18)

    def stop(_event):
        canvas._cyl_drag_x = None

    def zoom(event):
        num = getattr(event, "num", None)
        positive = (num == 4 if num in (4, 5) else
                    getattr(event, "delta", 0) > 0)
        factor = 1.09 if positive else 1 / 1.09
        gui._cylinder_view_zoom = max(
            0.65, min(1.55, getattr(gui, "_cylinder_view_zoom", 1.0) * factor)
        )
        gui._schedule_problem_scheme_redraw(20)
        return "break"

    canvas.bind("<ButtonPress-1>", start)
    canvas.bind("<B1-Motion>", move)
    canvas.bind("<ButtonRelease-1>", stop)
    canvas.bind("<Control-MouseWheel>", zoom)
    canvas.bind("<Control-Button-4>", zoom)
    canvas.bind("<Control-Button-5>", zoom)


def render_cylinder_view(gui, canvas):
    """Draw the selected setup in a bounded, resize-safe Tk canvas."""
    canvas.delete("all")
    w = max(240, canvas.winfo_width())
    h = max(150, canvas.winfo_height())
    angle = getattr(gui, "_cylinder_view_angle", 0.0)
    zoom = getattr(gui, "_cylinder_view_zoom", 1.0)

    def value(key, fallback=""):
        item = gui.vars.get(key)
        try:
            return str(item.get()).strip() if item is not None else str(fallback)
        except Exception:
            return str(fallback)

    def number(key, fallback=0.0):
        try:
            return float(value(key)) if value(key) else float(fallback)
        except (ValueError, TypeError):
            return float(fallback)

    theory = value("theory", "CUF")
    model = value("cuf_model") if theory == "CUF" else ""
    dynamic = value("calculation_kind", "STATIC") == "DYNAMIC"
    micro = value("scale_model", "MACRO") == "MICRO"
    flexo = micro and value("micro_flexoelectric", "OFF") == "ON"
    load = value("load_type", "UNIFORM_RADIAL_FACE").upper()
    direction = value("load_pressure_direction", "INTERNAL").upper()

    canvas.create_text(
        10, 10, anchor="nw", fill="#17324d",
        font=("Segoe UI Semibold", 10),
        text=f"{theory} {model} · {'Динамика' if dynamic else 'Статика'}",
    )
    physics = ("G3D + флексо" if flexo else
               "G3D (флексо ВЫКЛ)" if micro else "Классическая упругость")
    canvas.create_text(w - 10, 10, anchor="ne", fill="#52687a",
                       font=("Segoe UI", 9), text=physics)

    length = max(85.0, min(w - 115.0, w * 0.62 * zoom))
    x0, x1 = (w - length) / 2, (w + length) / 2
    cy = h * 0.48
    radius = min(80.0, h * 0.20 * zoom)
    cap = max(10.0, min(36.0, radius * 0.5))

    # Tube, end contours and a rotating axial guide.  The guide makes it
    # obvious that dragging changes the displayed azimuth, not the load.
    canvas.create_rectangle(x0, cy - radius, x1, cy + radius,
                            fill="#eaf2fa", outline="")
    canvas.create_rectangle(x0, cy - radius * .65, x1, cy - radius * .25,
                            fill="#d8e8f6", outline="")
    canvas.create_line(x0, cy - radius, x1, cy - radius,
                       fill="#24425a", width=2)
    canvas.create_line(x0, cy + radius, x1, cy + radius,
                       fill="#24425a", width=2)
    canvas.create_oval(x0 - cap, cy - radius, x0 + cap, cy + radius,
                       outline="#24425a", fill="#e5f0f8", width=2)
    canvas.create_oval(x1 - cap, cy - radius, x1 + cap, cy + radius,
                       outline="#24425a", fill="#cddfed", width=2)
    seam_y = cy + radius * 0.75 * sin(angle)
    canvas.create_line(x0 + cap, seam_y, x1 - cap, seam_y,
                       fill="#88a9c5", dash=(3, 5))

    # Highlight the axial footprint for localized load specifications.
    L = max(number("L", 1.0), 1e-30)
    localized = load in ("PATCH_RADIAL_FACE", "HARMONIC_RADIAL_FACE",
                         "LINE_RADIAL_GENERATOR")
    if localized:
        a = max(0., min(1., number("load_x1", 0.) / L))
        b = max(0., min(1., number("load_x2", L) / L))
        a, b = min(a, b), max(a, b)
        xa, xb = x0 + length * a, x0 + length * b
        canvas.create_rectangle(xa, cy - radius + 4, xb, cy + radius - 4,
                                outline="#0877bd", width=2, dash=(4, 3))
    elif load == "POINT_RADIAL_FORCE":
        pos = max(0., min(1., number("load_point_x0", L / 2) / L))
        xa = xb = x0 + length * pos
    else:
        xa, xb = x0, x1

    # Representative radial arrows, not a computed continuous load contour.
    if load != "EXTERNAL_NPZ":
        if load == "POINT_RADIAL_FORCE":
            positions = [xa]
        else:
            positions = [xa + (xb - xa) * frac for frac in
                         (0.10, 0.30, 0.50, 0.70, 0.90)]
        for xx in positions:
            theta = number("load_point_theta0", 0.0) if load == "POINT_RADIAL_FORCE" else 0.0
            offset = 0.28 * radius * sin(theta + angle) if load == "POINT_RADIAL_FORCE" else 0.
            y = cy - radius + offset
            if direction == "EXTERNAL":
                start, end = y - 21, y + 5
            else:
                start, end = y + 13, y - 21
            canvas.create_line(xx, start, xx, end, arrow=tk.LAST,
                               fill="#b65318", width=2)
    else:
        canvas.create_text((x0 + x1) / 2, cy, fill="#926100",
                           font=("Segoe UI", 9),
                           text="NPZ: точное распределение здесь не показано")

    # End restraints: display their actual selected codes; no fabricated
    # physical constraints for unset/custom BCs.
    def bc_label(side):
        raw = value(side)
        if not raw:
            return "Ручные КУ" if value("bc_mode") == "INDIVIDUAL" else "не задано"
        # The reference editor already validates the physical end supports.
        # Never import a removed classical preset registry for a drawing.
        return str(raw)

    left, right = bc_label("bc_left"), bc_label("bc_right")
    canvas.create_text(x0 + 8, cy + radius + 17, anchor="nw",
                       font=("Segoe UI Semibold", 9), fill="#17324d",
                       width=max(65, length * .45), text=f"x=0: {left}")
    canvas.create_text(x1 - 8, cy + radius + 17, anchor="ne",
                       font=("Segoe UI Semibold", 9), fill="#17324d",
                       width=max(65, length * .45), text=f"x=L: {right}")

    units = "м" if value("unit_system") == "SI_N_M_PA" else "мм"
    dimensions = ", ".join(
        f"{caption}={value(key)}" for key, caption in
        (("L", "L"), ("R", "R"), ("h", "h")) if value(key)
    )
    load_caption = _LOAD_NAMES.get(load, load)
    if load != "EXTERNAL_NPZ":
        load_caption += " · внутрь" if direction == "EXTERNAL" else " · наружу"
    if dynamic:
        load_caption += f" · F(t): {value('dyn_law', 'HARMONIC')}"
    canvas.create_text(w / 2, h - 36, fill="#17324d",
                       font=("Segoe UI", 9), width=max(180, w - 25),
                       text=f"{dimensions} [{units}]" if dimensions else "Размеры не заданы")
    canvas.create_text(w / 2, h - 14, fill="#52687a",
                       font=("Segoe UI", 9), width=max(180, w - 25),
                       text=load_caption)
