"""GUI descriptors tied to the actual seven-theory variational BC assembler.

No guessed combinations: component order, support selections and higher normal
trace restrictions are derived from cylinder_variational_boundary.
"""
from __future__ import annotations

from cylinder_variational_boundary import (
    _fixed_components, HIGHER_TRACES, MECHANICAL_SUPPORTS, ELECTRIC_MODES,
)


def theory_from_case(case):
    theory = str(case.get("theory", "")).upper()
    if theory == "CUF":
        theory = "CUF_" + str(case.get("cuf_model", "")).upper()
    if theory not in ("KL", "FSDT", "REDDY",
                      "CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4"):
        raise ValueError("Unknown shell theory")
    return theory


def gradient_from_case(case):
    if str(case.get("scale_model", "")).upper() != "MICRO":
        return "OFF"
    gradient = str((case.get("micro") or {}).get("gradient_model", "G3D")).upper()
    if gradient not in ("G3D", "GZ"):
        raise ValueError("Unknown G3D/Gz gradient model")
    return "G3D" if gradient == "G3D" else "Gz"


def boundary_ui_model(case):
    """Describe EXACTLY the boundary unknowns and available input choices."""
    theory = theory_from_case(case)
    if theory == "KL":
        names = ("u0", "v0", "w0")
    elif theory in ("FSDT", "REDDY"):
        names = ("u0", "v0", "w0", "theta_x", "theta_s")
    else:
        m = int(theory[-1]) + 1
        names = tuple(f"{component}{k}" for component in ("u", "v", "w")
                      for k in range(m))
    micro = case.get("micro") or {}
    enabled = (str(case.get("scale_model", "")).upper() == "MICRO"
               and bool(micro.get("enable_flexoelectric")))
    gradient = gradient_from_case(case)
    higher = tuple(mode for mode in HIGHER_TRACES
                   if mode != "FIX_FIRST_AND_SECOND_NORMAL"
                   or not theory.startswith("CUF_"))
    supports = {}
    for name in MECHANICAL_SUPPORTS:
        if name == "CUSTOM":
            continue
        supports[name] = tuple(names[i] for i in
                               _fixed_components(name, theory, len(names), {}))
    order = int(micro.get("potential_order", 0)) if enabled else 0
    return {
        "theory": theory,
        "components": names,
        "n_components": len(names),
        "support_components": supports,
        "support_choices": MECHANICAL_SUPPORTS,
        "higher_choices": higher,
        "gradient_model": gradient,
        "gradient_caption": {
            "G3D": "G3D — полная градиентная постановка (18 компонент)",
            "Gz": "Gz — усечённая градиентная постановка (6 компонент)",
            "OFF": "MACRO — градиентная постановка выключена",
        }[gradient],
        "electric_enabled": enabled,
        "electric_potential_order": order,
        "electric_components": tuple(f"psi{k}" for k in range(order + 1))
            if enabled else (),
        "electric_sites": ("left", "right", "outer", "inner")
            if enabled else (),
        "electric_modes": ELECTRIC_MODES if enabled else ("INSULATED",),
    }


__all__ = ("theory_from_case", "gradient_from_case", "boundary_ui_model")
