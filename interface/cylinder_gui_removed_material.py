# -*- coding: utf-8 -*-
"""GUI-only compatibility for removed, unsupported material editors.

Historical widgets in interface/FINAL_GUI.py still refer to a non-elastic
material curve editor, although the live solver is elastic-only. Keep those
widgets constructible so START_PROGRAM can open the original GUI layout.
Any attempt to compute an unavailable curve fails explicitly; there is
NO surrogate constitutive law or numerical fallback.
"""
from __future__ import annotations
from types import SimpleNamespace


CURVE_MODES = ("TABLE", "CSV", "FORMULA", "PIECEWISE", "EXTERNAL_PYTHON")
EXTRAPOLATIONS = ("ERROR", "LINEAR", "HOLD")


def _unsupported(*args, **kwargs):
    raise RuntimeError(
        "This historical material-curve module was removed from the "
        "elastic-only cylinder-flexo1 repository. The active physical "
        "backend supports HOOKE only. No old curve or material law "
        "has been substituted."
    )


curve_ui = SimpleNamespace(
    CURVE_MODES=CURVE_MODES,
    EXTRAPOLATIONS=EXTRAPOLATIONS,
    sample_curve=_unsupported,
    build_curve=_unsupported,
    read_csv_points=_unsupported,
)
material_validation_ui = SimpleNamespace(
    validate_material_diagram_near_origin=_unsupported,
)

__all__ = ("curve_ui", "material_validation_ui")
