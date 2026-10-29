"""Canonical constraint row coefficients for dense and sparse solvers.

Duplicate indices are algebraically summed before inserting the row. This
ensures the boundary matrix has the same meaning in Newton, Newmark and the
matrix-free sparse projector. No changes to physical boundary equations.
"""
from __future__ import annotations

from numbers import Integral
import math
import numpy as np


def combined_constraint_entries(indices, weights, field_size, *, offset=0, label="constraint row"):
    if (isinstance(field_size, bool) or not isinstance(field_size, Integral)
            or field_size < 1 or isinstance(offset, bool)
            or not isinstance(offset, Integral) or offset < 0):
        raise ValueError("invalid constraint component extent or offset")
    if len(indices) != len(weights):
        raise ValueError(f"{label} index/weight count mismatch")
    combined = {}
    for index, weight in zip(indices, weights):
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
                or not 0 <= index < field_size):
            raise ValueError(f"{label} index out of range")
        if isinstance(weight, (bool, np.bool_)):
            raise ValueError(f"{label} weight must be finite")
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} weight must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} weight must be finite")
        key = int(index) + int(offset)
        combined[key] = combined.get(key, 0.) + value
    if not all(map(math.isfinite, combined.values())):
        raise ValueError(f"nonfinite summed {label} weight")
    ordered = [(i, value) for i, value in sorted(combined.items()) if value != 0.]
    return (np.asarray([i for i, _ in ordered], dtype=np.intp),
            np.asarray([v for _, v in ordered], dtype=float))


__all__ = ("combined_constraint_entries",)
