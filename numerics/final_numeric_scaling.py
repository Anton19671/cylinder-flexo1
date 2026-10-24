#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared diagonal row/column equilibration for strong-form FDM systems.

This is algebraic scaling only.  It does not regularize the operator and it
uses the same ``np.linalg.solve`` as before.  For

    A x = b

we solve

    (D_r A D_c) y = D_r b,   x = D_c y,

with max-absolute row/column scaling.  Singular/degenerate systems remain
singular and raise ``np.linalg.LinAlgError``.
"""
from __future__ import annotations
import numpy as np


def equilibrate_matrix(A):
    """Return (A_scaled, row_scale, col_scale) for a square finite matrix."""
    A = np.asarray(A, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("A contains non-finite values")

    row = np.max(np.abs(A), axis=1)
    if np.any(row == 0.0):
        bad = int(np.flatnonzero(row == 0.0)[0])
        raise np.linalg.LinAlgError(f"ZERO_ROW_IN_OPERATOR / row={bad}")

    As = A / row[:, None]
    col = np.max(np.abs(As), axis=0)
    if np.any(col == 0.0):
        bad = int(np.flatnonzero(col == 0.0)[0])
        raise np.linalg.LinAlgError(f"ZERO_COLUMN_IN_OPERATOR / column={bad}")

    As = As / col[None, :]
    return As, row, col


def equilibrated_solve(A, b):
    """Solve A x=b by exact diagonal equilibration and np.linalg.solve.

    Returns ``(x, A_scaled, row_scale, col_scale)`` so callers that already
    report a condition number can diagnose the matrix that was actually
    solved.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if b.ndim != 1 or A.ndim != 2 or A.shape != (b.size, b.size):
        raise ValueError("A/b dimension mismatch")
    if not np.all(np.isfinite(b)):
        raise ValueError("b contains non-finite values")

    As, row, col = equilibrate_matrix(A)
    y = np.linalg.solve(As, b / row)
    x = y / col
    return x, As, row, col
