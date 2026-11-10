#!/usr/bin/env python3
"""No-physics regression: Gauss MVI endpoint fixed traces are amplitude-scaled.

Run from repository root: python ci/check_mvi_gauss_endpoint_traces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from independent_mvi.coupled_rank_endpoint import _trace_roundoff_limit


def check():
    for amplitude in (1.0, 1.e-3, 1.e-6, 1.e-9, 1.e-15):
        target = np.array([[0., amplitude, 0.]], dtype=float)
        limit = _trace_roundoff_limit(target)
        assert np.isfinite(limit) and limit > 0
        assert np.isclose(limit / amplitude,
                          64 * np.finfo(float).eps, rtol=2.e-14)
        # A nonzero supported-end displacement larger than floating
        # roundoff must be rejected regardless of endpoint fit tolerance.
        trial = target.copy()
        trial[0, 0] = 1.e-6 * amplitude
        assert abs(trial[0, 0]) > _trace_roundoff_limit(trial)
        # Exactly zero essential traces remain admissible.
        assert max(abs(target[0, 0]), abs(target[0, -1])) <= limit

    # Regression for the former max(1, amplitude) error:
    # this micro-displacement cannot be accepted at a clamped end.
    physical = np.array([5.e-10, 1.e-6, 0.])
    assert physical[0] > _trace_roundoff_limit(physical)
    assert physical[0] < 1.e-9  # Former tolerance incorrectly accepted it.
    assert _trace_roundoff_limit(np.zeros((3, 9))) > 0
    print("PASS: Gauss MVI fixed traces enforce amplitude-scaled roundoff only")


if __name__ == "__main__":
    check()
