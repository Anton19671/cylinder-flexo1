#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast generator-only check; it performs no numerical shell solution."""
from __future__ import annotations

import numpy as np

from cylinder_final_stage11a_dispatcher import _build_auto_mvi_profiles


def _max_periodic_correlation(Y: np.ndarray) -> float:
    if Y.ndim == 3:  # KL: component, rank, s
        banks = (Y[a] for a in range(Y.shape[0]))
    elif Y.ndim == 4:  # CUF: component, tau, rank, s
        banks = (Y[a, tau] for a in range(Y.shape[0]) for tau in range(Y.shape[1]))
    else:
        raise AssertionError(Y.shape)
    result = 0.0
    for bank in banks:
        norms = np.linalg.norm(bank, axis=1)
        corr = (bank @ bank.T) / np.outer(norms, norms)
        result = max(result, float(np.max(np.abs(corr - np.eye(bank.shape[0])))))
    return result


def main() -> None:
    common = {"L": 2000.0, "R": 1000.0, "h": 50.0, "nx": 41, "ns": 41, "stencil": 7}
    kl = {**common, "theory": "KL", "cuf_model": None}
    te1 = {**common, "theory": "CUF", "cuf_model": "TE1"}

    # Preserve the accepted MVI-1 starting profile exactly.
    _, y1 = _build_auto_mvi_profiles(te1, 1)
    _, y8 = _build_auto_mvi_profiles(te1, 8)
    assert np.array_equal(y1, y8[:, :, :1, :])

    for label, cfg in (("KL", kl), ("CUF-TE1", te1)):
        _x, y = _build_auto_mvi_profiles(cfg, 8)
        corr = _max_periodic_correlation(y)
        assert corr <= 1.0e-12, (label, corr)
        print(f"{label}: max |corr offdiag| = {corr:.3e}")
    print("MVI_PROFILE_BANK_SELF_CHECK: PASS")


if __name__ == "__main__":
    main()
