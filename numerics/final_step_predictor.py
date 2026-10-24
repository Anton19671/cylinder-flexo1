#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

PREDICTOR_MODES = ("ERROR", "LINEAR")


def normalize_mode(mode):
    m = str(mode or "ERROR").upper()
    if m not in PREDICTOR_MODES:
        raise ValueError(f"step_predictor.mode must be one of {PREDICTOR_MODES}")
    return m


def _scale(f_next, f_last, f_prev):
    try:
        a = float(f_next)
        b = float(f_last)
        c = float(f_prev)
    except Exception:
        return 1.0
    den = b - c
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c) or abs(den) <= 1.0e-30:
        return 1.0
    return (a - b) / den


def predict_array(prev, last, next_factor, last_factor, prev_factor, *, mode="ERROR"):
    m = normalize_mode(mode)
    out = np.asarray(last, float).copy()
    if m != "LINEAR" or prev is None:
        return out
    a = np.asarray(prev, float)
    if a.shape != out.shape:
        return out
    s = _scale(next_factor, last_factor, prev_factor)
    pred = out + s * (out - a)
    if not np.all(np.isfinite(pred)):
        return out
    return pred


def predict_state(prev_state, last_state, next_factor, last_factor, prev_factor, *, mode="ERROR"):
    m = normalize_mode(mode)
    state = last_state.copy()
    if m != "LINEAR" or prev_state is None:
        return state
    if not all(hasattr(x, "X") and hasattr(x, "Y") for x in (prev_state, last_state, state)):
        return state
    if np.asarray(prev_state.X).shape != np.asarray(last_state.X).shape:
        return state
    if np.asarray(prev_state.Y).shape != np.asarray(last_state.Y).shape:
        return state
    s = _scale(next_factor, last_factor, prev_factor)
    X = np.asarray(last_state.X, float) + s * (np.asarray(last_state.X, float) - np.asarray(prev_state.X, float))
    Y = np.asarray(last_state.Y, float) + s * (np.asarray(last_state.Y, float) - np.asarray(prev_state.Y, float))
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        return state
    state.X[...] = X
    state.Y[...] = Y
    return state
