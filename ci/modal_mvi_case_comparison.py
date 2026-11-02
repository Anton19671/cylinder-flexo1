#!/usr/bin/env python3
"""Mass-metric eigenmode comparison gate for the MVI modal CI suite.

The comparator only reads independently computed full FDM and MVI fields.
An exact repeated eigenvalue has a rotation-invariant modal subspace.
Comparisons in this file validate the same discrete problem, NOT a FEM or
literature benchmark. No mode is supplied to the MVI as an initial guess.
"""
from __future__ import annotations

import numpy as np


def _mass_gram(left, right, functional, density):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    zero = np.zeros_like(right[0])
    actions = np.stack([
        functional.mass_action(field, density_kg_m3=density,
                               mechanical=zero, velocity=zero)
        for field in right
    ])
    return left.reshape(len(left), -1) @ actions.reshape(len(right), -1).T


def _inverse_gram_sqrt(matrix, label):
    gram = (matrix + matrix.T) * 0.5
    eigenvalues, vectors = np.linalg.eigh(gram)
    if not np.isfinite(eigenvalues).all() or not eigenvalues.size:
        raise ValueError("MODAL_COMPARISON_INVALID_MASS_GRAM: " + label)
    largest = float(np.max(eigenvalues))
    if largest <= 0. or float(np.min(eigenvalues)) <= 1.e-10 * largest:
        raise ValueError("MODAL_COMPARISON_SINGULAR_MASS_SUBSPACE: " + label)
    return (vectors / np.sqrt(eigenvalues)) @ vectors.T


def principal_mass_mac(reference, trial, functional, density):
    """Squared principal overlaps in physical mass inner product.

    Whiten BOTH Gramians: a pair of nonorthogonal vectors spanning the
    same eigenspace must still match perfectly. Never compare individual
    eigenvectors of an exactly repeated eigenvalue.
    """
    reference = np.asarray(reference, float)
    trial = np.asarray(trial, float)
    if not len(reference) or len(reference) != len(trial):
        raise ValueError("MODAL_COMPARISON_BAD_SUBSPACE_DIMENSION")
    rr = _mass_gram(reference, reference, functional, density)
    tt = _mass_gram(trial, trial, functional, density)
    rt = _mass_gram(reference, trial, functional, density)
    overlap = (_inverse_gram_sqrt(rr, "FDM") @ rt @
               _inverse_gram_sqrt(tt, "MVI"))
    values = np.linalg.eigvalsh(overlap.T @ overlap)
    return np.clip(values[::-1], 0.0, 1.0)


def compare_modal_results(reference_frequencies, trial_frequencies,
                          reference_modes, trial_modes, functional, density,
                          *, frequency_rtol=2.e-5, frequency_atol_hz=1.e-4,
                          minimum_mass_mac=0.999):
    """Report FAIL when a frequency OR a modal subspace fails verification.

    The FDM spectral gaps delimit consecutive repeated eigenvalue groups.
    Report the absolute tolerance independently because a merely successful
    eigensolver call is not proof that the lowest modes were discovered.
    """
    ref = np.asarray(reference_frequencies, float)
    got = np.asarray(trial_frequencies, float)
    qref = np.asarray(reference_modes, float)
    qgot = np.asarray(trial_modes, float)
    if (ref.ndim != 1 or got.ndim != 1 or ref.shape != got.shape or
            ref.size == 0 or qref.ndim != 4 or qgot.ndim != 4 or
            qref.shape != qgot.shape or qref.shape[0] != ref.size or
            not np.isfinite(ref).all() or not np.isfinite(got).all() or
            not np.isfinite(qref).all() or not np.isfinite(qgot).all() or
            np.any(ref <= 0.) or np.any(got <= 0.)):
        raise ValueError("MODAL_COMPARISON_INVALID_SPECTRUM_OR_MODES")
    if np.any(np.diff(ref) < -frequency_atol_hz):
        raise ValueError("MODAL_COMPARISON_UNSORTED_REFERENCE")
    relative = np.abs(got - ref) / ref
    close = np.isclose(got, ref, rtol=frequency_rtol,
                       atol=frequency_atol_hz)
    issues = []
    if not bool(np.all(close)):
        issues.append("FREQUENCY_MISMATCH_MODES:" +
                      str([int(i + 1) for i in np.flatnonzero(~close)]))
    groups = []
    start = 0
    while start < len(ref):
        stop = start + 1
        while stop < len(ref) and abs(ref[stop] - ref[stop-1]) <= max(
                1.e-5, 5.e-7 * max(ref[stop], ref[stop-1])):
            stop += 1
        mac = principal_mass_mac(qref[start:stop], qgot[start:stop],
                                 functional, density)
        min_mac = float(np.min(mac))
        good = bool(min_mac >= minimum_mass_mac)
        group = {
            "modes": [int(i + 1) for i in range(start, stop)],
            "fdm_hz": [float(x) for x in ref[start:stop]],
            "mvi_hz": [float(x) for x in got[start:stop]],
            "principal_mass_mac": [float(x) for x in mac],
            "minimum_mass_mac": min_mac,
            "shape_verified": good,
        }
        groups.append(group)
        if not good:
            issues.append("MODE_SUBSPACE_MISMATCH:" + str(group["modes"]))
        start = stop
    return {
        "status": "FAIL" if issues else "PASS",
        "frequency_verified": bool(np.all(close)),
        "all_mode_subspaces_verified": bool(all(g["shape_verified"] for g in groups)),
        "maximum_relative_frequency_error": float(np.max(relative)),
        "relative_frequency_errors": [float(x) for x in relative],
        "frequency_rtol": frequency_rtol,
        "frequency_atol_hz": frequency_atol_hz,
        "minimum_required_mass_mac": minimum_mass_mac,
        "mode_subspaces": groups,
        "issues": issues,
        "scope": "identical_discretization_not_independent_FEM_validation",
    }
