"""Non-solver checks of MVI modal rank and spectrum selection.

No cylinder equations are assembled and no eigenproblem is solved here.
"""
import copy

import numpy as np
import pytest

from cylinder_modal_reference import (
    _component_separation_ranks, _use_harmonic_mvi_spectrum,
)


def _case():
    return {"method": "MVI", "scale_model": "MACRO",
            "mvi_grid": {"nx": 13, "ns": 13,
                         "operator_scheme": "SBP42_FOURIER_ODD"},
            "boundary_conditions": {"variational": {
                "mechanical": {
                    "left": {"support": "CLAMPED", "higher": "NATURAL"},
                    "right": {"support": "CLAMPED", "higher": "NATURAL"}},
                "electrical": {}}},
            "modal": {"n_modes": 8, "density_kg_m3": 7850.0}}


def test_auto_keeps_classical_unrestricted_mvi():
    assert not _use_harmonic_mvi_spectrum(_case(), 8)
    assert not _use_harmonic_mvi_spectrum(_case(), 9)


def test_explicit_unrestricted_never_forces_fourier_modes():
    case = _case()
    case["modal"]["mvi_spectrum_method"] = "UNRESTRICTED"
    assert not _use_harmonic_mvi_spectrum(case, 8)


def test_fourier_family_selection_requires_physical_symmetry():
    case = _case()
    case["modal"]["mvi_spectrum_method"] = "FOURIER_FAMILIES"
    assert _use_harmonic_mvi_spectrum(case, 8)
    other = copy.deepcopy(case)
    other["scale_model"] = "MICRO"
    assert not _use_harmonic_mvi_spectrum(other, 8)
    other = copy.deepcopy(case)
    other["boundary_conditions"]["variational"]["electrical"] = {
        "left": {"support": "FIXED"}}
    assert not _use_harmonic_mvi_spectrum(other, 8)


def test_effective_rank_can_be_less_than_selected_mvi_rank():
    x = np.linspace(-1, 1, 9)
    theta = 2.0 * np.pi * np.arange(13) / 13
    fields = np.zeros((2, 2, 9, 13), dtype=float)
    fields[0, 0] = np.outer(x**2, np.cos(3 * theta))
    fields[1, 0] = (np.outer(x**2, np.cos(3 * theta)) +
                    np.outer(x**3, np.sin(4 * theta)))
    assert _component_separation_ranks(fields) == [[1, 0], [2, 0]]


def test_effective_rank_rejects_bad_shapes():
    with pytest.raises(ValueError, match="MODAL_RANK_AUDIT_INVALID_FIELDS"):
        _component_separation_ranks(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="MODAL_RANK_AUDIT_INVALID_FIELDS"):
        _component_separation_ranks(np.full((1, 1, 2, 3), np.nan))
