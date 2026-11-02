"""Unit tests for modal CI comparison; no cylinder eigenproblem is solved."""
import numpy as np
import pytest

from ci.modal_mvi_case_comparison import compare_modal_results, principal_mass_mac


class IdentityMass:
    def mass_action(self, field, *, density_kg_m3, mechanical, velocity):
        return field


def modes():
    return np.array([100., 100., 200., 200.]), np.eye(5)[:4].reshape(4, 1, 1, 5)


def test_rotated_doublets_match_mass_subspace():
    freq, reference = modes()
    trial = reference.copy()
    trial[0] = (reference[0] + reference[1]) / np.sqrt(2)
    trial[1] = (reference[0] - reference[1]) / np.sqrt(2)
    trial[2] = (reference[2] + reference[3]) / np.sqrt(2)
    trial[3] = (reference[2] - reference[3]) / np.sqrt(2)
    result = compare_modal_results(freq, freq, reference, trial, IdentityMass(), 1.)
    assert result["status"] == "PASS"
    for pair in result["mode_subspaces"]:
        np.testing.assert_allclose(pair["principal_mass_mac"], [1., 1.], atol=1.e-12)


def test_nonorthogonal_vectors_in_same_doublet_are_accepted():
    freq, reference = modes()
    trial = reference.copy()
    trial[0] = reference[0] + .3 * reference[1]
    trial[1] = reference[0] - .6 * reference[1]
    result = compare_modal_results(freq, freq, reference, trial, IdentityMass(), 1.)
    assert result["status"] == "PASS"


def test_missing_shape_fails_even_if_all_frequencies_match():
    freq, reference = modes()
    trial = reference.copy()
    trial[3] = np.eye(5)[4].reshape(1, 1, 5)
    result = compare_modal_results(freq, freq, reference, trial, IdentityMass(), 1.)
    assert result["status"] == "FAIL"
    assert result["frequency_verified"]
    assert not result["all_mode_subspaces_verified"]


def test_wrong_frequency_fails_even_when_shapes_match():
    freq, reference = modes()
    incorrect = freq.copy()
    incorrect[-1] += 2.
    result = compare_modal_results(freq, incorrect, reference, reference,
                                   IdentityMass(), 1.)
    assert result["status"] == "FAIL"
    assert not result["frequency_verified"]
    assert result["all_mode_subspaces_verified"]


def test_duplicate_vectors_are_singular_not_a_valid_doublet():
    freq, reference = modes()
    trial = reference.copy()
    trial[1] = trial[0]
    with pytest.raises(ValueError, match="MODAL_COMPARISON_SINGULAR_MASS_SUBSPACE"):
        compare_modal_results(freq, freq, reference, trial, IdentityMass(), 1.)


def test_reject_wrong_shape():
    freq, reference = modes()
    with pytest.raises(ValueError, match="MODAL_COMPARISON_INVALID_SPECTRUM_OR_MODES"):
        compare_modal_results(freq, freq, reference, reference[:2],
                              IdentityMass(), 1.)
