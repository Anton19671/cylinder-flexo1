"""Conservative numerical stationarity checks for thin CUF shells.

These checks never solve a shell eigenproblem and do not change physical K/M.
"""
import pytest

from cylinder_modal_mvi import _modal_variational_step_precision


def test_cammalleri_13x21_false_projected_increase_is_bounded_by_physics():
    old = 484643330.474
    raised = 484643337.459
    allowed, mismatch = _modal_variational_step_precision(
        old, raised, old, old + 1.e-7)
    assert allowed
    assert mismatch + 1.e-6 >= raised - old


def test_true_physical_increase_cannot_be_excused_by_matrix_error():
    with pytest.raises(ValueError, match="MODAL_MVI_PROJECTED_VARIATIONAL_INCREASE"):
        _modal_variational_step_precision(
            484643330., 484643337., 484643330., 484643345.)


def test_projected_and_physical_increase_is_rejected():
    with pytest.raises(ValueError, match="MODAL_MVI_PROJECTED_VARIATIONAL_INCREASE"):
        _modal_variational_step_precision(100., 150., 100., 145.)


def test_valid_projected_decrease_needs_no_exception():
    allowed, mismatch = _modal_variational_step_precision(100., 99., 100., 99.)
    assert not allowed
    assert mismatch == 0.


def test_nonfinite_variational_data_is_rejected():
    with pytest.raises(ValueError, match="MODAL_MVI_NONFINITE_VARIATIONAL_ENERGY"):
        _modal_variational_step_precision(100., float("nan"), 100., 100.)


def test_large_matrix_field_disagreement_cannot_be_excused():
    with pytest.raises(ValueError, match="MODAL_MVI_PROJECTED_VARIATIONAL_INCREASE"):
        _modal_variational_step_precision(100., 200., 100., 100.)
