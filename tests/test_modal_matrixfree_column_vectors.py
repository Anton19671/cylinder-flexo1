"""SciPy generalized eigensolvers may pass (n,1) to LinearOperator.matvec.

This is an interface test, not a replacement for a real shell calculation.
The Cammalleri CUF TE1 13x21 one-mode physical regression is run separately
by the modal-mvi-cpu-profile GitHub Actions workflow.
"""
import numpy as np
import pytest

from cylinder_modal_matrixfree import _modal_linear_operator


def test_single_column_and_vector_use_same_physical_action():
    n = 17
    weights = np.arange(1., n + 1.)
    calls = []

    def action(v):
        assert v.shape == (n,)
        calls.append(tuple(v.shape))
        return weights * v

    op = _modal_linear_operator(action, n)
    x = np.linspace(-1., 1., n)
    expected = weights * x
    np.testing.assert_allclose(op @ x, expected, rtol=0, atol=0)
    matrix = op @ x[:, None]
    assert matrix.shape == (n, 1)
    np.testing.assert_allclose(matrix[:, 0], expected, rtol=0, atol=0)
    pair = np.column_stack((x, x[::-1]))
    np.testing.assert_allclose(op @ pair, weights[:, None] * pair,
                               rtol=0, atol=0)
    assert all(shape == (n,) for shape in calls)


def test_single_column_generalized_eigen_mass_gram():
    n = 9
    diagonal = np.linspace(0.2, 2.5, n)
    mass = _modal_linear_operator(lambda v: diagonal * v, n)
    vectors = np.arange(1., n + 1.)[:, None]
    mass_vectors = mass @ vectors
    assert mass_vectors.shape == vectors.shape
    gram = vectors.T @ mass_vectors
    assert gram.shape == (1, 1)
    np.testing.assert_allclose(gram[0, 0], np.dot(vectors[:, 0],
                                                  diagonal * vectors[:, 0]))


def test_bad_shape_not_silently_broadcast():
    op = _modal_linear_operator(lambda v: v, 7)
    with pytest.raises(ValueError, match="VECTOR_SHAPE|dimension mismatch"):
        op.matvec(np.ones((7, 2)))
    with pytest.raises(ValueError, match="MATRIX_SHAPE|dimension mismatch"):
        op.matmat(np.ones((6, 2)))
    with pytest.raises(ValueError, match="ACTION_SHAPE"):
        _modal_linear_operator(lambda v: v[:, None], 7) @ np.ones(7)
