"""Regression contracts for the independent Cylinder MODAL variational solver.

These tests intentionally use mocked modal operators and do not execute a
cylinder computation. They assert the X/Y equations, order, and independence.
"""
import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cylinder_modal_mvi import _SeparatedModalProblem, _mass_orthogonal_null


def test_modal_direction_order_matches_static_variational_contract():
    source = (Path(__file__).resolve().parents[1] / "numerics" /
              "cylinder_modal_mvi.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    solve = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "solve_modal_mvi")
    axes = [node for node in ast.walk(solve) if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name) and node.target.id == "axis"]
    assert len(axes) == 1
    assert ast.literal_eval(axes[0].iter) == ("Y", "X")


def test_modal_mass_constraint_nullspace():
    constraints = np.array([[1., 0., 0.], [0., 1., 0.]])
    z = _mass_orthogonal_null(constraints)
    assert z.shape == (3, 1)
    np.testing.assert_allclose(constraints @ z, 0., atol=1e-14)
    np.testing.assert_allclose(z.T @ z, np.eye(1), atol=1e-14)


def test_admissible_start_does_not_request_full_grid_modal_eigenvectors():
    """The seed is obtained from admissible separated X directions only."""
    operator = object.__new__(_SeparatedModalProblem)
    operator.n_components, operator.rank = 1, 1
    operator.nx, operator.ns, operator.nm, operator.ne = 4, 3, 12, 0
    operator.shape = (1, 4, 3)
    operator.scales = np.ones(12)
    operator.mass = lambda q: np.asarray(q, float)
    # Dirichlet x-end nodes must remain zero.
    admissible = np.zeros((12, 1))
    admissible[[3, 4, 5, 6, 7, 8], 0] = 1.
    operator.projected_bases = MagicMock(return_value=(admissible, np.empty((0, 0))))
    x, y, q, p = operator.admissible_start(0, [])
    assert p is None
    np.testing.assert_allclose(q.reshape(4, 3)[[0, -1]], 0., atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(q), 1., atol=1e-14)
    np.testing.assert_allclose(np.einsum("crx,crs->cxs", x, y).ravel(), q, atol=1e-12)
    assert operator.projected_bases.call_args.args[-1] == "X"


def test_modal_algorithm_does_not_change_static_solver_file():
    static = (Path(__file__).resolve().parents[1] / "numerics" /
              "cylinder_variational_mvi.py").read_text(encoding="utf-8")
    assert 'for axis in ("Y","X"):' in static
    modal = (Path(__file__).resolve().parents[1] / "numerics" /
             "cylinder_modal_mvi.py").read_text(encoding="utf-8")
    assert '"mvi_modal_algorithm": "fixed_rank_constrained_variational_YX_stationarity"' in modal
    assert "ritz_relative=" not in modal
