"""Regression contracts for the independent Cylinder MODAL variational solver.

These tests intentionally use mocked modal operators and do not execute a
cylinder computation. They assert the X/Y equations, order, and independence.
"""
import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cylinder_modal_mvi import (_SeparatedModalProblem, _mass_orthogonal_null,
                                _free_dof_count, _mass_project_previous_mode)


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


@pytest.mark.parametrize("scale", [1.e-200, 1.e-12, 1.e12, 1.e200])
def test_modal_orthogonality_rank_does_not_depend_on_row_scale(scale):
    # Both equations exclude a different preceding mode. An unscaled QR
    # drops one row when their magnitudes differ by more than its cutoff.
    constraints = np.array([[scale, 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    z = _mass_orthogonal_null(constraints)
    assert z.shape == (3, 1)
    np.testing.assert_allclose(z[:2], 0., atol=1.e-14)
    np.testing.assert_allclose(z.T @ z, np.eye(1), atol=1.e-14)


def test_modal_zero_orthogonality_rows_leave_all_directions():
    np.testing.assert_array_equal(_mass_orthogonal_null(np.zeros((2, 3))), np.eye(3))


def test_modal_free_dofs_exclude_only_independent_essential_traces():
    # Prescribing a trace twice does not remove two independent unknowns.
    constraints = np.array([[1., 0., 0., 0.], [2., 0., 0., 0.],
                            [0., 1.e-15, 0., 0.]])
    assert _free_dof_count(constraints) == 2
    assert _free_dof_count(np.empty((0, 4))) == 4
    assert _free_dof_count(np.empty((0, 0))) == 0


def test_admissible_start_does_not_request_full_grid_modal_eigenvectors():
    """The seed is obtained from admissible separated X directions only."""
    operator = object.__new__(_SeparatedModalProblem)
    operator.n_components, operator.rank = 1, 1
    operator.nx, operator.ns, operator.nm, operator.ne = 4, 3, 12, 0
    operator.shape = (1, 4, 3)
    operator.scales = np.ones(12)
    operator.mass = lambda q: np.asarray(q, float)
    # Construct an actual rank-1 frozen-Y variation space. Arbitrary
    # nodal fields cannot be recovered by a rank-1 separated product.
    from cylinder_boundary_spatial import separated_profile_directions
    def admissible_x_basis(x, y, axis):
        assert axis == "X"
        raw = separated_profile_directions(x, y, "X")
        return raw[:, [1, 2]], np.empty((0, 0))
    operator.projected_bases = MagicMock(side_effect=admissible_x_basis)
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


def test_variational_check_uses_one_projected_form():
    source = (Path(__file__).resolve().parents[1] / "numerics" /
              "cylinder_modal_mvi.py").read_text(encoding="utf-8")
    assert "old_projected_value = float(reduced @ kz @ reduced / old_mass)" in source
    assert "projected_value > old_projected + tolerance" in source
    assert "MODAL_MVI_VARIATIONAL_INCREASE:" not in source
    assert 'stage="MVI_VARIATIONAL_CHECK"' in source


def test_failed_reference_run_is_archived_after_exit():
    source = (Path(__file__).resolve().parents[1] / "interface" /
              "FINAL_GUI.py").read_text(encoding="utf-8")
    assert "if reference and rc != 0:" in source
    assert "archive_failed_run(" in source


def test_modal_already_satisfied_trace_does_not_create_false_constraints():
    from cylinder_boundary_spatial import affine_trace_space, separated_profile_directions
    from cylinder_variational_mvi import _normalize_partner, reconstruct_profiles

    # The boundary has only roundoff left after the admissible X seed.
    # Both frozen-partner halfsteps must retain the same rank-1 field.
    x = np.array([[[1.e-16, 1., 2., -1.e-16]]])
    y = np.array([[[1., 2., -1.]]])
    constraints = np.zeros((2, 12))
    constraints[0, 0] = 1.
    constraints[1, 9] = 1.
    x, y = _normalize_partner(x, y, "Y")
    directions = separated_profile_directions(x, y, "Y")
    space = affine_trace_space(constraints, np.zeros(2), np.zeros(12),
                               directions=directions, constraint_rank_floor=1.e-12)
    field = reconstruct_profiles(x, y).ravel()
    coords, *_ = np.linalg.lstsq(space.basis, field, rcond=None)
    np.testing.assert_allclose(space.basis @ coords, field, atol=1.e-13)
    assert space.basis.shape[1] == 3


def test_termination_archives_case_and_journal(tmp_path):
    from cylinder_stop_archive import archive_failed_run, archive_stopped_run
    case = tmp_path / "case.json"
    journal = tmp_path / "journal.log"
    case.write_text("{}", encoding="utf-8")
    journal.write_text("trace", encoding="utf-8")
    for archive, status in ((archive_failed_run, "FAILED"), (archive_stopped_run, "STOPPED")):
        out = tmp_path / status
        record = archive(out, case_path=case, journal_path=journal)
        assert (out / "INPUT" / (status + "_CASE.json")).read_text() == "{}"
        assert (out / record["files"]["journal"]).read_text() == "trace"
        assert (out / (status + "_RUN.json")).is_file()


def test_mass_projection_preserves_old_mode_in_current_admissible_space():
    bm = np.eye(3)
    z = np.array([[1., 0.], [0., 1.], [0., 0.]])
    mz = np.diag([2., 3.])
    old = np.array([0.4, -0.8, 2.e-5])
    mass = lambda q: np.array([2., 3., 5.]) * q
    reduced = _mass_project_previous_mode(bm, z, mz, old, mass)
    np.testing.assert_allclose(reduced, old[:2], atol=1.e-14)
    assert np.linalg.norm(bm @ z @ reduced - old) > 1.e-5


def test_mass_projection_handles_coupled_physical_mass():
    bm = np.eye(3)
    z = np.array([[1.], [0.], [0.]])
    physical_m = np.array([[2., .5, 0.], [.5, 3., 0.], [0., 0., 1.]])
    old = np.array([1., 1., 0.])
    projected = _mass_project_previous_mode(
        bm, z, np.array([[2.]]), old, lambda q: physical_m @ q)
    np.testing.assert_allclose(projected, [1.25], rtol=1.e-13)


def test_mass_projection_rejects_genuinely_lost_mode():
    bm = np.eye(3)
    z = np.array([[1.], [0.], [0.]])
    with pytest.raises(ValueError, match="OLD_MODE_LOST"):
        _mass_project_previous_mode(
            bm, z, np.array([[1.]]), np.array([0., 1., 0.]),
            lambda q: q)
