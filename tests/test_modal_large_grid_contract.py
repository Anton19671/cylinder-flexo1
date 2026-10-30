"""No-physics-run contracts for large-grid MODAL and DYNAMIC FDM routing.

Dense-grid modal eigenvalue calculations are intentionally not launched here.
"""
import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cylinder_modal_matrixfree import _projected_actions


def test_all_seven_theories_stay_in_modal_registry():
    from cylinder_modal_reference import THEORIES
    assert THEORIES == (
        "KL", "FSDT", "REDDY", "CUF_TE1", "CUF_TE2",
        "CUF_TE3", "CUF_TE4")


def test_modal_fdm_matrixfree_uses_sparse_constraint_projector():
    path = Path(__file__).resolve().parents[1] / "numerics/cylinder_modal_reference.py"
    source = path.read_text(encoding="utf-8")
    assert 'prepared.fdm_backend == "MATRIX_FREE_GMRES"' in source
    assert "SparseTraceProjector.from_problem(" in source
    assert "solve_modal_matrixfree(" in source
    assert 'grid.get("operator_scheme") != "SBP21_PERIODIC_ODD_CSR"' in source


def test_modal_mvi_keeps_original_separated_branch():
    path = Path(__file__).resolve().parents[1] / "numerics/cylinder_modal_reference.py"
    source = path.read_text(encoding="utf-8")
    assert 'if prepared.method == "MVI":' in source
    assert 'from cylinder_modal_mvi import solve_modal_mvi' in source


def test_large_grid_dynamic_matrixfree_route_remains_unchanged():
    root = Path(__file__).resolve().parents[1]
    case = (root / "interface/cylinder_reference_case.py").read_text(encoding="utf-8")
    matrixfree = (root / "numerics/cylinder_matrixfree_fdm.py").read_text(encoding="utf-8")
    assert "max_surface_nodes" in case
    assert "solve_matrixfree_dynamic_fdm" in matrixfree
    assert "MATRIX_FREE_GMRES" in case


def test_modal_operator_writes_no_time_steps_or_dense_matrix():
    root = Path(__file__).resolve().parents[1] / "numerics/cylinder_modal_matrixfree.py"
    tree = ast.parse(root.read_text(encoding="utf-8"))
    methods = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "_projected_actions" in methods
    assert "solve_modal_matrixfree" in methods
    source = root.read_text(encoding="utf-8")
    assert "SparseTraceProjector.from_problem" in source
    assert "np.linalg.eigh(" not in source
    assert "np.zeros((n, n))" not in source


def test_high_grid_preflight_rejects_non_csr_method():
    from cylinder_modal_reference import _modal_spec
    case = {
        "solver_backend": "VARIATIONAL_REFERENCE",
        "calculation_kind": "MODAL", "mode": "LINEAR_ELASTIC",
        "geometry_model": "LINEAR", "material_model": "HOOKE",
        "theory": "CUF", "cuf_model": "TE4",
        "method": "FDM", "fdm_backend": "MATRIX_FREE_GMRES",
        "scale_model": "MACRO", "modal": {"density_kg_m3": 7850, "n_modes": 6},
        "fdm_grid": {"nx": 25, "ns": 25,
                     "operator_scheme": "SBP21_PERIODIC_ODD"}
    }
    with pytest.raises(ValueError, match="MATRIX_FREE_MODAL_REQUIRES_SBP21_PERIODIC_ODD_CSR"):
        _modal_spec(case)


def test_large_grid_case_is_consistent_with_cuf_te4_dofs():
    from cylinder_matrixfree_fdm import MatrixFreeNewtonSettings
    settings = MatrixFreeNewtonSettings()
    for count in (17, 25, 33, 41):
        assert count * count < settings.max_surface_nodes
        assert 15 * count * count < settings.max_total_dofs
    assert 15 * 41 * 41 > 512
