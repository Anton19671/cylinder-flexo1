"""Compare cached and direct operator actions without running a modal solve."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cylinder_matrixfree_fdm import SparseTraceProjector
from cylinder_modal_matrixfree import _projected_actions
from cylinder_modal_reference import _modal_spec
from cylinder_variational_fdm import state_scales


@pytest.mark.parametrize("order, spring, grid", [
    (1, False, 5), (2, False, 5), (3, False, 5),
    (4, False, 13), (4, True, 5),
])
def test_cuf_cached_fdm_action_matches_full_variation(order, spring, grid):
    root = Path(__file__).resolve().parents[1]
    case_path = root / "config/verification/rawat2016/modal_linear/fdm_13x13/RAWAT2016_CUF_TE4_13x13_MODAL_FDM.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case = copy.deepcopy(case)
    case["cuf_model"] = f"TE{order}"
    case["fdm_grid"].update(nx=grid, ns=grid)
    if spring:
        case["boundary_conditions"]["variational"]["mechanical"]["right"] = {
            "support": "FREE", "higher": "NATURAL", "spring": [10000.] * (3 * (order + 1))}
    prepared, density, _ = _modal_spec(case)
    q0 = prepared.mechanical
    functional = prepared.problem.functional
    scales = state_scales(functional, q0, None, prepared.settings)
    projector = SparseTraceProjector.from_problem(
        prepared.problem, q0, None, scales, time_s=None,
        max_rows=prepared.settings.max_constraint_rows)
    k, _, _, _, _ = _projected_actions(prepared, density, projector)
    rng = np.random.default_rng(2301)
    for _ in range(2):
        vector = projector.project_null(rng.normal(size=q0.size))
        physical = (scales * vector).reshape(q0.shape)
        step = prepared.settings.tangent_step_scaled
        _, zero, _ = prepared.problem.evaluate_effective_residual(np.zeros_like(q0), None)
        _, shifted, _ = prepared.problem.evaluate_effective_residual(step * physical, None)
        stiffness_scale = functional.material.young_pa * functional.h * np.sum(functional.W)
        expected = projector.project_null(scales * (shifted - zero) / step / stiffness_scale)
        np.testing.assert_allclose(k @ vector, expected, rtol=1e-8, atol=1e-10)
