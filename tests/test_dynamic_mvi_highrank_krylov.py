"""Opt-in matrix-free Newton within *frozen MVI X/Y* variational spaces."""
from __future__ import annotations

import numpy as np
import pytest

from cylinder_variational_dynamic import solve_dynamic_mvi
from cylinder_variational_fdm import StaticFDMSettings
from cylinder_variational_mvi import StaticMVISettings
from test_mvi_grid_free_dof_gate import GlobalVsReducedDofTests


def test_frozen_partner_krylov_above_dense_limit_matches_dense_model():
    setup=GlobalVsReducedDofTests()
    problem,q,p,kw=setup.setup_problem(n=7)
    kw["settings"]=StaticFDMSettings(
        relative_tolerance=1.e-7,absolute_tolerance_scaled=1.e-12,
        max_iterations=8,max_free_dofs=256)
    dense=solve_dynamic_mvi(
        problem,q,p,**kw,
        mvi_settings=StaticMVISettings(rank=1,max_sweeps=8))
    assert dense.converged,dense.status

    kw["settings"]=StaticFDMSettings(
        relative_tolerance=1.e-7,absolute_tolerance_scaled=1.e-12,
        max_iterations=8,max_free_dofs=10)
    krylov=solve_dynamic_mvi(
        problem,q,p,**kw,
        mvi_settings=StaticMVISettings(
            rank=1,max_sweeps=8,dynamic_krylov_halfsteps=True,
            dynamic_krylov_rtol=1.e-8,dynamic_krylov_restart=35,
            dynamic_krylov_max_matvecs=350))
    assert krylov.converged,krylov.status
    assert krylov.mvi_certificate["accepted"]
    assert krylov.mvi_certificate["fixed_rank"]==1
    assert any(row.get("linear_solver")=="MVI_FROZEN_PARTNER_GMRES"
               for row in krylov.history)
    np.testing.assert_allclose(krylov.mechanical,dense.mechanical,
                               rtol=1.e-5,atol=1.e-8)
    np.testing.assert_allclose(krylov.velocity,dense.velocity,
                               rtol=1.e-5,atol=1.e-8)
    np.testing.assert_allclose(krylov.acceleration,dense.acceleration,
                               rtol=1.e-5,atol=1.e-8)


def test_dense_gate_still_fails_closed_without_explicit_optin():
    setup=GlobalVsReducedDofTests()
    problem,q,p,kw=setup.setup_problem(n=7)
    kw["settings"]=StaticFDMSettings(max_free_dofs=10)
    with pytest.raises(ValueError,match="free DOFs; limit=10"):
        solve_dynamic_mvi(problem,q,p,**kw,
            mvi_settings=StaticMVISettings(rank=1,max_sweeps=2))


def test_krylov_option_type_and_bounds_are_strict():
    with pytest.raises(ValueError,match="boolean"):
        StaticMVISettings(dynamic_krylov_halfsteps=1)
    with pytest.raises(ValueError,match="dynamic_krylov_rtol"):
        StaticMVISettings(dynamic_krylov_rtol=0.)
    with pytest.raises(ValueError,match="dynamic_krylov_max_matvecs"):
        StaticMVISettings(dynamic_krylov_restart=50,
                          dynamic_krylov_max_matvecs=40)
