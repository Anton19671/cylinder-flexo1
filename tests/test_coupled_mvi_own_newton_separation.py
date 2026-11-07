"""Algebra-only architecture guarantees of coupled MVI own Newton route.

No shell solve, nonlinear iteration, time integration or numerical inversion
is executed by these tests. Method boundaries are checked by AST and the
own frozen-halfstep fail-closed entrypoint.
"""
from __future__ import annotations
import ast
from dataclasses import fields
from pathlib import Path
import pytest

from cylinder_mvi_subspace_newton import (
    MVIHalfstepSettings,MVIHalfstepResult,
    mvi_settings_from_options,mvi_solve_static_subspace,
)
from cylinder_variational_fdm import StaticFDMSettings,StaticFDMResult


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name",(
    "cylinder_variational_mvi.py",
    "cylinder_mvi_subspace_newton.py",
))
def test_coupled_mvi_static_numerics_never_imports_fdm_solver(name):
    source=(ROOT/"numerics"/name).read_text(encoding="utf-8")
    tree=ast.parse(source)
    forbidden=("cylinder_variational_fdm","independent_mkr",
               "cylinder_direct_strong_fdm")
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert (node.module or "") not in forbidden
        if isinstance(node,ast.Import):
            assert all(not alias.name.startswith(forbidden)
                       for alias in node.names)
    assert "solve_static_fdm(" not in source
    assert "solve_dynamic_fdm(" not in source


def test_coupled_mvi_owns_exact_newton_setting_and_result_fields():
    assert {f.name for f in fields(MVIHalfstepSettings)}=={
        f.name for f in fields(StaticFDMSettings)}
    assert {f.name for f in fields(MVIHalfstepResult)}=={
        f.name for f in fields(StaticFDMResult)}


def test_legacy_parameter_passport_converted_but_numerical_method_is_own():
    old=StaticFDMSettings(max_iterations=3,max_free_dofs=88,
                          relative_tolerance=3.e-8)
    own=mvi_settings_from_options(old)
    assert isinstance(own,MVIHalfstepSettings)
    assert own is not old
    assert own.max_iterations==old.max_iterations
    assert own.max_free_dofs==88
    assert own.relative_tolerance==old.relative_tolerance
    assert mvi_settings_from_options(own) is own
    assert isinstance(mvi_settings_from_options(None),MVIHalfstepSettings)
    with pytest.raises(TypeError,match="MVI_NEWTON_SETTINGS_TYPE_INVALID"):
        mvi_settings_from_options({"max_iterations":10})


def test_mvi_subproblem_rejects_full_nodal_newton_invocation():
    class FakeProblem:
        kind="STATIC"
    with pytest.raises(ValueError,match="MVI_OWN_HALFSTEP_REQUIRES_XY_DIRECTIONS"):
        mvi_solve_static_subspace(FakeProblem(),None,None,directions=None)


def test_coupled_dynamic_mvi_uses_own_frozen_halfstep_newton_only():
    import cylinder_variational_dynamic as dynamic
    from cylinder_mvi_dynamic_subspace import (
        MVIDynamicHalfstepResult,mvi_solve_dynamic_subspace)
    p=ROOT/"numerics"/"cylinder_mvi_dynamic_subspace.py"
    source=p.read_text(encoding="utf-8")
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert (node.module or "") not in (
                "cylinder_variational_fdm","cylinder_variational_dynamic",
                "independent_mkr","cylinder_direct_strong_fdm")
        elif isinstance(node,ast.Import):
            assert all(not item.name.startswith((
                "cylinder_variational_fdm","independent_mkr"))
                for item in node.names)
    assert "mvi_solve_dynamic_subspace(" in source
    assert "solve_dynamic_fdm(" not in source
    assert dynamic.solve_dynamic_mvi.__code__.co_names.count(
        "mvi_solve_dynamic_subspace")==1

    class FakeDynamicProblem:
        kind="DYNAMIC"
    with pytest.raises(ValueError,match="MVI_OWN_DYNAMIC_REQUIRES_XY_DIRECTIONS"):
        mvi_solve_dynamic_subspace(
            FakeDynamicProblem(),None,None,
            q_predictor=None,velocity_predictor=None,time_s=0.,
            dt_s=.001,beta=.25,gamma=.5,density_kg_m3=1000.,
            directions=None)


def test_gauss2_coupled_mvi_adapter_never_imports_fdm_numerics():
    p=ROOT/"numerics"/"cylinder_gauss2_variational_mvi.py"
    source=p.read_text(encoding="utf-8")
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert (node.module or "") not in (
                "cylinder_variational_fdm","cylinder_gauss2_variational_fdm",
                "independent_mkr","cylinder_direct_strong_fdm")
        elif isinstance(node,ast.Import):
            assert all(not item.name.startswith((
                "cylinder_variational_fdm","cylinder_gauss2_variational_fdm",
                "independent_mkr")) for item in node.names)
    assert "mvi_settings_from_options" in source
    assert "mvi_state_scales" in source


@pytest.mark.parametrize("filename",(
    "cylinder_variational_mvi.py",
    "cylinder_mvi_dynamic_subspace.py",
    "cylinder_gauss2_variational_mvi.py",
))
def test_own_mvi_static_dynamic_gauss_never_import_fdm_solver(filename):
    tree=ast.parse((ROOT/"numerics"/filename).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert (node.module or "") not in (
                "cylinder_variational_fdm",
                "cylinder_direct_strong_fdm",
                "independent_mkr")
        elif isinstance(node,ast.Import):
            assert all(not alias.name.startswith((
                "cylinder_variational_fdm","independent_mkr"))
                for alias in node.names)
