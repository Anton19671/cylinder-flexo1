"""Protect already separated NATIVE coupled FDM and MVI numerical routes.

Only the local continuum material and kinematic models can be common.
The optional legacy *diagnostic* functions in two strong-FD modules
are intentionally not part of either executable native solver route.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FDM_FILES = (
    ROOT / "independent_mkr" / "run_coupled_research.py",
    ROOT / "numerics" / "cylinder_direct_strong_model.py",
    ROOT / "numerics" / "cylinder_direct_strong_boundary.py",
    ROOT / "numerics" / "cylinder_mkr_field_jets.py",
    *sorted((ROOT / "numerics").glob("cylinder_native_mkr_*.py")),
)

MVI_FILES = (
    ROOT / "independent_mvi" / "run_coupled_research.py",
    *sorted((ROOT / "independent_mvi").glob("coupled_*.py")),
)

SHARED_REFERENCE = (
    "cylinder_reference_case", "cylinder_functional_assembly",
    "cylinder_boundary_spatial", "cylinder_variational_fdm",
    "cylinder_variational_mvi", "cylinder_variational_dynamic",
)

def imports_in(node):
    result = []
    for item in ast.walk(node):
        if isinstance(item, ast.Import):
            result.extend(alias.name for alias in item.names)
        elif isinstance(item, ast.ImportFrom) and item.module:
            result.append(item.module)
    return result


@pytest.mark.parametrize(
    "file,forbidden",
    [(file, ("independent_mvi",) + SHARED_REFERENCE) for file in FDM_FILES]
    + [(file, ("independent_mkr", "cylinder_direct_strong_fdm",
               "cylinder_direct_strong_model",
               "cylinder_direct_strong_newton",
               "cylinder_direct_strong_boundary",
               "cylinder_mkr_field_jets") + SHARED_REFERENCE)
       for file in MVI_FILES],
    ids=[str(file.relative_to(ROOT)) for file in FDM_FILES + MVI_FILES],
)
def test_actual_native_solver_modules_have_no_cross_method_imports(file, forbidden):
    tree = ast.parse(file.read_text(encoding="utf-8"))
    for name in imports_in(tree):
        assert not any(name == prefix or name.startswith(prefix + ".")
                       for prefix in forbidden), (str(file), name)


@pytest.mark.parametrize(
    "path,functions",
    [
      ("numerics/cylinder_direct_strong_fdm.py",
       ("direct_coefficient_derivative", "direct_strong_jet_balance")),
      ("numerics/cylinder_direct_strong_newton.py",
       ("solve_strong_nonlinear_equations",)),
    ],
)
def test_live_strong_fdm_functions_have_no_reference_or_mvi_calls(path, functions):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = {node.name: node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert all(name in selected for name in functions)
    for name in functions:
        node = selected[name]
        imports = imports_in(node)
        assert not any(module.startswith(("independent_mvi",) + SHARED_REFERENCE)
                       for module in imports), (path, name, imports)
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        for call in calls:
            if isinstance(call.func, ast.Name):
                assert call.func.id not in (
                    "direct_strong_from_functional", "solve_direct_strong_reference",
                    "run_reference_case", "run_reference_mvi_series"
                ), (path, name, call.func.id)


def test_native_coupled_case_schemas_route_to_distinct_packages():
    from cylinder_independent_dispatch import SCHEMAS, _own_runner
    from independent_mkr import run_coupled_research as fdm
    from independent_mvi import run_coupled_research as mvi

    fdm_schema = "INDEPENDENT_MKR_COUPLED_RESEARCH_V1"
    mvi_schema = "INDEPENDENT_MVI_COUPLED_RESEARCH_V1"
    assert SCHEMAS[fdm_schema] == "INDEPENDENT_MKR"
    assert SCHEMAS[mvi_schema] == "INDEPENDENT_MVI"
    assert _own_runner(SCHEMAS[fdm_schema], fdm_schema) is fdm
    assert _own_runner(SCHEMAS[mvi_schema], mvi_schema) is mvi
    assert fdm is not mvi
    assert fdm.METHOD != mvi.METHOD
