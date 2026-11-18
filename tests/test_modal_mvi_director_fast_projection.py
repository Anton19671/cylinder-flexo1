"""FSDT/Reddy MVI-1..8: exact separated K and gradient-dependent M.

Independent reference: original physical stiffness and mass weak actions.
Tests do not claim all global eigenpairs are captured by low-rank MVI.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from cylinder_modal_reference import _modal_spec
from cylinder_modal_mvi import _SeparatedModalProblem
from cylinder_modal_separated_projector import (
    exact_separated_cuf_projection, _director_inertial_jet_matrix)

ROOT = Path(__file__).resolve().parents[1]
RAWAT = ROOT / "config" / "verification" / "rawat2016" / "modal_linear"


def _rel(a, b):
    return float(np.linalg.norm(a - b) /
                 max(float(np.linalg.norm(b)), 1.e-25))


@pytest.mark.parametrize("theory", ["FSDT", "REDDY"])
def test_real_operator_exact_KM_ranks_one_through_eight(theory):
    path = (RAWAT / "mvi_13x13" /
            f"RAWAT2016_{theory}_13x13_MODAL_MVI1.json")
    prepared, density, _ = _modal_spec(
        json.loads(path.read_text(encoding="utf-8")))
    op = _SeparatedModalProblem(prepared, density, rank=1)
    f = op.f
    rng = np.random.default_rng(40000 + (theory == "REDDY"))

    # Independent physical mass check: before using the new fast K/M path.
    field = rng.normal(size=op.shape)
    slow = f.mass_action(field, density_kg_m3=density,
                         mechanical=op.q0, velocity=op.velocity0).ravel()
    assert _rel(op.mass(field.ravel()), slow) < 2.e-10

    for rank in range(1, 9):
        count = rank + 2
        basis = np.zeros((op.nm, count))
        for j in range(count):
            c = (j * 3 + rank) % 5
            trial = np.zeros(op.shape)
            trial[c] = np.outer(
                rng.normal(size=op.nx), rng.normal(size=op.ns))
            basis[:, j] = trial.ravel()
        projected = exact_separated_cuf_projection(
            prepared.problem, density, op._local_tangent, basis)
        assert projected is not None, (
            f"{theory} MVI-{rank} did not use exact contractions")
        k_fast, m_fast = projected
        for j in sorted({0, count // 2, count - 1}):
            k_ref = basis.T @ op.stiffness(basis[:, j])
            m_ref = basis.T @ f.mass_action(
                basis[:, j].reshape(op.shape),
                density_kg_m3=density,
                mechanical=op.q0, velocity=op.velocity0).ravel()
            assert _rel(k_fast[:, j], k_ref) < 3.e-8
            assert _rel(m_fast[:, j], m_ref) < 2.e-10


@pytest.mark.parametrize("theory", ["FSDT", "REDDY"])
def test_local_director_mass_hessian_equals_original_virtual_work(theory):
    from types import SimpleNamespace
    from cylinder_director_g3d import (
        DirectorSurfaceJets, kinetic_weak_coefficients)
    radius, thickness, density, ngauss = .3, .01, 7800., 5
    func = SimpleNamespace(
        theory=theory, R=radius, h=thickness, n_gauss=ngauss,
        n_mechanical=5, n_derivatives=10)
    tangent = SimpleNamespace()
    local = _director_inertial_jet_matrix(func, density, tangent)
    assert _director_inertial_jet_matrix(func, density, tangent) is local
    rng = np.random.default_rng(50000 + (theory == "REDDY"))
    velocity = [rng.normal(size=5) for _ in range(10)]
    jets = DirectorSurfaceJets(*velocity, *(np.zeros(1) for _ in range(3)))
    physical = kinetic_weak_coefficients(
        jets, theory=theory, radius_m=radius,
        thickness_m=thickness, density_kg_m3=density,
        n_gauss=ngauss).T
    val = (local @ np.column_stack(velocity).ravel()).reshape(5, 10).T[:3]
    assert _rel(val, physical) < 2.e-12


@pytest.mark.parametrize("theory", ["FSDT", "REDDY"])
def test_real_frozen_partner_bases_stay_in_fast_path_for_all_ranks(theory):
    """Use actual constrained MVI directions, not synthetic rank-one columns."""
    from cylinder_modal_mvi import _initial_factors
    from cylinder_modal_separated_space import normalize_modal_partner
    from time import perf_counter

    path = (RAWAT / "mvi_13x13" /
            f"RAWAT2016_{theory}_13x13_MODAL_MVI1.json")
    prepared, density, _ = _modal_spec(
        json.loads(path.read_text(encoding="utf-8")))
    op = _SeparatedModalProblem(prepared, density, rank=1)
    for rank in range(1, 9):
        op.rank = rank
        x, y = _initial_factors(op.n_components, rank, op.nx, op.ns, 3)
        for axis in ("Y", "X"):
            xn, yn = normalize_modal_partner(
                x.copy(), y.copy(), axis, getattr(op, "fixed_mask", None))
            bm, be = op.projected_bases(xn, yn, axis)
            assert be.shape[1] == 0
            start = perf_counter()
            projected = op.cached_cuf_projection(bm)
            elapsed = perf_counter() - start
            assert projected is not None, (
                f"{theory} rank={rank} axis={axis}: actual constrained "
                "MVI basis fell back to slow per-column 2D weak actions")
            k, m = projected
            assert k.shape == m.shape == (bm.shape[1], bm.shape[1])
            assert np.isfinite(k).all() and np.isfinite(m).all()
            print(
                f"[REAL-MVI-FAST] {theory} rank={rank} axis={axis} "
                f"directions={bm.shape[1]} elapsed_s={elapsed:.4f}",
                flush=True)
