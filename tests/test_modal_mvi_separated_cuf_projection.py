"""Exact CUF X/Y projected operator contraction versus original weak actions.

No full eigensolver is used: tests independently compare exact B^T K B and
B^T M B against the old columnwise assembled physical operator on real
Cammalleri and Rawat CASEs. CI owns numerical execution.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from cylinder_modal_reference import _modal_spec
from cylinder_modal_mvi import _SeparatedModalProblem
from cylinder_modal_separated_projector import (
    exact_separated_cuf_projection, _one_component_factors)
from cylinder_variational_mvi import _normalize_partner


ROOT=Path(__file__).resolve().parents[1]
CAM=ROOT/"config/verification/cammalleri2023/user_13x21"


@pytest.mark.parametrize("rank,axis",[(1,"Y"),(1,"X"),(2,"Y"),(2,"X")])
def test_real_cammalleri_13x21_projected_km_equal_original_weak_actions(rank,axis):
    file=CAM/f"CAMMALLERI2023_CUF_TE1_13x21_MVI_{rank}.json"
    prepared,density,_=_modal_spec(json.loads(file.read_text(encoding="utf-8")))
    problem=_SeparatedModalProblem(prepared,density,rank)
    x,y,_,_=problem.admissible_start(700+rank,[],harmonic_seed=2)
    x,y=_normalize_partner(x,y,axis)
    bm,_=problem.projected_bases(x,y,axis)
    contracted=exact_separated_cuf_projection(
        prepared.problem,density,problem._local_tangent,bm)
    assert contracted is not None, "Expected exact separated CUF projector"
    k,m=contracted
    assert k.shape==m.shape==(bm.shape[1],bm.shape[1])
    assert np.isfinite(k).all() and np.isfinite(m).all()
    for j in sorted({0,bm.shape[1]//2,bm.shape[1]-1}):
        old_k=bm.T@problem.stiffness(bm[:,j])
        old_m=bm.T@problem.mass(bm[:,j])
        np.testing.assert_allclose(k[:,j],old_k,rtol=5.e-9,atol=1.e-10)
        np.testing.assert_allclose(m[:,j],old_m,rtol=5.e-11,atol=1.e-12)


def test_rejects_a_nonseparable_physical_basis_without_enrichment():
    basis=np.zeros((1*5*7,1))
    basis[:,0]=np.arange(35).reshape(5,7).ravel()**2
    assert _one_component_factors(basis,1,5,7) is None


def test_extracts_existing_single_component_xy_factor_without_svd():
    rng=np.random.default_rng(1702)
    x=rng.normal(size=5)
    y=rng.normal(size=7)
    b=np.zeros((2,5,7))
    b[1]=x[:,None]*y[None,:]
    c,a,ang=_one_component_factors(b.reshape(-1,1),2,5,7)
    assert c.tolist()==[1]
    np.testing.assert_allclose(a[0,:,None]*ang[0,None,:],b[1],rtol=1.e-12,atol=1.e-12)


def test_spring_hessian_is_left_to_original_projection():
    from types import SimpleNamespace
    from cylinder_modal_separated_projector import _spring_free_modal_edges
    assert _spring_free_modal_edges(
        SimpleNamespace(spec={"mechanical":{"left":{"spring":[0.,0.]}}}))
    assert not _spring_free_modal_edges(
        SimpleNamespace(spec={"mechanical":{"left":{"spring":[1.,0.]}}}))


def _serial_one_component_factors(basis, ncomp, nx, ns):
    """Unoptimized implementation kept as an independent test oracle."""
    field = np.asarray(basis, float)
    if field.ndim != 2 or field.shape[0] != ncomp * nx * ns:
        return None
    count = field.shape[1]
    axes = np.zeros((count, nx))
    angles = np.zeros((count, ns))
    components = np.empty(count, dtype=int)
    values = field.T.reshape(count, ncomp, nx, ns)
    for j in range(count):
        candidate = values[j]
        norms = np.linalg.norm(candidate.reshape(ncomp, -1), axis=1)
        c = int(np.argmax(norms))
        if norms[c] <= 0. or not np.isfinite(norms).all():
            return None
        if np.max(np.delete(norms, c), initial=0.) > 1.e-12 * norms[c]:
            return None
        profile = candidate[c]
        index = int(np.argmax(np.abs(profile)))
        ix, is_ = divmod(index, ns)
        pivot = float(profile[ix, is_])
        if pivot == 0.:
            return None
        axial = profile[:, is_].copy()
        angular = profile[ix, :] / pivot
        difference = np.linalg.norm(
            profile - axial[:, None] * angular[None, :])
        if not np.isfinite(difference) or (
                difference > 1.e-10 * max(np.linalg.norm(profile), 1.e-30)):
            return None
        components[j] = c
        axes[j] = axial
        angles[j] = angular
    return components, axes, angles


@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("axis", ("X", "Y"))
def test_vectorized_factors_equal_original_at_every_rank(rank, axis):
    rng = np.random.default_rng(2084 + rank * 11 + (axis == "Y"))
    ncomp, nx, ns = 6, 13, 21
    count = rank * (nx if axis == "X" else ns)
    basis = np.zeros((ncomp * nx * ns, count))
    for j in range(count):
        c = (j * 5 + rank) % ncomp
        x = rng.standard_normal(nx)
        y = rng.standard_normal(ns)
        field = np.zeros((ncomp, nx, ns))
        field[c] = (0.000254 if c < 3 else 1.) * np.outer(x, y)
        basis[:, j] = field.ravel()
    slow = _serial_one_component_factors(basis, ncomp, nx, ns)
    fast = _one_component_factors(basis, ncomp, nx, ns)
    assert slow is not None and fast is not None
    np.testing.assert_array_equal(fast[0], slow[0])
    np.testing.assert_array_equal(fast[1], slow[1])
    np.testing.assert_array_equal(fast[2], slow[2])

    mixed = basis[:, :1].copy()
    other = (int(slow[0][0]) + 1) % ncomp
    mixed[other * nx * ns + (nx // 2) * ns + 1, 0] = 0.01
    assert _one_component_factors(mixed, ncomp, nx, ns) is None
    assert _serial_one_component_factors(mixed, ncomp, nx, ns) is None

    nonseparable = basis[:, :1].copy()
    original = int(slow[0][0])
    nonseparable[original * nx * ns + ns + 2, 0] += 0.02
    assert _one_component_factors(nonseparable, ncomp, nx, ns) is None
    assert _serial_one_component_factors(nonseparable, ncomp, nx, ns) is None


def test_vectorized_factors_reject_invalid_fields_without_approximating():
    for count in (1, 2):
        bad = np.zeros((2 * 5 * 7, count))
        assert _one_component_factors(bad, 2, 5, 7) is None
        bad[0, 0] = np.nan
        assert _one_component_factors(bad, 2, 5, 7) is None
    assert _one_component_factors(np.zeros((71, 3)), 2, 5, 7) is None
    empty = _one_component_factors(np.zeros((2 * 5 * 7, 0)), 2, 5, 7)
    assert empty is not None
    assert [v.shape for v in empty] == [(0,), (0, 5), (0, 7)]
