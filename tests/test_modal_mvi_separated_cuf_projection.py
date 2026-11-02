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
