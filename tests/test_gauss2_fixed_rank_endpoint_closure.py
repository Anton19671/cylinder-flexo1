"""Physical rank geometry: a Gauss-2 endpoint need not retain stage rank.

This is an algebraic contract regression, NOT a shell PDE solve.
Do not remove the strict endpoint fixed-rank acceptance to make it pass.
"""
import numpy as np
import pytest

from cylinder_gauss2_mvi import _rank_factors
from cylinder_gauss2_rk import GAUSS2_B


def _matrix_rank_pivoted_qr(z, threshold=1e-11):
    from scipy.linalg import qr
    _, r, _=qr(z.T, mode="economic", pivoting=True)
    d=np.abs(np.diag(r))
    return int(np.count_nonzero(d>threshold*max(float(d.max(initial=0)),1e-300)))


def test_two_exact_rank_one_stages_have_rank_two_gauss_endpoint():
    # Each stage has a perfectly representable rank-one displacement,
    # but the update combines two independent axial and angular factors.
    nx=ns=7
    x=np.arange(nx,dtype=float)
    s=np.arange(ns,dtype=float)
    first=np.outer(1+x,1+s)
    second=np.outer((1+x)**2,(1+s)**2)
    assert _matrix_rank_pivoted_qr(first)==1
    assert _matrix_rank_pivoted_qr(second)==1
    endpoint=(GAUSS2_B[0]*first+GAUSS2_B[1]*second)
    assert _matrix_rank_pivoted_qr(endpoint)==2
    with pytest.raises(ValueError,match="GAUSS2_MVI_FIELD_OUTSIDE_FIXED_RANK"):
        _rank_factors(endpoint[None,...],1,2.e-10,"GAUSS2_MVI_ENDPOINT")
    # Rank two is accepted because it can represent the actual field
    # without projection, discarded components, or changed equations.
    a,b=_rank_factors(endpoint[None,...],2,2.e-10,"GAUSS2_MVI_ENDPOINT")
    np.testing.assert_allclose(np.einsum("crx,crs->cxs",a,b)[0],
                               endpoint,rtol=1e-11,atol=1e-11)


def test_two_exact_rank_two_stages_can_require_rank_four_at_endpoint():
    nx=ns=7
    x=np.arange(nx,dtype=float)
    s=np.arange(ns,dtype=float)
    a=sum(np.outer((x+1.)**k,(s+1.)**k) for k in (0,1))
    b=sum(np.outer((x+1.)**k,(s+1.)**k) for k in (2,3))
    assert _matrix_rank_pivoted_qr(a)==2
    assert _matrix_rank_pivoted_qr(b)==2
    endpoint=GAUSS2_B[0]*a+GAUSS2_B[1]*b
    assert _matrix_rank_pivoted_qr(endpoint)==4
    with pytest.raises(ValueError,match="GAUSS2_MVI_FIELD_OUTSIDE_FIXED_RANK"):
        _rank_factors(endpoint[None,...],2,2.e-10,"GAUSS2_MVI_ENDPOINT")


def test_zero_component_still_stays_exact_zero_in_strict_rank_check():
    q=np.zeros((3,7,7))
    q[2]=np.outer(np.arange(7.)+1,np.ones(7))
    a,b=_rank_factors(q,1,2.e-10,"GAUSS2_MVI_ENDPOINT")
    np.testing.assert_allclose(np.einsum("crx,crs->cxs",a,b),q,atol=1e-14)
