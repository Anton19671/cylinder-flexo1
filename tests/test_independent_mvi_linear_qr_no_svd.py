"""Older independent linear MVI must not use SVD anywhere in rank seeding.

Algebra-only tests (no shell load/time-history simulation). Verify QR
endpoint reconstruction and strict fail-closed rank limit.
"""
import numpy as np
import pytest

from independent_mvi.linear_models import IndependentVariationalModel
from test_independent_temporal_scientific import ScalarVariational


def test_scalar_mvi_qr_endpoint_reconstructs_without_svd(monkeypatch):
    model=ScalarVariational(amplitude=0.)
    monkeypatch.setattr(np.linalg,"svd",
                        lambda *args,**kwargs:pytest.fail("SVD forbidden"))
    target=np.array([[[.3125]]],dtype=float)
    x,y=model._endpoint_factors(target)
    np.testing.assert_allclose(
        model.field(x,y),target,rtol=0.,atol=1.e-13)
    seed_x,seed_y=model._seed_from_admissible_drive(target)
    assert seed_x.shape[1]==model.rank
    assert seed_y.shape[1]==model.rank


def test_rank_one_endpoint_rejects_genuine_rank_two_field_without_svd(
        monkeypatch):
    from independent_mvi.linear_models import VariationalModelCase
    case=VariationalModelCase(theory="KL",nx=9,ns=5)
    model=IndependentVariationalModel(case,rank=1)
    B=model.basis[0]
    assert B.shape[1]>=2
    target=np.zeros((model.n,case.nx,case.ns))
    a=np.zeros((B.shape[1],case.ns))
    a[0,0]=1.
    a[1,1]=1.
    target[0]=B@a
    monkeypatch.setattr(np.linalg,"svd",
                        lambda *args,**kwargs:pytest.fail("SVD forbidden"))
    with pytest.raises(RuntimeError,match="RANK_FAILURE"):
        model._endpoint_factors(target)
