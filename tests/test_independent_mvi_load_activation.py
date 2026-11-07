"""Independent separated-field excitation tests; algebra only.

No physical shell solve, no transient step, no Newton iteration.
"""
from __future__ import annotations
import numpy as np
import pytest
from independent_mvi.linear_models import (
    MODELS, IndependentVariationalModel, VariationalModelCase)
from independent_mvi.linear_te4 import VariationalCase, VariationalTE4


@pytest.mark.parametrize("theory",MODELS)
@pytest.mark.parametrize("rank",(1,2,4,8))
def test_seven_models_use_admissible_nonlocal_axial_load_seed(theory,rank):
    c=VariationalModelCase(theory=theory,nx=15,ns=9)
    solver=IndependentVariationalModel(c,rank)
    radial=(2 if theory in ("KL","FSDT","REDDY")
            else 2*(int(theory[-1])+1))
    drive=np.zeros((solver.n,c.nx,c.ns))
    drive[radial,c.nx-3,c.ns//2]=7.
    X,Y=solver._seed_from_admissible_drive(drive)
    assert X.shape==(solver.n,rank,c.nx)
    assert Y.shape==(solver.n,rank,c.ns)
    assert not np.any(Y)
    assert np.max(np.abs(X[:,:,[0,-1]]))<2.e-13
    assert not np.any(solver.field(X,Y))
    assert np.linalg.norm(X[radial]@drive[radial])>0
    for k,B in enumerate(solver.basis):
        np.testing.assert_allclose(B@(B.T@X[k].T),X[k].T,
                                   atol=1.e-12,rtol=1.e-12)


@pytest.mark.parametrize("rank",(1,2,4,6))
def test_te4_seed_captures_localized_load_without_increasing_rank(rank):
    c=VariationalCase(nx=11,ns=9)
    solver=VariationalTE4(c,rank)
    drive=np.zeros((15,c.nx,c.ns))
    drive[10,7,4]=11.
    x,y=solver._seed_from_admissible_drive(drive)
    assert x.shape==(15,rank,c.nx)
    assert y.shape==(15,rank,c.ns)
    assert not np.any(y)
    assert not np.any(solver._field(x,y))
    np.testing.assert_array_equal(x[:,:,[0,-1]],0.)
    assert np.linalg.norm(x[10]@drive[10])>0


def test_zero_driving_state_is_preserved():
    solver=IndependentVariationalModel(VariationalModelCase(),1)
    zero=np.zeros((solver.n,solver.case.nx,solver.case.ns))
    X,Y=solver._seed_from_admissible_drive(zero)
    a,b=solver._first_factors()
    np.testing.assert_array_equal(X,a)
    np.testing.assert_array_equal(Y,b)
    te4=VariationalTE4(VariationalCase(),rank=1)
    zero=np.zeros((15,te4.case.nx,te4.case.ns))
    x,y=te4._seed_from_admissible_drive(zero)
    a,b=te4._initial()
    np.testing.assert_array_equal(x,a)
    np.testing.assert_array_equal(y,b)


def test_invalid_drive_is_rejected_before_halfstep():
    a=IndependentVariationalModel(VariationalModelCase(),1)
    with pytest.raises(ValueError,match="FINITE_ADMISSIBLE_DRIVE"):
        a._seed_from_admissible_drive(np.full(
            (a.n,a.case.nx,a.case.ns),np.nan))
    b=VariationalTE4(VariationalCase(),rank=1)
    with pytest.raises(ValueError,match="BAD_SEED_DRIVE"):
        b._seed_from_admissible_drive(np.zeros((14,b.case.nx,b.case.ns)))
