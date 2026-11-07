"""Global free-surface equation checks for isolated rank-fixed MVI.

These are manufactured algebraic values only; NO shell assembly,
physical solve or time integration is executed here.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from independent_mvi.linear_models import IndependentVariationalModel
from independent_mvi.linear_te4 import VariationalTE4


def _manufactured_model():
    """Avoid constructing any physical stiffness, mass or load operator."""
    model=object.__new__(IndependentVariationalModel)
    model.n=1
    model.rank=1
    model.case=SimpleNamespace(nx=3,ns=5,frequency_hz=1.)
    model.area=np.ones((3,5))
    model.force=np.zeros((1,3,5))
    model.force[0,1,3]=1.
    model.pack=lambda field: np.asarray(field,dtype=float).ravel()
    model.mass_action=lambda field: np.zeros_like(field)
    X=np.zeros((1,1,3));X[0,0,1]=1.
    Y=np.zeros((1,1,5));Y[0,0,0]=1.
    model._normalize=lambda a,b,freeze: (a,b)
    model._directions=lambda a,b,active: (np.zeros((15,1)),[1])
    return model,(X,Y)


def test_global_free_surface_equations_reject_false_stationary_state():
    model,factors=_manufactured_model()
    field=model.field(*factors)
    model.stiffness_action=lambda q: np.zeros_like(q)
    assert not model._all_free_surface_equations_hold(
        model.force,model.stiffness_action(field),1.e-7)
    with pytest.raises(RuntimeError,match="RANK_NOT_SUFFICIENT_FOR_FULL_2D"):
        model.solve_static(previous=factors,max_sweeps=2)
    with pytest.raises(RuntimeError,match="RANK_NOT_SUFFICIENT_FOR_FULL_2D"):
        model._effective_step(np.zeros_like(field),.25,4.,previous=factors,
                              max_sweeps=2)


def test_global_free_surface_equations_accept_exact_stationary_state():
    model,factors=_manufactured_model()
    model.stiffness_action=lambda q: model.force.copy()
    field,other,_=model.solve_static(previous=factors,max_sweeps=2)
    np.testing.assert_array_equal(field,model.field(*factors))
    assert other[0].shape==factors[0].shape
    assert model._all_free_surface_equations_hold(
        model.force,model.stiffness_action(field),1.e-7)


def test_full_surface_criterion_checks_all_free_angular_modes():
    model,_=_manufactured_model()
    load=np.zeros((1,3,5));load[0,1,4]=1.
    action=load.copy()
    assert model._all_free_surface_equations_hold(load,action,1.e-7)
    action[0,1,4]=0.
    assert not model._all_free_surface_equations_hold(load,action,1.e-7)
    with pytest.raises(ValueError,match="ACCURACY_INVALID"):
        model._all_free_surface_equations_hold(load,action,0.)


def test_te4_global_2d_free_equations_ignore_fixed_axial_values_only():
    load=np.zeros((15,5,7))
    load[10,2,4]=1.
    actual=load.copy()
    assert VariationalTE4._all_free_surface_equations_hold(load,actual,1.e-8)
    actual[10,2,4]=0.
    assert not VariationalTE4._all_free_surface_equations_hold(
        load,actual,1.e-8)
    actual=load.copy()
    actual[:,0,:]=1.e10
    actual[:,-1,:]=-1.e10
    assert VariationalTE4._all_free_surface_equations_hold(
        load,actual,1.e-8)
    with pytest.raises(ValueError,match="CHECK_INVALID"):
        VariationalTE4._all_free_surface_equations_hold(
            load,actual,-1.)
