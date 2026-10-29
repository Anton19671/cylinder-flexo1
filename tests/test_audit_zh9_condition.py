"""Audit Zh-9: near-singular tangents and safe indefinite LU."""
import numpy as np
import pytest
from cylinder_variational_fdm import StaticFDMSettings, solve_checked_reduced_tangent

def test_reject_near_singular_tangent():
    with pytest.raises(np.linalg.LinAlgError):
        solve_checked_reduced_tangent(np.diag([1.,1.e-14]),np.ones(2))

def test_user_defined_condition_floor():
    a=np.diag([1.,1.e-5])
    _,cond=solve_checked_reduced_tangent(a,np.ones(2))
    assert cond==pytest.approx(1.e5)
    with pytest.raises(np.linalg.LinAlgError):
        solve_checked_reduced_tangent(a,np.ones(2),min_reciprocal_condition=1.e-4)

def test_indefinite_electric_saddle_is_not_regularized():
    a=np.array([[2.,.5],[.5,-1.]])
    step,_=solve_checked_reduced_tangent(a,np.array([1.,-2.]))
    np.testing.assert_allclose(a@step,[-1.,2.],atol=1.e-13)

@pytest.mark.parametrize("floor",[-1.,1.,float("nan"),True])
def test_invalid_floor_rejected(floor):
    with pytest.raises(ValueError):
        solve_checked_reduced_tangent(np.eye(2),np.ones(2),min_reciprocal_condition=floor)

def test_settings_control():
    assert StaticFDMSettings(min_reciprocal_condition=1.e-10).min_reciprocal_condition==1.e-10
    with pytest.raises(ValueError):
        StaticFDMSettings(min_reciprocal_condition=1.)
