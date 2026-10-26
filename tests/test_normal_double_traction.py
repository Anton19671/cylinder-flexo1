"""Pointwise mechanical double traction is NOT a complete face traction."""
import numpy as np
from cylinder_normal_double_traction import (
    symmetric_tensor_from_engineering_conjugate,
    recover_normal_double_traction)


def test_engineering_shear_is_not_halved_twice():
    sigma=np.array([1.,2.,3.,4.,5.,6.])
    S=symmetric_tensor_from_engineering_conjugate(sigma)
    assert S[0,1]==4. and S[0,2]==5. and S[1,2]==6.


def test_normal_normal_double_force_is_same_on_inner_and_outer_faces():
    F=np.eye(3)
    sigma=np.zeros(6)
    m=np.zeros((6,3))
    m[2,2]=3.
    m[4,2]=-4.
    m[5,2]=5.
    _,g=recover_normal_double_traction(F,sigma,m)
    np.testing.assert_allclose(g,[-4.,5.,3.])
    # n_i n_k carries TWO outward signs, cancelling on either face.
    for outward in (-1.,1.):
        np.testing.assert_allclose(
            np.array([outward*outward*g[0],
                      outward*outward*g[1],
                      outward*outward*g[2]]),g)


def test_double_force_vanishes_when_hyperstress_zero():
    F=np.array([[1.,.1,0.],[-.02,1.,.03],[.05,0.,.9]])
    nominal,g=recover_normal_double_traction(
        F,np.array([2.,4.,6.,8.,10.,12.]),np.zeros((6,3)))
    np.testing.assert_allclose(g,0.)
    assert np.isfinite(nominal).all()
