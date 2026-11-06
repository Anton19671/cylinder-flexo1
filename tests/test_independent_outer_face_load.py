"""Outer-reference-face pressure work audit against Krysko-Zhong Eq. (83)-(85).

All checks are static/algebraic. No displacement solution and no time step
is computed. This protects all radial CUF amplitudes and the cylindrical
outer-face Jacobian separately in the MKR and MVI implementations.
"""
from math import pi

import numpy as np
import pytest

from independent_mkr.linear_models import (
    MODELS, IndependentStrongModel, StrongModelCase,
)
from independent_mvi.linear_models import (
    IndependentVariationalModel, VariationalModelCase,
)


def _prescribed_window(case):
    if case.pressure_kind=="UNIFORM":
        return np.ones((case.nx,case.ns))
    def front(distance,width):
        location=np.clip(distance/width+0.5,0.0,1.0)
        return np.sin(0.5*pi*location)**2
    axial_x=np.linspace(0,case.length_m,case.nx)
    angles=2*pi*np.arange(case.ns)/case.ns
    start,end=case.x_interval_m
    lower,upper=case.theta_interval_rad
    axial=front(axial_x-start,case.x_edge_m)*front(
        end-axial_x,case.x_edge_m)
    shortest=(angles-(lower+upper)/2+pi)%(2*pi)-pi
    angular=front((upper-lower)/2-np.abs(shortest),case.theta_edge_rad)
    return np.outer(axial,angular)


@pytest.mark.parametrize("method",("MKR","MVI"))
@pytest.mark.parametrize("theory",MODELS)
@pytest.mark.parametrize("load_kind",("UNIFORM","PATCH"))
def test_exact_outer_face_projection_for_every_shell_theory(method,theory,load_kind):
    cls=StrongModelCase if method=="MKR" else VariationalModelCase
    case=cls(theory=theory,pressure_kind=load_kind,
             thickness_m=.06,radius_m=.9,pressure_pa=73000.)
    solver=(IndependentStrongModel(case) if method=="MKR"
            else IndependentVariationalModel(case,rank=1))
    Hout=1+case.thickness_m/(2*case.radius_m)
    expected=-case.pressure_pa*Hout*_prescribed_window(case)
    if theory.startswith("CUF_TE"):
        count=int(theory[-1])+1
        assert solver.n==3*count
        expected_radial=np.broadcast_to(expected,(count,case.nx,case.ns))
        np.testing.assert_allclose(solver.force[2*count:3*count],
                                   expected_radial,rtol=0,atol=1e-9)
        np.testing.assert_array_equal(solver.force[:2*count],
                                      np.zeros((2*count,case.nx,case.ns)))
    else:
        np.testing.assert_allclose(solver.force[2],expected,
                                   rtol=0,atol=1e-9)
        np.testing.assert_array_equal(solver.force[:2],np.zeros_like(solver.force[:2]))
        np.testing.assert_array_equal(solver.force[3:],np.zeros_like(solver.force[3:]))


@pytest.mark.parametrize("method",("MKR","MVI"))
@pytest.mark.parametrize("theory",MODELS)
def test_outer_face_virtual_work_is_conjugate_to_displacement(method,theory):
    cls=StrongModelCase if method=="MKR" else VariationalModelCase
    case=cls(theory=theory,pressure_pa=1200.,pressure_kind="PATCH")
    solver=(IndependentStrongModel(case) if method=="MKR"
            else IndependentVariationalModel(case,rank=1))
    rng=np.random.default_rng(1919)
    virtual=rng.normal(size=(solver.n,case.nx,case.ns))
    if theory.startswith("CUF_TE"):
        count=int(theory[-1])+1
        displacement_at_face=np.sum(virtual[2*count:3*count],axis=0)
    else:
        displacement_at_face=virtual[2]
    weights=np.ones((case.nx,case.ns)) if method=="MKR" else solver.area
    computed=float(np.sum(weights[None]*solver.force*virtual))
    prescribed=-case.pressure_pa*(1+case.thickness_m/(2*case.radius_m))
    reference=float(prescribed*np.sum(
        weights*_prescribed_window(case)*displacement_at_face))
    np.testing.assert_allclose(computed,reference,rtol=1e-13,atol=1e-8)


@pytest.mark.parametrize("method",("MKR","MVI"))
@pytest.mark.parametrize("theory",("CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
def test_inner_face_parity_differs_from_outer_face_projection(method,theory):
    # This test guards the deliberate reference-face choice: F_k(+h/2)=+1
    # even when the polynomial basis has odd powers.
    cls=StrongModelCase if method=="MKR" else VariationalModelCase
    case=cls(theory=theory)
    solver=(IndependentStrongModel(case) if method=="MKR"
            else IndependentVariationalModel(case,rank=1))
    count=int(theory[-1])+1
    np.testing.assert_allclose(
        solver.force[2*count:3*count],
        np.broadcast_to(solver.force[2*count],(count,case.nx,case.ns)),
        rtol=0,atol=0)
    assert np.all(solver.force[2*count:3*count]<0)
