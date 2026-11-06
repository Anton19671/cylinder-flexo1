"""Krysko-Zhong CUF scientific algebra audit (source equations 10-55, 61-73, 88-131).

No displacement solution, time marching or physical production calculation.
Independent tests compare the cylindrical tensor connection with its full
Cartesian commutator and the coupled weak work with finite differences of
the *original enthalpy*, across all 24 TE/geometry/gradient combinations.
They do not certify boundary closure or full electromechanical PDE accuracy.
"""
from __future__ import annotations

import numpy as np
import pytest

from cylinder_paths import activate
activate()

from cuf_g3d_weak import (
    CUFSurfaceJets, ELECTRICAL_VARIATIONS, MECHANICAL_VARIATIONS,
    point_state, thickness_enthalpy, thickness_weak_coefficients,
)
from flexo_gradient_g3d import G3DMaterial, covariant_strain_gradients

THEORIES=(1,2,3,4)
GEOMETRIES=("LINEAR","VON_KARMAN","GREEN_LAGRANGE")
GRADIENTS=("G3D","Gz")
RADIUS=2.4
THICKNESS=.24
N_GAUSS=8


def _material(model):
    return G3DMaterial(120.,.27,.13,.18,-.11,2.7,gradient_model=model)


def _state(n, rng):
    p=n
    mechanics=[rng.normal(0.,.0005,(3,n+1))
               for _ in MECHANICAL_VARIATIONS]
    electric=[rng.normal(0.,.008,p+1)
              for _ in ELECTRICAL_VARIATIONS]
    return CUFSurfaceJets(*mechanics,*electric)


def _direction(base, rng, *, mechanical=True, electrical=True):
    fields={}
    for key in MECHANICAL_VARIATIONS:
        value=getattr(base,key)
        fields[key]=(rng.normal(0.,.006,value.shape) if mechanical
                     else np.zeros_like(value))
    for key in ELECTRICAL_VARIATIONS:
        value=getattr(base,key)
        fields[key]=(rng.normal(0.,.025,value.shape) if electrical
                     else np.zeros_like(value))
    return base.replaced(**fields)


def _offset(base, direction, coefficient):
    return base.replaced(**{
        key:getattr(base,key)+coefficient*getattr(direction,key)
        for key in (*MECHANICAL_VARIATIONS,*ELECTRICAL_VARIATIONS)
    })


def _virtual_work(coeff, direction):
    mechanical=sum(
        np.sum(coeff.mechanical[...,k]*getattr(direction,name))
        for k,name in enumerate(MECHANICAL_VARIATIONS))
    electric=sum(
        np.sum(coeff.electrical[...,k]*getattr(direction,name))
        for k,name in enumerate(ELECTRICAL_VARIATIONS))
    return float(mechanical+electric)


def _symmetric_tensor(e):
    return np.array(((e[0],e[3]/2,e[4]/2),
                     (e[3]/2,e[1],e[5]/2),
                     (e[4]/2,e[5]/2,e[2])),dtype=float)


def _engineering(t):
    return np.array((t[0,0],t[1,1],t[2,2],
                     2*t[0,1],2*t[0,2],2*t[1,2]),dtype=float)


@pytest.mark.parametrize("radius,z",[(1.8,-.13),(1.8,.14),(3.2,.06)])
def test_all_18_covariant_components_against_cartesian_tensor_connection(radius,z):
    """Cartesian rotating-basis derivative is an independent 3x3 oracle."""
    rng=np.random.default_rng(9812)
    e=rng.normal(size=6)
    dx=rng.normal(size=6)
    ds=rng.normal(size=6)
    dz=rng.normal(size=6)
    actual=covariant_strain_gradients(
        e,dx,ds,dz,radius_m=radius,z_m=z)
    # Basis [x, theta, r]: de_theta/dtheta=-e_r,
    # de_r/dtheta=e_theta. The material s-direction is R*theta.
    generator=np.zeros((3,3))
    generator[1,2]=1.
    generator[2,1]=-1.
    cartesian_rotation=(generator@_symmetric_tensor(e)
                        -_symmetric_tensor(e)@generator)
    reference_s=(_engineering(_symmetric_tensor(ds))/(1+z/radius)
                 +_engineering(cartesian_rotation)/(radius+z))
    np.testing.assert_allclose(actual[:,0],dx,rtol=0,atol=0)
    np.testing.assert_allclose(actual[:,1],reference_s,rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(actual[:,2],dz,rtol=0,atol=0)


@pytest.mark.parametrize("order",THEORIES)
@pytest.mark.parametrize("geometry",GEOMETRIES)
@pytest.mark.parametrize("gradient_model",GRADIENTS)
def test_exact_cuf_weak_work_matches_enthalpy_gateaux_all_modes(
    order,geometry,gradient_model
):
    rng=np.random.default_rng(12345+order+10*GEOMETRIES.index(geometry)
                              +100*GRADIENTS.index(gradient_model))
    state=_state(order,rng)
    direction=_direction(state,rng)
    material=_material(gradient_model)
    params=dict(radius_m=RADIUS,thickness_m=THICKNESS,
                n_gauss=N_GAUSS,geometry_model=geometry)
    coefficients=thickness_weak_coefficients(state,material,**params)
    h=1.e-4
    positive=thickness_enthalpy(
        _offset(state,direction,h),material,**params)
    negative=thickness_enthalpy(
        _offset(state,direction,-h),material,**params)
    finite_difference=(positive-negative)/(2*h)
    exact=_virtual_work(coefficients,direction)
    np.testing.assert_allclose(exact,finite_difference,
                               rtol=4.e-6,atol=5.e-8)
    assert coefficients.mechanical.shape==(3,order+1,6)
    assert coefficients.electrical.shape==(order+1,3)


@pytest.mark.parametrize("order",THEORIES)
@pytest.mark.parametrize("geometry",GEOMETRIES)
@pytest.mark.parametrize("gradient_model",GRADIENTS)
def test_direct_inverse_flexoelectric_tangent_reciprocity(
    order,geometry,gradient_model
):
    """The mixed q/phi derivative must be symmetric for one reference enthalpy."""
    rng=np.random.default_rng(83611+order+7*GEOMETRIES.index(geometry)
                              +70*GRADIENTS.index(gradient_model))
    state=_state(order,rng)
    dm=_direction(state,rng,electrical=False)
    de=_direction(state,rng,mechanical=False)
    material=_material(gradient_model)
    params=dict(radius_m=RADIUS,thickness_m=THICKNESS,
                n_gauss=N_GAUSS,geometry_model=geometry)
    h=1.e-3
    plus_e=thickness_weak_coefficients(
        _offset(state,de,h),material,**params)
    minus_e=thickness_weak_coefficients(
        _offset(state,de,-h),material,**params)
    cross_mechanical=(_virtual_work(plus_e,dm)
                      -_virtual_work(minus_e,dm))/(2*h)

    plus_m=thickness_weak_coefficients(
        _offset(state,dm,h),material,**params)
    minus_m=thickness_weak_coefficients(
        _offset(state,dm,-h),material,**params)
    cross_electric=(_virtual_work(plus_m,de)
                    -_virtual_work(minus_m,de))/(2*h)
    np.testing.assert_allclose(cross_mechanical,cross_electric,
                               rtol=4.e-5,atol=4.e-8)


@pytest.mark.parametrize("order",THEORIES)
@pytest.mark.parametrize("gradient_model",GRADIENTS)
def test_finite_rigid_rotation_uses_exact_gl_strain_and_gradients(
    order,gradient_model
):
    """Reference document Eq. (238)-(239): finite rigid rotation, TE1 embedded."""
    zeros=np.zeros((3,order+1))
    potential=np.zeros(order+1)
    state=CUFSurfaceJets(*(zeros.copy() for _ in MECHANICAL_VARIATIONS),
                         *(potential.copy() for _ in ELECTRICAL_VARIATIONS))
    theta=.37
    q=state.q.copy()
    q[1,:2]=(RADIUS*np.sin(theta),THICKNESS*.5*np.sin(theta))
    q[2,:2]=(RADIUS*(np.cos(theta)-1),
              THICKNESS*.5*(np.cos(theta)-1))
    state=state.replaced(q=q)
    for z in (-THICKNESS/2,-THICKNESS/3,0.,THICKNESS/3,THICKNESS/2):
        point=point_state(state,z_m=z,radius_m=RADIUS,
                          thickness_m=THICKNESS,
                          geometry_model="GREEN_LAGRANGE")
        np.testing.assert_allclose(point.strain6,0.,atol=5.e-14,rtol=0.)
        np.testing.assert_allclose(point.gradient6x3,0.,atol=5.e-14,rtol=0.)
        np.testing.assert_allclose(point.det_f,1.,atol=3.e-14,rtol=0.)
    result=thickness_enthalpy(
        state,_material(gradient_model),
        radius_m=RADIUS,thickness_m=THICKNESS,
        n_gauss=N_GAUSS,geometry_model="GREEN_LAGRANGE")
    assert abs(result)<1.e-20


@pytest.mark.parametrize("order",THEORIES)
@pytest.mark.parametrize("gradient_model",GRADIENTS)
def test_single_global_electric_gauge_invariance_cuf(order,gradient_model):
    rng=np.random.default_rng(11231+order)
    state=_state(order,rng)
    original=state.psi.copy()
    shifted=original.copy()
    shifted[0]+=2.5
    material=_material(gradient_model)
    params=dict(radius_m=RADIUS,thickness_m=THICKNESS,
                n_gauss=N_GAUSS,geometry_model="GREEN_LAGRANGE")
    h1=thickness_enthalpy(state,material,**params)
    h2=thickness_enthalpy(state.replaced(psi=shifted),material,**params)
    np.testing.assert_allclose(h1,h2,rtol=3.e-14,atol=3.e-14)
