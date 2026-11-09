"""Manufactured formal natural-end actions for independent coupled MVI.

No physical shell solve is performed. These tests check derivative order,
left/right normal, circumferential integration by parts and array shapes.
"""
import numpy as np
import pytest

from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid, NativeCoupledMVISurface, CoupledMVIWork,
)
from independent_mvi.coupled_end_actions import formal_mvi_axial_end_actions
from flexo_gradient_g3d import G3DMaterial


def _surface(theory, electric=False):
    grid=CoupledMVISurfaceGrid(11,7,2.,1.3)
    material=G3DMaterial(100.,.2,.1,
                         .01 if electric else 0.,
                         .02 if electric else 0.,
                         1. if electric else 0.)
    return NativeCoupledMVISurface(
        theory,material,grid=grid,thickness_m=.2,
        potential_order=1 if electric else None,n_gauss=2)


def _manufactured(surface,coefficient):
    shape=(surface.n_fields,len(surface.multi),
           surface.grid.nx,surface.grid.ns)
    assert coefficient.shape==shape
    q=np.zeros((surface.n_mechanical,surface.grid.nx,surface.grid.ns))
    p=(np.zeros((surface.n_electric,surface.grid.nx,surface.grid.ns))
       if surface.electrical_on else None)
    return CoupledMVIWork(0.,q,p,
        np.zeros((surface.n_fields,1,surface.grid.nx)),
        np.zeros((surface.n_fields,1,surface.grid.ns)),
        1,coefficient_jets=coefficient)


@pytest.mark.parametrize("theory,expected",[
    ("KL",3),("FSDT",3),("REDDY",3),
    ("CUF_TE1",2),("CUF_TE2",2),
    ("CUF_TE3",2),("CUF_TE4",2)])
def test_order_and_opposite_outer_normals(theory,expected):
    surface=_surface(theory)
    coeff=np.zeros((surface.n_fields,len(surface.multi),
                    surface.grid.nx,surface.grid.ns))
    coeff[0,1,:,:]=4.5  # first axial virtual derivative
    result=formal_mvi_axial_end_actions(surface,
                                        _manufactured(surface,coeff))
    assert len(result.left[0])==expected
    assert len(result.right[0])==expected
    np.testing.assert_allclose(result.left[0][0],-4.5)
    np.testing.assert_allclose(result.right[0][0],4.5)
    for k in range(1,expected):
        np.testing.assert_allclose(result.left[0][k],0.,atol=1.e-12)
        np.testing.assert_allclose(result.right[0][k],0.,atol=1.e-12)


def test_second_axial_order_contributes_slope_and_derivative_to_value():
    surface=_surface("CUF_TE1")
    x=np.linspace(0,2.,surface.grid.nx)
    coeff=np.zeros((surface.n_fields,len(surface.multi),
                    surface.grid.nx,surface.grid.ns))
    coeff[0,3,:,:]=3*x[:,None]
    output=formal_mvi_axial_end_actions(surface,
                                        _manufactured(surface,coeff))
    # B2=3x, so moment for slope is +/-3x and physical end shear
    # for value is n*(-d_x B2) = -3n at each end.
    np.testing.assert_allclose(output.left[0][0],3.,atol=1.e-10)
    np.testing.assert_allclose(output.right[0][0],-3.,atol=1.e-10)
    np.testing.assert_allclose(output.left[0][1],0.,atol=1.e-10)
    np.testing.assert_allclose(output.right[0][1],6.,atol=1.e-10)


def test_periodic_reduction_of_circumferential_gradient():
    surface=_surface("CUF_TE1")
    s=np.arange(surface.grid.ns)*2*np.pi*surface.grid.radius_m/surface.grid.ns
    coeff=np.zeros((surface.n_fields,len(surface.multi),
                    surface.grid.nx,surface.grid.ns))
    # A_(1,1)=sin(s/R) and B_1=-d_s A_(1,1).
    coeff[0,4,:,:]=np.sin(s/surface.grid.radius_m)[None,:]
    output=formal_mvi_axial_end_actions(surface,
                                        _manufactured(surface,coeff))
    expected=np.cos(s/surface.grid.radius_m)/surface.grid.radius_m
    np.testing.assert_allclose(output.left[0][0],expected,atol=2.e-12)
    np.testing.assert_allclose(output.right[0][0],-expected,atol=2.e-12)


def test_electric_modes_have_one_axial_flux_each():
    surface=_surface("CUF_TE1",electric=True)
    coeff=np.zeros((surface.n_fields,len(surface.multi),
                    surface.grid.nx,surface.grid.ns))
    coeff[surface.n_mechanical:,1,:,:]=2.
    result=formal_mvi_axial_end_actions(surface,
                                        _manufactured(surface,coeff))
    for c in range(surface.n_mechanical,surface.n_fields):
        assert len(result.left[c])==1
        np.testing.assert_allclose(result.left[c][0],-2.)
        np.testing.assert_allclose(result.right[c][0],2.)


def test_coefficients_are_mandatory():
    surface=_surface("KL")
    q=np.zeros((surface.n_mechanical,surface.grid.nx,surface.grid.ns))
    work=CoupledMVIWork(0.,q,None,np.zeros((surface.n_fields,1,11)),
                       np.zeros((surface.n_fields,1,7)),1)
    with pytest.raises(ValueError,match="REQUIRE_LOCAL_WEAK_COEFFICIENTS"):
        formal_mvi_axial_end_actions(surface,work)
