"""No-solve checks for independent MVI Kelvin-Voigt and inertial work.

All material evaluations are synthetic/monkeypatched. No loaded shell
case, time history or numerical physical benchmark is executed.
"""
from types import SimpleNamespace
import numpy as np
import pytest

import independent_mvi.coupled_viscosity as viscosity
from independent_mvi.coupled_viscosity import NativeCoupledMVIViscosity
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,NativeCoupledMVISurface,
)
from flexo_gradient_g3d import G3DMaterial


@pytest.fixture(params=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
def surface(request):
    return NativeCoupledMVISurface(
        request.param,G3DMaterial(90.,.24,.06,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(9,5,1.2,2.),
        thickness_m=.3,n_gauss=2)


def test_zero_eta_never_evaluates_local_physics(surface,monkeypatch):
    def never(*a,**kw):
        raise AssertionError("zero Kelvin-Voigt must bypass material evaluation")
    monkeypatch.setattr(viscosity,"kelvin_voigt_thickness_weak",never)
    q=np.zeros((surface.n_mechanical,9,5))
    result=NativeCoupledMVIViscosity(surface,eta_s=0.).apply(q,q)
    np.testing.assert_array_equal(result,np.zeros_like(q))


def test_constant_local_viscous_force_has_correct_surface_weights(surface,monkeypatch):
    n=surface.n_mechanical
    def local(*args,**kwargs):
        c=np.zeros((n,len(surface.mechanical_multi)))
        c[:,0]=2.
        return SimpleNamespace(mechanical=c)
    monkeypatch.setattr(viscosity,"kelvin_voigt_thickness_weak",local)
    q=np.zeros((n,9,5))
    v=np.zeros_like(q)
    out=NativeCoupledMVIViscosity(surface,eta_s=.04).apply(q,v)
    expected=np.broadcast_to(
        2.*surface.grid.weights[None,:,:],out.shape)
    np.testing.assert_allclose(
        out,expected,rtol=2.e-13,atol=2.e-13)


def test_own_mvi_high_derivative_weighted_adjoint(surface,monkeypatch):
    n=surface.n_mechanical
    local_index=len(surface.mechanical_multi)-1
    shape=(n,9,5)
    x=np.arange(9)/8.
    s=2*np.pi*np.arange(5)/5.
    amplitude=np.sin(x[:,None])*np.cos(s[None,:])
    def local(*args,**kwargs):
        c=np.zeros((n,len(surface.mechanical_multi)))
        c[:,local_index]=amplitude[local.i,local.j]
        return SimpleNamespace(mechanical=c)
    # Assign node number to the synthetic callback without material physics.
    local.i=0;local.j=0
    def local_with_nodes(*args,**kwargs):
        local.j=local.calls%5
        local.i=local.calls//5
        local.calls+=1
        return local()
    local.calls=0
    monkeypatch.setattr(viscosity,"kelvin_voigt_thickness_weak",local_with_nodes)
    out=NativeCoupledMVIViscosity(surface,eta_s=.02).apply(
        np.zeros(shape),np.zeros(shape))
    ax,ang=surface.mechanical_multi[local_index]
    source=np.broadcast_to(amplitude,shape)
    expected=np.einsum(
        "ip,cij,jq->cpq",surface._dx_power[ax],
        source*surface.grid.weights[None,:,:],
        surface._ds_power[ang],optimize=True)
    np.testing.assert_allclose(out,expected,rtol=1.e-12,atol=1.e-12)


def test_constant_cuf_mass_density_is_positive(surface):
    if not surface.theory.startswith("CUF_TE"):
        pytest.skip("exact TE mass-moment test applies to CUF only")
    inertia=NativeCoupledMVIInertia(surface,density_kg_m3=3.)
    m=inertia._mass_te
    np.testing.assert_allclose(m,m.T,rtol=0.,atol=1.e-13)
    assert np.linalg.eigvalsh(m).min()>0.


def test_kelvin_voigt_bad_parameter_is_rejected(surface):
    for eta in (-1.,float("nan"),float("inf"),True):
        with pytest.raises(ValueError,match="MVI_KV_INVALID_TIME"):
            NativeCoupledMVIViscosity(surface,eta_s=eta)
