"""Independent coupled MVI SBP42/Fourier operator identity tests.

No physical shell calculation and no reference MKR/FDM assembler imported.
"""
import numpy as np
import pytest

from independent_mvi.coupled_operators import (
    coupled_mvi_sbp42, coupled_mvi_odd_fourier,
)


@pytest.mark.parametrize("n",(9,11,21))
def test_axial_sbp_first_derivative_exact_linear_and_discrete_green(n):
    L=2.7
    d,w=coupled_mvi_sbp42(n,L)
    x=np.linspace(0.,L,n)
    np.testing.assert_allclose(d@np.ones(n),0.,atol=4e-14)
    np.testing.assert_allclose(d@x,1.,atol=4e-14)
    boundary=np.zeros((n,n))
    boundary[0,0]=-1.
    boundary[-1,-1]=1.
    np.testing.assert_allclose(w[:,None]*d+d.T*w[None,:],
                               boundary,atol=1e-13)


@pytest.mark.parametrize("n",(5,7,9,13))
def test_odd_circumferential_fourier_skew_and_trigonometric_modes(n):
    R=1.7
    d,w=coupled_mvi_odd_fourier(n,R)
    s=np.arange(n)*2*np.pi*R/n
    np.testing.assert_allclose(d+d.T,0.,atol=5e-13)
    np.testing.assert_allclose(d@np.ones(n),0.,atol=5e-13)
    np.testing.assert_allclose(w,2*np.pi*R/n)
    for k in range(1,(n-1)//2+1):
        freq=k/R
        np.testing.assert_allclose(
            d@np.sin(freq*s),freq*np.cos(freq*s),atol=2e-12)
        np.testing.assert_allclose(
            d@np.cos(freq*s),-freq*np.sin(freq*s),atol=2e-12)


@pytest.mark.parametrize("nx,ns",((9,5),(11,7)))
def test_tensor_product_derivatives_commute(nx,ns):
    Dx,_=coupled_mvi_sbp42(nx,2.)
    Ds,_=coupled_mvi_odd_fourier(ns,1.)
    rng=np.random.default_rng(101)
    q=rng.normal(size=(nx,ns))
    np.testing.assert_allclose(Dx@(q@Ds.T),
                               (Dx@q)@Ds.T,atol=1e-13)


def test_reject_invalid_grid_and_geometry():
    with pytest.raises(ValueError):
        coupled_mvi_sbp42(8,2.)
    with pytest.raises(ValueError):
        coupled_mvi_sbp42(9,-1.)
    with pytest.raises(ValueError):
        coupled_mvi_odd_fourier(6,1.)
    with pytest.raises(ValueError):
        coupled_mvi_odd_fourier(5,0.)
