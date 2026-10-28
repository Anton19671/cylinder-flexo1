"""Audit V-3: optional Fourier derivative and operator-only certification."""
import copy
import numpy as np
import pytest
from cylinder_sbp_derivatives import periodic_sbp_odd_fourier_gradient, require_green_identity
from cylinder_functional_assembly import surface_sbp21, surface_sbp21_fourier, CylinderFunctional
from cylinder_spatial_spectrum_audit import spatial_operator_diagnostics
from flexo_gradient_g3d import G3DMaterial
from cylinder_dynamic_case_passport import dynamic_physics_sha256

@pytest.mark.parametrize("n",[5,9,15,31])
def test_fourier_all_harmonics(n):
    L=2*np.pi*2.5
    d,w=periodic_sbp_odd_fourier_gradient(n,L)
    assert np.max(np.abs(d+d.T))<1.e-12
    assert np.max(np.abs(d@np.ones(n)))<1.e-12
    for k in range(1,(n-1)//2+1):
        j=np.arange(n)
        np.testing.assert_allclose(d@np.sin(2*np.pi*k*j/n),
            (2*np.pi*k/L)*np.cos(2*np.pi*k*j/n),rtol=0,atol=1.e-12)

@pytest.mark.parametrize("n",[5,9,15])
def test_high_frequency_operator_weakness_and_green_pair(n):
    fresh=surface_sbp21_fourier(7,n,1.,2.)
    prior=surface_sbp21(7,n,1.,2.)
    assert require_green_identity(fresh.Dx,fresh.Ds,fresh.wx,fresh.ws)["relative_max"]<3.e-12
    a=spatial_operator_diagnostics(fresh,"SBP21_FOURIER_ODD")
    b=spatial_operator_diagnostics(prior,"SBP21_PERIODIC_ODD")
    assert a["circumferential_highest_mode_bending_indicator"]==pytest.approx(1.,rel=1.e-12)
    assert b["circumferential_highest_mode_bending_indicator"]<.01
    assert a["axial_alternating_mode_bending_indicator"]==pytest.approx(b["axial_alternating_mode_bending_indicator"])
    assert a["axial_high_frequency_warning"]
    assert not a["mesh_convergence_verified"]

def test_even_grid_rejected():
    with pytest.raises(ValueError,match="odd ns"):
        surface_sbp21_fourier(5,6,1.,2.)

@pytest.mark.parametrize("mode",["G3D","Gz"])
def test_coupled_enthalpy_variational_derivative(mode):
    ops=surface_sbp21_fourier(4,5,1.2,2.)
    m=G3DMaterial(100.,.25,.08,.0001,-.0002,1.5,mode)
    f=CylinderFunctional("CUF_TE1",m,radius_m=2.,thickness_m=.2,
                         surface_operators=ops,n_gauss=3)
    rng=np.random.default_rng(2301)
    q=rng.normal(0,.005,(6,4,5));p=rng.normal(0,.006,(1,4,5))
    dq=rng.normal(size=q.shape);dp=rng.normal(size=p.shape)
    current=f.evaluate(q,p)
    derivative=np.sum(current.mechanical_residual*dq)+np.sum(current.electrical_residual*dp)
    eps=1.e-6
    energy=(f.evaluate(q+eps*dq,p+eps*dp).total_enthalpy_j -
            f.evaluate(q-eps*dq,p-eps*dp).total_enthalpy_j)/(2*eps)
    assert energy==pytest.approx(derivative,rel=2.e-6,abs=2.e-7)

def test_dynamic_case_fingerprint_tracks_operator():
    a={"method":"MVI","calculation_kind":"DYNAMIC","mvi_grid":{"nx":5,"ns":9,
       "operator_scheme":"SBP21_PERIODIC_ODD"},"dynamics":{"n_steps":1,"time_step_s":.01,
       "density_kg_m3":1300.}}
    b=copy.deepcopy(a);b["mvi_grid"]["operator_scheme"]="SBP21_FOURIER_ODD"
    assert dynamic_physics_sha256(a)!=dynamic_physics_sha256(b)
