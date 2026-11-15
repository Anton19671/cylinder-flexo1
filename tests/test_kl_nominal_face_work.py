"""Analytic work and method-isolation checks for physical KL face traction."""
import numpy as np
import pytest
from types import SimpleNamespace
from cylinder_paths import activate
activate()
from cylinder_kl_face_load import compile_kl_face_load, kl_radial_displacement


def load(shape=(2,3),face="OUTER",traction=-7.,model="GREEN_LAGRANGE"):
    return compile_kl_face_load(dict(face=face,traction_pa=traction),
        theory="KL",shape=shape,radius_m=2.,thickness_m=.2,geometry_model=model)


def test_uniform_axial_tilt_has_exact_offset_face_work():
    q=np.zeros((3,2,3))
    qx=np.zeros_like(q)
    qs=np.zeros_like(q)
    slope=.4
    qx[2]=slope
    pressure=load()
    density,coeff=pressure.local_work(q,qx,qs)
    z=.1
    H=1.+z/2.
    expected=z*(1./np.sqrt(1.+slope*slope)-1.)
    np.testing.assert_allclose(density,-7.*H*expected,rtol=1.e-13)
    np.testing.assert_allclose(coeff[2,1],
        -7.*H*z*(-slope/(1.+slope*slope)**1.5),rtol=1.e-13)
    np.testing.assert_allclose(kl_radial_displacement(q,qx,qs,
        radius_m=2.,z_m=z,geometry_model="GREEN_LAGRANGE"),expected,rtol=1.e-13)


@pytest.mark.parametrize("face",("INNER","MIDDLE","OUTER"))
def test_face_coefficients_are_derivative_of_the_same_work(face):
    rng=np.random.default_rng(419)
    jets=tuple(rng.normal(0.,.04,(3,2,3)) for _ in range(3))
    variations=tuple(rng.normal(size=(3,2,3)) for _ in range(3))
    pressure=load(face=face)
    _,coeff=pressure.local_work(*jets)
    expected=sum(np.sum(coeff[:,k]*variations[k],axis=0) for k in range(3))
    step=1.e-6
    plus,_=pressure.local_work(*(q+step*d for q,d in zip(jets,variations)))
    minus,_=pressure.local_work(*(q-step*d for q,d in zip(jets,variations)))
    np.testing.assert_allclose((plus-minus)/(2*step),expected,rtol=2.e-7,atol=1.e-8)


@pytest.mark.parametrize("model",("LINEAR","VON_KARMAN"))
def test_small_rotation_models_have_only_radial_value_load(model):
    jets=tuple(np.full((3,2,3),.1) for _ in range(3))
    density,coeff=load(model=model).local_work(*jets)
    expected=np.zeros_like(coeff)
    expected[2,0]=-7.*1.05
    np.testing.assert_allclose(coeff,expected)
    np.testing.assert_allclose(density,-7.*1.05*jets[0][2])


def test_own_mvi_assembly_differentiates_the_physical_face_potential():
    from independent_mvi.coupled_surface import (
        NativeCoupledMVISurface,CoupledMVISurfaceGrid,CoupledMVIFactors)
    from flexo_gradient_g3d import G3DMaterial
    surface=NativeCoupledMVISurface("KL",G3DMaterial(90.,.2,.01,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(9,5,1.,2.),thickness_m=.2,n_gauss=2)
    surface.kl_face_load=load(shape=(9,5))
    surface._local=lambda jets:SimpleNamespace(
        energy_j_per_m2=0.,mechanical=np.zeros((3,10)))
    x=np.zeros((3,1,9))
    x[2,0]=.02*np.sin(np.linspace(0.,np.pi,9))
    y=np.ones((3,1,5))
    factors=CoupledMVIFactors(x,y)
    work=surface.weak_work(factors)
    delta=np.zeros_like(x)
    delta[2,0,4]=1.
    step=1.e-6
    plus=surface.weak_work(CoupledMVIFactors(x+step*delta,y)).energy_j
    minus=surface.weak_work(CoupledMVIFactors(x-step*delta,y)).energy_j
    assert work.gradient_x[2,0,4]!=0.
    assert (plus-minus)/(2*step)==pytest.approx(
        work.gradient_x[2,0,4],rel=2.e-6,abs=1.e-8)


def test_own_fdm_coefficients_include_physical_face_jet_work():
    from cylinder_direct_strong_model import DirectStrongShell
    from cylinder_direct_strong_fdm import DirectStrongGrid
    from flexo_gradient_g3d import G3DMaterial
    model=DirectStrongShell("KL",G3DMaterial(90.,.2,.01,0.,0.,0.),
        grid=DirectStrongGrid(9,5,1.,2.),thickness_m=.2,n_gauss=2)
    model.kl_face_load=load(shape=(9,5))
    model._local=lambda jets:SimpleNamespace(mechanical=np.zeros((3,10)))
    q=np.zeros((3,9,5))
    q[2]=.02*np.sin(np.linspace(0.,np.pi,9))[:,None]
    coefficients,_=model.coefficients(q)
    own_jets=model.field_jets.prepare(q,model.derivative_orders)
    _,expected=model.kl_face_load.local_work(*own_jets[:3])
    np.testing.assert_allclose(coefficients[:3],-np.swapaxes(expected,0,1))


@pytest.mark.parametrize("method",("FDM","MVI"))
@pytest.mark.parametrize("kind",("STATIC","DYNAMIC"))
def test_campaign_compiles_physical_face_in_both_native_routes(tmp_path,method,kind):
    from ci import native_gradient_fdm_mvi_campaign as campaign
    rank=None if method=="FDM" else 1
    file,forcing,_=campaign.write_input(tmp_path/"input",
        ("KL","G3D",kind,method,rank))
    np.testing.assert_array_equal(forcing,0.)
    solver=campaign.fdm if method=="FDM" else campaign.mvi
    prepared=solver.load_case(file)
    owner=prepared["model" if method=="FDM" else "surface"]
    assert owner.kl_face_load is not None
    np.testing.assert_allclose(owner.kl_face_load.traction_pa,
        np.broadcast_to(-campaign.pressure_profile(campaign.GRID[1]),campaign.GRID))


def test_campaign_recovery_uses_rotating_director():
    from ci import native_gradient_fdm_mvi_campaign as campaign
    q=np.zeros((3,*campaign.GRID))
    qx=np.zeros_like(q)
    qs=np.zeros_like(q)
    qx[2]=.4
    middle,outer,inner=campaign.radial_recovery(
        q,"KL",mechanical_jets=(q,qx,qs))
    np.testing.assert_array_equal(middle,0.)
    assert np.max(np.abs(outer))>0.
    np.testing.assert_allclose(inner,-outer)
