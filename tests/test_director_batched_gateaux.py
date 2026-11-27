"""Batched Green-Lagrange director assembly vs the original scalar law."""
import itertools
import numpy as np
import pytest
from cylinder_paths import activate
activate()
import cylinder_director_g3d as d
from flexo_gradient_g3d import G3DMaterial


def scalar_work(F,Fx,Fs,Fz,library,sigma,hyper,R,z):
    result=np.empty(library.shape[:2])
    for k,i in np.ndindex(result.shape):
        de,der=d._strain_virtual(F,Fx,Fs,Fz,*library[k,i],"GREEN_LAGRANGE")
        dg=d.covariant_strain_gradients(de,*der,radius_m=R,z_m=z)
        result[k,i]=sigma@de+np.sum(hyper*dg)
    return result


def assert_roundoff_equal(a,b):
    scale=max(np.linalg.norm(a.ravel()),np.linalg.norm(b.ravel()),1.e-30)
    assert np.linalg.norm((a-b).ravel())<=3.e-14*scale
    np.testing.assert_allclose(a,b,rtol=3.e-13,atol=5.e-15*scale)


@pytest.mark.parametrize("theory",d.THEORIES)
@pytest.mark.parametrize("gradient",("Gz","G3D"))
@pytest.mark.parametrize("electric",(False,True))
@pytest.mark.parametrize("gauss",(3,7))
def test_integrated_batched_gateaux_matches_scalar(theory,gradient,electric,gauss,monkeypatch):
    rng=np.random.default_rng(42)
    kw=dict(theory=theory,radius_m=2.,thickness_m=.08,n_gauss=gauss,
            geometry_model="GREEN_LAGRANGE")
    m=G3DMaterial(1.e9,.3,.02,.001 if electric else 0.,
                  -.0005 if electric else 0.,1.e-8 if electric else 0.,gradient)
    for pe,scale in itertools.product((0,1,4),(1.e-10,1.e-4,2.e-2)):
        jets=d.DirectorSurfaceJets(*[rng.normal(0,scale,5) for _ in d.FIELDS],
            *[rng.normal(0,.01,pe+1) for _ in d.ELECTRICAL_FIELDS])
        actual=d.thickness_weak_coefficients(jets,m,**kw)
        with monkeypatch.context() as patch:
            patch.setattr(d,"_green_lagrange_virtual_work",scalar_work)
            expected=d.thickness_weak_coefficients(jets,m,**kw)
        assert actual.energy_j_per_m2==expected.energy_j_per_m2
        assert actual.gauss_points==expected.gauss_points==gauss
        np.testing.assert_array_equal(actual.electrical,expected.electrical)
        assert_roundoff_equal(actual.mechanical,expected.mechanical)


@pytest.mark.parametrize("theory",d.THEORIES)
@pytest.mark.parametrize("geometry",("LINEAR","VON_KARMAN"))
def test_other_strain_measures_do_not_enter_batched_gl(theory,geometry,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Only Green-Lagrange uses the batched GL law")
    monkeypatch.setattr(d,"_green_lagrange_virtual_work",forbidden)
    rng=np.random.default_rng(37)
    jets=d.DirectorSurfaceJets(*[rng.normal(0,.002,5) for _ in d.FIELDS],
                               *[rng.normal(0,.01,2) for _ in d.ELECTRICAL_FIELDS])
    m=G3DMaterial(1.e9,.3,.02,.001,-.0005,1.e-8,"G3D")
    result=d.thickness_weak_coefficients(jets,m,theory=theory,radius_m=2.,
        thickness_m=.08,n_gauss=3,geometry_model=geometry)
    assert np.isfinite(result.mechanical).all()
    assert np.isfinite(result.electrical).all()
