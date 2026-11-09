"""No-loaded-shell synthetic tests of full native MVI dynamic end work.

Seven theories, electric ON/OFF, q/qx/qs inertia, high-order viscosity,
outward normal signs, missing electric mass, and a corrupted physical
nodal force. All synthetic material coefficients are prescribed;
no physical cylinder case or time-step is launched.
"""
import numpy as np
import pytest

from independent_mvi.coupled_dynamic_green import (
    audit_mvi_dynamic_discrete_green,_nodal_from_coefficients,
    _work_from_coefficients,
)
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_viscosity import NativeCoupledMVIViscosity
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,NativeCoupledMVISurface,
)
from flexo_gradient_g3d import G3DMaterial


def _case(theory,electric):
    surf=NativeCoupledMVISurface(
        theory,G3DMaterial(100.,.22,.08,
            .01 if electric else 0.,
            .02 if electric else 0.,
            1. if electric else 0.),
        grid=CoupledMVISurfaceGrid(11,7,2.,1.3),
        thickness_m=.2,n_gauss=2,
        potential_order=1 if electric else None)
    x=np.zeros((surf.n_fields,1,surf.grid.nx))
    x[:,:,1:-1]=.01
    y=np.ones((surf.n_fields,1,surf.grid.ns))
    factors=CoupledMVIFactors(x,y)
    inertia=NativeCoupledMVIInertia(surf,density_kg_m3=1000.)
    kv=NativeCoupledMVIViscosity(surf,eta_s=.1)
    return surf,factors,inertia,kv


def _manufacture(monkeypatch,surf,factors,inertia,kv):
    shape=(surf.n_fields,len(surf.multi),surf.grid.nx,surf.grid.ns)
    c=np.zeros(shape)
    c[0,1]=.04
    if surf.electrical_on:
        c[surf.n_mechanical:,1]=.002
    conservative=_work_from_coefficients(surf,factors,c)
    monkeypatch.setattr(surf,"weak_work",lambda *a,**kw:conservative)
    rng=np.random.default_rng(107)
    ci=rng.normal(0.,.06,size=(surf.n_mechanical,
                               len(surf.mechanical_multi),
                               surf.grid.nx,surf.grid.ns))
    ck=rng.normal(0.,.02,size=ci.shape)
    monkeypatch.setattr(inertia,"local_weak_coefficients",
                        lambda *a,**kw:ci.copy())
    monkeypatch.setattr(kv,"local_weak_coefficients",
                        lambda *a,**kw:ck.copy())
    def nodal(local):
        full=np.zeros(shape)
        full[:surf.n_mechanical,:len(surf.mechanical_multi)]=local
        return _nodal_from_coefficients(surf,full)[:surf.n_mechanical]
    monkeypatch.setattr(inertia,"apply",lambda *a,**kw:nodal(ci))
    monkeypatch.setattr(kv,"apply",lambda *a,**kw:nodal(ck))
    return ci,ck


@pytest.mark.parametrize("theory",(
    "KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
@pytest.mark.parametrize("electric",(False,True))
def test_dynamic_green_exactly_splits_all_mechanical_electric_terms(
        monkeypatch,theory,electric):
    surf,factors,inertia,kv=_case(theory,electric)
    _manufacture(monkeypatch,surf,factors,inertia,kv)
    a=np.zeros((surf.n_mechanical,surf.grid.nx,surf.grid.ns))
    result=audit_mvi_dynamic_discrete_green(
        surf,factors,a,a,inertia=inertia,viscosity=kv)
    assert result.status=="MVI_DYNAMIC_EXACT_DISCRETE_GREEN_DIAGNOSTIC"
    assert result.mechanical_nodal_work_error<1e-11
    assert result.electric_nodal_work_error==0.
    for term in (result.conservative,result.inertia,
                 result.kelvin_voigt,result.total):
        np.testing.assert_allclose(
            term.volume_gradient+term.boundary_gradient,
            term.full_gradient,rtol=1e-11,atol=1e-11)
        assert term.identity_relative_error<1e-11
    np.testing.assert_allclose(
        result.total.full_gradient,
        result.conservative.full_gradient+
        result.inertia.full_gradient+
        result.kelvin_voigt.full_gradient,
        rtol=1e-11,atol=1e-11)
    if electric:
        np.testing.assert_array_equal(
            result.inertia.full_gradient[surf.n_mechanical:],0.)
        np.testing.assert_array_equal(
            result.kelvin_voigt.full_gradient[surf.n_mechanical:],0.)


def test_dynamic_green_rejects_mismatched_viscous_nodal_assembly(monkeypatch):
    surf,factors,inertia,kv=_case("REDDY",True)
    _manufacture(monkeypatch,surf,factors,inertia,kv)
    original=kv.apply
    monkeypatch.setattr(kv,"apply",lambda *a,**kw:original(*a,**kw)+1.)
    zero=np.zeros((surf.n_mechanical,surf.grid.nx,surf.grid.ns))
    with pytest.raises(ValueError,match="MASS_OR_KV_FORCE_MISMATCH"):
        audit_mvi_dynamic_discrete_green(
            surf,factors,zero,zero,inertia=inertia,viscosity=kv)


def test_dynamic_green_never_accepts_mismatched_surface_objects():
    surf,factors,inertia,kv=_case("KL",False)
    other,_,_,_= _case("KL",False)
    other_kv=NativeCoupledMVIViscosity(other,eta_s=0.)
    zero=np.zeros((surf.n_mechanical,surf.grid.nx,surf.grid.ns))
    with pytest.raises(TypeError,match="MATCHING_PHYSICAL_OPERATORS"):
        audit_mvi_dynamic_discrete_green(
            surf,factors,zero,zero,inertia=inertia,viscosity=other_kv)


def test_dynamic_green_rejects_invalid_rate_field_before_material():
    surf,factors,inertia,kv=_case("FSDT",False)
    zero=np.zeros((surf.n_mechanical,surf.grid.nx,surf.grid.ns))
    wrong=np.zeros((surf.n_mechanical+1,surf.grid.nx,surf.grid.ns))
    with pytest.raises(ValueError,match="FINITE_MECHANICAL_RATES"):
        audit_mvi_dynamic_discrete_green(
            surf,factors,wrong,zero,inertia=inertia,viscosity=kv)
