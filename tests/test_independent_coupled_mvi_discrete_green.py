"""No-solve exact discrete SBP/Fourier Green identity for all seven models.

Manufactured weak coefficients and random virtual variations; no nonlinear
physical material evaluation or cylinder simulation. Tests all distinct
axial/circumferential jet terms, outward normal parity and electric flux.
"""
import numpy as np
import pytest

from independent_mvi.coupled_discrete_green import (
    coupled_mvi_discrete_green_identity,
)
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,NativeCoupledMVISurface,CoupledMVIWork,
)
from flexo_gradient_g3d import G3DMaterial


def surface(theory,electric=False):
    return NativeCoupledMVISurface(
        theory,G3DMaterial(100.,.22,.08,
                .01 if electric else 0.,
                .02 if electric else 0.,
                1. if electric else 0.),
        grid=CoupledMVISurfaceGrid(11,7,2.,1.3),
        thickness_m=.2,n_gauss=2,
        potential_order=1 if electric else None)


def work_of_coefficients(surf,coefficient):
    """Independent discrete transpose assembly of virtual work."""
    g=np.zeros(surf.shape)
    for k,(ax,ang) in enumerate(surf.multi):
        g+=np.einsum(
            "ip,cij,jq->cpq",
            surf._dx_power[ax],
            coefficient[:,k]*surf.grid.weights[None,:,:],
            surf._ds_power[ang],optimize=True)
    mechanical=np.zeros((surf.n_mechanical,surf.grid.nx,surf.grid.ns))
    electric=(np.zeros((surf.n_electric,surf.grid.nx,surf.grid.ns))
              if surf.electrical_on else None)
    return CoupledMVIWork(0.,mechanical,electric,
              np.zeros((surf.n_fields,1,surf.grid.nx)),
              np.zeros((surf.n_fields,1,surf.grid.ns)),
              1,nodal_gradient=g,coefficient_jets=coefficient)


@pytest.mark.parametrize("theory",(
    "KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
@pytest.mark.parametrize("electric",(False,True))
def test_green_exact_weighted_nodal_work_and_boundary_virtual_power(
        theory,electric):
    surf=surface(theory,electric)
    rng=np.random.default_rng(904)
    a=rng.normal(size=(surf.n_fields,len(surf.multi),
                       surf.grid.nx,surf.grid.ns))
    if surf.theory.startswith("CUF_TE"):
        a[:surf.n_mechanical,len(surf.mechanical_multi):]=0.
    if electric:
        a[surf.n_mechanical:,3:]=0.
    work=work_of_coefficients(surf,a)
    identity=coupled_mvi_discrete_green_identity(surf,work)
    np.testing.assert_allclose(
        identity.volume_gradient+identity.boundary_gradient,
        work.nodal_gradient,rtol=1e-11,atol=1e-11)
    rng=np.random.default_rng(65)
    variation=rng.normal(size=surf.shape)
    lhs=np.sum(work.nodal_gradient*variation)
    rhs=np.sum(identity.volume_gradient*variation)
    for c in range(surf.n_fields):
        for k,traction in enumerate(identity.coordinate_left[c]):
            trace=(surf._dx_power[k]@variation[c])[0]
            rhs+=np.dot(traction*surf.grid.ws,trace)
        for k,traction in enumerate(identity.coordinate_right[c]):
            trace=(surf._dx_power[k]@variation[c])[-1]
            rhs+=np.dot(traction*surf.grid.ws,trace)
    np.testing.assert_allclose(rhs,lhs,rtol=1e-11,atol=1e-11)
    assert identity.identity_relative_error<1e-11
    assert not identity.full_gradient.flags.writeable


def test_outward_normal_double_traction_left_sign_is_even():
    surf=surface("KL")
    a=np.zeros((surf.n_fields,len(surf.multi),surf.grid.nx,surf.grid.ns))
    a[0,3]=2.3 # axial order 2
    work=work_of_coefficients(surf,a)
    identity=coupled_mvi_discrete_green_identity(surf,work)
    np.testing.assert_allclose(identity.coordinate_left[0][1],-2.3)
    np.testing.assert_allclose(identity.coordinate_right[0][1],2.3)
    np.testing.assert_allclose(identity.normal_left[0][1],2.3)
    np.testing.assert_allclose(identity.normal_right[0][1],2.3)


def test_outward_normal_triple_traction_sign_reverses_on_left():
    surf=surface("REDDY")
    a=np.zeros((surf.n_fields,len(surf.multi),surf.grid.nx,surf.grid.ns))
    a[0,6]=4.2 # axial order 3
    identity=coupled_mvi_discrete_green_identity(
        surf,work_of_coefficients(surf,a))
    np.testing.assert_allclose(identity.normal_left[0][2],-4.2)
    np.testing.assert_allclose(identity.normal_right[0][2],4.2)


def test_electric_flux_has_only_one_axial_trace_order():
    surf=surface("CUF_TE1",True)
    a=np.zeros((surf.n_fields,len(surf.multi),surf.grid.nx,surf.grid.ns))
    a[surf.n_mechanical:,1]=2.1
    out=coupled_mvi_discrete_green_identity(
        surf,work_of_coefficients(surf,a))
    for c in range(surf.n_mechanical,surf.n_fields):
        assert len(out.normal_left[c])==1
        assert len(out.normal_right[c])==1
        np.testing.assert_allclose(out.normal_left[c][0],-2.1)
        np.testing.assert_allclose(out.normal_right[c][0],2.1)


def test_discrete_green_rejects_tampered_nodal_variation():
    surf=surface("CUF_TE1")
    a=np.zeros((surf.n_fields,len(surf.multi),surf.grid.nx,surf.grid.ns))
    a[0,4]=.2
    good=work_of_coefficients(surf,a)
    wrong=good.nodal_gradient.copy()
    wrong[0,5,3]+=.01
    forged=CoupledMVIWork(
        0.,good.mechanical,None,good.gradient_x,good.gradient_y,
        1,nodal_gradient=wrong,coefficient_jets=a)
    with pytest.raises(ValueError,match="FIRST_VARIATION_IDENTITY_FAILED"):
        coupled_mvi_discrete_green_identity(surf,forged)


def test_native_case_green_audit_is_conservative_and_not_physical_certification(
        monkeypatch):
    from independent_mvi import run_coupled_research as runner
    surf=surface("CUF_TE2",True)
    a=np.zeros((surf.n_fields,len(surf.multi),surf.grid.nx,surf.grid.ns))
    a[0,4]=.08
    a[surf.n_mechanical:,1]=.12
    work=work_of_coefficients(surf,a)
    monkeypatch.setattr(surf,"weak_work",lambda *arg,**kw:work)
    from independent_mvi.coupled_surface import CoupledMVIFactors
    x=np.zeros((surf.n_fields,1,surf.grid.nx))
    y=np.ones((surf.n_fields,1,surf.grid.ns))
    factors=CoupledMVIFactors(x,y)
    output=runner._green_audit(surf,factors,None,None)
    assert output["status"]=="MVI_EXACT_DISCRETE_SBP_FOURIER_GREEN_IDENTITY"
    assert output["conservative_only"] is True
    assert output["physical_boundary_closure_certified"] is False
    assert output["identity_relative_error"]<1.e-11
    assert output["field_trace_orders"][-1]==1
    assert output["field_trace_orders"][0]==2


def test_native_case_green_audit_rejects_wrong_nodal_material_work(
        monkeypatch):
    from independent_mvi import run_coupled_research as runner
    from independent_mvi.coupled_surface import CoupledMVIFactors
    surf=surface("KL")
    a=np.zeros((surf.n_fields,len(surf.multi),surf.grid.nx,surf.grid.ns))
    a[0,6]=.4
    base=work_of_coefficients(surf,a)
    wrong=base.nodal_gradient.copy()
    wrong[0,4,3]+=.02
    forged=CoupledMVIWork(
        0.,base.mechanical,None,base.gradient_x,base.gradient_y,
        1,nodal_gradient=wrong,coefficient_jets=a)
    monkeypatch.setattr(surf,"weak_work",lambda *arg,**kw:forged)
    x=np.zeros((surf.n_fields,1,surf.grid.nx))
    y=np.ones((surf.n_fields,1,surf.grid.ns))
    with pytest.raises(ValueError,match="FIRST_VARIATION_IDENTITY_FAILED"):
        runner._green_audit(surf,CoupledMVIFactors(x,y),None,None)
