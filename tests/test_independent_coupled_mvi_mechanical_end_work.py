"""No physical shell solve: exact native MVI generalized end virtual work.

Checks directional derivatives, left/right normal orientation, model
trace capacities, fail-closed CASE options, and conservative/dynamic
Green separation without using any MKR stencil or SVD.
"""
import json
import numpy as np
import pytest
from flexo_gradient_g3d import G3DMaterial
from independent_mvi.coupled_surface import (
    NativeCoupledMVISurface,CoupledMVISurfaceGrid,CoupledMVIFactors,
    CoupledMVIWork,
)
from independent_mvi.coupled_mechanical_end_work import compile_mechanical_end_work
from independent_mvi.coupled_discrete_green import coupled_mvi_discrete_green_identity
from independent_mvi.coupled_dynamic_green import _work_from_coefficients
from independent_mvi import run_coupled_research as native
from test_independent_coupled_mvi_case_dispatch import make_case

THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")


def setup(theory):
    surf=NativeCoupledMVISurface(
        theory,G3DMaterial(100.,.23,.03,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(11,7,1.5,2.),
        thickness_m=.2,n_gauss=2)
    rng=np.random.default_rng(203)
    x=np.zeros((surf.n_fields,2,surf.grid.nx))
    x[:,:,1:-1]=rng.normal(size=x[:,:,1:-1].shape)*.01
    y=rng.normal(size=(surf.n_fields,2,surf.grid.ns))
    return surf,CoupledMVIFactors(x,y)


@pytest.mark.parametrize("theory",THEORIES)
def test_normal_slope_end_potential_has_exact_x_y_adjoint(theory):
    surf,f=setup(theory)
    loads=compile_mechanical_end_work(surf,{
        "left":{"normal_double_traction":2.},
        "right":{"normal_double_traction":3.}})
    rng=np.random.default_rng(42)
    dx=rng.normal(size=f.x.shape)
    dy=rng.normal(size=f.y.shape)
    nodal=loads.nodal_gradient()
    assert nodal.shape==surf.shape
    np.testing.assert_allclose(
        loads.left[:,1,:],-2.)
    np.testing.assert_allclose(
        loads.right[:,1,:],3.)
    gx=np.einsum("cij,crj->cri",nodal,f.y)
    gy=np.einsum("cij,cri->crj",nodal,f.x)
    h=1e-7
    num_x=(loads.potential(CoupledMVIFactors(f.x+h*dx,f.y))-
           loads.potential(CoupledMVIFactors(f.x-h*dx,f.y)))/(2*h)
    num_y=(loads.potential(CoupledMVIFactors(f.x,f.y+h*dy))-
           loads.potential(CoupledMVIFactors(f.x,f.y-h*dy)))/(2*h)
    np.testing.assert_allclose(num_x,np.sum(gx*dx),rtol=2e-8,atol=1e-8)
    np.testing.assert_allclose(num_y,np.sum(gy*dy),rtol=2e-8,atol=1e-8)


@pytest.mark.parametrize("theory",("KL","FSDT","REDDY"))
def test_triple_normal_end_load_is_even_in_left_normal(theory):
    surf,_=setup(theory)
    load=compile_mechanical_end_work(surf,{
        "left":{"normal_triple_traction":4.},
        "right":{"normal_triple_traction":5.}})
    np.testing.assert_allclose(load.left[:,2,:],4.)
    np.testing.assert_allclose(load.right[:,2,:],5.)


@pytest.mark.parametrize("theory",("CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
def test_cuf_rejects_third_order_end_trace(theory):
    surf,_=setup(theory)
    with pytest.raises(ValueError,match="TRACE_ORDER_UNAVAILABLE"):
        compile_mechanical_end_work(surf,{
            "left":{"normal_triple_traction":1.},"right":{}})


@pytest.mark.parametrize("bad",[
    {"left":{"normal_double_traction":1.,"double_traction":1.},"right":{}},
    {"left":{"traction":1.},"right":{}},
    {"left":{"normal_double_traction":float("nan")},"right":{}},
])
def test_invalid_end_data_rejected(bad):
    surf,_=setup("REDDY")
    with pytest.raises(ValueError,match="MVI_"):
        compile_mechanical_end_work(surf,bad)


def test_material_green_does_not_double_count_external_end_gradient():
    surf,f=setup("KL")
    surf.mechanical_end_work=compile_mechanical_end_work(surf,{
        "left":{"normal_double_traction":1.},"right":{}})
    coeff=np.zeros((surf.n_fields,len(surf.multi),surf.grid.nx,surf.grid.ns))
    physical=_work_from_coefficients(surf,f,coeff)
    extension=surf.mechanical_end_work.nodal_gradient()
    fake=CoupledMVIWork(
        physical.energy_j+surf.mechanical_end_work.potential(f),
        physical.mechanical,None,
        physical.gradient_x+np.einsum("cij,crj->cri",extension,f.y),
        physical.gradient_y+np.einsum("cij,cri->crj",extension,f.x),
        f.rank,nodal_gradient=physical.nodal_gradient+extension,
        coefficient_jets=coeff,external_end_gradient=extension)
    green=coupled_mvi_discrete_green_identity(surf,fake)
    np.testing.assert_allclose(green.full_gradient,0.,atol=1.e-12)
    np.testing.assert_allclose(
        fake.nodal_gradient,extension,rtol=0.,atol=1.e-12)


@pytest.mark.parametrize("theory",THEORIES)
def test_optional_end_work_case_preflight_has_no_output(tmp_path,theory):
    file,document=make_case(tmp_path,theory=theory)
    document["mechanical_end_work"]={
        "left":{"normal_double_traction":2.},"right":{}}
    file.write_text(json.dumps(document))
    result=native.load_case(file)
    assert result["mechanical_end_work"] is not None
    assert result["surface"].mechanical_end_work is result["mechanical_end_work"]
    assert not (tmp_path/"output").exists()
