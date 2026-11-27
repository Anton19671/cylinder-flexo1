"""Regression: Gauss endpoint rank/trace checks at physical roundoff scales."""
import numpy as np
import pytest
from scipy.linalg import norm

from independent_mvi.linear_models import (
    MODELS, IndependentVariationalModel, VariationalModelCase,
)


def make_model(theory,rank=1,n=13):
    return IndependentVariationalModel(VariationalModelCase(
        theory=theory,nx=n,ns=n,length_m=1.2,radius_m=2.,
        thickness_m=.08,young_pa=1e9,poisson=.3,density_kg_m3=2500.,
        pressure_pa=1000.,pressure_face="MIDDLE",frequency_hz=25.,
        x_interval_m=(.2,.95),theta_interval_rad=(.55,2.55),
        x_edge_m=.25,theta_edge_rad=.5),rank)


def rank_two(model,field,k,amplitude):
    B=model.basis[k]
    field[k,:,0]=amplitude*B[:,0]
    field[k,:,1]=amplitude*B[:,1]


@pytest.mark.parametrize("theory",MODELS)
def test_roundoff_in_dormant_same_unit_component(theory):
    model=make_model(theory)
    q=np.zeros_like(model.force)
    q[0,:,0]=model.basis[0][:,0]*1.e-8
    rank_two(model,q,1,1.e-24)
    factors=model._endpoint_factors(q)
    assert factors[0].shape==(model.n,1,13)
    assert factors[1].shape==(model.n,1,13)
    assert norm((model.field(*factors)-q).ravel())<2.e-24


@pytest.mark.parametrize("theory",MODELS)
@pytest.mark.parametrize("amplitude",[1.,1.e-12,1.e-180])
def test_genuine_rank_two_is_rejected_at_every_field_scale(theory,amplitude):
    model=make_model(theory)
    q=np.zeros_like(model.force)
    rank_two(model,q,0,amplitude)
    with pytest.raises(RuntimeError,match="ENDPOINT_RANK_FAILURE"):
        model._endpoint_factors(q)


@pytest.mark.parametrize("theory",["FSDT","REDDY"])
@pytest.mark.parametrize("dominant,small",[(0,3),(3,0)])
def test_translation_and_rotation_scales_are_not_mixed(theory,dominant,small):
    model=make_model(theory)
    q=np.zeros_like(model.force)
    q[dominant,:,0]=model.basis[dominant][:,0]
    rank_two(model,q,small,1.e-16)
    with pytest.raises(RuntimeError,match="ENDPOINT_RANK_FAILURE"):
        model._endpoint_factors(q)


@pytest.mark.parametrize("theory",MODELS)
def test_nonzero_clamped_trace_is_rejected(theory):
    model=make_model(theory)
    q=np.zeros_like(model.force)
    q[0,:,0]=model.basis[0][:,0]*1.e-8
    q[0,0,0]=1.e-12
    with pytest.raises(RuntimeError,match="ENDPOINT_CONSTRAINT_FAILURE"):
        model._endpoint_factors(q)


@pytest.mark.parametrize("value",[float('nan'),float('inf')])
def test_nonfinite_endpoint_is_rejected(value):
    model=make_model("FSDT")
    q=np.zeros_like(model.force)
    q[0,0,0]=value
    with pytest.raises(RuntimeError,match="ENDPOINT_INVALID_FIELD"):
        model._endpoint_factors(q)


@pytest.mark.parametrize("rank",range(1,9))
def test_physical_uniform_fsdt_gauss_two_steps_all_ranks(rank):
    model=make_model("FSDT",rank=rank)
    q=np.zeros_like(model.force);v=q.copy();a=q.copy();factors=None
    dt=.0002
    for i in range(2):
        _,(q,v,a),factors,_=model.step_gauss_irk4(
            q,v,a,i*dt,dt,previous=factors)
        assert all(np.isfinite(z).all() for z in (q,v,a))
        assert norm((model.field(*factors)-q).ravel()) < 2.e-12*norm(q.ravel())
    np.testing.assert_allclose(q[2,6,0],-8.36896397410662e-9,rtol=2.e-6)
