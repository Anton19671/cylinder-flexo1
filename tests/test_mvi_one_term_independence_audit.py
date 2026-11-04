"""Strict algebraic regression for classical rank-fixed X/Y MVI.

No physical shell simulations. This test rejects hidden FDM dispatch, wrong
rank counting, broken partner QR, and a rank-1 representation of rank-2 data.
A frozen-partner Galerkin variation is *the discrete 1D variational step*:
T_X^T R = 0 or T_Y^T R = 0. It does not increase separation rank.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import cylinder_bathe_mvi as bathe
from cylinder_boundary_spatial import separated_profile_directions
from cylinder_variational_mvi import (
    StaticMVISettings, _initial_profiles, _normalize_partner,
    reconstruct_profiles,
)


def test_rank_one_has_one_separated_term_but_one_unknown_per_coordinate_node():
    rng=np.random.default_rng(4821)
    nc,nx,ns=15,9,9
    x=rng.standard_normal((nc,1,nx))
    y=rng.standard_normal((nc,1,ns))
    for axis in ("Y","X"):
        xx,yy=_normalize_partner(x,y,axis)
        np.testing.assert_allclose(
            reconstruct_profiles(xx,yy),reconstruct_profiles(x,y),
            rtol=3.e-13,atol=2.e-13)
        t=separated_profile_directions(xx,yy,axis)
        # 15 independent CUF components x 1 rank x 9 nodal values.
        # A 1215 x 135 matrix is NOT a rank-135 MVI approximation.
        assert t.shape==(15*9*9,15*1*9)
        np.testing.assert_allclose(t.T@t,np.eye(135),rtol=0.,atol=2.e-14)
        delta_coords=rng.normal(size=135)
        np.testing.assert_allclose(
            t.T@(t@delta_coords),delta_coords,rtol=2.e-14,atol=2.e-14)
        # Frozen-partner test variations are exactly contractions of
        # the 2D first variation with the opposite 1D profile.
        gradient=rng.normal(size=(nc,nx,ns))
        projected=(t.T@gradient.ravel()).reshape(nc,1,9)
        if axis=="X":
            expected=np.einsum("cs,cxs->cx",yy[:,0],gradient)[:,None,:]
        else:
            expected=np.einsum("cx,cxs->cs",xx[:,0],gradient)[:,None,:]
        np.testing.assert_allclose(projected,expected,rtol=2.e-14,atol=2.e-14)


def test_qr_preserves_rank_two_product_including_inactive_columns():
    rng=np.random.default_rng(937)
    x=rng.standard_normal((15,2,9))
    y=rng.standard_normal((15,2,9))
    x[0,1]=0.
    y[1,1]=y[1,0]
    original=reconstruct_profiles(x,y)
    for axis in ("Y","X"):
        xx,yy=_normalize_partner(x,y,axis)
        np.testing.assert_allclose(reconstruct_profiles(xx,yy),original,
                                   atol=5.e-14,rtol=5.e-14)
        assert xx.shape==x.shape and yy.shape==y.shape
        for component in range(15):
            assert np.linalg.matrix_rank(reconstruct_profiles(
                xx[component:component+1],
                yy[component:component+1])[0],tol=1.e-10)<=2


def test_exact_rank_two_cannot_be_hidden_in_one_mvi_term():
    nx=ns=9
    x_coord=np.linspace(0.,1.,nx)
    theta=np.linspace(0.,2*np.pi,ns,endpoint=False)
    # Axially clamped sine profiles, different wave numbers in x and theta.
    x=np.stack((np.sin(np.pi*x_coord),np.sin(2*np.pi*x_coord)))
    y=np.stack((np.cos(theta),np.sin(theta)))
    x=np.linalg.qr(x.T,mode="reduced")[0].T
    y=np.linalg.qr(y.T,mode="reduced")[0].T
    w=.004*np.outer(x[0],y[0])+.0012*np.outer(x[1],y[1])
    singular=np.linalg.svd(w,compute_uv=False)
    assert singular[1]/singular[0]==pytest.approx(.3,rel=1.e-12)
    best_rank1_error=singular[1]
    assert best_rank1_error/np.linalg.norm(w)>.28
    field=np.zeros((15,nx,ns))
    field[10]=w
    scales=np.linalg.norm(field,axis=(1,2))
    with pytest.raises(ValueError,match="initial field cannot be represented"):
        _initial_profiles(field,rank=1,roundoff_scale=scales)
    xx,yy=_initial_profiles(field,rank=2,roundoff_scale=scales)
    np.testing.assert_allclose(reconstruct_profiles(xx,yy),field,
                               rtol=0.,atol=1.e-13)


def test_bathe_mvi_uses_two_MVI_substeps_and_no_full_nodal_fdm():
    q=np.zeros((3,5,5))
    v=q.copy()
    a=q.copy()
    prior=[]
    def forbidden_fdm(*unused_args,**unused_kw):
        raise AssertionError("HIDDEN_FDM_CALL_INSIDE_MVI")
    def fake_mvi(problem,mechanical,electrical,**kwargs):
        assert kwargs["mvi_settings"].rank==1
        prior.append(kwargs["previous_mvi_result"])
        return SimpleNamespace(
            converged=True,status="CONVERGED_PROJECTED",
            mechanical=np.array(mechanical,copy=True),electrical=None,
            velocity=np.zeros_like(mechanical),
            acceleration=np.zeros_like(mechanical))
    with (patch.object(bathe,"solve_dynamic_mvi",side_effect=fake_mvi) as call_mvi,
          patch.object(bathe,"solve_dynamic_fdm",side_effect=forbidden_fdm),
          patch.object(bathe,"audit_bathe_kinematics",return_value=None)):
        result=bathe.solve_bathe_variational_step(
            SimpleNamespace(kind="DYNAMIC"),q,None,v,a,
            time_s=0.,h=.002,density_kg_m3=7800.,
            loads_at=lambda t:(q.copy(),None),
            method="MVI",fdm_backend="DENSE_REFERENCE",
            mvi_settings=StaticMVISettings(rank=1))
    assert call_mvi.call_count==2
    assert prior[0] is None
    assert prior[1] is result.first
    assert result.first is not result.final
    assert result.first.mechanical is not result.final.mechanical
    assert result.time_final_s==pytest.approx(.002)
