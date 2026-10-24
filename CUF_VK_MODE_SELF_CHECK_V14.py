#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, non-solving checks for the V14 CUF von-Karman branch.

This file deliberately does not start a shell calculation or an MVI load
series.  It checks only local mechanics, the exact directional tangent and
the dispatcher mode table.
"""
from __future__ import annotations

import numpy as np

import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7b_general_mvi_profiles as gp
import cylinder_cuf_stage7f_general_linear as gl
import cylinder_cuf_stage7h_general_vk as vk
import cylinder_final_stage11a_dispatcher as dispatch
import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage2b as s2
import cylinder_te1_stage3a_connector as c3
import final_te_basis_provider as tebasis


def _flat(r):
    return np.concatenate([np.asarray(getattr(r,k),float).reshape(-1)
                           for k in g7.RESULTANT_KEYS])


def main():
    rng=np.random.default_rng(20260822)
    basis=tebasis.get_basis("TE2")
    quad=s2.gauss_thickness_quadrature(2.0,5)
    shape=(2,3)
    q=2.0e-3*rng.standard_normal((3,basis.m)+shape)
    qx=8.0e-4*rng.standard_normal(q.shape)
    qs=8.0e-4*rng.standard_normal(q.shape)
    dq=rng.standard_normal(q.shape)
    dqx=rng.standard_normal(q.shape)
    dqs=rng.standard_normal(q.shape)
    kw=dict(R=100.0,K0=8.0e4,GB=3.0e4,quadrature=quad,basis=basis)

    # With zero radial slopes the new branch is exactly the old linear branch.
    zeros=np.zeros_like(q)
    rlin=gl.general_linear_thickness_resultants(q,zeros,zeros,**kw)
    rvk=vk.general_vk_thickness_resultants(q,zeros,zeros,**kw)
    if not np.array_equal(_flat(rlin),_flat(rvk)):
        raise AssertionError("VK does not reduce exactly to linear CUF at zero slopes")

    # Cylindrical membrane term w/R must remain present at the midsurface.
    q0=np.zeros((3,basis.m)); qx0=np.zeros_like(q0); qs0=np.zeros_like(q0)
    q0[2,0]=3.0
    local=vk.general_vk_local_kernel(q0,qx0,qs0,0.0,100.0,8.0e4,3.0e4,basis)
    if not np.isclose(local.E[1,1],3.0/100.0,rtol=0.0,atol=1e-15):
        raise AssertionError("cylindrical w/R term was lost")

    # Exact directional tangent versus a central finite difference.
    ctx=vk.prepare_vk_tangent_context(q,qx,qs,**kw)
    exact=_flat(vk.vk_resultants_tangent_from_context(ctx,dq,dqx,dqs))
    eps=1.0e-7
    plus=_flat(vk.general_vk_thickness_resultants(
        q+eps*dq,qx+eps*dqx,qs+eps*dqs,**kw))
    minus=_flat(vk.general_vk_thickness_resultants(
        q-eps*dq,qx-eps*dqx,qs-eps*dqs,**kw))
    fd=(plus-minus)/(2.0*eps)
    rel=float(np.linalg.norm(exact-fd)/max(np.linalg.norm(fd),1.0))
    if rel>2.0e-8:
        raise AssertionError(f"VK exact tangent mismatch: rel={rel:.3e}")

    # Every approved GUI combination must map to a distinct existing kernel.
    expected_cuf={
        ("LINEAR","HOOKE"):"CUF_LINEAR_ELASTIC",
        ("LINEAR","BIRGER"):"CUF_LINEAR_BIRGER",
        ("VON_KARMAN","HOOKE"):"CUF_VK_ELASTIC",
        ("VON_KARMAN","BIRGER"):"CUF_VK_BIRGER",
        ("GREEN_LAGRANGE","HOOKE"):"CUF_NL_ELASTIC",
        ("GREEN_LAGRANGE","BIRGER"):"CUF_BIRGER",
    }
    for (geometry,material),expected in expected_cuf.items():
        got=dispatch._cuf_child_mode({"geometry_model":geometry,
                                      "material_model":material})
        if got!=expected:
            raise AssertionError((geometry,material,got,expected))
    expected_kl={
        ("LINEAR","HOOKE"):"KL_LINEAR_ELASTIC",
        ("LINEAR","BIRGER"):"KL_LINEAR_BIRGER",
        ("VON_KARMAN","HOOKE"):"KL_VK_ELASTIC",
        ("VON_KARMAN","BIRGER"):"KL_VK_BIRGER",
    }
    for (geometry,material),expected in expected_kl.items():
        got=dispatch._kl_child_mode({"geometry_model":geometry,
                                     "material_model":material})
        if got!=expected:
            raise AssertionError((geometry,material,got,expected))

    # MVI-1 and MVI-2 connection check: construct their real separated
    # states and evaluate the new physical residual, but do not start a load
    # step or iterate the solver.
    grid=mvi.CylinderGrid(100.0,50.0,7,7)
    ops=c3.CylinderSpatialOperators.build(grid,7)
    basis1=tebasis.get_basis("TE1")
    bc=dispatch._make_cuf_simple_bc(("CLAMPED","CLAMPED"),basis1.m)
    nonzero_load=np.zeros((3,basis1.m,grid.ns,grid.nx),float)
    nonzero_load[2,0]=1.0e-4
    for rank in (1,2):
        X,Y=dispatch._build_auto_mvi_profiles({
            "theory":"CUF","cuf_model":"TE1","nx":grid.nx,
            "ns":grid.ns,"stencil":7,"L":grid.length_x,
            "R":grid.radius,"h":1.0,
        },rank)
        state=gp.ProfileStateCUF(X,Y)
        residual=vk.make_vk_hooke_residual_function(
            ops,E0=70000.0,nu0=0.3,quadrature=s2.gauss_thickness_quadrature(1.0,3),
            basis=basis1,generalized_load=nonzero_load,bc=bc
        )(state)
        if residual.shape!=(3*basis1.m*grid.ns*grid.nx,) or not np.all(np.isfinite(residual)):
            raise AssertionError(f"CUF VK MVI-{rank} residual connection failed")

    print("CUF_VK_MODE_SELF_CHECK_V14: PASS")
    print(f"  exact tangent relative error = {rel:.3e}")
    print("  cylindrical w/R term         = PASS")
    print("  CUF menu routes              = 6/6")
    print("  KL menu routes               = 4/4")
    print("  CUF VK MVI residual          = ranks 1,2 PASS (no solve)")


if __name__=="__main__":
    main()
