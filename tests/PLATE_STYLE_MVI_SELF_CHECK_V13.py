#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import cylinder_cuf_stage7d_general_internal_mvi as internal
import cylinder_cuf_stage7b_general_mvi_profiles as profiles
import cylinder_te1_mvi_core as core
from cylinder_final_stage11a_dispatcher import (
    _build_auto_mvi_profiles,
    _make_cuf_simple_bc,
)


def _max_partner_orthogonality_error(array,mask):
    worst=0.0
    for component in range(array.shape[0]):
        for tau in range(array.shape[1]):
            free=np.flatnonzero(mask[component,tau,0])
            matrix=array[component,tau][:,free]
            if np.linalg.norm(matrix)<=1e-30:
                continue
            worst=max(worst,float(np.max(np.abs(
                matrix@matrix.T-np.eye(matrix.shape[0])
            ))))
    return worst


def run_checks(verbose=True):
    cfg={
        "theory":"CUF","cuf_model":"TE1","nx":41,"ns":41,
        "stencil":7,"L":2000.0,"R":1000.0,"h":50.0,
    }
    X,Y=_build_auto_mvi_profiles(cfg,8)
    state=profiles.ProfileStateCUF(X,Y)
    grid=core.CylinderGrid(cfg["L"],cfg["R"],cfg["nx"],cfg["ns"])
    bc=_make_cuf_simple_bc(("CLAMPED","CLAMPED"),state.m)

    if np.any(Y!=0.0):
        raise AssertionError("plate-style initial Y is not exactly zero")
    if np.any(profiles.reconstruct_fields(state)!=0.0):
        raise AssertionError("plate-style initial physical field is not zero")
    for component in range(3):
        for tau in range(state.m):
            if np.linalg.matrix_rank(X[component,tau])!=8:
                raise AssertionError("plate-style X seed bank is rank deficient")

    rng=np.random.default_rng(20260822)
    state.Y[:]=rng.normal(size=state.Y.shape)
    before=profiles.reconstruct_fields(state).copy()
    internal._orthonormalise_frozen_partner(state,grid,bc,"X")
    after_x=profiles.reconstruct_fields(state)
    err_x=float(np.max(np.abs(after_x-before)))
    mask_x=internal._active_mask_for_direction(state,grid,bc,"X")
    orth_x=_max_partner_orthogonality_error(state.X,mask_x)

    internal._orthonormalise_frozen_partner(state,grid,bc,"Y")
    after_y=profiles.reconstruct_fields(state)
    err_y=float(np.max(np.abs(after_y-after_x)))
    mask_y=internal._active_mask_for_direction(state,grid,bc,"Y")
    orth_y=_max_partner_orthogonality_error(state.Y,mask_y)

    before_balance=after_y.copy()
    internal._balance_all_profile_pairs(state)
    err_balance=float(np.max(np.abs(
        profiles.reconstruct_fields(state)-before_balance
    )))

    scale=max(float(np.max(np.abs(before))),1.0)
    if err_x>2e-13*scale or err_y>2e-13*scale or err_balance>2e-13*scale:
        raise AssertionError(
            f"gauge transform changed field: X={err_x}, Y={err_y}, balance={err_balance}"
        )
    if orth_x>2e-13 or orth_y>2e-13:
        raise AssertionError(f"partner QR failed: X={orth_x}, Y={orth_y}")

    report={
        "status":"PASS","rank":8,"initial_Y_zero":True,
        "initial_field_zero":True,"qr_field_error_X":err_x,
        "qr_field_error_Y":err_y,"balance_field_error":err_balance,
        "orthogonality_error_X":orth_x,"orthogonality_error_Y":orth_y,
    }
    if verbose:
        print("PLATE_STYLE_MVI_SELF_CHECK_V13: PASS")
        for key,value in report.items():
            if key!="status":print(f"  {key}: {value}")
    return report


if __name__=="__main__":
    run_checks(True)
