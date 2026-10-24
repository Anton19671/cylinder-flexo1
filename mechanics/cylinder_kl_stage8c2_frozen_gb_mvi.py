#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8C-2: KL weak projected residual + strict MVI
for a FROZEN Birger secant field GB(z,s,x).

GB is immutable during the whole inner MVI solve.

This stage reuses without modification:
- Stage8B-4 weak KL variational residual;
- Stage8B-5 strict Y->X MVI.

Only the material/resultant evaluator is replaced:
elastic N/M -> frozen-Birger N/M from Stage8C-1.
"""
from __future__ import annotations
import numpy as np

import cylinder_kl_stage8b4_weak_projected as kw
import cylinder_kl_stage8b5_strict_mvi as km
import cylinder_kl_stage8c1_birger_material as bm


def frozen_birger_projected_residual(
    state,ops,maps,bc,*,K0,GB,quadrature,geometry,
    generalized_load,direction
):
    # Stage8C1 evaluates both current KL kinematics and frozen-GB resultants.
    bout,kin=bm.frozen_birger_resultants_from_profiles(
        state,ops,K0=K0,GB=GB,quadrature=quadrature,geometry=geometry
    )
    variations=kw.direction_variations(state,ops,maps,direction)
    r,labels=kw.projected_residual_from_variations(
        kin,bout.resultants,variations,ops,bc,
        geometry=geometry,generalized_load=generalized_load
    )
    return r,labels,kin,bout


def make_frozen_birger_projected_function(
    ops,maps,bc,*,K0,GB,quadrature,geometry,generalized_load
):
    G=np.asarray(GB,float).copy()
    p=np.asarray(generalized_load,float).copy()

    expected=(len(quadrature.z),ops.grid.ns,ops.grid.nx)
    if G.shape!=expected:
        raise ValueError(f"GB must have shape {expected}")
    if np.any(~np.isfinite(G)) or np.any(G<=0):
        raise ValueError("GB must be finite and positive")

    load_shape=(3,ops.grid.ns,ops.grid.nx)
    if p.shape!=load_shape:
        raise ValueError(f"generalized_load must have shape {load_shape}")

    # Keep a private immutable copy for this inner solve.
    G.setflags(write=False)
    p.setflags(write=False)

    def projected(state,direction):
        r,labels,_,_=frozen_birger_projected_residual(
            state,ops,maps,bc,K0=K0,GB=G,quadrature=quadrature,
            geometry=geometry,generalized_load=p,direction=direction
        )
        return r,labels

    def _prepare(state,direction):
        variations=kw.direction_variations(state,ops,maps,direction)
        labels=tuple(v.label for v in variations)
        norms=np.asarray([np.linalg.norm(v.dU.reshape(-1)) for v in variations],float)
        return {"variations":variations,"labels":labels,"norms":norms}

    def _evaluate_prepared(state,direction,prepared):
        bout,kin=bm.frozen_birger_resultants_from_profiles(
            state,ops,K0=K0,GB=G,quadrature=quadrature,geometry=geometry
        )
        r,labels=kw.projected_residual_from_variations(
            kin,bout.resultants,prepared["variations"],ops,bc,
            geometry=geometry,generalized_load=p
        )
        return r,labels

    projected.mvi_prepare_direction=_prepare
    projected.mvi_evaluate_prepared=_evaluate_prepared

    projected.frozen_GB=G
    projected.generalized_load=p
    projected.mvi_projected_reference_scale=km._surface_load_resultant_scale(
        ops.grid,p
    )
    return projected


def solve_frozen_birger_inner(
    initial_state,ops,maps,bc,*,K0,GB,quadrature,geometry,
    generalized_load,mvi_settings
):
    pf=make_frozen_birger_projected_function(
        ops,maps,bc,K0=K0,GB=GB,quadrature=quadrature,
        geometry=geometry,generalized_load=generalized_load
    )
    return km.solve_internal_mvi(
        initial_state,ops,maps,bc,pf,mvi_settings,quadrature=quadrature
    )


def evaluate_frozen_material_state(
    state,ops,*,K0,GB,quadrature,geometry
):
    return bm.evaluate_material_points_from_profiles(
        state,ops,K0=K0,GB=GB,quadrature=quadrature,geometry=geometry
    )
