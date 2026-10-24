#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUF-TE1 closed-cylinder Stage 4B:
GEOMETRICALLY NONLINEAR ELASTIC branch.

Physics:
- full TE1 kinematics;
- full Green-Lagrange strain;
- Total Lagrangian S-E elastic law with the INITIAL elastic constants;
- P=F*S;
- fixed G=G0, K=K0 throughout the whole calculation;
- NO Birger update and NO material outer cycle.

This is a separate solver branch from Stage 4A LINEAR_ELASTIC and from the
future TE1_BIRGER branch.
"""
from __future__ import annotations

import numpy as np

import cylinder_te1_local_kernel as nk
import cylinder_te1_stage3c_internal_mvi as c4


def elastic_constants(E0,nu0):
    return nk.ElasticConstants(float(E0),float(nu0))


def make_nonlinear_elastic_residual_function(
    ops,
    *,
    E0,
    nu0,
    quadrature,
    generalized_load,
    bc,
):
    ec=elastic_constants(E0,nu0)
    return c4.make_physical_residual_function(
        ops,
        K0=ec.K0,
        GB=ec.G0,
        quadrature=quadrature,
        generalized_load=np.asarray(generalized_load,float),
        bc=bc,
    )


def solve_nonlinear_elastic_mvi(
    initial_state,
    ops,
    *,
    E0,
    nu0,
    quadrature,
    generalized_load,
    bc,
    settings,
):
    residual_fn=make_nonlinear_elastic_residual_function(
        ops,
        E0=E0,
        nu0=nu0,
        quadrature=quadrature,
        generalized_load=generalized_load,
        bc=bc,
    )
    return c4.solve_internal_mvi(
        initial_state,
        ops.grid,
        bc,
        residual_fn,
        settings,
        quadrature=quadrature,
    )
