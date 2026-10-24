#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KL Green--Lagrange bridge to finite-strain incremental J2.

The KL GL equilibrium uses Green strain virtual work, so this bridge integrates
the material second Piola stress, not Cauchy stress.  At every z quadrature
point the in-plane GL strains define C_{alpha beta}; the local finite J2
plane-stress solver finds the thickness stretch from S_33=0.

The bridge is non-mutating.  A dispatcher commits its returned Fp/kappa state
only after the global load step has converged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cylinder_kl_full_gl_kinematics as fullgl
import cylinder_kl_stage8a_elastic_mechanics as kl
from final_j2_incremental import J2MaterialError, J2Parameters
from final_j2_finite_strain import (
    FiniteJ2State,
    FiniteJ2Response,
    update_finite_plane_stress,
)


@dataclass
class KLGLJ2MaterialState:
    plastic_deformation_gradient: np.ndarray
    kappa: np.ndarray
    equivalent_stress_trial: np.ndarray
    equivalent_stress: np.ndarray
    yield_stress: np.ndarray
    plastic_multiplier: np.ndarray
    plastic_mask: np.ndarray
    unloading_mask: np.ndarray
    reloading_mask: np.ndarray
    green_strain: np.ndarray
    second_piola: np.ndarray
    cauchy: np.ndarray


@dataclass
class KLGLJ2Resultants:
    resultants: kl.KLResultants
    material_state: KLGLJ2MaterialState
    trial_state: FiniteJ2State


def resultants_from_gl_kinematics(
    kin, *, committed: FiniteJ2State, parameters: J2Parameters, quadrature,
) -> KLGLJ2Resultants:
    if kin.full_gl is None or not isinstance(kin.full_gl, fullgl.KLFullGLState):
        raise J2MaterialError("finite KL J2 requires GREEN_LAGRANGE kinematics")
    surface_shape = tuple(kin.state.eps_x.shape)
    expected = (len(quadrature.z),) + surface_shape
    if committed.validate() != expected:
        raise J2MaterialError(f"finite KL J2 history shape must be {expected}")

    nx = np.zeros(surface_shape, float)
    ns = np.zeros(surface_shape, float)
    nxs = np.zeros(surface_shape, float)
    mx = np.zeros(surface_shape, float)
    ms = np.zeros(surface_shape, float)
    mxs = np.zeros(surface_shape, float)

    fp = np.empty(expected + (3, 3), float)
    kappa = np.empty(expected, float)
    qtrial = np.empty(expected, float)
    qnew = np.empty(expected, float)
    sy = np.empty(expected, float)
    dg = np.empty(expected, float)
    plastic = np.zeros(expected, bool)
    unloading = np.zeros(expected, bool)
    reloading = np.zeros(expected, bool)
    green = np.empty(expected + (3, 3), float)
    second_piola = np.empty_like(green)
    cauchy = np.empty_like(green)

    for g, (z, wz) in enumerate(zip(np.asarray(quadrature.z, float), np.asarray(quadrature.w, float))):
        ex, es, gamma = fullgl.through_thickness_strains(kin.full_gl, float(z))
        response: FiniteJ2Response = update_finite_plane_stress(
            1.0 + 2.0 * ex,
            1.0 + 2.0 * es,
            gamma,
            FiniteJ2State(
                np.asarray(committed.Fp[g], float),
                np.asarray(committed.kappa[g], float),
                np.asarray(committed.equivalent_kirchhoff_stress[g], float),
            ),
            parameters,
        )
        stress = response.second_piola
        nx += float(wz) * stress[..., 0, 0]
        ns += float(wz) * stress[..., 1, 1]
        nxs += float(wz) * stress[..., 0, 1]
        mx += float(wz) * float(z) * stress[..., 0, 0]
        ms += float(wz) * float(z) * stress[..., 1, 1]
        mxs += float(wz) * float(z) * stress[..., 0, 1]

        fp[g] = response.state.Fp
        kappa[g] = response.state.kappa
        qtrial[g] = response.equivalent_trial
        qnew[g] = response.equivalent_stress
        sy[g] = response.yield_stress
        dg[g] = response.plastic_multiplier
        plastic[g] = response.plastic_mask
        unloading[g] = response.unloading_mask
        reloading[g] = response.reloading_mask
        green[g, ..., 0, 0] = ex
        green[g, ..., 1, 1] = es
        green[g, ..., 0, 1] = green[g, ..., 1, 0] = 0.5 * gamma
        green[g, ..., 2, 2] = 0.0
        green[g, ..., 0, 2] = green[g, ..., 2, 0] = 0.0
        green[g, ..., 1, 2] = green[g, ..., 2, 1] = 0.0
        second_piola[g] = stress
        cauchy[g] = response.cauchy

    material = KLGLJ2MaterialState(
        plastic_deformation_gradient=fp,
        kappa=kappa,
        equivalent_stress_trial=qtrial,
        equivalent_stress=qnew,
        yield_stress=sy,
        plastic_multiplier=dg,
        plastic_mask=plastic,
        unloading_mask=unloading,
        reloading_mask=reloading,
        green_strain=green,
        second_piola=second_piola,
        cauchy=cauchy,
    )
    return KLGLJ2Resultants(
        kl.KLResultants(nx, ns, nxs, mx, ms, mxs),
        material,
        FiniteJ2State(fp, kappa, qnew),
    )


__all__ = ["KLGLJ2MaterialState", "KLGLJ2Resultants", "resultants_from_gl_kinematics"]
