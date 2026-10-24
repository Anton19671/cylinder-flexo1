#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KL material bridge for small-strain incremental J2 plane-stress plasticity.

The bridge consumes the accepted J2 history at the thickness integration
points, evaluates a non-mutating J2 trial response, and integrates only the
physical in-plane stresses into KL resultants.  It contains no Birger modulus,
no deformation-theory update and no global-solver policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8b2_profiles as kp
from final_j2_incremental import (
    J2MaterialError, J2Parameters, J2State, update_plane_stress,
    plane_stress_algorithmic_tangent,
)


@dataclass
class KLJ2MaterialState:
    plastic_strain: np.ndarray
    kappa: np.ndarray
    equivalent_stress_trial: np.ndarray
    equivalent_stress: np.ndarray
    yield_stress: np.ndarray
    plastic_multiplier: np.ndarray
    plastic_mask: np.ndarray
    unloading_mask: np.ndarray
    reloading_mask: np.ndarray
    total_strain: np.ndarray
    stress: np.ndarray


@dataclass
class KLJ2Resultants:
    resultants: kl.KLResultants
    material_state: KLJ2MaterialState
    trial_state: J2State
    consistent_tangent: np.ndarray | None = None


def _shape(state: J2State, quadrature, surface_shape):
    expected = (len(quadrature.z),) + tuple(surface_shape)
    if state.validate() != expected:
        raise J2MaterialError(f"KL J2 history shape must be {expected}")
    return expected


def resultants_from_kinematics(
    kin, *, committed: J2State, parameters: J2Parameters, quadrature,
    with_tangent: bool = False,
) -> KLJ2Resultants:
    """Integrate a trial J2 plane-stress state through the KL thickness."""

    surf_shape = tuple(kin.state.eps_x.shape)
    nz, *_ = _shape(committed, quadrature, surf_shape)
    Nx = np.zeros(surf_shape, float)
    Ns = np.zeros(surf_shape, float)
    Nxs = np.zeros(surf_shape, float)
    Mx = np.zeros(surf_shape, float)
    Ms = np.zeros(surf_shape, float)
    Mxs = np.zeros(surf_shape, float)

    ep = np.empty((nz,) + surf_shape + (3, 3), float)
    kappa = np.empty((nz,) + surf_shape, float)
    qtrial = np.empty_like(kappa)
    qnew = np.empty_like(kappa)
    sy = np.empty_like(kappa)
    dg = np.empty_like(kappa)
    plastic = np.zeros_like(kappa, bool)
    unloading = np.zeros_like(kappa, bool)
    reloading = np.zeros_like(kappa, bool)
    total = np.empty((nz,) + surf_shape + (3, 3), float)
    stress = np.empty_like(total)
    calg = np.empty((nz,) + surf_shape + (3, 3), float) if with_tangent else None

    for g, (z, wz) in enumerate(zip(np.asarray(quadrature.z, float), np.asarray(quadrature.w, float))):
        ex, es, gamma = kl.through_thickness_strains(kin.state, float(z))
        old = J2State(
            np.asarray(committed.plastic_strain[g], float),
            np.asarray(committed.kappa[g], float),
            np.asarray(committed.equivalent_stress[g], float),
        )
        response = update_plane_stress(ex, es, gamma, old, parameters)
        if calg is not None:
            calg[g] = plane_stress_algorithmic_tangent(
                ex, es, gamma, old, parameters, response=response
            )
        sig = response.stress
        Nx += float(wz) * sig[..., 0, 0]
        Ns += float(wz) * sig[..., 1, 1]
        Nxs += float(wz) * sig[..., 0, 1]
        Mx += float(wz) * float(z) * sig[..., 0, 0]
        Ms += float(wz) * float(z) * sig[..., 1, 1]
        Mxs += float(wz) * float(z) * sig[..., 0, 1]

        ep[g] = response.state.plastic_strain
        kappa[g] = response.state.kappa
        qtrial[g] = response.equivalent_stress_trial
        qnew[g] = response.equivalent_stress
        sy[g] = response.yield_stress
        dg[g] = response.plastic_multiplier
        plastic[g] = response.plastic_mask
        unloading[g] = response.unloading_mask
        reloading[g] = response.reloading_mask
        total[g] = response.total_strain
        stress[g] = sig

    resultants = kl.KLResultants(Nx, Ns, Nxs, Mx, Ms, Mxs)
    material = KLJ2MaterialState(
        plastic_strain=ep,
        kappa=kappa,
        equivalent_stress_trial=qtrial,
        equivalent_stress=qnew,
        yield_stress=sy,
        plastic_multiplier=dg,
        plastic_mask=plastic,
        unloading_mask=unloading,
        reloading_mask=reloading,
        total_strain=total,
        stress=stress,
    )
    trial = J2State(ep, kappa, qnew)
    return KLJ2Resultants(resultants, material, trial, calg)


def resultants_from_profiles(state, ops, *, committed: J2State, parameters: J2Parameters, quadrature, geometry: str):
    U = kp.reconstruct_fields(state)
    kin = kl.field_kinematics(U, ops, geometry=geometry)
    return resultants_from_kinematics(
        kin, committed=committed, parameters=parameters, quadrature=quadrature
    ), kin


__all__ = [
    "KLJ2MaterialState",
    "KLJ2Resultants",
    "resultants_from_kinematics",
    "resultants_from_profiles",
]
