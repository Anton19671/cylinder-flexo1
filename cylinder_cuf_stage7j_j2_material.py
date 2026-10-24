#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incremental J2 material bridge for all CUF thickness bases (TE1--TE4).

LINEAR and von Karman use the small-strain three-dimensional J2 return map.
GREEN_LAGRANGE uses the multiplicative finite-strain return map and returns
first Piola stress P=F S.  CUF is a three-dimensional expansion, therefore no
shell plane-stress elimination is imposed here.

Every call reads one committed integration-point history and returns a trial
history.  Neither J2 nor Birger state is changed in residual/Jacobian probes.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7h_general_vk as gvk
from final_j2_incremental import J2MaterialError, J2Parameters, J2State, update_small_strain
from final_j2_finite_strain import FiniteJ2State, update_finite_strain


@dataclass
class CUFJ2MaterialState:
    kind: str
    kappa: np.ndarray
    equivalent_stress_trial: np.ndarray
    equivalent_stress: np.ndarray
    yield_stress: np.ndarray
    plastic_multiplier: np.ndarray
    plastic_mask: np.ndarray
    unloading_mask: np.ndarray
    reloading_mask: np.ndarray
    stress: np.ndarray
    total_strain: np.ndarray | None
    plastic_strain: np.ndarray | None
    deformation_gradient: np.ndarray | None
    second_piola: np.ndarray | None
    plastic_deformation_gradient: np.ndarray | None


@dataclass
class CUFJ2Resultants:
    resultants: g7.GeneralResultants
    material_state: CUFJ2MaterialState
    trial_state: J2State | FiniteJ2State


def _history_shape(committed, quadrature, surface_shape, *, finite: bool):
    expected = (len(quadrature.z),) + tuple(surface_shape)
    if finite:
        if not isinstance(committed, FiniteJ2State) or committed.validate() != expected:
            raise J2MaterialError(f"finite CUF J2 history shape must be {expected}")
    else:
        if not isinstance(committed, J2State) or committed.validate() != expected:
            raise J2MaterialError(f"small-strain CUF J2 history shape must be {expected}")
    return expected


def _tensor_to_points(tensor):
    """(3,3,*surface)->(*surface,3,3), preserving material component order."""
    return np.moveaxis(np.asarray(tensor, float), (0, 1), (-2, -1))


def _points_to_tensor(points):
    """(*surface,3,3)->(3,3,*surface)."""
    return np.moveaxis(np.asarray(points, float), (-2, -1), (0, 1))


def _geometry_strain(geometry, q, qx, qs, z, R, basis):
    geometry = str(geometry).upper()
    H, Ft, Fp, abc = g7._batched_general_gradient_quantities(
        q, qx, qs, z, R, basis
    )
    if geometry == "LINEAR":
        return H, Ft, Fp, g7._batched_linear_strain_from_abc(abc), None
    if geometry == "VON_KARMAN":
        slope = gvk._batched_vk_slope(qx, qs, H=H, Ft=Ft)
        return H, Ft, Fp, gvk._batched_vk_strain(abc, slope), slope
    raise ValueError("small-strain CUF J2 geometry must be LINEAR or VON_KARMAN")


def _new_accumulator(basis, surface_shape):
    return {
        key: np.zeros((basis.m,) + tuple(surface_shape), float)
        for key in g7.RESULTANT_KEYS
    }


def resultants_small_strain(
    q, qx, qs, *, geometry, R, committed: J2State, parameters: J2Parameters,
    quadrature, basis,
) -> CUFJ2Resultants:
    """CUF LINEAR/VK resultants from the independent small-strain J2 law."""
    q, qx, qs = g7._check_general_surface_arrays(q, qx, qs, basis)
    surface_shape = q.shape[2:]
    expected = _history_shape(committed, quadrature, surface_shape, finite=False)
    acc = _new_accumulator(basis, surface_shape)

    plastic_strain = np.empty(expected + (3, 3), float)
    kappa = np.empty(expected, float)
    qtrial = np.empty(expected, float)
    qnew = np.empty(expected, float)
    yield_stress = np.empty(expected, float)
    multiplier = np.empty(expected, float)
    plastic = np.zeros(expected, bool)
    unloading = np.zeros(expected, bool)
    reloading = np.zeros(expected, bool)
    strain_out = np.empty(expected + (3, 3), float)
    stress_out = np.empty_like(strain_out)

    for g, (z, wz) in enumerate(zip(np.asarray(quadrature.z, float), np.asarray(quadrature.w, float))):
        H, Ft, Fp, strain, slope = _geometry_strain(
            geometry, q, qx, qs, float(z), float(R), basis
        )
        response = update_small_strain(
            _tensor_to_points(strain),
            J2State(
                np.asarray(committed.plastic_strain[g], float),
                np.asarray(committed.kappa[g], float),
                np.asarray(committed.equivalent_stress[g], float),
            ),
            parameters,
        )
        stress = _points_to_tensor(response.stress)
        g7._accumulate_general_resultants(
            acc, stress, H, Ft, Fp, float(wz), float(R), surface_shape
        )
        if slope is not None:
            radial = np.einsum("i...,ij...->j...", slope, stress, optimize=False)
            gvk._add_vk_radial_work(
                acc, radial, H=H, Ft=Ft, wz=float(wz), surface_shape=surface_shape
            )

        plastic_strain[g] = response.state.plastic_strain
        kappa[g] = response.state.kappa
        qtrial[g] = response.equivalent_stress_trial
        qnew[g] = response.equivalent_stress
        yield_stress[g] = response.yield_stress
        multiplier[g] = response.plastic_multiplier
        plastic[g] = response.plastic_mask
        unloading[g] = response.unloading_mask
        reloading[g] = response.reloading_mask
        strain_out[g] = response.total_strain
        stress_out[g] = response.stress

    material = CUFJ2MaterialState(
        kind="SMALL_STRAIN", kappa=kappa, equivalent_stress_trial=qtrial,
        equivalent_stress=qnew, yield_stress=yield_stress,
        plastic_multiplier=multiplier, plastic_mask=plastic,
        unloading_mask=unloading, reloading_mask=reloading, stress=stress_out,
        total_strain=strain_out, plastic_strain=plastic_strain,
        deformation_gradient=None, second_piola=None,
        plastic_deformation_gradient=None,
    )
    return CUFJ2Resultants(
        g7.GeneralResultants(**acc), material,
        J2State(plastic_strain, kappa, qnew),
    )


def resultants_finite_strain(
    q, qx, qs, *, R, committed: FiniteJ2State, parameters: J2Parameters,
    quadrature, basis,
) -> CUFJ2Resultants:
    """CUF full-GL resultants from finite J2: P=F S at every point."""
    q, qx, qs = g7._check_general_surface_arrays(q, qx, qs, basis)
    surface_shape = q.shape[2:]
    expected = _history_shape(committed, quadrature, surface_shape, finite=True)
    acc = _new_accumulator(basis, surface_shape)

    Fout = np.empty(expected + (3, 3), float)
    Fpout = np.empty_like(Fout)
    second_piola = np.empty_like(Fout)
    stress_out = np.empty_like(Fout)
    kappa = np.empty(expected, float)
    qtrial = np.empty(expected, float)
    qnew = np.empty(expected, float)
    yield_stress = np.empty(expected, float)
    multiplier = np.empty(expected, float)
    plastic = np.zeros(expected, bool)
    unloading = np.zeros(expected, bool)
    reloading = np.zeros(expected, bool)

    for g, (z, wz) in enumerate(zip(np.asarray(quadrature.z, float), np.asarray(quadrature.w, float))):
        H, Ft, Fp, abc = g7._batched_general_gradient_quantities(
            q, qx, qs, float(z), float(R), basis
        )
        F = abc.copy()
        F[0, 0] += 1.0
        F[1, 1] += 1.0
        F[2, 2] += 1.0
        det = g7._batched_det3(F)
        g7._raise_if_invalid_batched_F(det, float(z), surface_shape)
        response = update_finite_strain(
            _tensor_to_points(F),
            FiniteJ2State(
                np.asarray(committed.Fp[g], float),
                np.asarray(committed.kappa[g], float),
                np.asarray(committed.equivalent_kirchhoff_stress[g], float),
            ),
            parameters,
        )
        S = _points_to_tensor(response.second_piola)
        P = np.einsum("ik...,kj...->ij...", F, S, optimize=False)
        g7._accumulate_general_resultants(
            acc, P, H, Ft, Fp, float(wz), float(R), surface_shape
        )

        Fout[g] = _tensor_to_points(F)
        Fpout[g] = response.state.Fp
        second_piola[g] = response.second_piola
        stress_out[g] = response.cauchy
        kappa[g] = response.state.kappa
        qtrial[g] = response.equivalent_trial
        qnew[g] = response.equivalent_stress
        yield_stress[g] = response.yield_stress
        multiplier[g] = response.plastic_multiplier
        plastic[g] = response.plastic_mask
        unloading[g] = response.unloading_mask
        reloading[g] = response.reloading_mask

    material = CUFJ2MaterialState(
        kind="FINITE_STRAIN", kappa=kappa, equivalent_stress_trial=qtrial,
        equivalent_stress=qnew, yield_stress=yield_stress,
        plastic_multiplier=multiplier, plastic_mask=plastic,
        unloading_mask=unloading, reloading_mask=reloading, stress=stress_out,
        total_strain=None, plastic_strain=None, deformation_gradient=Fout,
        second_piola=second_piola, plastic_deformation_gradient=Fpout,
    )
    return CUFJ2Resultants(
        g7.GeneralResultants(**acc), material,
        FiniteJ2State(Fpout, kappa, qnew),
    )


__all__ = [
    "CUFJ2MaterialState", "CUFJ2Resultants",
    "resultants_small_strain", "resultants_finite_strain",
]
