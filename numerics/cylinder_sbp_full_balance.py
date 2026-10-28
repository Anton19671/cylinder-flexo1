#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent complete SBP virtual-work ledger for G3D/Flexo shell solvers.

The exact nodal equilibrium residual is decomposed WITHOUT imposing
additional or double-counted end conditions:

  R_effective
      = R_internal_bulk + R_internal_natural_end
      - W * F_applied_reference
      + R_prescribed_end_and_face_work.

This holds for both mechanical and electrical components. Internal
includes nonlinear Green-Lagrange, covariant G3D, direct+inverse flexo,
Kelvin-Voigt viscous force, director/CUF inertia and mass-proportional
damping, exactly as selected by the current model. Electrical charge
has no artificial mass or viscosity.

The internal components come from the *constitutive* weak jets and
an exact SBP Green reduction; the applied generalized body/face loads
use precisely the area quadrature of CylinderFunctional; prescribed
end tractions, springs, face electrodes and free charges use precisely
compile_boundary_program's residual corrections. The total is checked
against BoundarySpatialProblem.evaluate_effective_residual, which is
the same unprojected operator used inside FDM/MVI Newton.

This is an optional fail-closed diagnostic. It does NOT claim that the
full residual itself is small, that essential BCs were satisfied, that
a MVI rank converged, or that numerical mesh/time refinement passed.
It must not be inserted as additional boundary equations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from cylinder_boundary_spatial import BoundarySpatialProblem


@dataclass(frozen=True)
class FieldWorkLedger:
    """Nodal force/charge residual arrays; reference integration included."""
    internal_bulk: np.ndarray
    internal_natural_end: np.ndarray
    external_distributed: np.ndarray
    prescribed_end_face_work: np.ndarray
    reconstructed: np.ndarray
    independent_full: np.ndarray
    absolute_defect: float
    relative_defect: float
    absolute_tolerance: float
    relative_tolerance: float

    def metrics(self):
        return {
            "absolute_defect":self.absolute_defect,
            "relative_defect":self.relative_defect,
            "absolute_tolerance":self.absolute_tolerance,
            "relative_tolerance":self.relative_tolerance,
            "internal_bulk_peak":float(np.max(np.abs(self.internal_bulk))),
            "internal_natural_end_peak":float(np.max(np.abs(
                self.internal_natural_end))),
            "external_distributed_peak":float(np.max(np.abs(
                self.external_distributed))),
            "prescribed_end_face_peak":float(np.max(np.abs(
                self.prescribed_end_face_work))),
            "independent_full_peak":float(np.max(np.abs(self.independent_full))),
        }


@dataclass(frozen=True)
class FullSBPWorkLedger:
    """One physical time/load state, not a convergence certification."""
    mechanical: FieldWorkLedger
    electrical: FieldWorkLedger | None
    conservative_potential_j: float
    natural_end_fluxes: object
    constraints_count: int
    electric_gauge_required: bool
    time_s: float | None
    status: str = "FULL_REFERENCE_SBP_WORK_IDENTITY_NOT_MESH_VERIFIED"

    def summary(self):
        return {
            "status":self.status,
            "conservative_potential_j":self.conservative_potential_j,
            "time_s":self.time_s,
            "mechanical":self.mechanical.metrics(),
            "electrical":None if self.electrical is None else
                self.electrical.metrics(),
            "constraints_count":self.constraints_count,
            "electric_gauge_required":self.electric_gauge_required,
            "natural_end_flux_magnitudes":self.natural_end_fluxes.magnitudes(),
            "equilibrium_converged":None,
            "mesh_convergence_verified":False,
            "time_step_convergence_verified":False,
        }


def _validated_field(value, shape, name):
    if value is None:
        return np.zeros(shape,dtype=float)
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: finite generalized field {shape} required")
    return a


def _positive_tol(value,name):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(f"{name}: nonnegative finite tolerance required")
    try: out=float(value)
    except (TypeError,ValueError) as exc:
        raise ValueError(f"{name}: finite tolerance required") from exc
    if not math.isfinite(out) or out<0:
        raise ValueError(f"{name}: finite nonnegative tolerance required")
    return out


def _ledger(bulk,natural,distributed,boundary,reference,
            *,rel_tol,abs_tol,name):
    arrays=(np.asarray(bulk,float),np.asarray(natural,float),
            np.asarray(distributed,float),np.asarray(boundary,float),
            np.asarray(reference,float))
    if arrays[0].size==0 or any(
            a.shape!=arrays[0].shape or not np.isfinite(a).all()
            for a in arrays):
        raise ValueError(f"{name}: shape/nonfinite in full virtual-work ledger")
    reconstructed=arrays[0]+arrays[1]+arrays[2]+arrays[3]
    deviation=float(np.max(np.abs(reconstructed-arrays[4])))
    scale=max(float(np.max(np.abs(a))) for a in arrays+(reconstructed,))
    # A dimensionful absolute floor is specified by the caller for the
    # type of residual: newtons for mechanics, coulombs for electronics.
    rel=deviation/max(scale,np.finfo(float).tiny)
    limit=abs_tol+rel_tol*scale
    if deviation>limit:
        raise ValueError(
            "FULL_SBP_WORK_BALANCE_DRIFT: "
            f"{name} error {deviation:.6e} > {limit:.6e}; "
            f"reference scale {scale:.6e}")
    return FieldWorkLedger(
        arrays[0].copy(),arrays[1].copy(),arrays[2].copy(),
        arrays[3].copy(),reconstructed,arrays[4].copy(),
        deviation,rel,abs_tol,rel_tol)


def audit_full_sbp_work(problem,mechanical,electrical=None,*,time_s=None,
                        dead_load=None,free_charge=None,
                        velocity=None,acceleration=None,
                        density_kg_m3=None,eta_s=0.0,
                        mass_damping_rate_per_s=0.0,
                        relative_tolerance=2.e-9,
                        mechanical_absolute_tolerance_n=2.e-10,
                        electrical_absolute_tolerance_c=2.e-12):
    """Reconstruct the full effective solver residual, with no SVD projection.

    The audit uses BOTH independently assembled weak-jet Green components
    and BoundarySpatialProblem's effective full Newton residual. It
    includes time-dependent conservative generalized end work, springs
    and charge BCs, as well as distributed reference loads. Essential
    supports and prescribed electrode potentials are constraints and
    are deliberately NOT transformed into artificial traction forces.
    """
    if not isinstance(problem,BoundarySpatialProblem):
        raise TypeError("audit requires BoundarySpatialProblem")
    rel_tol=_positive_tol(relative_tolerance,"relative_tolerance")
    mtol=_positive_tol(mechanical_absolute_tolerance_n,
                       "mechanical_absolute_tolerance_n")
    etol=_positive_tol(electrical_absolute_tolerance_c,
                       "electrical_absolute_tolerance_c")
    if rel_tol==mtol==etol==0:
        raise ValueError("at least one full-work tolerance must be positive")
    f=problem.functional
    q=np.asarray(mechanical,float)
    if q.ndim!=3 or q.shape!=(f.n_mechanical,*f.ops.shape) or (
            not np.isfinite(q).all()):
        raise ValueError("finite mechanical state required")
    if f.electrical_on:
        if electrical is None:
            raise ValueError("electric potential must be provided")
        p=np.asarray(electrical,float)
        if p.ndim!=3 or p.shape[1:]!=f.ops.shape or not np.isfinite(p).all():
            raise ValueError("finite electric coefficients required")
    else:
        if electrical is not None or free_charge is not None:
            raise ValueError("electric field disabled in this case")
        p=None
    # The following two paths are intentionally assembled separately.
    # Split: constitutive coefficients + exact SBP Green identity.
    split=f.sbp_green_reduction(
        q,p,velocity=velocity,acceleration=acceleration,
        density_kg_m3=density_kg_m3,eta_s=eta_s,
        mass_damping_rate_per_s=mass_damping_rate_per_s)
    # Reference: the actual FDM/MVI full effective Newton residual.
    potential,flat,edge=problem.evaluate_effective_residual(
        q,p,time_s=time_s,dead_load=dead_load,free_charge=free_charge,
        velocity=velocity,acceleration=acceleration,
        density_kg_m3=density_kg_m3,eta_s=eta_s,
        mass_damping_rate_per_s=mass_damping_rate_per_s)
    mload=_validated_field(dead_load,q.shape,"distributed mechanical load")
    mwork=-f.W[None,...]*mload
    mel=_ledger(
        split.mechanical.bulk,split.mechanical.boundary,
        mwork,edge.mechanical_residual_correction,
        flat[:q.size].reshape(q.shape),rel_tol=rel_tol,
        abs_tol=mtol,name="mechanical")

    elec=None
    if p is not None:
        if split.electrical is None or edge.electrical_residual_correction is None:
            raise ValueError("electric split and boundary residual required")
        ch=_validated_field(free_charge,p.shape,"distributed free charge")
        elec=_ledger(
            split.electrical.bulk,split.electrical.boundary,
            +f.W[None,...]*ch,edge.electrical_residual_correction,
            flat[q.size:].reshape(p.shape),rel_tol=rel_tol,
            abs_tol=etol,name="electrical")
    if not math.isfinite(float(potential)):
        raise ValueError("nonfinite conservative work functional")
    return FullSBPWorkLedger(
        mel,elec,float(potential),split.natural_ends,
        len(edge.constraints),bool(edge.gauge_required),
        None if time_s is None else float(time_s))


__all__=("FieldWorkLedger","FullSBPWorkLedger","audit_full_sbp_work")
