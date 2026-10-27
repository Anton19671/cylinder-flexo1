#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUF-TE1 closed-cylinder Stage 3B: end boundary-condition layer.

Source: supplied verified Word document, Appendix G.7.2, equations (722)-(725).

For each generalized TE1 field at each end, exactly one condition is assigned:
- GEOMETRIC: prescribed generalized displacement value;
- NATURAL: prescribed generalized end traction.

Natural conditions:
  nx*Ixx0 = T_u0, nx*Ixx1 = T_u1
  nx*Isx0 = T_v0, nx*Isx1 = T_v1
  nx*Irx0 = T_w0, nx*Irx1 = T_w1

with nx=-1 at x=0 and nx=+1 at x=L.

No rule for rigid-mode removal of a fully free cylinder is invented here.
If both ends are entirely natural, the layer reports RIGID_MODE_HANDLING_REQUIRED.

For the MVI profile solver, homogeneous geometric end conditions are converted
to inactive X-profile endpoint parameters, following the already accepted
direct-elimination philosophy of the strict MVI code. Nonzero geometric data
are represented exactly as boundary residual equations but are not silently
converted to per-profile constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage2b as s2

FIELD_NAMES = ("u0","u1","v0","v1","w0","w1")
FIELD_INDEX = {name:i for i,name in enumerate(FIELD_NAMES)}

NATURAL_CACHE_KEY = {
    "u0":"Ixx0",
    "u1":"Ixx1",
    "v0":"Isx0",
    "v1":"Isx1",
    "w0":"Irx0",
    "w1":"Irx1",
}

class BoundaryConditionError(RuntimeError):
    pass

class RigidModeHandlingRequired(BoundaryConditionError):
    pass

@dataclass(frozen=True)
class ScalarOrCircumferentialData:
    value: object = 0.0

    def array(self, ns: int) -> np.ndarray:
        a = np.asarray(self.value, dtype=float)
        if a.ndim == 0:
            out = np.full(ns, float(a), dtype=float)
        elif a.shape == (ns,):
            out = a.astype(float, copy=True)
        else:
            raise BoundaryConditionError(
                f"BC data must be scalar or shape ({ns},), got {a.shape}"
            )
        if not np.all(np.isfinite(out)):
            raise BoundaryConditionError("BC data contains non-finite values")
        return out

@dataclass(frozen=True)
class FieldEndCondition:
    kind: str                 # "GEOMETRIC" or "NATURAL"
    data: ScalarOrCircumferentialData = ScalarOrCircumferentialData(0.0)

    def __post_init__(self):
        k = str(self.kind).upper()
        if k not in ("GEOMETRIC","NATURAL"):
            raise BoundaryConditionError(
                "kind must be GEOMETRIC or NATURAL"
            )
        object.__setattr__(self, "kind", k)

@dataclass(frozen=True)
class EndBoundarySet:
    """
    One complete end: exactly six conditions, one for each TE1 field.
    """
    conditions: tuple

    def __post_init__(self):
        if len(self.conditions) != 6:
            raise BoundaryConditionError("exactly six TE1 end conditions required")
        for c in self.conditions:
            if not isinstance(c, FieldEndCondition):
                raise BoundaryConditionError("invalid end condition object")

    @classmethod
    def all_geometric_zero(cls):
        return cls(tuple(
            FieldEndCondition("GEOMETRIC", ScalarOrCircumferentialData(0.0))
            for _ in FIELD_NAMES
        ))

    @classmethod
    def all_free(cls):
        return cls(tuple(
            FieldEndCondition("NATURAL", ScalarOrCircumferentialData(0.0))
            for _ in FIELD_NAMES
        ))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, FieldEndCondition]):
        missing = [f for f in FIELD_NAMES if f not in mapping]
        extra = [f for f in mapping if f not in FIELD_INDEX]
        if missing or extra:
            raise BoundaryConditionError(
                f"mapping must contain exactly {FIELD_NAMES}; missing={missing}, extra={extra}"
            )
        return cls(tuple(mapping[f] for f in FIELD_NAMES))

    def all_natural(self) -> bool:
        return all(c.kind == "NATURAL" for c in self.conditions)

    def all_geometric(self) -> bool:
        return all(c.kind == "GEOMETRIC" for c in self.conditions)

@dataclass(frozen=True)
class CylinderEndBC:
    left: EndBoundarySet
    right: EndBoundarySet

    @classmethod
    def clamped_free(cls):
        # "clamped" here means all six TE1 generalized displacement functions
        # are geometrically zero at that end; "free" means all six generalized
        # tractions from (725) are zero.
        return cls(
            left=EndBoundarySet.all_geometric_zero(),
            right=EndBoundarySet.all_free(),
        )

    @classmethod
    def free_clamped(cls):
        return cls(
            left=EndBoundarySet.all_free(),
            right=EndBoundarySet.all_geometric_zero(),
        )

    @classmethod
    def clamped_clamped(cls):
        return cls(
            left=EndBoundarySet.all_geometric_zero(),
            right=EndBoundarySet.all_geometric_zero(),
        )

    @classmethod
    def free_free(cls):
        return cls(
            left=EndBoundarySet.all_free(),
            right=EndBoundarySet.all_free(),
        )

    def check_rigid_mode_requirement(self):
        if self.left.all_natural() and self.right.all_natural():
            raise RigidModeHandlingRequired(
                "RIGID_MODE_HANDLING_REQUIRED: both cylinder ends are entirely natural; "
                "the supplied derivation requires separate rigid-mode treatment and "
                "does not prescribe its form."
            )

def _end_index_and_normal(side: str, nx: int):
    side = str(side).upper()
    if side == "LEFT":
        return 0, -1.0
    if side == "RIGHT":
        return nx-1, +1.0
    raise BoundaryConditionError("side must be LEFT or RIGHT")

def end_boundary_residual(q, cache: s2.ThicknessCache,
                          end: EndBoundarySet, side: str) -> np.ndarray:
    """
    Return the six boundary equations at one end, shape (6,ns).

    GEOMETRIC:
        q_f - qbar_f = 0
    NATURAL (eqs. 722-724):
        nx*I_f - Tbar_f = 0
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 3 or q.shape[0] != 6:
        raise BoundaryConditionError("q must have shape (6,ns,nx)")
    ns, nx = q.shape[1], q.shape[2]
    ix, normal = _end_index_and_normal(side, nx)

    out = np.empty((6,ns), dtype=float)
    cd = cache.as_dict()

    for f, name in enumerate(FIELD_NAMES):
        bc = end.conditions[f]
        prescribed = bc.data.array(ns)
        if bc.kind == "GEOMETRIC":
            out[f] = q[f,:,ix] - prescribed
        else:
            key = NATURAL_CACHE_KEY[name]
            field = np.asarray(cd[key], dtype=float)
            if field.shape != (ns,nx):
                raise BoundaryConditionError(
                    f"cache field {key} has shape {field.shape}, expected {(ns,nx)}"
                )
            out[f] = normal*field[:,ix] - prescribed
    return out

def full_boundary_residuals(q, cache: s2.ThicknessCache,
                            bc: CylinderEndBC):
    return {
        "left": end_boundary_residual(q, cache, bc.left, "LEFT"),
        "right": end_boundary_residual(q, cache, bc.right, "RIGHT"),
    }

def assemble_strong_residual_with_end_bcs(raw_residual, q,
                                          cache: s2.ThicknessCache,
                                          bc: CylinderEndBC,
                                          *,
                                          reject_fully_free: bool = True):
    """
    Interior columns are the verified Stage 2B strong residual.
    At x=0 and x=L the corresponding six boundary equations replace the
    endpoint strong equations.

    This function assembles the strong equation set; it does not itself solve it.
    """
    raw = np.asarray(raw_residual, dtype=float)
    q = np.asarray(q, dtype=float)
    if raw.shape != q.shape or raw.ndim != 3 or raw.shape[0] != 6:
        raise BoundaryConditionError(
            "raw_residual and q must both have shape (6,ns,nx)"
        )
    if reject_fully_free:
        bc.check_rigid_mode_requirement()

    b = full_boundary_residuals(q, cache, bc)
    out = raw.copy()
    out[:,:,0] = b["left"]
    out[:,:,-1] = b["right"]
    return out

def homogeneous_geometric_x_active_mask(
    state: mvi.ProfileStateTE1,
    grid: mvi.CylinderGrid,
    bc: CylinderEndBC,
    *,
    zero_tolerance: float = 0.0,
):
    """
    MVI X-profile parameter mask for HOMOGENEOUS geometric end data only.

    For a zero geometric condition q_f=0 at an end, setting every
    X[f,r,end]=0 enforces the condition for any periodic Y partner.

    Nonzero geometric data are not silently decomposed into profiles.
    """
    state.validate_grid(grid)
    mask = np.ones_like(state.X, dtype=bool)
    ns = grid.ns

    for side, end in (("LEFT",bc.left),("RIGHT",bc.right)):
        ix = 0 if side == "LEFT" else grid.nx-1
        for f, cond in enumerate(end.conditions):
            if cond.kind != "GEOMETRIC":
                continue
            data = cond.data.array(ns)
            if np.max(np.abs(data)) > zero_tolerance:
                raise BoundaryConditionError(
                    f"NONZERO_GEOMETRIC_BC_NOT_DECOMPOSED: {side} {FIELD_NAMES[f]}"
                )
            mask[f,:,ix] = False
    return mask

def enforce_homogeneous_geometric_x_profiles(
    state: mvi.ProfileStateTE1,
    grid: mvi.CylinderGrid,
    bc: CylinderEndBC,
):
    """
    Set constrained homogeneous X endpoint profile parameters exactly to zero.
    """
    mask = homogeneous_geometric_x_active_mask(state, grid, bc)
    state.X[~mask] = 0.0
    return mask

def boundary_status(bc: CylinderEndBC) -> str:
    if bc.left.all_natural() and bc.right.all_natural():
        return "RIGID_MODE_HANDLING_REQUIRED"
    return "BOUNDARY_SET_DEFINED"
