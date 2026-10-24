#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7C: GENERAL CUF end boundary-condition layer.

Source: supplied verified Word derivation, general CUF boundary conditions
(section 22.2, equations 351-354), plus TE1 Appendix G.7.2 regression.

For each generalized CUF amplitude q_{alpha,tau} at each x-end exactly one:
- GEOMETRIC value qbar_{alpha,tau}(s), or
- NATURAL generalized end traction Tbar_{alpha,tau}(s).

Natural end conditions:
    nx * Au_x[tau] = T_u_tau
    nx * Av_x[tau] = T_v_tau
    nx * Aw_x[tau] = T_w_tau

with nx=-1 at x=0 and nx=+1 at x=L.

For m=2 and F0=1,F1=z this must reduce exactly to verified TE1 Stage 3B:
    Ixx0,Ixx1, Isx0,Isx1, Irx0,Irx1.

No rigid-mode normalization is invented for fully natural both ends.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7b_general_mvi_profiles as g8


class GeneralBoundaryConditionError(RuntimeError):
    pass


class GeneralRigidModeHandlingRequired(GeneralBoundaryConditionError):
    pass


@dataclass(frozen=True)
class ScalarOrCircumferentialData:
    value: object=0.0
    def array(self,ns):
        a=np.asarray(self.value,float)
        if a.ndim==0:
            out=np.full(ns,float(a),float)
        elif a.shape==(ns,):
            out=a.astype(float,copy=True)
        else:
            raise GeneralBoundaryConditionError(
                f"BC data must be scalar or shape ({ns},), got {a.shape}"
            )
        if not np.all(np.isfinite(out)):
            raise GeneralBoundaryConditionError("BC data contains non-finite values")
        return out


@dataclass(frozen=True)
class GeneralFieldEndCondition:
    kind: str
    data: ScalarOrCircumferentialData=ScalarOrCircumferentialData(0.0)
    stiffness: ScalarOrCircumferentialData | None = None
    reference: ScalarOrCircumferentialData = ScalarOrCircumferentialData(0.0)
    def __post_init__(self):
        k=str(self.kind).upper()
        if k not in ("GEOMETRIC","NATURAL","ELASTIC"):
            raise GeneralBoundaryConditionError("kind must be GEOMETRIC, NATURAL or ELASTIC")
        object.__setattr__(self,"kind",k)
        if k=="ELASTIC":
            if self.stiffness is None:
                raise GeneralBoundaryConditionError("ELASTIC boundary condition requires stiffness")
            a=np.asarray(self.stiffness.value,float)
            if not np.all(np.isfinite(a)) or np.any(a<0.0):
                raise GeneralBoundaryConditionError(
                    "ELASTIC boundary stiffness must be finite and nonnegative"
                )

    def stiffness_array(self,ns):
        if self.kind!="ELASTIC":
            return np.zeros(int(ns),float)
        return self.stiffness.array(ns)

    def reference_array(self,ns):
        return self.reference.array(ns)


@dataclass(frozen=True)
class GeneralEndBoundarySet:
    """
    conditions shape conceptually (3,m), stored tuple-of-tuples:
      component 0=u, 1=v, 2=w; tau second.
    """
    conditions: tuple

    def __post_init__(self):
        if len(self.conditions)!=3:
            raise GeneralBoundaryConditionError("conditions must contain 3 components")
        m=None
        for row in self.conditions:
            if m is None: m=len(row)
            if len(row)!=m or m<1:
                raise GeneralBoundaryConditionError("inconsistent/empty CUF tau rows")
            if not all(isinstance(c,GeneralFieldEndCondition) for c in row):
                raise GeneralBoundaryConditionError("invalid end condition object")

    @property
    def m(self):
        return len(self.conditions[0])

    @classmethod
    def all_geometric_zero(cls,m):
        m=int(m)
        return cls(tuple(tuple(
            GeneralFieldEndCondition("GEOMETRIC",ScalarOrCircumferentialData(0.0))
            for _ in range(m)
        ) for _ in range(3)))

    @classmethod
    def all_free(cls,m):
        m=int(m)
        return cls(tuple(tuple(
            GeneralFieldEndCondition("NATURAL",ScalarOrCircumferentialData(0.0))
            for _ in range(m)
        ) for _ in range(3)))

    def all_natural(self):
        return all(c.kind=="NATURAL" for row in self.conditions for c in row)

    def all_geometric(self):
        return all(c.kind=="GEOMETRIC" for row in self.conditions for c in row)


@dataclass(frozen=True)
class GeneralCylinderEndBC:
    left: GeneralEndBoundarySet
    right: GeneralEndBoundarySet

    def __post_init__(self):
        if self.left.m!=self.right.m:
            raise GeneralBoundaryConditionError("left/right CUF m mismatch")

    @property
    def m(self):
        return self.left.m

    @classmethod
    def clamped_free(cls,m):
        return cls(
            GeneralEndBoundarySet.all_geometric_zero(m),
            GeneralEndBoundarySet.all_free(m),
        )

    @classmethod
    def free_clamped(cls,m):
        return cls(
            GeneralEndBoundarySet.all_free(m),
            GeneralEndBoundarySet.all_geometric_zero(m),
        )

    @classmethod
    def clamped_clamped(cls,m):
        return cls(
            GeneralEndBoundarySet.all_geometric_zero(m),
            GeneralEndBoundarySet.all_geometric_zero(m),
        )

    @classmethod
    def free_free(cls,m):
        return cls(
            GeneralEndBoundarySet.all_free(m),
            GeneralEndBoundarySet.all_free(m),
        )

    def check_rigid_mode_requirement(self):
        def restrains_linearized(cond):
            if cond.kind=="GEOMETRIC":
                return True
            if cond.kind=="ELASTIC":
                a=np.asarray(cond.stiffness.value,float)
                return bool(np.any(a>0.0))
            return False

        if self.left.all_natural() and self.right.all_natural():
            raise GeneralRigidModeHandlingRequired(
                "RIGID_MODE_HANDLING_REQUIRED: both cylinder ends are entirely natural; "
                "the supplied general CUF derivation requires separate identification "
                "and removal of represented rigid modes and does not prescribe its form."
            )

        def any_restrained(end, component, taus):
            return any(
                restrains_linearized(end.conditions[component][tau])
                for tau in taus
                if tau < end.m
            )

        # q_{u,0}=const is the axial translation.  Rotation about the cylinder
        # axis is v=omega*(R+z), hence it contains both v0=omega*R and
        # v1=omega.  Checking only v0 rejects valid CUF supports and does not
        # represent the actual rigid mode.
        if not (
            any_restrained(self.left,0,(0,))
            or any_restrained(self.right,0,(0,))
        ):
            raise GeneralRigidModeHandlingRequired(
                "RIGID_MODE_HANDLING_REQUIRED: CUF axial rigid translation u0=const is unrestrained; "
                "restrain u0 on at least one end, add a nonzero u0 spring, or use an explicit rigid-mode gauge"
            )
        if not (
            any_restrained(self.left,1,(0,1))
            or any_restrained(self.right,1,(0,1))
        ):
            raise GeneralRigidModeHandlingRequired(
                "RIGID_MODE_HANDLING_REQUIRED: CUF rotation about the axis "
                "(v0=omega*R, v1=omega) is unrestrained; restrain v0 or v1 "
                "on at least one end, add a nonzero spring, or use an explicit rigid-mode gauge"
            )


def _end_index_and_normal(side,nx):
    s=str(side).upper()
    if s=="LEFT": return 0,-1.0
    if s=="RIGHT": return nx-1,+1.0
    raise GeneralBoundaryConditionError("side must be LEFT or RIGHT")


def end_boundary_residual(q,resultants,end,side):
    """
    q shape (3,m,ns,nx), output (3,m,ns).
    """
    q=np.asarray(q,float)
    if q.ndim!=4 or q.shape[0]!=3:
        raise GeneralBoundaryConditionError("q must have shape (3,m,ns,nx)")
    m,ns,nx=q.shape[1],q.shape[2],q.shape[3]
    if end.m!=m:
        raise GeneralBoundaryConditionError("BC m does not match q m")
    if not isinstance(resultants,g7.GeneralResultants) or resultants.m!=m:
        raise GeneralBoundaryConditionError("resultants m mismatch")
    if resultants.surface_shape!=(ns,nx):
        raise GeneralBoundaryConditionError("resultants surface shape mismatch")

    ix,normal=_end_index_and_normal(side,nx)
    natural=(resultants.Au_x,resultants.Av_x,resultants.Aw_x)
    out=np.empty((3,m,ns),float)

    for a in range(3):
        for tau in range(m):
            bc=end.conditions[a][tau]
            prescribed=bc.data.array(ns)
            geom=q[a,tau,:,ix]
            nat=normal*natural[a][tau,:,ix]
            if bc.kind=="GEOMETRIC":
                out[a,tau]=geom-prescribed
            elif bc.kind=="NATURAL":
                out[a,tau]=nat-prescribed
            elif bc.kind=="ELASTIC":
                out[a,tau]=(nat+bc.stiffness_array(ns)*(geom-bc.reference_array(ns))
                            -prescribed)
            else:
                raise GeneralBoundaryConditionError(
                    f"unsupported CUF boundary kind: {bc.kind}"
                )
    return out


def full_boundary_residuals(q,resultants,bc):
    return {
        "left":end_boundary_residual(q,resultants,bc.left,"LEFT"),
        "right":end_boundary_residual(q,resultants,bc.right,"RIGHT"),
    }


def assemble_strong_residual_with_end_bcs(
    raw_residual,q,resultants,bc,*,reject_fully_free=True
):
    raw=np.asarray(raw_residual,float)
    q=np.asarray(q,float)
    if raw.shape!=q.shape or raw.ndim!=4 or raw.shape[0]!=3:
        raise GeneralBoundaryConditionError(
            "raw_residual and q must have shape (3,m,ns,nx)"
        )
    if raw.shape[1]!=bc.m:
        raise GeneralBoundaryConditionError("BC m mismatch")
    if reject_fully_free:
        bc.check_rigid_mode_requirement()

    b=full_boundary_residuals(q,resultants,bc)
    out=raw.copy()
    out[:,:,:,0]=b["left"]
    out[:,:,:,-1]=b["right"]
    return out


def homogeneous_geometric_x_active_mask(
    state,grid,bc,*,zero_tolerance=0.0
):
    """
    Homogeneous geometric q_{alpha,tau}=0 at an end is enforced exactly by
    X[alpha,tau,rank,end]=0 for every retained MVI term.
    """
    if not isinstance(state,g8.ProfileStateCUF):
        raise GeneralBoundaryConditionError("state must be ProfileStateCUF")
    state.validate_grid(grid)
    if state.m!=bc.m:
        raise GeneralBoundaryConditionError("state/BC m mismatch")

    mask=np.ones_like(state.X,dtype=bool)
    ns=grid.ns

    for side,end in (("LEFT",bc.left),("RIGHT",bc.right)):
        ix=0 if side=="LEFT" else grid.nx-1
        for a in range(3):
            for tau in range(state.m):
                cond=end.conditions[a][tau]
                if cond.kind!="GEOMETRIC":
                    continue
                data=cond.data.array(ns)
                if np.max(np.abs(data))>zero_tolerance:
                    raise GeneralBoundaryConditionError(
                        f"NONZERO_GEOMETRIC_BC_NOT_DECOMPOSED: "
                        f"{side} component={a} tau={tau}"
                    )
                mask[a,tau,:,ix]=False
    return mask


def enforce_homogeneous_geometric_x_profiles(state,grid,bc):
    mask=homogeneous_geometric_x_active_mask(state,grid,bc)
    state.X[~mask]=0.0
    return mask


def boundary_status(bc):
    if bc.left.all_natural() and bc.right.all_natural():
        return "RIGID_MODE_HANDLING_REQUIRED"
    return "BOUNDARY_SET_DEFINED"
