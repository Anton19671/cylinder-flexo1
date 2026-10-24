#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8B-1: end-boundary mechanics for displacement-only Kirchhoff-Love
closed circular cylindrical shell.

x-end outward normal:
    n_x=-1 left, +1 right.

Membrane natural quantities:
    T_u = n_x Nx
    T_v = n_x Nxs

Transverse membrane flux:
    A_x = 0                                      LINEAR
    A_x = Nx*w_x + Nxs*w_s                      VON_KARMAN

After integration by parts, using periodicity along the circular end:
    T_w = n_x*(A_x + Mx,x + 2*Mxs,s)

Normal slope and its conjugate moment:
    theta_n = n_x*w_x
    T_theta = -Mx

Free end:
    T_u=T_v=T_w=T_theta=0.

Homogeneous clamped end:
    u=v=w=theta_n=0.

This stage defines and verifies the BC mechanics only.
It does not yet impose the fourth-order w conditions in the MVI system.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_kl_full_gl_kinematics as fullgl

class KLRigidModeHandlingRequired(RuntimeError):
    pass

@dataclass(frozen=True)
class KLBoundaryData:
    value: object
    def array(self,ns):
        a=np.asarray(self.value,float)
        if a.ndim==0:
            return np.full(ns,float(a))
        if a.shape!=(ns,):
            raise ValueError(f"boundary data must be scalar or shape ({ns},)")
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite boundary data")
        return a.copy()

@dataclass(frozen=True)
class KLBoundaryCondition:
    kind: str
    data: KLBoundaryData
    stiffness: KLBoundaryData | None = None
    reference: KLBoundaryData = KLBoundaryData(0.0)
    def __post_init__(self):
        k=str(self.kind).upper()
        if k not in ("GEOMETRIC","NATURAL","ELASTIC"):
            raise ValueError("kind must be GEOMETRIC, NATURAL or ELASTIC")
        object.__setattr__(self,"kind",k)
        if k=="ELASTIC":
            if self.stiffness is None:
                raise ValueError("ELASTIC boundary condition requires stiffness")
            a=np.asarray(self.stiffness.value,float)
            if not np.all(np.isfinite(a)) or np.any(a<0.0):
                raise ValueError("ELASTIC boundary stiffness must be finite and nonnegative")

    def stiffness_array(self,ns):
        if self.kind!="ELASTIC":
            return np.zeros(int(ns),float)
        return self.stiffness.array(ns)

    def reference_array(self,ns):
        return self.reference.array(ns)

@dataclass(frozen=True)
class KLEndBoundarySet:
    u: KLBoundaryCondition
    v: KLBoundaryCondition
    w: KLBoundaryCondition
    slope: KLBoundaryCondition

    @classmethod
    def homogeneous_clamped(cls):
        z=KLBoundaryData(0.0)
        return cls(
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("GEOMETRIC",z),
        )

    @classmethod
    def free(cls):
        z=KLBoundaryData(0.0)
        return cls(
            KLBoundaryCondition("NATURAL",z),
            KLBoundaryCondition("NATURAL",z),
            KLBoundaryCondition("NATURAL",z),
            KLBoundaryCondition("NATURAL",z),
        )

    @classmethod
    def simply_supported_s1(cls):
        """S1: Nx=0, v=0, w=0, Mx=0 (axially movable diaphragm)."""
        z=KLBoundaryData(0.0)
        return cls(
            KLBoundaryCondition("NATURAL",z),
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("NATURAL",z),
        )

    @classmethod
    def simply_supported_s2(cls):
        """S2: u=0, Nxs=0, w=0, Mx=0 (circumferentially movable)."""
        z=KLBoundaryData(0.0)
        return cls(
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("NATURAL",z),
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("NATURAL",z),
        )

    @classmethod
    def simply_supported_s3(cls):
        """S3: u=0, v=0, w=0, Mx=0 (immovable simple support)."""
        z=KLBoundaryData(0.0)
        return cls(
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("NATURAL",z),
        )

    @classmethod
    def simply_supported_s4(cls):
        """S4: Nx=0, Nxs=0, w=0, Mx=0 (fully movable in-plane)."""
        z=KLBoundaryData(0.0)
        return cls(
            KLBoundaryCondition("NATURAL",z),
            KLBoundaryCondition("NATURAL",z),
            KLBoundaryCondition("GEOMETRIC",z),
            KLBoundaryCondition("NATURAL",z),
        )

@dataclass(frozen=True)
class KLCylinderEndBC:
    left: KLEndBoundarySet
    right: KLEndBoundarySet

    @classmethod
    def clamped_clamped(cls):
        return cls(KLEndBoundarySet.homogeneous_clamped(),
                   KLEndBoundarySet.homogeneous_clamped())

    @classmethod
    def clamped_free(cls):
        return cls(KLEndBoundarySet.homogeneous_clamped(),
                   KLEndBoundarySet.free())

    @classmethod
    def free_clamped(cls):
        return cls(KLEndBoundarySet.free(),
                   KLEndBoundarySet.homogeneous_clamped())

    @classmethod
    def free_free(cls):
        return cls(KLEndBoundarySet.free(),KLEndBoundarySet.free())

    def check_rigid_mode_requirement(self):
        def all_nat(end):
            return all(getattr(end,k).kind=="NATURAL"
                       for k in ("u","v","w","slope"))

        def restrains_linearized(cond):
            # Rigid-mode uniqueness concerns the homogeneous perturbation of
            # the boundary equation.  GEOMETRIC fixes that perturbation.
            # ELASTIC fixes it only when the spring stiffness is nonzero; an
            # ELASTIC row with k=0 is mechanically NATURAL.
            if cond.kind=="GEOMETRIC":
                return True
            if cond.kind=="ELASTIC":
                a=np.asarray(cond.stiffness.value,float)
                return bool(np.any(a>0.0))
            return False

        if all_nat(self.left) and all_nat(self.right):
            raise KLRigidModeHandlingRequired(
                "RIGID_MODE_HANDLING_REQUIRED: both KL cylinder ends are fully natural"
            )

        # Two exact zero-energy KL rigid modes are especially easy to leave
        # behind with sliding/simple supports: u=const (axial translation)
        # and v=const (rotation about the cylinder axis).  A singular/near-
        # singular solve must not be reported as an ordinary converged CASE.
        if not (restrains_linearized(self.left.u) or restrains_linearized(self.right.u)):
            raise KLRigidModeHandlingRequired(
                "RIGID_MODE_HANDLING_REQUIRED: axial rigid translation u=const is unrestrained; "
                "restrain u on at least one end, add a nonzero u spring, or use an explicit rigid-mode gauge"
            )
        if not (restrains_linearized(self.left.v) or restrains_linearized(self.right.v)):
            raise KLRigidModeHandlingRequired(
                "RIGID_MODE_HANDLING_REQUIRED: circumferential rigid rotation v=const is unrestrained; "
                "restrain v on at least one end, add a nonzero v spring, or use an explicit rigid-mode gauge"
            )

@dataclass(frozen=True)
class KLEndMechanicalQuantities:
    T_u: np.ndarray
    T_v: np.ndarray
    T_w: np.ndarray
    T_theta: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    theta_n: np.ndarray

def _edge_index(side):
    s=str(side).upper()
    if s=="LEFT": return 0,-1.0
    if s=="RIGHT": return -1,+1.0
    raise ValueError("side must be LEFT or RIGHT")

def transverse_membrane_flux(kin,resultants,geometry,*,ops=None):
    mode=str(geometry).upper()
    if mode=="LINEAR":
        return np.zeros_like(resultants.Nx)
    if mode=="VON_KARMAN":
        return resultants.Nx*kin.wx + resultants.Nxs*kin.ws
    if mode=="GREEN_LAGRANGE":
        if ops is None:
            raise ValueError("ops is required for GREEN_LAGRANGE membrane flux")
        b=kin.ws-kin.U[1]/ops.grid.radius
        return resultants.Nx*kin.wx+resultants.Nxs*b
    raise ValueError("geometry must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")

def end_mechanical_quantities(kin,resultants,ops,*,geometry,side):
    ix,n=_edge_index(side)
    mode=str(geometry).upper()
    if mode=="GREEN_LAGRANGE":
        if kin.full_gl is None:
            raise ValueError("full Green-Lagrange KL kinematics is missing")
        traction,moment=fullgl.exact_end_force_and_moment(
            kin.full_gl,resultants,ops,side=side
        )
        return KLEndMechanicalQuantities(
            T_u=np.asarray(traction[0],float),
            T_v=np.asarray(traction[1],float),
            T_w=np.asarray(traction[2],float),
            T_theta=np.asarray(moment,float),
            u=np.asarray(kin.U[0,:,ix],float),
            v=np.asarray(kin.U[1,:,ix],float),
            w=np.asarray(kin.U[2,:,ix],float),
            theta_n=np.asarray(-n*kin.full_gl.normal[0,:,ix],float),
        )

    flux_u=resultants.Nx
    flux_v=resultants.Nxs
    Ax=transverse_membrane_flux(kin,resultants,geometry)
    Mx_x=ops.Dx(resultants.Mx)
    Mxs_s=ops.Ds(resultants.Mxs)

    return KLEndMechanicalQuantities(
        T_u=np.asarray(n*flux_u[:,ix],float),
        T_v=np.asarray(n*flux_v[:,ix],float),
        T_w=np.asarray(n*(Ax[:,ix]+Mx_x[:,ix]+2.0*Mxs_s[:,ix]),float),
        T_theta=np.asarray(-resultants.Mx[:,ix],float),
        u=np.asarray(kin.U[0,:,ix],float),
        v=np.asarray(kin.U[1,:,ix],float),
        w=np.asarray(kin.U[2,:,ix],float),
        theta_n=np.asarray(n*kin.wx[:,ix],float),
    )

def boundary_residual(end_bc,quantities,ns):
    out=[]
    pairs=(
        (end_bc.u,quantities.u,quantities.T_u),
        (end_bc.v,quantities.v,quantities.T_v),
        (end_bc.w,quantities.w,quantities.T_w),
        (end_bc.slope,quantities.theta_n,quantities.T_theta),
    )
    for bc,g,nat in pairs:
        target=bc.data.array(ns)
        if bc.kind=="GEOMETRIC":
            residual=np.asarray(g,float)-target
        elif bc.kind=="NATURAL":
            residual=np.asarray(nat,float)-target
        else:
            residual=(np.asarray(nat,float)
                      +bc.stiffness_array(ns)*(np.asarray(g,float)-bc.reference_array(ns))
                      -target)
        out.append(np.asarray(residual,float))
    return tuple(out)

def both_end_boundary_residuals(kin,resultants,ops,*,geometry,bc):
    bc.check_rigid_mode_requirement()
    ql=end_mechanical_quantities(kin,resultants,ops,geometry=geometry,side="LEFT")
    qr=end_mechanical_quantities(kin,resultants,ops,geometry=geometry,side="RIGHT")
    return (
        boundary_residual(bc.left,ql,ops.grid.ns),
        boundary_residual(bc.right,qr,ops.grid.ns),
        ql,qr,
    )
