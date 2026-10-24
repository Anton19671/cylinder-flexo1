#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 9A: independent full-grid KL finite-difference solver,
LINEAR ELASTIC only.

There is NO separated representation:
    no rank
    no X/Y profiles
    no Y->X halfsteps

Unknowns are the complete nodal displacement fields:
    u(s_j,x_i), v(s_j,x_i), w(s_j,x_i)

Homogeneous essential x-end BC are eliminated exactly with the already
verified linear x-constraint maps C q = 0:
    q_a(s,:) = c_a(s,:) B_a^T

This is only boundary-coordinate elimination.  Each circumferential row
has its own independent free coefficients, so no separation is imposed.

The discrete equilibrium residual is the first variation of the same
Kirchhoff-Love energy, with spatial derivatives evaluated by the verified
7-point finite-difference operators.

For LINEAR ELASTIC the residual is affine:
    R(c) = K c + R(0)
so K is assembled exactly by unit-coordinate residual differences and
the global square system is solved once with np.linalg.solve.

No Newton, no MVI, no FEM, no rank reduction.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8b4_weak_projected as kw


class KLFDMError(RuntimeError):
    pass


class KLFDMLinearSystemSingular(KLFDMError):
    pass


@dataclass(frozen=True)
class KLFullGridCoordinates:
    maps: object
    ns: int

    @property
    def component_sizes(self):
        return tuple(self.ns*m.n_free for m in self.maps.maps)

    @property
    def ndof(self):
        return int(sum(self.component_sizes))


@dataclass(frozen=True)
class KLFullGridVariation:
    dU: np.ndarray
    dUx: np.ndarray
    dUs: np.ndarray
    dUxx: np.ndarray
    dUss: np.ndarray
    dUxs: np.ndarray
    label: tuple


@dataclass
class KLFDMLinearResult:
    U: np.ndarray
    coefficients: tuple
    converged: bool
    status: str
    ndof: int
    residual_inf: float
    K_symmetry_relative: float


def make_fullgrid_coordinates(maps,grid):
    return KLFullGridCoordinates(maps=maps,ns=grid.ns)


def zero_coefficients(coords):
    return tuple(
        np.zeros((coords.ns,m.n_free),float)
        for m in coords.maps.maps
    )


def copy_coefficients(coeffs):
    return tuple(np.asarray(x,float).copy() for x in coeffs)


def coefficients_to_vector(coeffs,coords):
    if len(coeffs)!=3:
        raise ValueError("need three component coefficient arrays")
    blocks=[]
    for a,(c,m) in enumerate(zip(coeffs,coords.maps.maps)):
        q=np.asarray(c,float)
        expected=(coords.ns,m.n_free)
        if q.shape!=expected:
            raise ValueError(
                f"coefficients[{a}] shape {q.shape}, expected {expected}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("non-finite full-grid coefficients")
        blocks.append(q.ravel())
    return np.concatenate(blocks)


def vector_to_coefficients(vec,coords):
    x=np.asarray(vec,float)
    if x.shape!=(coords.ndof,):
        raise ValueError(f"coefficient vector must have shape ({coords.ndof},)")
    out=[]
    pos=0
    for m in coords.maps.maps:
        n=coords.ns*m.n_free
        out.append(x[pos:pos+n].reshape(coords.ns,m.n_free).copy())
        pos+=n
    return tuple(out)


def reconstruct_fullgrid(coeffs,coords,grid):
    U=np.empty((3,grid.ns,grid.nx),float)
    for a,(c,m) in enumerate(zip(coeffs,coords.maps.maps)):
        q=np.asarray(c,float)
        if q.shape!=(grid.ns,m.n_free):
            raise ValueError("full-grid coefficient shape mismatch")
        U[a]=q@m.B.T
    return U


def fullgrid_variations(coords,ops):
    """
    One variation for every independent full-grid coefficient.
    Ordering: component -> circumferential node -> free x coordinate.
    """
    ns,nx=ops.grid.ns,ops.grid.nx
    D1x=ops.D1x
    D2x=D1x@D1x
    D1s=ops.D1s
    D2s=D1s@D1s

    out=[]
    for a,m in enumerate(coords.maps.maps):
        for js in range(ns):
            y=np.zeros(ns,float)
            y[js]=1.0
            y1=D1s[:,js]
            y2=D2s[:,js]

            for jx in range(m.n_free):
                x=m.B[:,jx]
                x1=D1x@x
                x2=D2x@x

                arrays=[np.zeros((3,ns,nx),float) for _ in range(6)]
                dU,dUx,dUs,dUxx,dUss,dUxs=arrays

                dU[a]=np.outer(y,x)
                dUx[a]=np.outer(y,x1)
                dUs[a]=np.outer(y1,x)
                dUxx[a]=np.outer(y,x2)
                dUss[a]=np.outer(y2,x)
                dUxs[a]=np.outer(y1,x1)

                out.append(
                    KLFullGridVariation(
                        dU,dUx,dUs,dUxx,dUss,dUxs,
                        (a,js,jx)
                    )
                )
    return tuple(out)


def _check_labels(variations,coords):
    if len(variations)!=coords.ndof:
        raise KLFDMError("variation count/ndof mismatch")


def linear_elastic_fullgrid_residual(
    coeffs,coords,ops,bc,variations,
    *,E,nu,h,generalized_load,resultants_method="ANALYTIC",quadrature=None
):
    _check_labels(variations,coords)
    U=reconstruct_fullgrid(coeffs,coords,ops.grid)
    kin=kl.field_kinematics(U,ops,geometry="LINEAR")
    const=kl.KLElasticConstants(E,nu,h)

    method=str(resultants_method).upper()
    if method=="ANALYTIC":
        nm=kl.analytic_elastic_resultants(kin.state,const)
    elif method=="QUADRATURE":
        if quadrature is None:
            raise ValueError("quadrature required")
        nm=kl.quadrature_elastic_resultants(
            kin.state,const,quadrature
        )
    else:
        raise ValueError("resultants_method must be ANALYTIC or QUADRATURE")

    r,labels=kw.projected_residual_from_variations(
        kin,nm,variations,ops,bc,
        geometry="LINEAR",
        generalized_load=generalized_load,
    )
    return r,labels,U,kin,nm


def assemble_linear_system(
    coords,ops,bc,*,E,nu,h,generalized_load,
    resultants_method="ANALYTIC",quadrature=None
):
    variations=fullgrid_variations(coords,ops)
    c0=zero_coefficients(coords)

    r0,labels,_,_,_=linear_elastic_fullgrid_residual(
        c0,coords,ops,bc,variations,
        E=E,nu=nu,h=h,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )

    n=coords.ndof
    K=np.empty((n,n),float)

    # LINEAR ELASTIC residual is affine. A unit coordinate exactly extracts
    # the corresponding stiffness column; no finite-difference step is used.
    for j in range(n):
        e=np.zeros(n,float)
        e[j]=1.0
        cj=vector_to_coefficients(e,coords)
        rj,labels_j,_,_,_=linear_elastic_fullgrid_residual(
            cj,coords,ops,bc,variations,
            E=E,nu=nu,h=h,
            generalized_load=generalized_load,
            resultants_method=resultants_method,
            quadrature=quadrature,
        )
        if labels_j!=labels:
            raise KLFDMError("residual ordering changed during assembly")
        K[:,j]=rj-r0

    return K,r0,variations,labels


def solve_linear_elastic_fdm(
    ops,maps,bc,*,E,nu,h,generalized_load,
    resultants_method="ANALYTIC",quadrature=None
):
    bc.check_rigid_mode_requirement()
    coords=make_fullgrid_coordinates(maps,ops.grid)
    K,r0,variations,_labels=assemble_linear_system(
        coords,ops,bc,E=E,nu=nu,h=h,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )

    try:
        cvec=np.linalg.solve(K,-r0)
    except np.linalg.LinAlgError as exc:
        raise KLFDMLinearSystemSingular(
            "SINGULAR_FULLGRID_FDM_LINEAR_SYSTEM"
        ) from exc

    coeffs=vector_to_coefficients(cvec,coords)
    rfin,_,U,_,_=linear_elastic_fullgrid_residual(
        coeffs,coords,ops,bc,variations,
        E=E,nu=nu,h=h,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )

    sym=float(
        np.linalg.norm(K-K.T)/max(np.linalg.norm(K),1e-300)
    )
    return KLFDMLinearResult(
        U=U,
        coefficients=coeffs,
        converged=True,
        status="CONVERGED_FULLGRID_FDM_LINEAR",
        ndof=coords.ndof,
        residual_inf=float(np.max(np.abs(rfin))),
        K_symmetry_relative=sym,
    )
