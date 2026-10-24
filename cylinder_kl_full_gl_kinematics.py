#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometrically exact classical Kirchhoff--Love cylinder kinematics.

The midsurface placement is described in the local orthonormal basis
``(e_x, e_theta, e_r)`` by the displacements ``(u,v,w)``.  This module uses
the exact first and second fundamental forms of the deformed midsurface.
No Donnell or von Karman curvature approximation is used.

The engineering membrane shear is ``gamma_xs = 2 E_xs`` and the curvature
convention is ``kappa = B - b``.  It therefore reduces to ``-w_xx`` for a
small transverse displacement of a flat generator while retaining every
cylindrical displacement and finite-rotation term.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class KLFullGLKinematicsError(RuntimeError):
    pass


@dataclass(frozen=True)
class KLFullGLState:
    U: np.ndarray
    Ux: np.ndarray
    Us: np.ndarray
    Uxx: np.ndarray
    Uss: np.ndarray
    Uxs: np.ndarray
    g_x: np.ndarray
    g_s: np.ndarray
    normal: np.ndarray
    area_ratio: np.ndarray
    y_xx: np.ndarray
    y_ss: np.ndarray
    y_xs: np.ndarray
    eps_x: np.ndarray
    eps_s: np.ndarray
    gamma_xs: np.ndarray
    kappa_x: np.ndarray
    kappa_s: np.ndarray
    kappa_xs: np.ndarray


@dataclass(frozen=True)
class KLFullGLVariation:
    d_eps_x: np.ndarray
    d_eps_s: np.ndarray
    d_gamma_xs: np.ndarray
    d_kappa_x: np.ndarray
    d_kappa_s: np.ndarray
    d_kappa_xs: np.ndarray
    d_normal: np.ndarray


@dataclass(frozen=True)
class KLFullGLInternalCoefficients:
    """Coefficients of dU, dU_x, dU_s, dU_xx, dU_ss and dU_xs."""
    A0: np.ndarray
    Ax: np.ndarray
    As: np.ndarray
    Axx: np.ndarray
    Ass: np.ndarray
    Axs: np.ndarray


def _field3(value, name):
    a = np.asarray(value, float)
    if a.ndim < 2 or a.shape[0] != 3:
        raise ValueError(f"{name} must have shape (3,...)")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _same_fields(*named):
    arrays = [_field3(value, name) for name, value in named]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("KL displacement derivative shape mismatch")
    return arrays


def _dot(a, b):
    return np.sum(a*b, axis=0)


def _cross(a, b):
    return np.moveaxis(
        np.cross(np.moveaxis(a, 0, -1), np.moveaxis(b, 0, -1)),
        -1, 0,
    )


def state_from_derivatives(U, Ux, Us, Uxx, Uss, Uxs, *, R, jacobian_tol=1e-12):
    """Build exact midsurface metric, normal and curvature tensors."""
    U, Ux, Us, Uxx, Uss, Uxs = _same_fields(
        ("U", U), ("Ux", Ux), ("Us", Us),
        ("Uxx", Uxx), ("Uss", Uss), ("Uxs", Uxs),
    )
    R = float(R)
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be finite and positive")
    jacobian_tol = float(jacobian_tol)
    if not np.isfinite(jacobian_tol) or jacobian_tol <= 0.0:
        raise ValueError("jacobian_tol must be finite and positive")

    u, v, w = U
    ux, vx, wx = Ux
    us, vs, ws = Us
    uxx, vxx, wxx = Uxx
    uss, vss, wss = Uss
    uxs, vxs, wxs = Uxs

    g_x = np.stack((1.0 + ux, vx, wx), axis=0)
    g_s = np.stack((us, 1.0 + vs + w/R, ws - v/R), axis=0)

    cross = _cross(g_x, g_s)
    area_ratio = np.sqrt(_dot(cross, cross))
    if np.any(~np.isfinite(area_ratio)) or np.any(area_ratio <= jacobian_tol):
        minimum = float(np.nanmin(area_ratio))
        raise KLFullGLKinematicsError(
            f"degenerate deformed midsurface: min area ratio={minimum:.17g}"
        )
    normal = cross/area_ratio[None, ...]

    # Physical second derivatives in the rotating cylindrical basis.
    y_xx = np.stack((uxx, vxx, wxx), axis=0)
    y_xs = np.stack((
        uxs,
        vxs + wx/R,
        wxs - vx/R,
    ), axis=0)
    y_ss = np.stack((
        uss,
        vss + 2.0*ws/R - v/(R*R),
        wss - 2.0*vs/R - 1.0/R - w/(R*R),
    ), axis=0)

    # Exact Green--Lagrange membrane strains in the reference orthonormal
    # (x,s) basis. s is midsurface arc length, hence A=identity there.
    eps_x = 0.5*(_dot(g_x, g_x) - 1.0)
    eps_s = 0.5*(_dot(g_s, g_s) - 1.0)
    gamma_xs = _dot(g_x, g_s)

    b_x = _dot(normal, y_xx)
    b_s = _dot(normal, y_ss)
    b_xs = _dot(normal, y_xs)
    # Reference second fundamental form: B_xx=0, B_ss=-1/R, B_xs=0.
    kappa_x = -b_x
    kappa_s = -1.0/R - b_s
    kappa_xs = -b_xs

    return KLFullGLState(
        U, Ux, Us, Uxx, Uss, Uxs,
        g_x, g_s, normal, area_ratio,
        y_xx, y_ss, y_xs,
        eps_x, eps_s, gamma_xs,
        kappa_x, kappa_s, kappa_xs,
    )


def field_kinematics(U, ops, *, jacobian_tol=1e-12):
    """Build the exact state using the solver's canonical FD operators."""
    U = _field3(U, "U")
    expected = (3, ops.grid.ns, ops.grid.nx)
    if U.shape != expected:
        raise ValueError(f"U must have shape {expected}")
    Ux = np.stack([ops.Dx(q) for q in U])
    Us = np.stack([ops.Ds(q) for q in U])
    Uxx = np.stack([ops.Dxx(q) for q in U])
    Uss = np.stack([ops.Dss(q) for q in U])
    Uxs = np.stack([ops.Ds(ops.Dx(q)) for q in U])
    return state_from_derivatives(
        U, Ux, Us, Uxx, Uss, Uxs,
        R=ops.grid.radius, jacobian_tol=jacobian_tol,
    )


def directional_variation(
    state, dU, dUx, dUs, dUxx, dUss, dUxs, *, R
):
    """Exact directional derivative of all membrane and curvature measures."""
    if not isinstance(state, KLFullGLState):
        raise ValueError("state must be KLFullGLState")
    dU, dUx, dUs, dUxx, dUss, dUxs = _same_fields(
        ("dU", dU), ("dUx", dUx), ("dUs", dUs),
        ("dUxx", dUxx), ("dUss", dUss), ("dUxs", dUxs),
    )
    if dU.shape != state.U.shape:
        raise ValueError("variation/state shape mismatch")
    R = float(R)
    du, dv, dw = dU
    dux, dvx, dwx = dUx
    dus, dvs, dws = dUs
    duxx, dvxx, dwxx = dUxx
    duss, dvss, dwss = dUss
    duxs, dvxs, dwxs = dUxs

    dg_x = np.stack((dux, dvx, dwx), axis=0)
    dg_s = np.stack((dus, dvs + dw/R, dws - dv/R), axis=0)
    dcross = _cross(dg_x, state.g_s) + _cross(state.g_x, dg_s)
    normal_component = _dot(state.normal, dcross)
    dnormal = (
        dcross - state.normal*normal_component[None, ...]
    )/state.area_ratio[None, ...]

    dy_xx = np.stack((duxx, dvxx, dwxx), axis=0)
    dy_xs = np.stack((
        duxs,
        dvxs + dwx/R,
        dwxs - dvx/R,
    ), axis=0)
    dy_ss = np.stack((
        duss,
        dvss + 2.0*dws/R - dv/(R*R),
        dwss - 2.0*dvs/R - dw/(R*R),
    ), axis=0)

    d_eps_x = _dot(state.g_x, dg_x)
    d_eps_s = _dot(state.g_s, dg_s)
    d_gamma = _dot(dg_x, state.g_s) + _dot(state.g_x, dg_s)
    d_kappa_x = -(_dot(dnormal, state.y_xx) + _dot(state.normal, dy_xx))
    d_kappa_s = -(_dot(dnormal, state.y_ss) + _dot(state.normal, dy_ss))
    d_kappa_xs = -(_dot(dnormal, state.y_xs) + _dot(state.normal, dy_xs))
    return KLFullGLVariation(
        d_eps_x, d_eps_s, d_gamma,
        d_kappa_x, d_kappa_s, d_kappa_xs, dnormal,
    )


def through_thickness_strains(state, z):
    """Classical Koiter KL strain distribution using exact surface measures."""
    if not isinstance(state, KLFullGLState):
        raise ValueError("state must be KLFullGLState")
    z = float(z)
    return (
        state.eps_x + z*state.kappa_x,
        state.eps_s + z*state.kappa_s,
        state.gamma_xs + 2.0*z*state.kappa_xs,
    )


def _internal_density(resultants, variation):
    return (
        resultants.Nx*variation.d_eps_x
        + resultants.Ns*variation.d_eps_s
        + resultants.Nxs*variation.d_gamma_xs
        + resultants.Mx*variation.d_kappa_x
        + resultants.Ms*variation.d_kappa_s
        + 2.0*resultants.Mxs*variation.d_kappa_xs
    )


def internal_coefficients(state, resultants, *, R):
    """Return pointwise coefficients of the exact first variation.

    The six derivative slots are treated as independent at each surface point.
    This provides a compact reverse-mode representation used by both the full
    nodal residual and exact natural edge-force evaluation.
    """
    shape=state.U.shape
    slots=[]
    for active in range(6):
        coefficient=np.empty(shape,float)
        for component in range(3):
            variations=[np.zeros(shape,float) for _ in range(6)]
            variations[active][component]=1.0
            dv=directional_variation(state,*variations,R=R)
            coefficient[component]=_internal_density(resultants,dv)
        slots.append(coefficient)
    return KLFullGLInternalCoefficients(*slots)


def surface_weights(grid):
    wx=np.full(grid.nx,grid.dx,float)
    wx[0]*=0.5; wx[-1]*=0.5
    return np.full((grid.ns,1),grid.ds,float)*wx[None,:]


def variational_strong_residual(state, resultants, ops, generalized_load):
    """Full GL nodal Euler residual obtained from the exact first variation.

    The canonical derivative matrices are differentiated by their algebraic
    transposes. Division by the nodal area weights returns force-per-area rows
    compatible with the existing strong-form scaling and load convention.
    Boundary rows are subsequently replaced by the selected physical BCs.
    """
    p=np.asarray(generalized_load,float)
    if p.shape!=state.U.shape or not np.all(np.isfinite(p)):
        raise ValueError(f"generalized_load must have shape {state.U.shape}")
    c=internal_coefficients(state,resultants,R=ops.grid.radius)
    W=surface_weights(ops.grid)
    residual=np.empty_like(state.U)
    for a in range(3):
        grad=W*c.A0[a]
        grad+=(W*c.Ax[a])@ops.D1x
        grad+=ops.D1s.T@(W*c.As[a])
        grad+=(W*c.Axx[a])@ops.D2x
        grad+=ops.D2s.T@(W*c.Ass[a])
        grad+=ops.D1s.T@(W*c.Axs[a])@ops.D1x
        residual[a]=grad/W-p[a]
    return residual,c


def exact_end_force_and_moment(state, resultants, ops, *, side, coefficients=None):
    """Exact total force vector and normal bending moment at an x-end."""
    side=str(side).upper()
    if side=="LEFT":
        ix=0
        outward=-1.0
    elif side=="RIGHT":
        ix=-1
        outward=1.0
    else:
        raise ValueError("side must be LEFT or RIGHT")
    c=coefficients or internal_coefficients(state,resultants,R=ops.grid.radius)
    force=np.empty_like(c.Ax)
    for a in range(3):
        force[a]=c.Ax[a]-ops.Dx(c.Axx[a])-ops.Ds(c.Axs[a])
    traction=outward*force[:, :, ix]
    # Axx is parallel to the deformed normal and equals -Mx*n exactly.
    moment=np.sum(c.Axx[:, :, ix]*state.normal[:, :, ix],axis=0)
    return traction,moment


__all__ = (
    "KLFullGLKinematicsError", "KLFullGLState", "KLFullGLVariation",
    "KLFullGLInternalCoefficients",
    "state_from_derivatives", "field_kinematics", "directional_variation",
    "through_thickness_strains", "internal_coefficients", "surface_weights",
    "variational_strong_residual", "exact_end_force_and_moment",
)
