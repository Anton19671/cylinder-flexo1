#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUF-TE1 Stage 2B for a closed circular cylindrical shell.

STRICT SCOPE:
- thickness Gauss quadrature;
- exact cached thickness fields from equations (698)-(706);
- six TE1 residuals from equations (708),(710),(712),(714),(716),(718);
- fixed residual order [u0,u1,v0,v1,w0,w1];
- radial nominal face traction generalized loads from (641)-(643);
- immediate det(F)>0 rejection before material evaluation.

NO spatial stencil is invented here.
Dx and Ds are supplied by the accepted spatial/MVI discretization.
NO literature benchmark values are embedded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

import cylinder_te1_local_kernel as lk


class InvalidDeformationMapping(RuntimeError):
    pass


@dataclass(frozen=True)
class ThicknessQuadrature:
    z: np.ndarray
    w: np.ndarray

    def __post_init__(self):
        z = np.asarray(self.z, dtype=float)
        w = np.asarray(self.w, dtype=float)
        if z.ndim != 1 or w.ndim != 1 or z.shape != w.shape or z.size < 1:
            raise ValueError("z and w must be same non-empty 1D shape")
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(w)):
            raise ValueError("non-finite quadrature data")
        if np.any(w <= 0.0):
            raise ValueError("quadrature weights must be positive")
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "w", w)

    def integrate(self, values):
        v = np.asarray(values, dtype=float)
        if v.shape[0] != self.z.size:
            raise ValueError("quadrature leading dimension mismatch")
        return np.tensordot(self.w, v, axes=(0, 0))


def gauss_thickness_quadrature(h: float, n_gauss: int) -> ThicknessQuadrature:
    """
    Equations (692)-(693):
        z_g = h/2 * xi_g
        w_g^z = h/2 * omega_g
    """
    h = float(h)
    n_gauss = int(n_gauss)
    if h <= 0.0:
        raise ValueError("h must be positive")
    if n_gauss < 1:
        raise ValueError("n_gauss must be >=1")
    xi, omega = np.polynomial.legendre.leggauss(n_gauss)
    return ThicknessQuadrature(0.5*h*xi, 0.5*h*omega)


def layered_thickness_quadrature(
    h: float,
    layer_bounds: Sequence[float],
    n_gauss_per_layer: Sequence[int] | int
) -> ThicknessQuadrature:
    """
    Equations (694)-(697), with explicit sublayer mapping.
    layer_bounds must run strictly from -h/2 to +h/2.
    """
    h = float(h)
    if h <= 0.0:
        raise ValueError("h must be positive")
    b = np.asarray(layer_bounds, dtype=float)
    if b.ndim != 1 or b.size < 2:
        raise ValueError("layer_bounds must contain at least two values")
    if not np.all(np.diff(b) > 0.0):
        raise ValueError("layer_bounds must be strictly increasing")
    tol = 1e-12 * max(1.0, abs(h))
    if abs(b[0] + 0.5*h) > tol or abs(b[-1] - 0.5*h) > tol:
        raise ValueError("layer_bounds must start at -h/2 and end at +h/2")

    n_layers = b.size - 1
    if isinstance(n_gauss_per_layer, int):
        ns = [int(n_gauss_per_layer)] * n_layers
    else:
        ns = [int(x) for x in n_gauss_per_layer]
        if len(ns) != n_layers:
            raise ValueError("n_gauss_per_layer length mismatch")
    if any(n < 1 for n in ns):
        raise ValueError("all layer Gauss orders must be >=1")

    zs, ws = [], []
    for zl, zr, ng in zip(b[:-1], b[1:], ns):
        xi, omega = np.polynomial.legendre.leggauss(ng)
        z = 0.5*(zr + zl) + 0.5*(zr - zl)*xi
        w = 0.5*(zr - zl)*omega
        zs.append(z)
        ws.append(w)
    return ThicknessQuadrature(np.concatenate(zs), np.concatenate(ws))


CACHE_KEYS = (
    "Ixx0", "Ixx1",
    "Ixs0", "Ixs1",
    "Ixr1",
    "Isx0", "Isx1",
    "Iss0", "Iss1",
    "Isr1",
    "Irs0", "Irs1",
    "Irx0", "Irx1",
    "Irr1",
)


@dataclass(frozen=True)
class ThicknessCache:
    Ixx0: np.ndarray
    Ixx1: np.ndarray
    Ixs0: np.ndarray
    Ixs1: np.ndarray
    Ixr1: np.ndarray
    Isx0: np.ndarray
    Isx1: np.ndarray
    Iss0: np.ndarray
    Iss1: np.ndarray
    Isr1: np.ndarray
    Irs0: np.ndarray
    Irs1: np.ndarray
    Irx0: np.ndarray
    Irx1: np.ndarray
    Irr1: np.ndarray

    def as_dict(self):
        return {k: getattr(self, k) for k in CACHE_KEYS}


def _check_mid_surface_arrays(q, qx, qs):
    q = np.asarray(q, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qs = np.asarray(qs, dtype=float)
    if q.shape != qx.shape or q.shape != qs.shape:
        raise ValueError("q, qx, qs shape mismatch")
    if q.ndim < 1 or q.shape[0] != 6:
        raise ValueError("first dimension must be 6: [u0,u1,v0,v1,w0,w1]")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qx)) and np.all(np.isfinite(qs))):
        raise ValueError("non-finite q/qx/qs")
    return q, qx, qs


def _local_point_before_material(q6, qx6, qs6, z, R):
    """
    Build H and F, then reject J<=0 BEFORE S/P are evaluated.
    This follows G.4.3 exactly.
    """
    H, abc = lk.te1_gradient_quantities(q6, qx6, qs6, z, R)
    F = lk.deformation_gradient_from_abc(abc)
    J = float(np.linalg.det(F))
    if not np.isfinite(J) or J <= 0.0:
        raise InvalidDeformationMapping(
            f"INVALID_DEFORMATION_MAPPING detF={J:.17g}, z={float(z):.17g}"
        )
    return H, abc, F, J


def thickness_cache_te1(
    q,
    qx,
    qs,
    *,
    R: float,
    K0: float,
    GB,
    quadrature: ThicknessQuadrature,
):
    """
    Compute equations (698)-(706) at every midsurface point.

    q,qx,qs shape: (6, ...surface_shape...)
    GB may be scalar, surface-shaped, or (n_z, ...surface_shape...).

    Matrix P row/column order follows the document:
    rows x,s,r; columns x,s,r(thickness direction).

    J-17 audit optimization: the only Python loop is over the thickness
    quadrature.  Every midsurface point of one Gauss plane is evaluated in a
    single NumPy batch.  Equations (698)-(706) are unchanged.
    """
    q, qx, qs = _check_mid_surface_arrays(q, qx, qs)
    R = float(R)
    K0 = float(K0)
    if R <= 0.0 or K0 <= 0.0:
        raise ValueError("R and K0 must be positive")

    surf_shape = q.shape[1:]
    nz = quadrature.z.size

    GBa = np.asarray(GB, dtype=float)
    if GBa.ndim == 0:
        GB_mode = "scalar"
    elif GBa.shape == surf_shape:
        GB_mode = "surface"
    elif GBa.shape == (nz,) + surf_shape:
        GB_mode = "full"
    else:
        raise ValueError("GB must be scalar, surface-shaped, or (nz,*surface_shape)")
    if np.any(GBa <= 0.0) or not np.all(np.isfinite(GBa)):
        raise ValueError("GB must be finite and positive")

    acc = {k: np.zeros(surf_shape, dtype=float) for k in CACHE_KEYS}
    I3 = np.eye(3, dtype=float)

    # Complete midsurface component fields.
    _u0,u1,v0,v1,w0,w1 = q
    u0x,u1x,v0x,v1x,w0x,w1x = qx
    u0s,u1s,v0s,v1s,w0s,w1s = qs

    for g, (z_raw, wz_raw) in enumerate(zip(quadrature.z, quadrature.w)):
        z = float(z_raw)
        wz = float(wz_raw)
        H = 1.0 + z/R
        if H <= 0.0:
            raise ValueError("H=1+z/R must be positive")

        v = v0 + z*v1
        w = w0 + z*w1
        ux = u0x + z*u1x
        vx = v0x + z*v1x
        wx = w0x + z*w1x
        us = u0s + z*u1s
        vs = v0s + z*v1s
        ws = w0s + z*w1s

        F = np.empty(surf_shape + (3,3), dtype=float)
        F[...,0,0] = 1.0 + ux
        F[...,0,1] = us/H
        F[...,0,2] = u1
        F[...,1,0] = vx
        F[...,1,1] = 1.0 + (vs + w/R)/H
        F[...,1,2] = v1
        F[...,2,0] = wx
        F[...,2,1] = (ws - v/R)/H
        F[...,2,2] = 1.0 + w1

        detF = np.linalg.det(F)
        invalid = (~np.isfinite(detF)) | (detF <= 0.0)
        if np.any(invalid):
            flat = np.asarray(detF).reshape(-1)
            bad_flat = np.asarray(invalid).reshape(-1)
            Jbad = float(flat[np.flatnonzero(bad_flat)[0]])
            raise InvalidDeformationMapping(
                f"INVALID_DEFORMATION_MAPPING detF={Jbad:.17g}, z={z:.17g}"
            )

        E = 0.5*(np.swapaxes(F,-1,-2) @ F - I3)
        theta = np.trace(E, axis1=-2, axis2=-1)
        devE = E - (theta[...,None,None]/3.0)*I3

        if GB_mode == "scalar":
            GBg = float(GBa)
            S = K0*theta[...,None,None]*I3 + 2.0*GBg*devE
        elif GB_mode == "surface":
            S = K0*theta[...,None,None]*I3 + 2.0*GBa[...,None,None]*devE
        else:
            GBg = GBa[g]
            S = K0*theta[...,None,None]*I3 + 2.0*GBg[...,None,None]*devE

        P = F @ S

        # P rows/columns: x,s,r.
        Pxx, Pxs, Pxr = P[...,0,0], P[...,0,1], P[...,0,2]
        Psx, Pss, Psr = P[...,1,0], P[...,1,1], P[...,1,2]
        Prx, Prs, Prr = P[...,2,0], P[...,2,1], P[...,2,2]

        # Equations (698)-(706), unchanged and evaluated for the whole surface.
        acc["Ixx0"] += wz * (H * Pxx)
        acc["Ixx1"] += wz * (H * z * Pxx)

        acc["Ixs0"] += wz * Pxs
        acc["Ixs1"] += wz * (z * Pxs)

        acc["Ixr1"] += wz * (H * Pxr)

        acc["Isx0"] += wz * (H * Psx)
        acc["Isx1"] += wz * (H * z * Psx)

        acc["Iss0"] += wz * Pss
        acc["Iss1"] += wz * (z * Pss)

        acc["Isr1"] += wz * (H * Psr)

        acc["Irs0"] += wz * Prs
        acc["Irs1"] += wz * (z * Prs)

        acc["Irx0"] += wz * (H * Prx)
        acc["Irx1"] += wz * (H * z * Prx)

        acc["Irr1"] += wz * (H * Prr)

    return ThicknessCache(**acc)


def zero_generalized_load(surface_shape=()):
    return np.zeros((6,) + tuple(surface_shape), dtype=float)


def radial_nominal_face_load(qr, *, zq: float, R: float):
    """
    Equations (641)-(643):
      Hq = 1 + zq/R
      pu0=pu1=pv0=pv1=0
      pw0 = qr*Hq
      pw1 = qr*Hq*zq

    qr may be scalar or a midsurface array.
    """
    R = float(R)
    zq = float(zq)
    if R <= 0.0:
        raise ValueError("R must be positive")
    Hq = 1.0 + zq/R
    if Hq <= 0.0:
        raise ValueError("face Hq must be positive")
    qr = np.asarray(qr, dtype=float)
    if not np.all(np.isfinite(qr)):
        raise ValueError("non-finite qr")
    out = np.zeros((6,) + qr.shape, dtype=float)
    out[4] = qr * Hq
    out[5] = qr * Hq * zq
    return out


def sum_generalized_loads(*loads):
    if not loads:
        raise ValueError("at least one load required")
    arrs = [np.asarray(x, dtype=float) for x in loads]
    shape = arrs[0].shape
    if any(a.shape != shape for a in arrs):
        raise ValueError("load shape mismatch")
    return np.sum(arrs, axis=0)


DerivativeOperator = Callable[[np.ndarray], np.ndarray]


def te1_residuals(
    cache: ThicknessCache,
    *,
    Dx: DerivativeOperator,
    Ds: DerivativeOperator,
    R: float,
    generalized_load,
):
    """
    Exact cached forms:
      (708) Ru0
      (710) Ru1
      (712) Rv0
      (714) Rv1
      (716) Rw0
      (718) Rw1

    Fixed output order is equation (719):
      [Ru0,Ru1,Rv0,Rv1,Rw0,Rw1].

    Dx and Ds are deliberately injected. This module does NOT invent a
    spatial finite-difference or spectral operator.
    """
    R = float(R)
    if R <= 0.0:
        raise ValueError("R must be positive")

    c = cache
    p = np.asarray(generalized_load, dtype=float)
    surf_shape = np.asarray(c.Ixx0).shape
    if p.shape != (6,) + surf_shape:
        raise ValueError("generalized_load shape mismatch")

    def _D(op, A, name):
        out = np.asarray(op(np.asarray(A, dtype=float)), dtype=float)
        if out.shape != surf_shape:
            raise ValueError(f"{name} returned wrong shape")
        return out

    Ru0 = -_D(Dx, c.Ixx0, "Dx") - _D(Ds, c.Ixs0, "Ds") - p[0]
    Ru1 = -_D(Dx, c.Ixx1, "Dx") - _D(Ds, c.Ixs1, "Ds") + c.Ixr1 - p[1]

    Rv0 = -_D(Dx, c.Isx0, "Dx") - _D(Ds, c.Iss0, "Ds") - c.Irs0/R - p[2]
    Rv1 = -_D(Dx, c.Isx1, "Dx") - _D(Ds, c.Iss1, "Ds") + c.Isr1 - c.Irs1/R - p[3]

    Rw0 = -_D(Dx, c.Irx0, "Dx") - _D(Ds, c.Irs0, "Ds") + c.Iss0/R - p[4]
    Rw1 = -_D(Dx, c.Irx1, "Dx") - _D(Ds, c.Irs1, "Ds") + c.Irr1 + c.Iss1/R - p[5]

    return np.stack([Ru0, Ru1, Rv0, Rv1, Rw0, Rw1], axis=0)
