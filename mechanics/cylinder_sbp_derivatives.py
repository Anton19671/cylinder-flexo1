#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagonal-norm SBP(2,1) / periodic odd-grid reference derivatives.

The exact semidiscrete axial and circumferential Green identities are

  Dx.T @ Wx + Wx @ Dx = e_R e_R.T - e_L e_L.T = Bx,
  Ds.T @ Ws + Ws @ Ds = 0,

where Wx = diag(wx), Ws = diag(ws). This is a physical-coordinate
summation-by-parts pair, not a high-order production solver.

Axial first derivative: second order in the interior, first order at
the end nodes. Circumferential derivative: second-order centered and
periodic. In the scalar periodic nodal space a REAL skew-adjoint
derivative with D_s 1=0 necessarily has an extra zero eigenvalue for
even node count. Therefore this implementation *requires odd ns>=5*,
so the periodic constant-mode gauge is the only null direction.

Higher surface derivatives are implemented by the same Dx**a Ds**b
matrix products as the variational functional. Repeated SBP identities
give an exact DISCRETE Green formula for independent mixed jets.
The scheme does not claim pointwise physical boundary stress
convergence or sparse production-scale performance.
"""
from __future__ import annotations

import math
import numpy as np

SBP_OPERATOR_SCHEME="SBP21_PERIODIC_ODD"


def _positive_dimension(value,label,minimum):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    if value<minimum:
        raise ValueError(f"{label} must be >= {minimum}")


def _positive_length(value,label):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(f"{label} must be positive finite SI length")
    try:
        out=float(value)
    except (TypeError,ValueError) as exc:
        raise ValueError(f"{label} must be positive finite SI length") from exc
    if not math.isfinite(out) or out<=0:
        raise ValueError(f"{label} must be positive finite SI length")
    return out


def axial_sbp21_gradient(n,length_m):
    """Return (Dx,wx) with diagonal-norm first-derivative SBP identity."""
    _positive_dimension(n,"axial node count",4)
    L=_positive_length(length_m,"axial length")
    h=L/(n-1)
    Dx=np.zeros((n,n),dtype=float)
    Dx[0,0]=-1./h
    Dx[0,1]=1./h
    Dx[-1,-2]=-1./h
    Dx[-1,-1]=1./h
    for i in range(1,n-1):
        Dx[i,i-1]=-.5/h
        Dx[i,i+1]=.5/h
    wx=np.full(n,h)
    wx[0]=wx[-1]=.5*h
    return Dx,wx


def periodic_sbp_odd_gradient(n,perimeter_m):
    """Return (Ds,ws); reject even grids rather than hiding checkerboards."""
    _positive_dimension(n,"periodic node count",5)
    if n%2==0:
        raise ValueError(
            "SBP periodic centered gradient requires odd ns; "
            "even ns introduces a checkerboard derivative null mode")
    L=_positive_length(perimeter_m,"periodic perimeter")
    h=L/n
    Ds=np.zeros((n,n),dtype=float)
    for i in range(n):
        Ds[i,(i-1)%n]=-.5/h
        Ds[i,(i+1)%n]=.5/h
    return Ds,np.full(n,h)


def periodic_sbp_odd_fourier_gradient(n, perimeter_m):
    """Exact trigonometric derivative on the odd periodic nodal space.

    A dense reference operator, NOT a sparse SBP or a higher-order axial
    closure. It uses all supported Fourier harmonics without the near-Nyquist
    weakness of the centered periodic difference. For uniform weights it is
    real, skew-adjoint and annihilates constants to roundoff.
    """
    _positive_dimension(n, "periodic node count", 5)
    if n % 2 == 0:
        raise ValueError("Fourier periodic derivative requires odd ns >= 5")
    L = _positive_length(perimeter_m, "periodic perimeter")
    frequency = 2. * np.pi * np.fft.fftfreq(n, d=L/n)
    identity = np.eye(n)
    derivative = np.fft.ifft(
        (1j * frequency)[:, None] * np.fft.fft(identity, axis=0),
        axis=0).real
    # Remove only floating-point antisymmetry noise; this is an exact
    # coordinate identity for odd-n trigonometric interpolation.
    derivative = .5 * (derivative - derivative.T)
    return derivative, np.full(n, L/n)



def _axial_sbp42_rows(n):
    """Fourth-order interior, second-order boundary diagonal-norm SBP D1.

    Strand's standard SBP(4,2) closure. D^T H + H D = eR*eR^T-eL*eL^T.
    For a Dirichlet interval, D^T H D couples adjacent node parities,
    unlike the two-grid-decoupled centered SBP(2,1) elastic operator.
    """
    _positive_dimension(n, "SBP42 axial node count", 9)
    closure = (
        ((0, -24./17), (1, 59./34), (2, -4./17), (3, -3./34)),
        ((0, -.5), (2, .5)),
        ((0, 4./43), (1, -59./86), (3, 59./86), (4, -4./43)),
        ((0, 3./98), (2, -59./98), (4, 32./49), (5, -4./49)),
    )
    for i in range(n):
        if i < 4:
            row = closure[i]
        elif i >= n - 4:
            row = tuple((n-1-j, -a) for j,a in closure[n-1-i])
        else:
            row = ((i-2, 1./12), (i-1, -2./3),
                   (i+1, 2./3), (i+2, -1./12))
        yield i, row


def _axial_sbp42_weights(n, h):
    weights = np.ones(n, float)
    weights[:4] = (17./48, 59./48, 43./48, 49./48)
    weights[-4:] = weights[:4][::-1]
    return weights*h


def axial_sbp42_gradient(n, length_m):
    """Non-checkerboard SBP42 axial D and its exact diagonal-norm weights."""
    _positive_dimension(n, "SBP42 axial node count", 9)
    length = _positive_length(length_m, "axial length")
    h = length/(n-1)
    d = np.zeros((n,n), float)
    for i,row in _axial_sbp42_rows(n):
        for j,v in row:
            d[i,j] = v/h
    return d, _axial_sbp42_weights(n,h)


def axial_sbp42_gradient_sparse(n, length_m):
    """O(n)-storage canonical CSR version of the same SBP42 axial D."""
    _positive_dimension(n, "SBP42 axial node count", 9)
    length = _positive_length(length_m, "axial length")
    from scipy import sparse
    h = length/(n-1)
    rows,cols,vals = [],[],[]
    for i,row in _axial_sbp42_rows(n):
        for j,v in row:
            rows.append(i);cols.append(j);vals.append(v/h)
    d = sparse.csr_matrix((vals,(rows,cols)),shape=(n,n))
    return d, _axial_sbp42_weights(n,h)


def periodic_sbp_odd_fourier_gradient_sparse(n, perimeter_m):
    """Canonical odd Fourier D_s as CSR (small/medium circumferential grids).

    This is dense in angular bandwidth; keep an explicit safety ceiling.
    It gives exact resolved periodic harmonics, not a new constitutive law.
    """
    _positive_dimension(n,"periodic node count",5)
    if n > 129:
        raise ValueError("FOURIER_CSR_ANGULAR_NODE_SAFETY_LIMIT: ns <= 129")
    from scipy import sparse
    dense, weights = periodic_sbp_odd_fourier_gradient(n,perimeter_m)
    d = sparse.csr_matrix(dense)
    d.eliminate_zeros()
    return d, weights


def axial_sbp21_gradient_sparse(n,length_m):
    """CSR SBP(2,1) axial derivative with O(n) nonzeros and diagonal norm."""
    _positive_dimension(n,"axial node count",4)
    L=_positive_length(length_m,"axial length")
    from scipy import sparse
    h=L/(n-1)
    lower=np.full(n-1,-.5/h)
    upper=np.full(n-1,.5/h)
    main=np.zeros(n)
    lower[-1]=-1./h
    upper[0]=1./h
    main[0]=-1./h
    main[-1]=1./h
    Dx=sparse.diags((lower,main,upper),(-1,0,1),
                    shape=(n,n),format="csr")
    wx=np.full(n,h)
    wx[[0,-1]]*=.5
    return Dx,wx


def periodic_sbp_odd_gradient_sparse(n,perimeter_m):
    """CSR centered odd-periodic derivative; no even-grid Nyquist null mode."""
    _positive_dimension(n,"periodic node count",5)
    if n%2==0:
        raise ValueError(
            "SBP periodic centered gradient requires odd ns; "
            "even ns introduces a checkerboard derivative null mode")
    L=_positive_length(perimeter_m,"periodic perimeter")
    from scipy import sparse
    h=L/n
    i=np.arange(n,dtype=int)
    rows=np.concatenate((i,i))
    cols=np.concatenate(((i-1)%n,(i+1)%n))
    values=np.concatenate((np.full(n,-.5/h),np.full(n,.5/h)))
    Ds=sparse.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr()
    return Ds,np.full(n,h)


def green_identity_defects(Dx,Ds,wx,ws):
    """Exact weighted Green-matrix defects without densifying CSR operators."""
    from cylinder_surface_linear_algebra import is_sparse_matrix
    sparse_x=is_sparse_matrix(Dx)
    sparse_s=is_sparse_matrix(Ds)
    if sparse_x!=sparse_s:
        raise ValueError("SBP Dx and Ds must both use dense or both CSR operators")
    dx=Dx if sparse_x else np.asarray(Dx,float)
    ds=Ds if sparse_s else np.asarray(Ds,float)
    wx=np.asarray(wx,float)
    ws=np.asarray(ws,float)
    if (dx.ndim!=2 or ds.ndim!=2 or
        dx.shape[0]!=dx.shape[1] or ds.shape[0]!=ds.shape[1]):
        raise ValueError("SBP derivative matrices must be square")
    if wx.shape!=(dx.shape[0],) or ws.shape!=(ds.shape[0],):
        raise ValueError("SBP quadrature weight shapes do not match derivatives")
    entries=(dx.data,ds.data) if sparse_x else (dx,ds)
    if (not all(np.isfinite(a).all() for a in (*entries,wx,ws)) or
        np.any(wx<=0) or np.any(ws<=0)):
        raise ValueError("SBP requires finite matrices and strictly positive weights")
    if sparse_x:
        from scipy import sparse
        qx=(sparse.diags(wx)@dx).tocsr()
        qs=(sparse.diags(ws)@ds).tocsr()
        bx=sparse.csr_matrix(
            ([-1.,1.],([0,len(wx)-1],[0,len(wx)-1])),
            shape=dx.shape)
        ax=(qx+qx.T-bx).tocsr()
        az=(qs+qs.T).tocsr()
        peak=lambda a:float(np.max(np.abs(a.data),initial=0.))
        axial=peak(ax);periodic=peak(az)
        scale=max(1.,peak(qx),peak(qs))
    else:
        bx=np.zeros_like(dx)
        bx[0,0]=-1.
        bx[-1,-1]=1.
        qx=wx[:,None]*dx
        qs=ws[:,None]*ds
        axial=float(np.max(np.abs(qx+qx.T-bx)))
        periodic=float(np.max(np.abs(qs+qs.T)))
        scale=max(1.,float(np.max(np.abs(qx))),
                  float(np.max(np.abs(qs))))
    return {"axial":axial,"periodic":periodic,
            "relative_max":max(axial,periodic)/scale}


def require_green_identity(Dx,Ds,wx,ws,*,relative_tolerance=3.e-12):
    """Fail closed when a purported SBP pair breaks either Green identity."""
    if isinstance(relative_tolerance,(bool,np.bool_)):
        raise ValueError("SBP tolerance must be finite positive")
    tol=float(relative_tolerance)
    if not math.isfinite(tol) or tol<=0:
        raise ValueError("SBP tolerance must be finite positive")
    report=green_identity_defects(Dx,Ds,wx,ws)
    if report["relative_max"]>tol:
        raise ValueError(
            "SBP_GREEN_IDENTITY_VIOLATED: "
            f"relative defect {report['relative_max']:.4e} > {tol:.4e}")
    return report


__all__=("SBP_OPERATOR_SCHEME","axial_sbp21_gradient",
         "axial_sbp42_gradient","axial_sbp42_gradient_sparse",
         "periodic_sbp_odd_fourier_gradient_sparse",
         "periodic_sbp_odd_gradient",
         "axial_sbp21_gradient_sparse","periodic_sbp_odd_gradient_sparse",
         "green_identity_defects",
         "require_green_identity")
