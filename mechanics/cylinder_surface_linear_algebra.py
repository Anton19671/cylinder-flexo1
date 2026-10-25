#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared tensor-product surface actions for dense or SciPy CSR derivatives.

The model-specific 3D Green--Lagrange constitutive routines never see
a matrix representation. They receive *identical* physical surface jets:
    D_x^a q (D_s^b)^T.
Its discrete weak adjoint is
    (D_x^a)^T (W A_ab) D_s^b.
Sparse CSR mode applies 1D stencils to each component instead of
materializing any Kronecker or nodal Jacobian matrix.

Dense mode retains the original einsum order for regression compatibility.
SciPy is only required if a sparse matrix is actually supplied.
"""
from __future__ import annotations

import numpy as np


def is_sparse_matrix(matrix):
    return hasattr(matrix, "tocsr") and hasattr(matrix, "getrow")


def check_matrix(matrix, size, label):
    if is_sparse_matrix(matrix):
        if matrix.shape != (size, size) or not np.isfinite(matrix.data).all():
            raise ValueError(f"{label}: finite square CSR differentiation required")
        return matrix.tocsr()
    a=np.asarray(matrix,float)
    if a.shape!=(size,size) or not np.isfinite(a).all():
        raise ValueError(f"{label}: finite square differentiation required")
    return a


def dense_row(matrix, row):
    """Return a single 1D row; sparse matrices may not be coerced globally."""
    if is_sparse_matrix(matrix):
        return np.asarray(matrix.getrow(row).toarray()).ravel()
    return np.asarray(matrix[row],float).ravel()


def apply_axial(field,dx):
    """D_x @ field, independently on each surface component."""
    arr=np.asarray(field,float)
    if arr.ndim!=3 or arr.shape[1]!=dx.shape[1]:
        raise ValueError("surface axial field shape mismatch")
    if not is_sparse_matrix(dx):
        return np.einsum("ij,cjk->cik",dx,arr,optimize=True)
    out=np.empty((arr.shape[0],dx.shape[0],arr.shape[2]),float)
    for c in range(len(arr)):
        out[c]=dx@arr[c]
    return out


def apply_periodic(field,ds):
    """field @ D_s.T; ds rows are the physical theta/s derivative."""
    arr=np.asarray(field,float)
    if arr.ndim!=3 or arr.shape[2]!=ds.shape[1]:
        raise ValueError("surface periodic field shape mismatch")
    if not is_sparse_matrix(ds):
        return np.einsum("ij,ckj->cki",ds,arr,optimize=True)
    out=np.empty((arr.shape[0],arr.shape[1],ds.shape[0]),float)
    for c in range(len(arr)):
        out[c]=(ds@arr[c].T).T
    return out


def apply_biaxial(field,dx,ds):
    """D_x @ field @ D_s.T for dense or sparse derivative powers."""
    arr=np.asarray(field,float)
    if (arr.ndim!=3 or arr.shape[1]!=dx.shape[1] or
            arr.shape[2]!=ds.shape[1]):
        raise ValueError("tensor-product surface derivative shape mismatch")
    if not is_sparse_matrix(dx) and not is_sparse_matrix(ds):
        return np.einsum("ij,cjk,lk->cil",dx,arr,ds,optimize=True)
    return apply_periodic(apply_axial(arr,dx),ds)


def apply_weighted_adjoint(coeff,weights,dx,ds):
    """Exactly the discrete Gateaux adjoint (D_x)^T W A (D_s)."""
    field=np.asarray(coeff,float)
    w=np.asarray(weights,float)
    if field.ndim!=3 or w.shape!=field.shape[1:]:
        raise ValueError("weighted surface adjoint shape mismatch")
    if not is_sparse_matrix(dx) and not is_sparse_matrix(ds):
        return np.einsum("ij,cjk,kl->cil",dx.T,field*w[None,...],
                         ds,optimize=True)
    return apply_biaxial(field*w[None,...],dx.T,ds.T)


def operator_power(matrix,order):
    """Power of a 1D derivative without dense materialization."""
    if isinstance(order,bool) or not isinstance(order,int) or order<0:
        raise ValueError("nonnegative integer derivative order required")
    if is_sparse_matrix(matrix):
        from scipy import sparse
        result=sparse.eye(matrix.shape[0],format="csr")
        for _ in range(order):
            result=(result@matrix).tocsr()
            result.eliminate_zeros()
        return result
    return np.linalg.matrix_power(np.asarray(matrix,float),order)


__all__=("is_sparse_matrix","check_matrix","dense_row",
         "apply_axial","apply_periodic","apply_biaxial",
         "apply_weighted_adjoint","operator_power")
