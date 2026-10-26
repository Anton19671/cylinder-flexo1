"""Sparse, fixed-rank X/Y variations and trace projector for MVI.

Node order: (component,x,s). Partners must be orthonormal; no SVD,
pseudo-inverse, full-grid nodal basis, or physical-tangent regularization.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.linalg import qr
from scipy.sparse.csgraph import connected_components

@dataclass(frozen=True)
class _Block:
    columns: np.ndarray
    q: np.ndarray

class SeparatedMVIDirections:
    def __init__(self,x,y,axis):
        x=np.asarray(x,float);y=np.asarray(y,float)
        if (x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=y.shape[:2]
            or not 1<=x.shape[1]<=8
            or x.shape[1]>min(x.shape[2],y.shape[2])
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("MVI_DIRECTIONS_INVALID_FIXED_RANK_SHAPES")
        axis=str(axis).upper()
        if axis not in ("X","Y"):
            raise ValueError("MVI_DIRECTIONS_EXPECT_X_OR_Y")
        self.x=x.copy();self.y=y.copy();self.axis=axis
        self.nc,self.rank,self.nx=x.shape;self.ns=y.shape[2]
        self.n=self.nc*self.nx*self.ns
        self.parameter_shape=(self.nc,self.rank,
                              self.nx if axis=="X" else self.ns)
        self.n_parameters=int(np.prod(self.parameter_shape))
        partner=y if axis=="X" else x
        for c in range(self.nc):
            gram=partner[c]@partner[c].T
            if not np.allclose(gram,np.eye(self.rank),atol=1.e-10,rtol=1.e-10):
                raise ValueError("MVI_FROZEN_PARTNERS_REQUIRE_ORTHONORMAL_QR")

    def push(self,parameters):
        a=np.asarray(parameters,float)
        if a.shape!=(self.n_parameters,) or not np.isfinite(a).all():
            raise ValueError("MVI_DIRECTION_PARAMETER_SHAPE")
        a=a.reshape(self.parameter_shape)
        if self.axis=="X":
            return np.einsum("crx,crs->cxs",a,self.y,optimize=True).ravel()
        return np.einsum("crx,crs->cxs",self.x,a,optimize=True).ravel()

    def pull(self,nodal):
        v=np.asarray(nodal,float)
        if v.shape!=(self.n,) or not np.isfinite(v).all():
            raise ValueError("MVI_DIRECTION_NODAL_SHAPE")
        field=v.reshape((self.nc,self.nx,self.ns))
        if self.axis=="X":
            return np.einsum("cxs,crs->crx",field,self.y,optimize=True).ravel()
        return np.einsum("crx,cxs->crs",self.x,field,optimize=True).ravel()

    def sparse_matrix(self):
        nc,r,nx,ns=self.nc,self.rank,self.nx,self.ns
        c=np.arange(nc)[:,None,None,None]
        rr=np.arange(r)[None,:,None,None]
        ix=np.arange(nx)[None,None,:,None]
        js=np.arange(ns)[None,None,None,:]
        shape=(nc,r,nx,ns)
        rows=np.broadcast_to((c*nx+ix)*ns+js,shape).ravel()
        if self.axis=="X":
            cols=np.broadcast_to((c*r+rr)*nx+ix,shape).ravel()
            data=np.broadcast_to(self.y[:,:,None,:],shape).ravel()
        else:
            cols=np.broadcast_to((c*r+rr)*ns+js,shape).ravel()
            data=np.broadcast_to(self.x[:,:,:,None],shape).ravel()
        T=sparse.coo_matrix((data,(rows,cols)),shape=(
            self.n,self.n_parameters)).tocsr()
        T.eliminate_zeros()
        return T

class SparseHalfstepTraceProjector:
    """Orthogonal projection onto ker(C*T), using small independent QR blocks.

    Rows with many nonzeros (e.g. global electric gauge) are projected after
    local blocks. Fail closed if a connected block exceeds the stated cap.
    """
    def __init__(self,trace_matrix,directions,*,rank_tolerance=1.e-11,
                 max_dense_block_entries=2_000_000,
                 global_row_nnz=256,max_constraint_rows=25_000):
        if not isinstance(directions,SeparatedMVIDirections):
            raise TypeError("directions must be SeparatedMVIDirections")
        C=sparse.csr_matrix(trace_matrix,dtype=float)
        if (C.ndim!=2 or C.shape[1]!=directions.n or
                C.shape[0]>max_constraint_rows or
                not np.isfinite(C.data).all() or
                not 0<rank_tolerance<1):
            raise ValueError("MVI_HALFSTEP_INVALID_PHYSICAL_TRACE_MATRIX")
        self.direction=directions
        A=(C@directions.sparse_matrix()).tocsr()
        A.eliminate_zeros()
        self._a=A
        self.n_parameters=directions.n_parameters
        self._blocks=[];self._global_q=[]
        lengths=np.diff(A.indptr)
        global_rows=np.flatnonzero(lengths>global_row_nnz)
        local_rows=np.flatnonzero((lengths>0)&(lengths<=global_row_nnz))
        local=A[local_rows]
        if local.shape[0] and local.nnz:
            nr,nc=local.shape
            incidence=local.copy()
            incidence.data=np.ones_like(incidence.data)
            graph=sparse.bmat([
                [sparse.csr_matrix((nr,nr)),incidence],
                [incidence.T,sparse.csr_matrix((nc,nc))]
            ],format="csr")
            _,labels=connected_components(graph,directed=False)
            for group in np.unique(labels[:nr]):
                rows=np.flatnonzero(labels[:nr]==group)
                cols=np.flatnonzero(labels[nr:]==group)
                if not len(cols):
                    continue
                if len(rows)*len(cols)>max_dense_block_entries:
                    raise ValueError("MVI_HALFSTEP_CONSTRAINT_BLOCK_TOO_LARGE")
                block=local[rows][:,cols].toarray()
                q,r,_=qr(block.T,mode="economic",pivoting=True,
                         check_finite=False)
                diagonals=np.abs(np.diag(r))
                if len(diagonals):
                    rank=int(np.count_nonzero(
                        diagonals>rank_tolerance*np.max(diagonals)))
                    if rank:
                        self._blocks.append(_Block(cols,q[:,:rank].copy()))
        for row_index in global_rows:
            row=A.getrow(int(row_index))
            g=self._project_local(row.toarray().reshape(-1))
            for v in self._global_q:
                g-=v*np.dot(v,g)
            norm=float(np.linalg.norm(g))
            if norm>rank_tolerance*max(1.,float(np.linalg.norm(row.data))):
                self._global_q.append(g/norm)
        self.active_constraint_rows=int(np.count_nonzero(lengths))
        self.local_qr_blocks=len(self._blocks)
        self.global_constraints=len(self._global_q)

    def _project_local(self,v):
        out=v.copy()
        for block in self._blocks:
            idx=block.columns
            out[idx]-=block.q@(block.q.T@out[idx])
        return out

    def project(self,values):
        v=np.asarray(values,float)
        if v.shape!=(self.n_parameters,) or not np.isfinite(v).all():
            raise ValueError("MVI_HALFSTEP_BAD_PARAMETER_VECTOR")
        out=self._project_local(v)
        for g in self._global_q:
            out-=g*float(g@out)
        residual=self._a@out
        bound=max(1.,float(np.max(np.abs(self._a.data),initial=0))
                  *float(np.linalg.norm(v)))
        if np.max(np.abs(residual),initial=0)>1.e-8*bound:
            raise ValueError("MVI_HALFSTEP_TRACE_PROJECTOR_INACCURATE")
        return out

    def projected_nodal(self,parameters):
        return self.direction.push(self.project(parameters))

__all__=("SeparatedMVIDirections","SparseHalfstepTraceProjector")
