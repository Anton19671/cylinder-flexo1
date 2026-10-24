#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent second-order FD operators for the full-grid MKR/FDM branch.

The MVI branch keeps its existing high-order profile operators.  This module
exists so MKR/FDM and MVI do not share the same differentiation matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class MKRFD2OperatorError(ValueError):
    pass


def derivative_matrix_nonperiodic_fd2(x, order=1):
    x=np.asarray(x,float)
    if x.ndim!=1 or x.size<4:
        raise MKRFD2OperatorError("uniform x grid with at least 4 nodes required")
    d=np.diff(x); h=float(d[0])
    if h<=0.0 or not np.allclose(d,h,rtol=1e-12,atol=1e-14*max(1.0,abs(h),float(np.max(np.abs(x))))):
        raise MKRFD2OperatorError("uniform increasing x grid required")
    n=x.size; D=np.zeros((n,n),float)
    if order==1:
        c=1.0/(2.0*h)
        for i in range(1,n-1):
            D[i,i-1]=-c; D[i,i+1]=c
        D[0,0]=-3*c; D[0,1]=4*c; D[0,2]=-c
        D[-1,-1]=3*c; D[-1,-2]=-4*c; D[-1,-3]=c
        return D
    if order==2:
        c=1.0/(h*h)
        for i in range(1,n-1):
            D[i,i-1]=c; D[i,i]=-2*c; D[i,i+1]=c
        D[0,0]=2*c; D[0,1]=-5*c; D[0,2]=4*c; D[0,3]=-c
        D[-1,-1]=2*c; D[-1,-2]=-5*c; D[-1,-3]=4*c; D[-1,-4]=-c
        return D
    raise MKRFD2OperatorError("only derivative orders 1 and 2 are supported")


def derivative_matrix_periodic_fd2(length,n,order=1):
    length=float(length); n=int(n)
    if length<=0.0 or n<3:
        raise MKRFD2OperatorError("positive periodic length and n>=3 required")
    h=length/n; D=np.zeros((n,n),float)
    if order==1:
        c=1.0/(2.0*h)
        for i in range(n):
            D[i,(i-1)%n]=-c; D[i,(i+1)%n]=c
        return D
    if order==2:
        c=1.0/(h*h)
        for i in range(n):
            D[i,(i-1)%n]=c; D[i,i]=-2*c; D[i,(i+1)%n]=c
        return D
    raise MKRFD2OperatorError("only derivative orders 1 and 2 are supported")


@dataclass(frozen=True)
class MKRSpatialOperators:
    grid: object
    D1x: np.ndarray
    D1s: np.ndarray
    D2x: np.ndarray
    D2s: np.ndarray
    scheme_name: str = "MKR_FDM_INDEPENDENT_FD2"
    formal_order: int = 2

    @classmethod
    def build(cls,grid):
        return cls(
            grid,
            derivative_matrix_nonperiodic_fd2(grid.x,1),
            derivative_matrix_periodic_fd2(grid.circumference,grid.ns,1),
            derivative_matrix_nonperiodic_fd2(grid.x,2),
            derivative_matrix_periodic_fd2(grid.circumference,grid.ns,2),
        )

    def Dx(self,A):
        A=np.asarray(A,float)
        if A.shape!=(self.grid.ns,self.grid.nx): raise ValueError('Dx shape')
        return A@self.D1x.T

    def Ds(self,A):
        A=np.asarray(A,float)
        if A.shape!=(self.grid.ns,self.grid.nx): raise ValueError('Ds shape')
        return self.D1s@A

    def Dxx(self,A):
        A=np.asarray(A,float)
        if A.shape!=(self.grid.ns,self.grid.nx): raise ValueError('Dxx shape')
        return A@self.D2x.T

    def Dss(self,A):
        A=np.asarray(A,float)
        if A.shape!=(self.grid.ns,self.grid.nx): raise ValueError('Dss shape')
        return self.D2s@A
