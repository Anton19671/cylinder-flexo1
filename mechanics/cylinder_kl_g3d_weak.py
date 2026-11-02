#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometrically exact Kirchhoff-Love normal director, full covariant G3D.

Independent surface unknowns: u0,v0,w0, plus electric potential coefficients
psi_0..psi_P. Reference coordinates: x and s=R*theta, z normal outward.
Physical reference basis: [axial, tangential, outward radial].

r(x,s,z)=r_mid(x,s)+z*n(x,s), where n is the normalized current
mid-surface normal, NOT the linearized rotation (-w_x,v/R-w_s,1).
Consequently F_z=n, E_xz=E_sz=E_zz=0 even for finite rotations.

The local normal and its first/second surface derivatives are computed
by exact truncated 2D Taylor jet arithmetic; third surface derivatives of
(u,v,w) enter G3D strain gradients. Mechanical variations are taken by
complex-step differentiation of that analytic geometry ONLY, then contracted
with analytic 3D material derivatives. No finite-difference sensitivity of
the material energy and no copying of CUF or FSDT director equations.

Only LOCAL thickness enthalpy/Gateaux coefficients are implemented.
A valid global KL G3D PDE, microfree/higher edge traces, independent MVI/FDM
and full dynamic mass operator are separate stages; no run is enabled here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

from cuf_g3d_weak import te_basis, gauss_thickness
from cylinder_kinematic_models import normalized_model, strain_jets, virtual_strain_jets
from flexo_gradient_g3d import (
    G3DMaterial, covariant_strain_gradients, reference_electric_field,
    enthalpy_parts, enthalpy_derivatives,
)

FIELDS = ("q","qx","qs","qxx","qxs","qss",
          "qxxx","qxxs","qxss","qsss")
ELEC_FIELDS = ("psi","psix","psis")
_MON = ((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))
_MON_INDEX = dict(zip(_MON,range(6)))
_DER_INDEX = dict(zip(
    ((0,0),(1,0),(0,1),(2,0),(1,1),(0,2),
     (3,0),(2,1),(1,2),(0,3)), FIELDS
))
_FACT = (1.,1.,1.,2.,1.,2.)
_EYE = np.eye(3)
_E_R = np.array((0.,0.,1.))


@dataclass(frozen=True)
class KLSurfaceJets:
    q: np.ndarray
    qx: np.ndarray
    qs: np.ndarray
    qxx: np.ndarray
    qxs: np.ndarray
    qss: np.ndarray
    qxxx: np.ndarray
    qxxs: np.ndarray
    qxss: np.ndarray
    qsss: np.ndarray
    psi: np.ndarray
    psix: np.ndarray
    psis: np.ndarray

    def __post_init__(self):
        for name in FIELDS:
            a=np.asarray(getattr(self,name),float)
            if a.shape!=(3,) or not np.isfinite(a).all():
                raise ValueError(f"{name} must be a finite three-vector")
            object.__setattr__(self,name,a.copy())
        p=np.asarray(self.psi,float)
        if p.ndim!=1 or not 1<=len(p)<=5:
            raise ValueError("psi must contain P+1 coefficients, P=0..4")
        for name in ELEC_FIELDS:
            a=np.asarray(getattr(self,name),float)
            if a.shape!=p.shape or not np.isfinite(a).all():
                raise ValueError(f"{name} must have shape {p.shape}")
            object.__setattr__(self,name,a.copy())

    @property
    def potential_order(self):
        return self.psi.size-1

    def replaced(self,**kwargs):
        names=(*FIELDS,*ELEC_FIELDS)
        if set(kwargs)-set(names):
            raise ValueError("unknown surface jet")
        return KLSurfaceJets(**{n:kwargs.get(n,getattr(self,n)) for n in names})


@dataclass(frozen=True)
class KLPoint:
    z_m: float
    H: float
    deformation_gradient: np.ndarray
    strain6: np.ndarray
    gradient6x3: np.ndarray
    electric3: np.ndarray
    determinant: float


@dataclass(frozen=True)
class KLWeak:
    energy_j_per_m2: float
    mechanical: np.ndarray # (3,10)
    electrical: np.ndarray # (P+1,3)
    gauss_points: int


def _geometry(R,h,z=None):
    R,h=float(R),float(h)
    if not all(math.isfinite(v) for v in (R,h)) or h<=0 or R<=h/2:
        raise ValueError("KL requires R > h/2 > 0")
    if z is not None and (not math.isfinite(float(z)) or abs(z)>h/2+1e-12*h):
        raise ValueError("z outside reference thickness")
    return R,h


def _mul(a,b):
    """Exact degree-2 bivariate Taylor product (six coefficient basis)."""
    a=np.asarray(a);b=np.asarray(b)
    return np.array((
        a[0]*b[0],
        a[0]*b[1]+a[1]*b[0],
        a[0]*b[2]+a[2]*b[0],
        a[0]*b[3]+a[1]*b[1]+a[3]*b[0],
        a[0]*b[4]+a[1]*b[2]+a[2]*b[1]+a[4]*b[0],
        a[0]*b[5]+a[2]*b[2]+a[5]*b[0],
    ),dtype=np.result_type(a,b))

def _dot(a,b):
    return sum((_mul(a[i],b[i]) for i in range(3)),np.zeros(6,dtype=np.result_type(a,b)))


def _cross(a,b):
    return np.stack((
        _mul(a[1],b[2])-_mul(a[2],b[1]),
        _mul(a[2],b[0])-_mul(a[0],b[2]),
        _mul(a[0],b[1])-_mul(a[1],b[0]),
    ))


def _poly_constant(number,dtype):
    out=np.zeros(6,dtype=dtype)
    out[0]=number
    return out


def _shift(data,component,dx,ds):
    return np.array([
        data[_DER_INDEX[(dx+a,ds+b)]][component] /fact
        for (a,b),fact in zip(_MON,_FACT)
    ])


def _normal_jets(data,R):
    """Finite-rotation director and derivatives through second order in x,s."""
    u=np.stack([_shift(data,i,1,0) for i in range(3)])
    v=np.stack([_shift(data,i,0,1) for i in range(3)])
    q=np.stack([_shift(data,i,0,0) for i in range(3)])
    dtype=np.result_type(u,v,q)
    one=_poly_constant(1.,dtype)
    A=np.stack((u[0]+one,u[1],u[2]))
    B=np.stack((v[0],one+v[1]+q[2]/R,v[2]-q[1]/R))
    N=_cross(A,B)
    norm2=_dot(N,N)
    if norm2[0].real<=1e-20 or not np.isfinite(norm2[0]):
        raise ValueError("INVALID_KL_MIDSURFACE_NORMAL: degenerate metric")
    delta=norm2/norm2[0]
    delta[0]-=1
    inv=(one-.5*delta+.375*_mul(delta,delta))/np.sqrt(norm2[0])
    normal=np.stack([_mul(n,inv) for n in N])
    if not np.isfinite(normal).all():
        raise ValueError("INVALID_KL_NORMAL")
    # Taylor-coefficient derivatives: factor 2 for pure second derivatives.
    return (normal[:,0],normal[:,1],normal[:,2],
            2*normal[:,3],normal[:,4],2*normal[:,5])


def _normal_jets_small_rotation(data,R):
    """Linear KL normal and its first/second material-coordinate derivatives."""
    a=data
    zero=lambda: np.zeros(3,dtype=np.result_type(a["q"]))
    def vec(ax,az):
        return np.array((ax,az,0.),dtype=np.result_type(a["q"]))
    n=np.array((-a["qx"][2],a["q"][1]/R-a["qs"][2],1.),
               dtype=np.result_type(a["q"]))
    nx=vec(-a["qxx"][2],a["qx"][1]/R-a["qxs"][2])
    ns=vec(-a["qxs"][2],a["qs"][1]/R-a["qss"][2])
    nxx=vec(-a["qxxx"][2],a["qxx"][1]/R-a["qxxs"][2])
    nxs=vec(-a["qxxs"][2],a["qxs"][1]/R-a["qxss"][2])
    nss=vec(-a["qxss"][2],a["qss"][1]/R-a["qsss"][2])
    return n,nx,ns,nxx,nxs,nss


def _kl_normal_gl_linearization(data,R):
    """Base 2D Taylor jets for the exact Gateaux normal variation."""
    u=np.stack([_shift(data,i,1,0) for i in range(3)])
    v=np.stack([_shift(data,i,0,1) for i in range(3)])
    q=np.stack([_shift(data,i,0,0) for i in range(3)])
    one=_poly_constant(1.,np.result_type(u,v,q))
    A=np.stack((u[0]+one,u[1],u[2]))
    B=np.stack((v[0],one+v[1]+q[2]/R,v[2]-q[1]/R))
    N=_cross(A,B)
    squared=_dot(N,N)
    if squared[0].real<=1e-20 or not np.isfinite(squared[0]):
        raise ValueError("INVALID_KL_MIDSURFACE_NORMAL: degenerate metric")
    shifted=squared/squared[0]
    shifted[0]-=1
    inv=(one-.5*shifted+.375*_mul(shifted,shifted))/np.sqrt(squared[0])
    inv_cubed=_mul(_mul(inv,inv),inv)
    return A,B,N,inv,inv_cubed


def _kl_normal_virtual(delta,R,model,linearization=None):
    """Exact first variation of the KL director and its surface derivatives."""
    if model!="GREEN_LAGRANGE":
        n,nx,ns,nxx,nxs,nss=_normal_jets_small_rotation(delta,R)
        n=n.copy()
        n[2]=0.
        return n,nx,ns,nxx,nxs,nss
    if linearization is None:
        raise ValueError("GL normal variation requires current base jets")
    A,B,N,inv,inv_cubed=linearization
    du=np.stack([_shift(delta,i,1,0) for i in range(3)])
    dv=np.stack([_shift(delta,i,0,1) for i in range(3)])
    dq=np.stack([_shift(delta,i,0,0) for i in range(3)])
    dA=du
    dB=np.stack((dv[0],dv[1]+dq[2]/R,dv[2]-dq[1]/R))
    dN=_cross(dA,B)+_cross(A,dB)
    dnorm2=2.*_dot(N,dN)
    dn=np.stack([
        _mul(dN[i],inv)-.5*_mul(_mul(N[i],dnorm2),inv_cubed)
        for i in range(3)])
    return (dn[:,0],dn[:,1],dn[:,2],
            2*dn[:,3],dn[:,4],2*dn[:,5])


def _kl_virtual_F_jets(delta,R,h,z,dnormal):
    """Virtual F and spatial derivatives without an identity contribution."""
    n,nx,ns,nxx,nxs,nss=dnormal
    H=1.+z/R
    u=delta['q']+z*n
    ux=delta['qx']+z*nx
    us=delta['qs']+z*ns
    uxx=delta['qxx']+z*nxx
    uxs=delta['qxs']+z*nxs
    uss=delta['qss']+z*nss
    uz=n
    def circ(a,b):
        return np.array((b[0],b[1]+a[2]/R,b[2]-a[1]/R))
    c=circ(u,us)
    return (np.column_stack((ux,c/H,uz)),
            np.column_stack((uxx,circ(ux,uxs)/H,nx)),
            np.column_stack((uxs,circ(us,uss)/H,ns)),
            np.column_stack((nx,circ(uz,ns)/H-c/(R*H*H),np.zeros(3))))


@lru_cache(maxsize=128)
def _kl_small_virtual_F_library(R,h,z,model):
    """Cache 30 exact linear-geometry virtual F-jets at each thickness point."""
    if model not in ("LINEAR","VON_KARMAN"):
        raise ValueError("small-rotation cache requires LINEAR or VON_KARMAN")
    output=np.empty((3,len(FIELDS),4,3,3))
    delta={key:np.zeros(3) for key in FIELDS}
    for k,key in enumerate(FIELDS):
        for comp in range(3):
            delta[key][comp]=1.
            dn=_kl_normal_virtual(delta,R,model)
            output[comp,k]=np.stack(_kl_virtual_F_jets(delta,R,h,z,dn))
            delta[key][comp]=0.
    output.setflags(write=False)
    return output


def _F_jets_raw(data,R,h,z,geometry_model="GREEN_LAGRANGE"):
    model=normalized_model(geometry_model)
    n,nx,ns,nxx,nxs,nss=(
        _normal_jets(data,R) if model=="GREEN_LAGRANGE"
        else _normal_jets_small_rotation(data,R))
    H=1.+z/R
    u=data["q"]+z*(n-_E_R)
    ux=data["qx"]+z*nx
    us=data["qs"]+z*ns
    uxx=data["qxx"]+z*nxx
    uxs=data["qxs"]+z*nxs
    uss=data["qss"]+z*nss
    uz=n-_E_R
    uxz=nx
    usz=ns
    uzz=np.zeros(3,dtype=np.result_type(u))

    def circ(a,b):
        return np.array((b[0],b[1]+a[2]/R,b[2]-a[1]/R))
    c=circ(u,us)
    F=_EYE+np.column_stack((ux,c/H,uz))
    Fx=np.column_stack((uxx,circ(ux,uxs)/H,uxz))
    Fs=np.column_stack((uxs,circ(us,uss)/H,usz))
    Fz=np.column_stack((uxz,circ(uz,usz)/H-c/(R*H*H),uzz))
    return F,Fx,Fs,Fz


def _eng(t):
    return np.array((t[0,0],t[1,1],t[2,2],
                     2*t[0,1],2*t[0,2],2*t[1,2]))


def _strain(F,Fx,Fs,Fz,geometry_model="GREEN_LAGRANGE"):
    if normalized_model(geometry_model)!="GREEN_LAGRANGE":
        return strain_jets(F,Fx,Fs,Fz,geometry_model)
    e=_eng(.5*(F.T@F-_EYE))
    der=[_eng(.5*(g.T@F+F.T@g)) for g in (Fx,Fs,Fz)]
    return e,der


def _variation(F,Fx,Fs,Fz,dF,dFx,dFs,dFz,
               geometry_model="GREEN_LAGRANGE"):
    if normalized_model(geometry_model)!="GREEN_LAGRANGE":
        return virtual_strain_jets(F,Fx,Fs,Fz,dF,dFx,dFs,dFz,geometry_model)
    de=_eng(.5*(dF.T@F+F.T@dF))
    der=[_eng(.5*(dg.T@F+g.T@dF+dF.T@g+F.T@dg))
         for g,dg in zip((Fx,Fs,Fz),(dFx,dFs,dFz))]
    return de,der


def _data(j):
    return {name:getattr(j,name) for name in FIELDS}


def kl_point(jets:KLSurfaceJets,*,z_m,radius_m,thickness_m,
             geometry_model="GREEN_LAGRANGE"):
    R,h=_geometry(radius_m,thickness_m,z_m)
    F,Fx,Fs,Fz=_F_jets_raw(_data(jets),R,h,float(z_m),geometry_model)
    det=float(np.linalg.det(F))
    if not math.isfinite(det) or det<=0:
        raise ValueError(f"INVALID_KL_DEFORMATION_MAPPING: detF={det}")
    e,der=_strain(F,Fx,Fs,Fz,geometry_model)
    g=covariant_strain_gradients(e,*der,radius_m=R,z_m=float(z_m))
    G,Gz,_=te_basis(float(z_m),h,jets.potential_order)
    electric=reference_electric_field(
        float(G@jets.psix),float(G@jets.psis),float(Gz@jets.psi),
        radius_m=R,z_m=float(z_m)
    )
    return KLPoint(float(z_m),1+float(z_m)/R,F,e,g,electric,det)


def thickness_enthalpy(jets:KLSurfaceJets,material:G3DMaterial,*,
                       radius_m,thickness_m,n_gauss=12,
                       geometry_model="GREEN_LAGRANGE"):
    R,h=_geometry(radius_m,thickness_m)
    result=0.
    z,w=gauss_thickness(h,n_gauss)
    for zz,ww in zip(z,w):
        p=kl_point(jets,z_m=zz,radius_m=R,thickness_m=h,
                   geometry_model=geometry_model)
        result+=ww*p.H*enthalpy_parts(
            p.strain6,p.gradient6x3,p.electric3,material)["total"]
    return float(result)


def thickness_weak_coefficients(jets:KLSurfaceJets,material:G3DMaterial,*,
                                radius_m,thickness_m,n_gauss=12,
                                geometry_model="GREEN_LAGRANGE"):
    """Thickness-integrated KL virtual work coefficients for all 10 jets.

    Complex step is used ONLY on the analytic normal geometry (F and F_i).
    The G3D material derivative is analytical, from enthalpy_derivatives.
    """
    R,h=_geometry(radius_m,thickness_m)
    mech=np.zeros((3,len(FIELDS)),float)
    electrical=np.zeros((len(jets.psi),len(ELEC_FIELDS)),float)
    energy=0.
    zs,ws=gauss_thickness(h,n_gauss)
    data=_data(jets)
    model=normalized_model(geometry_model)
    if model=="GREEN_LAGRANGE":
        linearization=_kl_normal_gl_linearization(data,R)
        virtual_data={}
        virtual_normal={}
        for k,name in enumerate(FIELDS):
            for comp in range(3):
                unit={key:np.zeros(3) for key in FIELDS}
                unit[name][comp]=1.
                virtual_data[comp,k]=unit
                virtual_normal[comp,k]=_kl_normal_virtual(
                    unit,R,model,linearization)
    for z,w in zip(zs,ws):
        point=kl_point(jets,z_m=z,radius_m=R,thickness_m=h,
                       geometry_model=geometry_model)
        F,Fx,Fs,Fz=_F_jets_raw(data,R,h,float(z),geometry_model)
        S,M,D=enthalpy_derivatives(point.strain6,point.gradient6x3,
                                   point.electric3,material)
        energy+=w*point.H*enthalpy_parts(
            point.strain6,point.gradient6x3,point.electric3,material)["total"]
        virtual_library=(
            None if model=="GREEN_LAGRANGE"
            else _kl_small_virtual_F_library(R,h,float(z),model))
        for k,name in enumerate(FIELDS):
            for comp in range(3):
                if virtual_library is None:
                    dF,dFx,dFs,dFz=_kl_virtual_F_jets(
                        virtual_data[comp,k],R,h,float(z),virtual_normal[comp,k])
                else:
                    dF,dFx,dFs,dFz=virtual_library[comp,k]
                de,der=_variation(F,Fx,Fs,Fz,dF,dFx,dFs,dFz,
                                  geometry_model)
                dg=covariant_strain_gradients(de,*der,radius_m=R,z_m=float(z))
                mech[comp,k]+=w*point.H*(float(S@de)+float(np.sum(M*dg)))
        G,Gz,_=te_basis(float(z),h,jets.potential_order)
        electrical[:,0]+=w*point.H*Gz*D[2]
        electrical[:,1]+=w*point.H*G*D[0]
        electrical[:,2]+=w*G*D[1]
    return KLWeak(float(energy),mech,electrical,len(zs))


__all__=("KLSurfaceJets","KLPoint","KLWeak","FIELDS","ELEC_FIELDS",
         "kl_point","thickness_enthalpy","thickness_weak_coefficients")
