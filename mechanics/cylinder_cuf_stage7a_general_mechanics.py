#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7A: GENERAL CUF mechanics for the closed circular cylindrical shell.

Source: supplied verified Word derivation, general CUF sections before TE1
specialization.

Unknown generalized fields:
    q[component,tau,...], component = 0:u, 1:v, 2:w

CUF expansion:
    u = sum_tau F_tau(z) u_tau
    v = sum_tau F_tau(z) v_tau
    w = sum_tau F_tau(z) w_tau

For arbitrary preselected thickness functions F_tau(z) and derivatives F'_tau(z),
this module computes:
    q,q_x,q_s -> a_i,b_i,c_i -> F -> detF check -> E -> S -> P
    -> generalized thickness resultants A^x, A^s, C
    -> 3m strong residual fields.

NO particular higher-order CUF family is invented here.
The only named concrete basis supplied is TE1:
    F0=1, F1=z,
because it is explicitly defined in the verified source and is used as a strict
regression target.

No MVI generalization is implemented in Stage 7A.
No end-BC generalization is implemented in Stage 7A.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import cylinder_te1_local_kernel as lk
import cylinder_te1_stage2b as s2


class CUFInputError(ValueError):
    pass


@dataclass(frozen=True)
class CallableThicknessBasis:
    """
    Arbitrary preselected CUF thickness basis.

    funcs[tau](z)  -> F_tau(z)
    dfuncs[tau](z) -> F'_tau(z)

    The functions themselves are supplied by the accepted CUF model.
    """
    funcs: tuple
    dfuncs: tuple
    names: tuple = ()

    def __post_init__(self):
        if len(self.funcs) < 1:
            raise CUFInputError("at least one thickness function is required")
        if len(self.funcs) != len(self.dfuncs):
            raise CUFInputError("funcs/dfuncs length mismatch")
        if self.names and len(self.names) != len(self.funcs):
            raise CUFInputError("names length mismatch")
        if not all(callable(f) for f in self.funcs):
            raise CUFInputError("all funcs must be callable")
        if not all(callable(f) for f in self.dfuncs):
            raise CUFInputError("all dfuncs must be callable")

    @property
    def m(self):
        return len(self.funcs)

    def values(self, z):
        zz=float(z)
        out=np.array([float(f(zz)) for f in self.funcs],dtype=float)
        if not np.all(np.isfinite(out)):
            raise CUFInputError("non-finite F_tau(z)")
        return out

    def derivatives(self, z):
        zz=float(z)
        out=np.array([float(f(zz)) for f in self.dfuncs],dtype=float)
        if not np.all(np.isfinite(out)):
            raise CUFInputError("non-finite F'_tau(z)")
        return out


def te1_basis():
    """
    The exact concrete TE1 basis stated in the source:
        F0=1, F1=z.
    """
    return CallableThicknessBasis(
        funcs=(lambda z: 1.0, lambda z: z),
        dfuncs=(lambda z: 0.0, lambda z: 1.0),
        names=("F0=1","F1=z"),
    )


@dataclass(frozen=True)
class GeneralLocalResult:
    H: float
    abc: np.ndarray
    F: np.ndarray
    E: np.ndarray
    strain_intensity: float
    S: np.ndarray
    stress_intensity: float
    P: np.ndarray
    detF: float


def _as_3m(a, m, name):
    a=np.asarray(a,dtype=float)
    if a.shape != (3,m):
        raise CUFInputError(
            f"{name} must have shape (3,{m}) in component order [u,v,w]"
        )
    if not np.all(np.isfinite(a)):
        raise CUFInputError(f"{name} contains non-finite values")
    return a


def general_gradient_quantities(q, qx, qs, z, R, basis):
    """
    Exact general-CUF nine gradient quantities.

    q/qx/qs shape: (3,m), rows [u,v,w], columns tau.
    """
    if not isinstance(basis,CallableThicknessBasis):
        raise CUFInputError("basis must be CallableThicknessBasis")
    m=basis.m
    q=_as_3m(q,m,"q")
    qx=_as_3m(qx,m,"qx")
    qs=_as_3m(qs,m,"qs")

    z=float(z); R=float(R)
    if R <= 0.0:
        raise CUFInputError("R must be positive")
    H=1.0+z/R
    if H <= 0.0:
        raise CUFInputError("H=1+z/R must be positive")

    Ft=basis.values(z)
    Fp=basis.derivatives(z)

    u=q[0]; v=q[1]; w=q[2]
    ux=qx[0]; vx=qx[1]; wx=qx[2]
    us=qs[0]; vs=qs[1]; ws=qs[2]

    a1=float(Ft@ux)
    a2=float(Ft@us)/H
    a3=float(Fp@u)

    b1=float(Ft@vx)
    b2=float(Ft@(vs+w/R))/H
    b3=float(Fp@v)

    c1=float(Ft@wx)
    c2=float(Ft@(ws-v/R))/H
    c3=float(Fp@w)

    abc=np.array([
        [a1,a2,a3],
        [b1,b2,b3],
        [c1,c2,c3],
    ],dtype=float)
    return H,abc


def general_local_kernel(q,qx,qs,z,R,K0,GB,basis):
    H,abc=general_gradient_quantities(q,qx,qs,z,R,basis)
    F=lk.deformation_gradient_from_abc(abc)
    J=float(np.linalg.det(F))

    # Mandatory admissibility check BEFORE constitutive evaluation.
    if not np.isfinite(J) or J <= 0.0:
        raise s2.InvalidDeformationMapping(
            f"INVALID_DEFORMATION_MAPPING detF={J:.17g}, z={float(z):.17g}"
        )

    E=lk.green_lagrange(F)
    S=lk.birger_second_piola(E,K0,GB)
    P=F@S

    return GeneralLocalResult(
        H=H,
        abc=abc,
        F=F,
        E=E,
        strain_intensity=lk.strain_intensity_from_E(E),
        S=S,
        stress_intensity=lk.stress_intensity_from_S(S),
        P=P,
        detF=J,
    )


RESULTANT_KEYS=(
    "Au_x","Au_s","Cu",
    "Av_x","Av_s","Cv",
    "Aw_x","Aw_s","Cw",
)


@dataclass(frozen=True)
class GeneralResultants:
    """
    Every array has shape (m,*surface_shape).
    """
    Au_x: np.ndarray
    Au_s: np.ndarray
    Cu: np.ndarray

    Av_x: np.ndarray
    Av_s: np.ndarray
    Cv: np.ndarray

    Aw_x: np.ndarray
    Aw_s: np.ndarray
    Cw: np.ndarray

    def as_dict(self):
        return {k:getattr(self,k) for k in RESULTANT_KEYS}

    @property
    def m(self):
        return int(np.asarray(self.Au_x).shape[0])

    @property
    def surface_shape(self):
        return tuple(np.asarray(self.Au_x).shape[1:])


def _check_general_surface_arrays(q,qx,qs,basis):
    q=np.asarray(q,float); qx=np.asarray(qx,float); qs=np.asarray(qs,float)
    if q.shape != qx.shape or q.shape != qs.shape:
        raise CUFInputError("q/qx/qs shape mismatch")
    if q.ndim < 2 or q.shape[0] != 3 or q.shape[1] != basis.m:
        raise CUFInputError(
            f"q/qx/qs must have shape (3,{basis.m},...surface_shape...)"
        )
    if not (
        np.all(np.isfinite(q))
        and np.all(np.isfinite(qx))
        and np.all(np.isfinite(qs))
    ):
        raise CUFInputError("non-finite q/qx/qs")
    return q,qx,qs


def _gb_mode(GB,nz,surface_shape):
    a=np.asarray(GB,float)
    if a.ndim == 0:
        mode="scalar"
    elif a.shape == surface_shape:
        mode="surface"
    elif a.shape == (nz,)+surface_shape:
        mode="full"
    else:
        raise CUFInputError(
            "GB must be scalar, surface-shaped, or (nz,*surface_shape)"
        )
    if np.any(a <= 0.0) or not np.all(np.isfinite(a)):
        raise CUFInputError("GB must be finite and positive")
    return a,mode


def general_thickness_resultants_scalar_reference(
    q,qx,qs,*,R,K0,GB,quadrature,basis
):
    """Pre-K2b scalar node-by-node reference retained only for regression tests."""
    q,qx,qs=_check_general_surface_arrays(q,qx,qs,basis)
    R=float(R); K0=float(K0)
    if R <= 0.0 or K0 <= 0.0:
        raise CUFInputError("R and K0 must be positive")

    surface_shape=q.shape[2:]
    nz=quadrature.z.size
    GBa,mode=_gb_mode(GB,nz,surface_shape)

    acc={
        k:np.zeros((basis.m,)+surface_shape,dtype=float)
        for k in RESULTANT_KEYS
    }

    for idx in np.ndindex(surface_shape):
        qpt=q[(slice(None),slice(None))+idx]
        qxpt=qx[(slice(None),slice(None))+idx]
        qspt=qs[(slice(None),slice(None))+idx]

        for g,(z,wz) in enumerate(zip(quadrature.z,quadrature.w)):
            if mode=="scalar":
                GBg=float(GBa)
            elif mode=="surface":
                GBg=float(GBa[idx])
            else:
                GBg=float(GBa[(g,)+idx])

            rr=general_local_kernel(
                qpt,qxpt,qspt,z,R,K0,GBg,basis
            )
            H=rr.H
            Ft=basis.values(z)
            Fp=basis.derivatives(z)
            P=rr.P

            Pxx,Pxs,Pxr=P[0,0],P[0,1],P[0,2]
            Psx,Pss,Psr=P[1,0],P[1,1],P[1,2]
            Prx,Prs,Prr=P[2,0],P[2,1],P[2,2]

            sl=(slice(None),)+idx

            acc["Au_x"][sl] += wz*(H*Ft*Pxx)
            acc["Au_s"][sl] += wz*(Ft*Pxs)
            acc["Cu"][sl]   += wz*(H*Fp*Pxr)

            acc["Av_x"][sl] += wz*(H*Ft*Psx)
            acc["Av_s"][sl] += wz*(Ft*Pss)
            acc["Cv"][sl]   += wz*(H*Fp*Psr-(Ft/R)*Prs)

            acc["Aw_x"][sl] += wz*(H*Ft*Prx)
            acc["Aw_s"][sl] += wz*(Ft*Prs)
            acc["Cw"][sl]   += wz*(H*Fp*Prr+(Ft/R)*Pss)

    return GeneralResultants(**acc)


def _batched_general_gradient_quantities(q,qx,qs,z,R,basis):
    """K2b: evaluate the nine CUF gradient quantities on all surface nodes."""
    z=float(z); R=float(R)
    if R <= 0.0:
        raise CUFInputError("R must be positive")
    H=1.0+z/R
    if H <= 0.0:
        raise CUFInputError("H=1+z/R must be positive")
    Ft=basis.values(z)
    Fp=basis.derivatives(z)

    # Result shape is (3,*surface_shape).  Only the small thickness-index
    # contraction is reduced; all surface nodes stay vectorized.
    fq=np.einsum('t,ct...->c...',Ft,q,optimize=False)
    fqx=np.einsum('t,ct...->c...',Ft,qx,optimize=False)
    fqs=np.einsum('t,ct...->c...',Ft,qs,optimize=False)
    fpq=np.einsum('t,ct...->c...',Fp,q,optimize=False)

    surface_shape=q.shape[2:]
    abc=np.empty((3,3)+surface_shape,dtype=float)
    abc[0,0]=fqx[0]
    abc[1,0]=fqx[1]
    abc[2,0]=fqx[2]
    abc[0,1]=fqs[0]/H
    abc[1,1]=(fqs[1]+fq[2]/R)/H
    abc[2,1]=(fqs[2]-fq[1]/R)/H
    abc[0,2]=fpq[0]
    abc[1,2]=fpq[1]
    abc[2,2]=fpq[2]
    return H,Ft,Fp,abc


def _batched_det3(F):
    """Determinant of F with shape (3,3,*surface_shape), without node loops."""
    return (
        F[0,0]*(F[1,1]*F[2,2]-F[1,2]*F[2,1])
        -F[0,1]*(F[1,0]*F[2,2]-F[1,2]*F[2,0])
        +F[0,2]*(F[1,0]*F[2,1]-F[1,1]*F[2,0])
    )


def _batched_green_lagrange(F):
    C=np.einsum('ki...,kj...->ij...',F,F,optimize=False)
    E=0.5*C
    E=E.copy()
    E[0,0]-=0.5
    E[1,1]-=0.5
    E[2,2]-=0.5
    return E


def _batched_linear_strain_from_abc(abc):
    """Linear cylindrical strain tensor on all surface nodes."""
    surface_shape=abc.shape[2:]
    E=np.empty((3,3)+surface_shape,dtype=float)
    E[0,0]=abc[0,0]
    E[1,1]=abc[1,1]
    E[2,2]=abc[2,2]
    E[0,1]=E[1,0]=0.5*(abc[0,1]+abc[1,0])
    E[0,2]=E[2,0]=0.5*(abc[0,2]+abc[2,0])
    E[1,2]=E[2,1]=0.5*(abc[1,2]+abc[2,1])
    return E


def _batched_tensor_deviatoric_intensity(A,factor):
    """sqrt(factor * dev(A):dev(A)) over all surface nodes."""
    tr=A[0,0]+A[1,1]+A[2,2]
    D=np.array(A,copy=True)
    third=tr/3.0
    D[0,0]-=third; D[1,1]-=third; D[2,2]-=third
    val=float(factor)*np.sum(D*D,axis=(0,1))
    return np.sqrt(np.maximum(val,0.0))


def _batched_strain_intensity(E):
    return _batched_tensor_deviatoric_intensity(E,2.0/3.0)


def _batched_stress_intensity(S):
    return _batched_tensor_deviatoric_intensity(S,3.0/2.0)


def _batched_birger_second_piola(E,K0,GB):
    """S=K0 tr(E)I+2 GB dev(E), vectorized over surface nodes."""
    K0=float(K0)
    G=np.asarray(GB,float)
    if K0 <= 0.0 or np.any(G <= 0.0) or not np.all(np.isfinite(G)):
        raise CUFInputError("K0 and frozen GB must be finite and positive")
    tr=E[0,0]+E[1,1]+E[2,2]
    S=2.0*G*E
    diag=(K0-(2.0/3.0)*G)*tr
    S=S.copy()
    S[0,0]+=diag
    S[1,1]+=diag
    S[2,2]+=diag
    return S


def _raise_if_invalid_batched_F(J,z,surface_shape):
    bad=(~np.isfinite(J)) | (J <= 0.0)
    if np.any(bad):
        flat=int(np.flatnonzero(np.asarray(bad).reshape(-1))[0])
        idx=np.unravel_index(flat,surface_shape) if surface_shape else ()
        val=float(np.asarray(J).reshape(-1)[flat])
        raise s2.InvalidDeformationMapping(
            f"INVALID_DEFORMATION_MAPPING detF={val:.17g}, "
            f"z={float(z):.17g}, node={idx}"
        )


def _accumulate_general_resultants(acc,P,H,Ft,Fp,wz,R,surface_shape):
    """Vectorized virtual-work accumulation for one thickness point."""
    shp=(Ft.size,)+(1,)*len(surface_shape)
    FtB=Ft.reshape(shp)
    FpB=Fp.reshape(shp)
    Pxx,Pxs,Pxr=P[0,0],P[0,1],P[0,2]
    Psx,Pss,Psr=P[1,0],P[1,1],P[1,2]
    Prx,Prs,Prr=P[2,0],P[2,1],P[2,2]

    acc["Au_x"] += wz*(H*FtB*Pxx)
    acc["Au_s"] += wz*(FtB*Pxs)
    acc["Cu"]   += wz*(H*FpB*Pxr)
    acc["Av_x"] += wz*(H*FtB*Psx)
    acc["Av_s"] += wz*(FtB*Pss)
    acc["Cv"]   += wz*(H*FpB*Psr-(FtB/R)*Prs)
    acc["Aw_x"] += wz*(H*FtB*Prx)
    acc["Aw_s"] += wz*(FtB*Prs)
    acc["Cw"]   += wz*(H*FpB*Prr+(FtB/R)*Pss)


def general_thickness_resultants(
    q,qx,qs,*,R,K0,GB,quadrature,basis
):
    """K2b production path: same CUF resultants, batched over surface nodes.

    The only Python loop left is over the thickness quadrature points.  The
    scalar pre-K2b implementation is kept in
    ``general_thickness_resultants_scalar_reference`` for self-checks.
    """
    q,qx,qs=_check_general_surface_arrays(q,qx,qs,basis)
    R=float(R); K0=float(K0)
    if R <= 0.0 or K0 <= 0.0:
        raise CUFInputError("R and K0 must be positive")

    surface_shape=q.shape[2:]
    nz=quadrature.z.size
    GBa,mode=_gb_mode(GB,nz,surface_shape)
    acc={
        k:np.zeros((basis.m,)+surface_shape,dtype=float)
        for k in RESULTANT_KEYS
    }

    for g,(z,wz) in enumerate(zip(quadrature.z,quadrature.w)):
        if mode=="scalar":
            GBg=float(GBa)
        elif mode=="surface":
            GBg=GBa
        else:
            GBg=GBa[g]

        H,Ft,Fp,abc=_batched_general_gradient_quantities(q,qx,qs,z,R,basis)
        F=abc.copy()
        F[0,0]+=1.0; F[1,1]+=1.0; F[2,2]+=1.0
        J=_batched_det3(F)
        # Same mandatory admissibility check, still before constitutive law.
        _raise_if_invalid_batched_F(J,z,surface_shape)
        E=_batched_green_lagrange(F)
        S=_batched_birger_second_piola(E,K0,GBg)
        P=np.einsum('ik...,kj...->ij...',F,S,optimize=False)
        _accumulate_general_resultants(
            acc,P,H,Ft,Fp,float(wz),R,surface_shape
        )

    return GeneralResultants(**acc)



@dataclass(frozen=True)
class GeneralElasticTangentPoint:
    """Precomputed base-state data for one thickness quadrature point."""
    z: float
    wz: float
    H: float
    Ft: np.ndarray
    Fp: np.ndarray
    F: np.ndarray
    S: np.ndarray
    GB: object


@dataclass(frozen=True)
class GeneralElasticTangentContext:
    """Base-state cache for the exact fixed-K0/fixed-GB Green-Lagrange tangent."""
    R: float
    K0: float
    basis: CallableThicknessBasis
    surface_shape: tuple
    points: tuple


def prepare_general_elastic_tangent_context(
    q,qx,qs,*,R,K0,GB,quadrature,basis
):
    """Prepare the exact material/geometric tangent at a valid base state.

    This is the V3b elastic / V3c frozen-GB tangent cache.  The deformation mapping check is
    performed at the base state before any tangent use.  No finite differences
    are involved.
    """
    q,qx,qs=_check_general_surface_arrays(q,qx,qs,basis)
    R=float(R); K0=float(K0)
    if R <= 0.0 or K0 <= 0.0:
        raise CUFInputError("R and K0 must be positive")
    surface_shape=q.shape[2:]
    nz=quadrature.z.size
    GBa,mode=_gb_mode(GB,nz,surface_shape)
    pts=[]
    for g,(z,wz) in enumerate(zip(quadrature.z,quadrature.w)):
        if mode=="scalar":
            GBg=float(GBa)
        elif mode=="surface":
            GBg=GBa
        else:
            GBg=GBa[g]
        H,Ft,Fp,abc=_batched_general_gradient_quantities(q,qx,qs,z,R,basis)
        F=abc.copy()
        F[0,0]+=1.0; F[1,1]+=1.0; F[2,2]+=1.0
        J=_batched_det3(F)
        _raise_if_invalid_batched_F(J,z,surface_shape)
        E=_batched_green_lagrange(F)
        S=_batched_birger_second_piola(E,K0,GBg)
        pts.append(GeneralElasticTangentPoint(
            z=float(z),wz=float(wz),H=float(H),Ft=np.asarray(Ft,float),
            Fp=np.asarray(Fp,float),F=np.asarray(F,float),S=np.asarray(S,float),
            GB=GBg,
        ))
    return GeneralElasticTangentContext(
        R=R,K0=K0,basis=basis,surface_shape=surface_shape,points=tuple(pts)
    )


def _batched_general_gradient_direction(dq,dqx,dqs,*,H,Ft,Fp,R):
    """Exact variation of the nine CUF gradient quantities."""
    dq=np.asarray(dq,float); dqx=np.asarray(dqx,float); dqs=np.asarray(dqs,float)
    if dq.shape!=dqx.shape or dq.shape!=dqs.shape or dq.ndim<2 or dq.shape[0]!=3:
        raise CUFInputError("dq/dqx/dqs shape mismatch")
    if not (np.all(np.isfinite(dq)) and np.all(np.isfinite(dqx)) and np.all(np.isfinite(dqs))):
        raise CUFInputError("non-finite dq/dqx/dqs")
    fq=np.einsum('t,ct...->c...',Ft,dq,optimize=False)
    fqx=np.einsum('t,ct...->c...',Ft,dqx,optimize=False)
    fqs=np.einsum('t,ct...->c...',Ft,dqs,optimize=False)
    fpq=np.einsum('t,ct...->c...',Fp,dq,optimize=False)
    surface_shape=dq.shape[2:]
    dabc=np.empty((3,3)+surface_shape,dtype=float)
    dabc[0,0]=fqx[0]
    dabc[1,0]=fqx[1]
    dabc[2,0]=fqx[2]
    dabc[0,1]=fqs[0]/H
    dabc[1,1]=(fqs[1]+fq[2]/R)/H
    dabc[2,1]=(fqs[2]-fq[1]/R)/H
    dabc[0,2]=fpq[0]
    dabc[1,2]=fpq[1]
    dabc[2,2]=fpq[2]
    return dabc


def _batched_green_lagrange_tangent(F,dF):
    """dE = sym(F^T dF), exact for Green-Lagrange strain."""
    a=np.einsum('ki...,kj...->ij...',dF,F,optimize=False)
    b=np.einsum('ki...,kj...->ij...',F,dF,optimize=False)
    return 0.5*(a+b)


def general_thickness_resultants_tangent_from_context(ctx,dq,dqx,dqs):
    """Exact directional derivative of nonlinear CUF resultants at fixed K0/GB."""
    if not isinstance(ctx,GeneralElasticTangentContext):
        raise CUFInputError("ctx must be GeneralElasticTangentContext")
    dq,dqx,dqs=_check_general_surface_arrays(dq,dqx,dqs,ctx.basis)
    if dq.shape[2:]!=ctx.surface_shape:
        raise CUFInputError("tangent direction surface shape mismatch")
    acc={
        k:np.zeros((ctx.basis.m,)+ctx.surface_shape,dtype=float)
        for k in RESULTANT_KEYS
    }
    for pt in ctx.points:
        dF=_batched_general_gradient_direction(
            dq,dqx,dqs,H=pt.H,Ft=pt.Ft,Fp=pt.Fp,R=ctx.R
        )
        dE=_batched_green_lagrange_tangent(pt.F,dF)
        # The constitutive law is linear in E for fixed K0/GB.  In V3b GB=G0;
        # in V3c GB is the frozen Birger field of the current outer iteration.
        dS=_batched_birger_second_piola(dE,ctx.K0,pt.GB)
        dP=(
            np.einsum('ik...,kj...->ij...',dF,pt.S,optimize=False)
            +np.einsum('ik...,kj...->ij...',pt.F,dS,optimize=False)
        )
        _accumulate_general_resultants(
            acc,dP,pt.H,pt.Ft,pt.Fp,pt.wz,ctx.R,ctx.surface_shape
        )
    return GeneralResultants(**acc)


def general_thickness_resultants_tangent(
    q,qx,qs,dq,dqx,dqs,*,R,K0,GB,quadrature,basis
):
    """Convenience wrapper for the exact V3b elastic CUF tangent resultants."""
    ctx=prepare_general_elastic_tangent_context(
        q,qx,qs,R=R,K0=K0,GB=GB,quadrature=quadrature,basis=basis
    )
    return general_thickness_resultants_tangent_from_context(ctx,dq,dqx,dqs)

def zero_generalized_load(m,surface_shape=()):
    return np.zeros((3,int(m))+tuple(surface_shape),dtype=float)


def radial_nominal_face_load(qr,*,zq,R,basis):
    """
    General CUF projection of nominal radial face traction:
        p^w_tau = q_r H_q F_tau(z_q),
        p^u_tau = p^v_tau = 0.

    For TE1 this reduces exactly to
        p_w0=q_r H_q,
        p_w1=q_r H_q z_q.
    """
    zq=float(zq); R=float(R)
    if R <= 0.0:
        raise CUFInputError("R must be positive")
    Hq=1.0+zq/R
    if Hq <= 0.0:
        raise CUFInputError("face Hq must be positive")
    qr=np.asarray(qr,float)
    if not np.all(np.isfinite(qr)):
        raise CUFInputError("non-finite qr")

    Ft=basis.values(zq)
    out=np.zeros((3,basis.m)+qr.shape,dtype=float)
    for tau in range(basis.m):
        out[(2,tau)+((slice(None),)*qr.ndim)] = qr*Hq*Ft[tau]
    return out


def sum_generalized_loads(*loads):
    if not loads:
        raise CUFInputError("at least one load required")
    arr=[np.asarray(a,float) for a in loads]
    if any(a.shape != arr[0].shape for a in arr):
        raise CUFInputError("load shape mismatch")
    return np.sum(arr,axis=0)


def general_residuals(
    resultants,*,Dx,Ds,generalized_load
):
    """
    Strong form for every tau:
      Ru_tau = -d_x Au_x_tau - d_s Au_s_tau + Cu_tau - pu_tau
      Rv_tau = -d_x Av_x_tau - d_s Av_s_tau + Cv_tau - pv_tau
      Rw_tau = -d_x Aw_x_tau - d_s Aw_s_tau + Cw_tau - pw_tau

    Output shape: (3,m,*surface_shape).
    No flatten ordering is imposed in Stage 7A.
    """
    if not isinstance(resultants,GeneralResultants):
        raise CUFInputError("resultants must be GeneralResultants")

    m=resultants.m
    surface_shape=resultants.surface_shape
    p=np.asarray(generalized_load,float)
    if p.shape != (3,m)+surface_shape:
        raise CUFInputError("generalized_load shape mismatch")

    out=np.empty_like(p)

    def D(op,a,label):
        b=np.asarray(op(np.asarray(a,float)),float)
        if b.shape != surface_shape:
            raise CUFInputError(f"{label} returned wrong shape")
        return b

    for tau in range(m):
        out[0,tau]=(
            -D(Dx,resultants.Au_x[tau],"Dx")
            -D(Ds,resultants.Au_s[tau],"Ds")
            +resultants.Cu[tau]
            -p[0,tau]
        )
        out[1,tau]=(
            -D(Dx,resultants.Av_x[tau],"Dx")
            -D(Ds,resultants.Av_s[tau],"Ds")
            +resultants.Cv[tau]
            -p[1,tau]
        )
        out[2,tau]=(
            -D(Dx,resultants.Aw_x[tau],"Dx")
            -D(Ds,resultants.Aw_s[tau],"Ds")
            +resultants.Cw[tau]
            -p[2,tau]
        )
    return out


def te1_general_to_legacy_order(a):
    """
    Map general CUF shape (3,2,...) to legacy TE1 order
    [u0,u1,v0,v1,w0,w1].
    Regression helper only.
    """
    a=np.asarray(a,float)
    if a.shape[0:2] != (3,2):
        raise CUFInputError("expected general TE1 shape (3,2,...)")
    return np.stack([
        a[0,0],a[0,1],
        a[1,0],a[1,1],
        a[2,0],a[2,1],
    ],axis=0)


def te1_legacy_to_general(a):
    """
    Map legacy TE1 order [u0,u1,v0,v1,w0,w1] to (3,2,...).
    Regression helper only.
    """
    a=np.asarray(a,float)
    if a.shape[0] != 6:
        raise CUFInputError("legacy TE1 first dimension must be 6")
    return np.stack([
        np.stack([a[0],a[1]],axis=0),
        np.stack([a[2],a[3]],axis=0),
        np.stack([a[4],a[5]],axis=0),
    ],axis=0)
