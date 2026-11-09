"""Independent fixed-rank nonlinear MVI surface-jet/weak-work assembler.

KL, FSDT, Reddy, CUF TE1..TE4; G3D/Gz and linear/VK/GL geometry.
The continuum material and thickness-integrated Gateaux laws are shared;
the two-dimensional spatial differentiation, quadrature, and separated
X/Y virtual work are OWNED by independent_mvi, never by MKR or the
reference CylinderFunctional.

Each field has the classical fixed-rank product
    U_c(x_i,s_j) = sum_{r=1}^R X[c,r,i] * Y[c,r,j], R in 1..8.
Derivatives act independently on the one-dimensional factors.
Mechanical and electric virtual work are projected directly into X
and Y directions with the opposite factors frozen; NO full-node
Newton/FDM solve, pseudoinverse, SVD, rank growth, or 3D spatial mesh.

This module evaluates weak work, not full microfree boundary equations.
It is research-only and is NOT connected to production GUI/CASE dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import math
import numpy as np

from flexo_gradient_g3d import G3DMaterial
from cylinder_kinematic_models import normalized_model
from independent_mvi.coupled_operators import (coupled_mvi_sbp42, coupled_mvi_odd_fourier)


THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
MECHANICAL_MULTI=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2),
                  (3,0),(2,1),(1,2),(0,3))
ELECTRIC_MULTI=((0,0),(1,0),(0,1))


def _positive(value,name):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(name+": positive finite number required")
    try:
        v=float(value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(name+": positive finite number required") from exc
    if not math.isfinite(v) or v<=0:
        raise ValueError(name+": positive finite number required")
    return v


def _shape(value,shape,name):
    data=np.asarray(value,dtype=float)
    if data.shape!=shape or not np.isfinite(data).all():
        raise ValueError(name+": finite shape "+str(shape)+" required")
    return data


@dataclass(frozen=True)
class CoupledMVISurfaceGrid:
    nx: int
    ns: int
    length_m: float
    radius_m: float
    dx: np.ndarray=field(init=False,repr=False)
    ds: np.ndarray=field(init=False,repr=False)
    wx: np.ndarray=field(init=False,repr=False)
    ws: np.ndarray=field(init=False,repr=False)

    def __post_init__(self):
        if (type(self.nx) is not int or type(self.ns) is not int or
                self.nx<9 or self.ns<5 or self.ns%2==0):
            raise ValueError("MVI requires nx>=9 and odd ns>=5 for SBP/Fourier")
        L=_positive(self.length_m,"length")
        R=_positive(self.radius_m,"radius")
        Dx,wx=coupled_mvi_sbp42(self.nx,L)
        Ds,ws=coupled_mvi_odd_fourier(self.ns,R)
        for name,raw in (("dx",Dx),("ds",Ds),("wx",wx),("ws",ws)):
            array=np.array(raw,dtype=float,copy=True)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite MVI derivative or quadrature")
            array.setflags(write=False)
            object.__setattr__(self,name,array)
        object.__setattr__(self,"length_m",L)
        object.__setattr__(self,"radius_m",R)

    @property
    def weights(self):
        return self.wx[:,None]*self.ws[None,:]


@dataclass(frozen=True)
class CoupledMVIFactors:
    """Copy-on-create factors; one independent rank-R pair per physical field."""
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        x=np.asarray(self.x,dtype=float)
        y=np.asarray(self.y,dtype=float)
        if (x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=y.shape[:2] or
                x.shape[0]<1 or not 1<=x.shape[1]<=8 or
                min(x.shape[-1],y.shape[-1])<1 or
                not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("finite fixed-rank X/Y arrays required")
        a=x.copy(); b=y.copy()
        a.setflags(write=False); b.setflags(write=False)
        object.__setattr__(self,"x",a)
        object.__setattr__(self,"y",b)

    @property
    def rank(self):
        return self.x.shape[1]


@dataclass(frozen=True)
class CoupledMVIWork:
    energy_j: float
    mechanical: np.ndarray
    electrical: np.ndarray | None
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    rank: int
    status: str="MVI_OWN_WEAK_WORK_NOT_BOUNDARY_CERTIFIED"


class NativeCoupledMVISurface:
    """Fully own separated spatial weak coefficients for seven theories.

    End traces, source-specific surface fluxes, electric gauge closure and
    principal differential orders must be certified independently before
    assembling a boundary-value nonlinear solver. No default supports.
    """

    def __init__(self,theory,material,*,grid:CoupledMVISurfaceGrid,
                 thickness_m,n_gauss=12,geometry_model="GREEN_LAGRANGE",
                 potential_order=None):
        if theory not in THEORIES:
            raise ValueError("MVI theory must be KL/FSDT/REDDY/CUF_TE1..4")
        if not isinstance(material,G3DMaterial):
            raise TypeError("G3DMaterial required")
        if not isinstance(grid,CoupledMVISurfaceGrid):
            raise TypeError("CoupledMVISurfaceGrid required")
        h=_positive(thickness_m,"thickness")
        if grid.radius_m<=h/2:
            raise ValueError("MVI cylindrical metric requires R>h/2")
        if type(n_gauss) is not int or n_gauss<2:
            raise ValueError("n_gauss must be an integer >=2")
        electricity=material.kappa_f_per_m>0
        if electricity and (type(potential_order) is not int or
                            not 0<=potential_order<=4):
            raise ValueError("active potential requires explicit order 0..4")
        if not electricity and potential_order is not None:
            raise ValueError("electricity OFF requires potential_order=None")
        self.theory=theory
        self.material=material
        self.grid=grid
        self.thickness_m=h
        self.geometry_model=normalized_model(geometry_model)
        self.n_gauss=n_gauss
        self.electrical_on=electricity
        self.potential_order=potential_order
        self.n_mechanical=(3 if theory=="KL" else 5 if theory in ("FSDT","REDDY")
                           else 3*(int(theory[-1])+1))
        self.n_electric=(potential_order+1 if electricity else 0)
        self.n_fields=self.n_mechanical+self.n_electric
        self.mechanical_multi=MECHANICAL_MULTI[:6 if theory.startswith("CUF") else 10]
        self.multi=MECHANICAL_MULTI
        self._dx_power=tuple(np.linalg.matrix_power(grid.dx,k) for k in range(4))
        self._ds_power=tuple(np.linalg.matrix_power(grid.ds,k) for k in range(4))
        params=dict(radius_m=grid.radius_m,thickness_m=h,
                    n_gauss=n_gauss,geometry_model=self.geometry_model)
        if theory.startswith("CUF"):
            from cuf_g3d_weak import CUFSurfaceJets,thickness_weak_coefficients
            self._jet_type=CUFSurfaceJets
        elif theory in ("FSDT","REDDY"):
            from cylinder_director_g3d import DirectorSurfaceJets,thickness_weak_coefficients
            self._jet_type=DirectorSurfaceJets
            params["theory"]=theory
        else:
            from cylinder_kl_g3d_weak import KLSurfaceJets,thickness_weak_coefficients
            self._jet_type=KLSurfaceJets
        self._local=partial(thickness_weak_coefficients,material=material,**params)

    @property
    def shape(self):
        return (self.n_fields,self.grid.nx,self.grid.ns)

    def _check_factors(self,factors):
        if not isinstance(factors,CoupledMVIFactors):
            raise TypeError("CoupledMVIFactors required")
        if (factors.x.shape!=(self.n_fields,factors.rank,self.grid.nx) or
                factors.y.shape!=(self.n_fields,factors.rank,self.grid.ns) or
                factors.rank>min(self.grid.nx,self.grid.ns)):
            raise ValueError("MVI factors must match fields, rank and surface grid")
        return factors

    def reconstruct(self,factors):
        self._check_factors(factors)
        return np.einsum("cri,crj->cij",factors.x,factors.y,optimize=True)

    def _surface_derivatives(self,factors):
        """Represented jets and derivative factors, independent of MKR stencils."""
        f=self._check_factors(factors)
        xd={i:np.einsum("ik,crk->cri",D,f.x,optimize=True)
            for i,D in enumerate(self._dx_power)}
        yd={j:np.einsum("jk,crk->crj",D,f.y,optimize=True)
            for j,D in enumerate(self._ds_power)}
        fields={(a,b):np.einsum("cri,crj->cij",xd[a],yd[b],optimize=True)
                for a,b in self.multi}
        return xd,yd,fields

    def _point(self,fields,i,j):
        values=[fields[order][:self.n_mechanical,i,j]
                for order in self.mechanical_multi]
        if self.theory.startswith("CUF"):
            values=[v.reshape(3,int(self.theory[-1])+1) for v in values]
        if self.electrical_on:
            electric=[fields[order][self.n_mechanical:,i,j]
                      for order in ELECTRIC_MULTI]
        else:
            electric=[np.zeros(1) for _ in ELECTRIC_MULTI]
        return self._jet_type(*values,*electric)

    def weak_work(self,factors,*,mechanical_load=None,free_charge=None):
        """Exact derivative of this MVI discrete surface enthalpy in X and Y.

        Mechanical load is a generalized midsurface force density; positive
        physical free charge enters electric enthalpy with PLUS sign. This
        matches div(D)=rho and does not invent a face-pressure conversion.
        """
        xd,yd,fields=self._surface_derivatives(factors)
        shape=(self.grid.nx,self.grid.ns)
        weights=self.grid.weights
        q=fields[(0,0)][:self.n_mechanical]
        p=fields[(0,0)][self.n_mechanical:] if self.electrical_on else None
        if mechanical_load is None:
            load_m=np.zeros((self.n_mechanical,*shape))
        else:
            load_m=_shape(mechanical_load,(self.n_mechanical,*shape),
                          "generalized mechanical load")
        if not self.electrical_on:
            if free_charge is not None:
                raise ValueError("electric charge with electricity OFF")
            load_e=None
        elif free_charge is None:
            load_e=np.zeros((self.n_electric,*shape))
        else:
            load_e=_shape(free_charge,(self.n_electric,*shape),
                          "generalized electric free charge")
        coeff=np.zeros((self.n_fields,len(self.multi),*shape))
        density=np.zeros(shape)
        for i in range(self.grid.nx):
            for j in range(self.grid.ns):
                local=self._local(self._point(fields,i,j))
                density[i,j]=local.energy_j_per_m2
                mc=np.asarray(local.mechanical,float)
                if mc.size!=self.n_mechanical*len(self.mechanical_multi):
                    raise ValueError("local mechanical weak coefficient count mismatch")
                coeff[:self.n_mechanical,:len(self.mechanical_multi),i,j]=mc.reshape(
                    self.n_mechanical,len(self.mechanical_multi))
                if self.electrical_on:
                    ec=np.asarray(local.electrical,float)
                    if ec.shape!=(self.n_electric,len(ELECTRIC_MULTI)):
                        raise ValueError("local electrical weak coefficient shape mismatch")
                    coeff[self.n_mechanical:,:len(ELECTRIC_MULTI),i,j]=ec
        if not np.isfinite(density).all() or not np.isfinite(coeff).all():
            raise ValueError("nonfinite MVI local coupled material work")
        energy=float(np.sum(weights*density))
        energy-=float(np.sum(weights*load_m*q))
        coeff[:self.n_mechanical,0]-=load_m
        if self.electrical_on:
            energy+=float(np.sum(weights*load_e*p))
            coeff[self.n_mechanical:,0]+=load_e
        # Vary the X family with Y fixed, and vice versa; only the
        # one-dimensional factor derivative matrices enter these sums.
        gx=np.zeros_like(factors.x)
        gy=np.zeros_like(factors.y)
        for k,(ax,ang) in enumerate(self.multi):
            coefficient=coeff[:,k]
            if not np.any(coefficient):
                continue
            wx_contracted=np.einsum("cij,crj,j->cri",coefficient,yd[ang],
                                   self.grid.ws,optimize=True)
            wy_contracted=np.einsum("cij,cri,i->crj",coefficient,xd[ax],
                                   self.grid.wx,optimize=True)
            gx+=np.einsum("ip,cri->crp",self._dx_power[ax],wx_contracted,
                          optimize=True)
            gy+=np.einsum("jp,crj->crp",self._ds_power[ang],wy_contracted,
                          optimize=True)
        if not (math.isfinite(energy) and np.isfinite(gx).all() and
                np.isfinite(gy).all()):
            raise ValueError("nonfinite MVI projected weak work")
        return CoupledMVIWork(energy,q.copy(),None if p is None else p.copy(),
                              gx,gy,factors.rank)


__all__=("THEORIES","MECHANICAL_MULTI","ELECTRIC_MULTI",
         "CoupledMVISurfaceGrid","CoupledMVIFactors","CoupledMVIWork",
         "NativeCoupledMVISurface")
