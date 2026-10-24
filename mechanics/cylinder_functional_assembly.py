#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference discrete *enthalpy stationarity* for cylindrical shell models.

Every mechanical/electrical residual is an EXACT DISCRETE first variation of
one quadrature functional, with its derivatives assembled by transposes of the
SAME surface differentiation matrices used to evaluate the local field jets.

KL (3 surface fields), FSDT/Reddy (5), CUF TE1..4 (6,9,12,15) use their own
kinematics but ONE G3D material enthalpy. For electric ON, a global gauge
constraint is available. Electric OFF drops all electric unknowns altogether.

This is a reproducible reference variational assembler, NOT a verified
engineering production solver. Higher-order end traces, actual MVI alternating
profile solves, spatial boundary enforcement and time-dependent KL inertia
are NOT implemented here. No existing static solver route is changed.

Reference coordinates x,s=R*theta; area dA=dx ds. Units are SI throughout.
See Krysko--Zhong corrected manuscript, sections 8, 18--24, 31--39, 48.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from flexo_gradient_g3d import G3DMaterial, enthalpy_parts
from cuf_g3d_weak import (
    CUFSurfaceJets, mass_matrix_te, gauss_thickness,
    thickness_weak_coefficients as _cuf_weak,
    point_state as _cuf_point,
)
from cylinder_director_g3d import (
    DirectorSurfaceJets, kinetic_weak_coefficients as _director_mass,
    thickness_weak_coefficients as _director_weak,
    director_point as _director_point,
)

MECHANICAL_DERIVATIVES = (
    ("q", 0, 0), ("qx", 1, 0), ("qs", 0, 1),
    ("qxx", 2, 0), ("qxs", 1, 1), ("qss", 0, 2),
    ("qxxx", 3, 0), ("qxxs", 2, 1),
    ("qxss", 1, 2), ("qsss", 0, 3),
)
ELECTRICAL_DERIVATIVES = (("psi",0,0),("psix",1,0),("psis",0,1))
_THEORIES = ("KL", "FSDT", "REDDY", "CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")


def periodic_face_gradient(n, perimeter_m):
    """First difference across periodic cell faces; no even-grid sawtooth null."""
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n < 4:
        raise ValueError("periodic grid requires at least four nodes")
    circumference = float(perimeter_m)
    if not math.isfinite(circumference) or circumference<=0:
        raise ValueError("periodic circumference must be positive")
    h = circumference/n
    matrix = np.zeros((n,n),float)
    for j in range(n):
        matrix[j,j] = -1./h
        matrix[j,(j+1)%n] = 1./h
    return matrix


def axial_fd_gradient(n,length_m):
    """Second-order test derivative with one-sided endpoint rows.

    Not a high-order SBP operator: the caller must establish boundary closure,
    convergence, and trace treatment before a production PDE dispatch.
    """
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n < 4:
        raise ValueError("axial grid requires at least four nodes")
    L=float(length_m)
    if not math.isfinite(L) or L<=0:raise ValueError("L must be positive")
    dx=L/(n-1)
    D=np.zeros((n,n),float)
    for i in range(1,n-1):
        D[i,i-1]=-1/(2*dx)
        D[i,i+1]=1/(2*dx)
    D[0,:3]=np.array([-3,4,-1])/(2*dx)
    D[-1,-3:]=np.array([1,-4,3])/(2*dx)
    return D


def surface_trapezoid(n_axial,n_circumference,length_m,radius_m):
    """Convenience verification mesh; does not imply certified SBP closures."""
    R=float(radius_m)
    if not math.isfinite(R) or R<=0:raise ValueError("R must be positive")
    L=float(length_m)
    Dx=axial_fd_gradient(n_axial,L)
    Ds=periodic_face_gradient(n_circumference,2*math.pi*R)
    wx=np.full(n_axial,L/(n_axial-1))
    wx[[0,-1]]*=.5
    ws=np.full(n_circumference,2*math.pi*R/n_circumference)
    return SurfaceOperators(Dx,Ds,wx,ws)


@dataclass(frozen=True)
class SurfaceOperators:
    Dx: np.ndarray
    Ds: np.ndarray
    wx: np.ndarray
    ws: np.ndarray

    def __post_init__(self):
        Dx=np.asarray(self.Dx,dtype=float)
        Ds=np.asarray(self.Ds,dtype=float)
        wx=np.asarray(self.wx,dtype=float)
        ws=np.asarray(self.ws,dtype=float)
        if Dx.ndim!=2 or Dx.shape[0]!=Dx.shape[1] or Dx.shape[0]<4:
            raise ValueError("Dx must be a square matrix of size >=4")
        if Ds.ndim!=2 or Ds.shape[0]!=Ds.shape[1] or Ds.shape[0]<4:
            raise ValueError("Ds must be a square matrix of size >=4")
        if wx.shape!=(Dx.shape[0],) or ws.shape!=(Ds.shape[0],):
            raise ValueError("integration weights must match differentiation grids")
        if any(not np.isfinite(x).all() for x in (Dx,Ds,wx,ws)):
            raise ValueError("surface operators contain nonfinite values")
        if np.any(wx<=0) or np.any(ws<=0):
            raise ValueError("surface quadrature weights must be positive")
        for name,D in (("Dx",Dx),("Ds",Ds)):
            if np.max(np.abs(D@np.ones(len(D))))>1.e-10*max(1.,np.linalg.norm(D,np.inf)):
                raise ValueError(f"{name}: derivative of a constant is not zero")
        # Electric gauge must be the ONLY null mode of the periodic first
        # derivative. A centred D1 on an even grid has a second (checkerboard)
        # zero mode: reject it explicitly.
        vals=np.linalg.svd(Ds,compute_uv=False)
        rank=int(np.count_nonzero(vals>max(vals[0],1.)*1.e-11))
        if rank!=Ds.shape[0]-1:
            raise ValueError("Ds has spurious null modes (possible even-grid checkerboard)")
        for name,a in (("Dx",Dx),("Ds",Ds),("wx",wx),("ws",ws)):
            object.__setattr__(self,name,a.copy())

    @property
    def shape(self):
        return (len(self.wx),len(self.ws))

    @property
    def area_weights(self):
        return np.outer(self.wx,self.ws)

    def derivative_powers(self,max_order):
        return ({k:np.linalg.matrix_power(self.Dx,k) for k in range(max_order+1)},
                {k:np.linalg.matrix_power(self.Ds,k) for k in range(max_order+1)})


def _theory_shape(theory):
    t=str(theory).upper()
    if t not in _THEORIES:
        raise ValueError("theory must be KL, FSDT, REDDY or CUF_TE1..4")
    return t, 3 if t=="KL" else (5 if t in ("FSDT","REDDY") else
                                   3*(int(t[-1])+1)), 10 if t in ("KL","FSDT","REDDY") else 6


def _check_fields(value,expected,name):
    a=np.asarray(value,dtype=float)
    if a.shape!=expected or not np.isfinite(a).all():
        raise ValueError(f"{name} must have finite shape {expected}, got {a.shape}")
    return a.copy()


def _surface_derivative(field,Dx,Ds):
    # Each component has a surface grid: (components,nx,ns).
    return np.einsum('ij,cjk,lk->cil',Dx,field,Ds,optimize=True)


def _assemble_adjoint(coeff,wxws,Dx,Ds):
    return np.einsum('ij,cjk,kl->cil',Dx.T,coeff*wxws[None,...],Ds,optimize=True)


def _basis_moments(potential_order,R,h):
    # I_m = int[-h/2,h/2] H(z) (2z/h)^m dz, with H=1+z/R.
    return np.array([h/(m+1) if m%2==0 else h*h/(2*R*(m+2))
                     for m in range(potential_order+1)],float)


@dataclass(frozen=True)
class FunctionalResult:
    theory: str
    total_enthalpy_j: float
    energy_parts_j: dict
    mechanical_residual: np.ndarray
    electrical_residual: np.ndarray | None
    electric_gauge_vector: np.ndarray | None
    electric_gauge_value: float | None
    boundary_condition_status: str = "HIGHER_ORDER_BC_NOT_ASSEMBLED"
    spatial_method_status: str = "REFERENCE_FUNCTIONAL_ONLY"

    @property
    def mechanical_max_abs(self):
        return float(np.max(np.abs(self.mechanical_residual)))

    @property
    def electrical_max_abs(self):
        return (None if self.electrical_residual is None else
                float(np.max(np.abs(self.electrical_residual))))

    def augmented_residual(self,lagrange_multiplier=0.):
        m=self.mechanical_residual.reshape(-1)
        if self.electrical_residual is None:return m.copy()
        lm=float(lagrange_multiplier)
        if not math.isfinite(lm):raise ValueError("gauge multiplier must be finite")
        return np.concatenate((m,(self.electrical_residual+lm*self.electric_gauge_vector).reshape(-1),
                               [self.electric_gauge_value]))


class CylinderFunctional:
    """Surface quadrature of a single stationary reference electric enthalpy.

    The potential is an electric *saddle variable*, so this is stationarity,
    NOT minimisation in all unknowns. Internal FE/finite-difference and PGD/MVI
    solutions must take derivatives of the same Π_h, without adding ad hoc
    electric field equations in postprocessing.
    """
    def __init__(self, theory, material, *, radius_m, thickness_m,
                 surface_operators:SurfaceOperators, n_gauss=4):
        self.theory,self.n_mechanical,self.n_derivatives=_theory_shape(theory)
        if not isinstance(material,G3DMaterial):raise TypeError("material must be G3DMaterial")
        self.material=material
        self.R=float(radius_m)
        self.h=float(thickness_m)
        if (not math.isfinite(self.R) or not math.isfinite(self.h) or
            self.h<=0 or self.R<=self.h/2):
            raise ValueError("require R > h/2 > 0 in SI units")
        if not isinstance(surface_operators,SurfaceOperators):
            raise TypeError("surface_operators must be SurfaceOperators")
        self.ops=surface_operators
        if (isinstance(n_gauss,bool) or not isinstance(n_gauss,int) or n_gauss<2):
            raise ValueError("n_gauss must be integer >=2")
        self.n_gauss=n_gauss
        self.electrical_on=(material.kappa_f_per_m>0 or material.mu1_c_per_m!=0
                             or material.mu2_c_per_m!=0)
        if self.electrical_on and material.kappa_f_per_m<=0:
            raise ValueError("electrical potential requires positive dielectric kappa")
        self.dxp,self.dsp=self.ops.derivative_powers(3)
        self.W=self.ops.area_weights

    def _make_jets(self, derivs, elec_derivs, i,j):
        if self.theory.startswith("CUF_"):
            n=int(self.theory[-1])+1
            arrays=[a[:,i,j].reshape((3,n)) for a in derivs]
            return CUFSurfaceJets(*arrays,*[a[:,i,j] for a in elec_derivs])
        if self.theory in ("FSDT","REDDY"):
            return DirectorSurfaceJets(*[a[:,i,j] for a in derivs],
                                       *[a[:,i,j] for a in elec_derivs])
        from cylinder_kl_g3d_weak import KLSurfaceJets
        return KLSurfaceJets(*[a[:,i,j] for a in derivs],
                             *[a[:,i,j] for a in elec_derivs])

    def _local(self,jets):
        args=dict(radius_m=self.R,thickness_m=self.h,n_gauss=self.n_gauss)
        if self.theory.startswith("CUF_"):
            return _cuf_weak(jets,self.material,**args)
        if self.theory in ("FSDT","REDDY"):
            return _director_weak(jets,self.material,theory=self.theory,**args)
        from cylinder_kl_g3d_weak import thickness_weak_coefficients
        return thickness_weak_coefficients(jets,self.material,**args)

    def _point(self,jets,z):
        args=dict(z_m=z,radius_m=self.R,thickness_m=self.h)
        if self.theory.startswith("CUF_"):
            return _cuf_point(jets,**args)
        if self.theory in ("FSDT","REDDY"):
            return _director_point(jets,theory=self.theory,**args)
        from cylinder_kl_g3d_weak import kl_point
        return kl_point(jets,**args)

    def _field_derivatives(self,field,spec):
        return [_surface_derivative(field,self.dxp[ax],self.dsp[bs])
                for _,ax,bs in spec]

    def evaluate(self,mechanical,electrical=None,*,dead_load=None,
                 free_charge=None,energy_breakdown=True):
        """Discrete Π and EXACT (for fixed quadrature) coefficient gradients.

        Mechanical nodal fields have shape (n_mechanical,nx,ns). Electrical
        nodal fields have shape (P+1,nx,ns), P=0..4. When electrical physics
        is OFF the electrical field MUST be None and is excluded from Π.

        dead_load and free_charge, if given, are generalized SURFACE loads
        conjugate to the respective nodal fields, not pressures in Pa. Real
        face-pressure pullbacks require a separately verified follower-load
        virtual work and are not interchangeable with these dead loads.
        """
        nx,ns=self.ops.shape
        mech=_check_fields(mechanical,(self.n_mechanical,nx,ns),"mechanical")
        if self.electrical_on:
            if electrical is None:raise ValueError("coupled enthalpy requires electrical potential")
            p=np.asarray(electrical,dtype=float)
            if p.ndim!=3 or p.shape[1:]!=(nx,ns) or not 1<=p.shape[0]<=5:
                raise ValueError("electric shape must be (P+1,nx,ns), P=0..4")
            phi=_check_fields(p,p.shape,"electrical")
        else:
            if electrical is not None or free_charge is not None:
                raise ValueError("electric coefficients must be omitted when flexo/dielectric are OFF")
            phi=np.zeros((1,nx,ns),float)  # dummy local zero for existing material kernels
        f=(np.zeros_like(mech) if dead_load is None else
           _check_fields(dead_load,mech.shape,"dead_load"))
        charge=(np.zeros_like(phi) if free_charge is None else
                _check_fields(free_charge,phi.shape,"free_charge"))
        if self.electrical_on:
            # In fully insulated mode a nonzero integral of the constant-mode
            # free charge cannot be balanced by div D; gauge must NOT hide it.
            net=float(np.sum(self.W*charge[0]))
            if abs(net)>1.e-9*max(1.,float(np.sum(np.abs(self.W*charge[0])))):
                raise ValueError("incompatible insulated free charge: nonzero net charge")
        spec=MECHANICAL_DERIVATIVES[:self.n_derivatives]
        mder=self._field_derivatives(mech,spec)
        eder=self._field_derivatives(phi,ELECTRICAL_DERIVATIVES)
        mcoef=np.zeros((self.n_derivatives,self.n_mechanical,nx,ns),float)
        ecoef=np.zeros((3,len(phi),nx,ns),float)
        names=("classical","gradient","dielectric","flexoelectric")
        parts=dict.fromkeys(names,0.)
        total=0.
        zgauss,wgauss=gauss_thickness(self.h,self.n_gauss)
        for i in range(nx):
            for j in range(ns):
                jets=self._make_jets(mder,eder,i,j)
                weak=self._local(jets)
                total+=self.W[i,j]*weak.energy_j_per_m2
                assert weak.mechanical.size==self.n_mechanical*self.n_derivatives
                mcoef[:,:,i,j]=weak.mechanical.reshape(self.n_mechanical,self.n_derivatives).T
                if self.electrical_on:
                    ecoef[:,:,i,j]=weak.electrical.T
                if energy_breakdown:
                    for z,w in zip(zgauss,wgauss):
                        point=self._point(jets,z)
                        ent=enthalpy_parts(point.strain6,point.gradient6x3,
                                           point.electric3,self.material)
                        for name in names:
                            parts[name]+=self.W[i,j]*w*point.H*ent[name]
        rmech=np.zeros_like(mech)
        for k,(_,dx,ds) in enumerate(spec):
            rmech+=_assemble_adjoint(mcoef[k],self.W,self.dxp[dx],self.dsp[ds])
        rmech-=self.W[None,...]*f
        total-=float(np.sum(self.W[None,...]*f*mech))
        if self.electrical_on:
            relec=np.zeros_like(phi)
            for k,(_,dx,ds) in enumerate(ELECTRICAL_DERIVATIVES):
                relec+=_assemble_adjoint(ecoef[k],self.W,self.dxp[dx],self.dsp[ds])
            relec-=self.W[None,...]*charge
            total-=float(np.sum(self.W[None,...]*charge*phi))
            moments=_basis_moments(len(phi)-1,self.R,self.h)
            gaugevec=moments[:,None,None]*self.W[None,...]
            gauge=float(np.sum(gaugevec*phi))
        else:
            relec=None
            gaugevec=None
            gauge=None
        if energy_breakdown:
            parts["total_internal"]=sum(parts.values())
            parts["external_dead_load"]=-float(np.sum(self.W[None,...]*f*mech))
            if self.electrical_on:
                parts["external_charge"]=-float(np.sum(self.W[None,...]*charge*phi))
            parts["total_stationary_functional"]=total
        return FunctionalResult(self.theory,total,parts,rmech,relec,gaugevec,gauge)

    def mass_action(self,acceleration,*,density_kg_m3):
        """Consistent mechanical inertia from the SAME director/CUF ansatz.

        No time integration is executed. KL finite-rotation kinetic energy
        requires configuration-dependent director mass and gyroscopic terms;
        rejecting KL prevents a false constant-mass approximation.
        """
        density=float(density_kg_m3)
        if not math.isfinite(density) or density<=0:
            raise ValueError("mass density must be finite positive SI kg/m3")
        nx,ns=self.ops.shape
        accel=_check_fields(acceleration,(self.n_mechanical,nx,ns),"acceleration")
        if self.theory=="KL":
            raise NotImplementedError(
                "exact nonlinear KL director inertia needs q-dependent mass/gyroscopic terms"
            )
        if self.theory.startswith("CUF_"):
            N=int(self.theory[-1])
            m=mass_matrix_te(radius_m=self.R,thickness_m=self.h,
                             rho_kg_m3=density,te_order=N)
            a=accel.reshape((3,N+1,nx,ns))
            transformed=np.einsum('ij,cjkl->cikl',m,a)
            return transformed.reshape(accel.shape)*self.W[None,...]
        # FSDT / Reddy mass contains gradient variations of the director.
        zeros=np.zeros((1,nx,ns))
        mder=self._field_derivatives(accel,MECHANICAL_DERIVATIVES)
        eder=self._field_derivatives(zeros,ELECTRICAL_DERIVATIVES)
        mcoef=np.zeros((3,5,nx,ns),float)
        for i in range(nx):
            for j in range(ns):
                jets=self._make_jets(mder,eder,i,j)
                mcoef[:,:,i,j]=_director_mass(
                    jets,theory=self.theory,radius_m=self.R,thickness_m=self.h,
                    density_kg_m3=density,n_gauss=self.n_gauss
                ).T
        residual=np.zeros_like(accel)
        for k,(_,ax,as_) in enumerate(MECHANICAL_DERIVATIVES[:3]):
            residual+=_assemble_adjoint(mcoef[k],self.W,self.dxp[ax],self.dsp[as_])
        return residual


__all__=("CylinderFunctional","SurfaceOperators","FunctionalResult",
         "surface_trapezoid","periodic_face_gradient","axial_fd_gradient",
         "MECHANICAL_DERIVATIVES","ELECTRICAL_DERIVATIVES")
