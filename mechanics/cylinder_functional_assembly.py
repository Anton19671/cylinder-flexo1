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
engineering production solver. Reference trace enforcement and projected
FDM/MVI assembly are provided separately by cylinder_boundary_spatial.
Actual MVI alternating profile solves and time-dependent KL inertia are NOT
implemented here. No existing static solver route is changed.

Reference coordinates x,s=R*theta; area dA=dx ds. Units are SI throughout.
See Krysko--Zhong corrected manuscript, sections 8, 18--24, 31--39, 48.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
import numpy as np

from flexo_gradient_g3d import G3DMaterial, enthalpy_parts
from cylinder_kl_exact_inertia import kl_inertial_weak_action, kl_inertial_jet_coefficients
from cylinder_kelvin_voigt import kelvin_voigt_thickness_weak, KelvinVoigtGlobal
from cylinder_g3d_natural_edges import axial_natural_end_tractions
from cylinder_sbp_green_reduction import split_coupled_sbp_residual
from cylinder_sbp_derivatives import (
    SBP_OPERATOR_SCHEME, axial_sbp21_gradient, periodic_sbp_odd_gradient,
    require_green_identity, axial_sbp21_gradient_sparse,
    periodic_sbp_odd_gradient_sparse,
)
from cylinder_surface_linear_algebra import (
    is_sparse_matrix, check_matrix, apply_biaxial,
    apply_weighted_adjoint, operator_power,
)
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


def surface_sbp21(n_axial,n_circumference,length_m,radius_m):
    """Explicit SBP21 / odd periodic verification discretization in SI."""
    R=float(radius_m)
    if not math.isfinite(R) or R<=0:
        raise ValueError("R must be finite positive")
    Dx,wx=axial_sbp21_gradient(n_axial,length_m)
    Ds,ws=periodic_sbp_odd_gradient(n_circumference,2*math.pi*R)
    require_green_identity(Dx,Ds,wx,ws)
    return SurfaceOperators(Dx,Ds,wx,ws)

def surface_sbp21_csr(n_axial,n_circumference,length_m,radius_m):
    """O(nx+ns) CSR SBP21/odd-periodic reference operators.

    This is opt-in for Newton--Krylov FDM only. The original dense
    operators and MVI ranks remain unchanged.
    """
    R=float(radius_m)
    if not math.isfinite(R) or R<=0:
        raise ValueError("R must be finite positive")
    Dx,wx=axial_sbp21_gradient_sparse(n_axial,length_m)
    Ds,ws=periodic_sbp_odd_gradient_sparse(n_circumference,2*math.pi*R)
    require_green_identity(Dx,Ds,wx,ws)
    return SurfaceOperators(Dx,Ds,wx,ws)


@dataclass(frozen=True)
class SurfaceOperators:
    Dx: object
    Ds: object
    wx: np.ndarray
    ws: np.ndarray

    def __post_init__(self):
        wx=np.asarray(self.wx,dtype=float)
        ws=np.asarray(self.ws,dtype=float)
        if wx.ndim!=1 or ws.ndim!=1 or len(wx)<4 or len(ws)<4:
            raise ValueError("surface quadrature requires at least 4 nodes")
        sparse_x=is_sparse_matrix(self.Dx)
        sparse_s=is_sparse_matrix(self.Ds)
        if sparse_x!=sparse_s:
            raise ValueError("Dx and Ds must both be dense or both CSR")
        Dx=check_matrix(self.Dx,len(wx),"Dx")
        Ds=check_matrix(self.Ds,len(ws),"Ds")
        if not np.isfinite(wx).all() or not np.isfinite(ws).all():
            raise ValueError("surface quadrature weights must be finite")
        if np.any(wx<=0) or np.any(ws<=0):
            raise ValueError("surface quadrature weights must be positive")
        for name,D in (("Dx",Dx),("Ds",Ds)):
            if np.max(np.abs(D@np.ones(D.shape[1])),initial=0)>1.e-10*max(
                    1.,float(np.max(np.abs(D.data),initial=0))
                    if sparse_x else float(np.linalg.norm(D,np.inf))):
                raise ValueError(f"{name}: derivative of a constant is not zero")
        if sparse_x:
            # Full dense SVD is forbidden. Accept only the explicit,
            # mathematically characterized SBP21/odd-periodic stencil.
            # Even periodic counts are rejected to avoid checkerboards.
            refx,refwx=axial_sbp21_gradient_sparse(len(wx),float(np.sum(wx)))
            refs,refws=periodic_sbp_odd_gradient_sparse(len(ws),float(np.sum(ws)))
            if not (np.allclose(wx,refwx,rtol=2.e-12,atol=0.) and
                    np.allclose(ws,refws,rtol=2.e-12,atol=0.)):
                raise ValueError("CSR surface weights must match diagonal SBP norm")
            for name,D,ref in (("Dx",Dx,refx),("Ds",Ds,refs)):
                error=(D-ref).tocsr()
                peak=float(np.max(np.abs(error.data),initial=0))
                scale=max(1.,float(np.max(np.abs(ref.data),initial=0)))
                if peak>2.e-12*scale:
                    raise ValueError(
                        f"{name}: CSR reference requires canonical SBP stencil")
            require_green_identity(Dx,Ds,wx,ws)
        else:
            # Original dense reference: retain rank validation of the
            # periodic derivative and its checkerboard-null protection.
            values=np.linalg.svd(Ds,compute_uv=False)
            rank=int(np.count_nonzero(
                values>max(values[0],1.)*1.e-11))
            if rank!=Ds.shape[0]-1:
                raise ValueError(
                    "Ds has spurious null modes (possible even-grid checkerboard)")
        for name,value in (("Dx",Dx),("Ds",Ds),("wx",wx),("ws",ws)):
            object.__setattr__(self,name,value.copy())

    @property
    def shape(self):
        return (len(self.wx),len(self.ws))

    @property
    def area_weights(self):
        return np.outer(self.wx,self.ws)

    def derivative_powers(self,max_order):
        return ({k:operator_power(self.Dx,k) for k in range(max_order+1)},
                {k:operator_power(self.Ds,k) for k in range(max_order+1)})


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
    return apply_biaxial(field,Dx,Ds)


def _assemble_adjoint(coeff,wxws,Dx,Ds):
    return apply_weighted_adjoint(coeff,wxws,Dx,Ds)


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
        params=dict(radius_m=self.R,thickness_m=self.h,n_gauss=self.n_gauss)
        point_params=dict(radius_m=self.R,thickness_m=self.h)
        if self.theory.startswith("CUF_"):
            self._weak_kernel=partial(_cuf_weak,material=self.material,**params)
            self._point_kernel=partial(_cuf_point,**point_params)
        elif self.theory in ("FSDT","REDDY"):
            self._weak_kernel=partial(_director_weak,material=self.material,
                                      theory=self.theory,**params)
            self._point_kernel=partial(_director_point,theory=self.theory,
                                       **point_params)
        else:
            from cylinder_kl_g3d_weak import thickness_weak_coefficients,kl_point
            self._weak_kernel=partial(thickness_weak_coefficients,
                                      material=self.material,**params)
            self._point_kernel=partial(kl_point,**point_params)

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
        return self._weak_kernel(jets)

    def _point(self,jets,z):
        return self._point_kernel(jets,z_m=z)

    def _field_derivatives(self,field,spec):
        return [_surface_derivative(field,self.dxp[ax],self.dsp[bs])
                for _,ax,bs in spec]

    def evaluate(self,mechanical,electrical=None,*,dead_load=None,
                 free_charge=None,energy_breakdown=True,check_charge_balance=True):
        """Discrete Π and EXACT (for fixed quadrature) coefficient gradients.

        Mechanical nodal fields have shape (n_mechanical,nx,ns). Electrical
        nodal fields have shape (P+1,nx,ns), P=0..4. When electrical physics
        is OFF the electrical field MUST be None and is excluded from Π.

        dead_load and free_charge, if given, are generalized SURFACE loads
        conjugate to the respective nodal fields, not pressures in Pa. Real
        face-pressure pullbacks require a separately verified follower-load
        virtual work and are not interchangeable with these dead loads.
        check_charge_balance=False is reserved for a combined bulk/boundary
        assembler that checks TOTAL charge, including electrode supply.
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
        if self.electrical_on and check_charge_balance:
            # In fully insulated mode a nonzero integral of the constant-mode
            # free charge cannot be balanced by div D; gauge must NOT hide it.
            net=float(np.sum(self.W*charge[0]))
            if abs(net)>1.e-10*max(1.e-30,float(np.sum(np.abs(self.W*charge[0])))):
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

    def _dynamic_g3d_weak_jet_coefficients(
            self,mechanical,electrical=None,*,velocity=None,acceleration=None,
            density_kg_m3=None,eta_s=0.0,mass_damping_rate_per_s=0.0):
        """One source of truth for conservative and *dynamic* surface jets.

        Coefficients are not multiplied by quadrature area. Their weighted
        adjoints are the SAME physical forces as evaluate(), mass_action(),
        kelvin_voigt_action(), and mass_proportional_damping_action().
        Boundary Green splitting may only be applied to these complete
        coefficients on an SBP-compatible grid. Neither external forces nor
        separately prescribed end work are included in local constitutive
        jets: those are added by BoundarySpatialProblem.
        """
        nx,ns=self.ops.shape
        q=_check_fields(mechanical,(self.n_mechanical,nx,ns),"G3D weak q")
        if self.electrical_on:
            if electrical is None:
                raise ValueError("G3D electric potential coefficients required")
            pe=np.asarray(electrical,float)
            if pe.ndim!=3 or pe.shape[1:]!=(nx,ns) or not 1<=len(pe)<=5:
                raise ValueError("G3D electric coefficient shape mismatch")
            p=_check_fields(pe,pe.shape,"G3D weak electric potential")
        else:
            if electrical is not None:
                raise ValueError("G3D electric coefficients must be disabled")
            p=np.zeros((1,nx,ns),float)

        if isinstance(eta_s,(bool,np.bool_)) or isinstance(
                mass_damping_rate_per_s,(bool,np.bool_)):
            raise ValueError("viscous parameters must be finite nonnegative numbers")
        eta=float(eta_s)
        alpha=float(mass_damping_rate_per_s)
        if not (math.isfinite(eta) and eta>=0 and
                math.isfinite(alpha) and alpha>=0):
            raise ValueError("viscosity and mass damping must be nonnegative")
        is_dynamic=(velocity is not None or acceleration is not None or
                    density_kg_m3 is not None or eta>0 or alpha>0)
        if is_dynamic and (velocity is None or acceleration is None
                           or density_kg_m3 is None):
            raise ValueError("dynamic G3D Green split requires velocity, "
                             "acceleration and density together")
        if is_dynamic:
            v=_check_fields(velocity,q.shape,"G3D weak velocity")
            a=_check_fields(acceleration,q.shape,"G3D weak acceleration")
            rho=float(density_kg_m3)
            if not math.isfinite(rho) or rho<=0:
                raise ValueError("G3D dynamic density must be positive")
        else:
            v=a=None
            rho=None

        spec=MECHANICAL_DERIVATIVES[:self.n_derivatives]
        mder=self._field_derivatives(q,spec)
        eder=self._field_derivatives(p,ELECTRICAL_DERIVATIVES)
        mc=np.zeros((len(spec),self.n_mechanical,nx,ns),float)
        ec=np.zeros((3,len(p),nx,ns)) if self.electrical_on else None
        if is_dynamic:
            zero_elec=self._field_derivatives(
                np.zeros((1,nx,ns)),ELECTRICAL_DERIVATIVES)
            vd=self._field_derivatives(v,spec)
            ad=self._field_derivatives(a,spec)
        for i in range(nx):
            for j in range(ns):
                jet=self._make_jets(mder,eder,i,j)
                conservative=self._local(jet)
                mc[:,:,i,j]=conservative.mechanical.reshape(
                    self.n_mechanical,len(spec)).T
                if ec is not None:
                    ec[:,:,i,j]=conservative.electrical.T
                if is_dynamic and eta:
                    visc=kelvin_voigt_thickness_weak(
                        self._make_jets(mder,zero_elec,i,j),
                        self._make_jets(vd,zero_elec,i,j),
                        self.material,theory=self.theory,radius_m=self.R,
                        thickness_m=self.h,eta_s=eta,n_gauss=self.n_gauss)
                    mc[:,:,i,j]+=visc.mechanical.T
                if is_dynamic and self.theory in ("FSDT","REDDY"):
                    inertia=_director_mass(
                        self._make_jets(ad,zero_elec,i,j),
                        theory=self.theory,radius_m=self.R,thickness_m=self.h,
                        density_kg_m3=rho,n_gauss=self.n_gauss)
                    mc[:3,:,i,j]+=inertia.T
                    if alpha:
                        damping=_director_mass(
                            self._make_jets(vd,zero_elec,i,j),
                            theory=self.theory,radius_m=self.R,thickness_m=self.h,
                            density_kg_m3=rho,n_gauss=self.n_gauss)
                        mc[:3,:,i,j]+=alpha*damping.T

        if is_dynamic and self.theory=="KL":
            inertia=kl_inertial_jet_coefficients(
                q,v,a,Dx=self.ops.Dx,Ds=self.ops.Ds,
                wx=self.ops.wx,ws=self.ops.ws,radius_m=self.R,
                thickness_m=self.h,density_kg_m3=rho)
            mc[:3]+=inertia
            if alpha:
                # M(q)*v only: no nonlinear gyroscopic velocity-square
                # term may be included inside Rayleigh alpha_M damping.
                damping=kl_inertial_jet_coefficients(
                    q,np.zeros_like(v),v,Dx=self.ops.Dx,Ds=self.ops.Ds,
                    wx=self.ops.wx,ws=self.ops.ws,radius_m=self.R,
                    thickness_m=self.h,density_kg_m3=rho)
                mc[:3]+=alpha*damping
        elif is_dynamic and self.theory.startswith("CUF_TE"):
            # CUF mass has zero derivative-jet coefficients. The exact
            # curved thickness mass moments already used by mass_action()
            # are a LOCAL contribution to the independent q jet.
            n=int(self.theory[-1])+1
            mass=mass_matrix_te(radius_m=self.R,thickness_m=self.h,
                                rho_kg_m3=rho,te_order=n-1)
            inertial_field=a.reshape((3,n,nx,ns))
            if alpha:
                inertial_field=inertial_field+alpha*v.reshape((3,n,nx,ns))
            local=np.einsum("ij,cjkl->cikl",mass,inertial_field)
            mc[0]+=local.reshape((self.n_mechanical,nx,ns))
        if not np.isfinite(mc).all() or (
                ec is not None and not np.isfinite(ec).all()):
            raise ValueError("nonfinite coupled dynamic G3D weak coefficients")
        return mc,ec

    def sbp_green_reduction(self,mechanical,electrical=None,*,velocity=None,
                            acceleration=None,density_kg_m3=None,
                            eta_s=0.0,mass_damping_rate_per_s=0.0):
        """Exact SBP bulk+edge split, including mechanical dynamics.

        This audit does NOT add boundary conditions to Newton/MVI/FDM.
        It yields unforced internal weak residuals and matching sampled
        natural fluxes; external distributed and end loading is separately
        included in the BoundarySpatialProblem full effective residual.
        """
        mc,ec=self._dynamic_g3d_weak_jet_coefficients(
            mechanical,electrical,velocity=velocity,
            acceleration=acceleration,density_kg_m3=density_kg_m3,
            eta_s=eta_s,
            mass_damping_rate_per_s=mass_damping_rate_per_s)
        return split_coupled_sbp_residual(
            self.ops,mc,MECHANICAL_DERIVATIVES[:self.n_derivatives],ec)

    def natural_end_fluxes(self,mechanical,electrical=None,*,velocity=None,
                           acceleration=None,density_kg_m3=None,eta_s=0.0,
                           mass_damping_rate_per_s=0.0):
        """Sample continuum-form conservative+dynamic end fluxes.

        For a NON-SBP mesh this remains continuum-form diagnostics only.
        For an SBP mesh use sbp_green_reduction to obtain exact nodal
        boundary work compatible with the *discrete* weak residual.
        """
        mc,ec=self._dynamic_g3d_weak_jet_coefficients(
            mechanical,electrical,velocity=velocity,
            acceleration=acceleration,density_kg_m3=density_kg_m3,
            eta_s=eta_s,
            mass_damping_rate_per_s=mass_damping_rate_per_s)
        return axial_natural_end_tractions(
            mc,MECHANICAL_DERIVATIVES[:self.n_derivatives],
            self.ops.Dx,self.ops.Ds,electrical_coefficients=ec)

    def kelvin_voigt_action(self,mechanical,velocity,*,eta_s):
        """Assemble the nonlinear material dissipative weak force for any theory.

        This is the velocity-gradient first variation of the *same* finite
        strain geometry as the conservative G3D/flexoelectric functional.
        Electrical equations and conservative enthalpy are unmodified.
        """
        eta=float(eta_s)
        if not math.isfinite(eta) or eta<0:
            raise ValueError("micro.eta_s must be finite and nonnegative [s]")
        nx,ns=self.ops.shape
        q=_check_fields(mechanical,(self.n_mechanical,nx,ns),"viscous mechanical state")
        v=_check_fields(velocity,q.shape,"viscous velocity")
        force=np.zeros_like(q)
        if eta==0:
            return KelvinVoigtGlobal(force,0.)
        electric_zeros=np.zeros((1,nx,ns))
        jets_q=self._field_derivatives(q,MECHANICAL_DERIVATIVES[:self.n_derivatives])
        jets_v=self._field_derivatives(v,MECHANICAL_DERIVATIVES[:self.n_derivatives])
        z_electric=self._field_derivatives(electric_zeros,ELECTRICAL_DERIVATIVES)
        coefficients=np.zeros((self.n_derivatives,self.n_mechanical,nx,ns))
        power=0.
        for i in range(nx):
            for j in range(ns):
                state=self._make_jets(jets_q,z_electric,i,j)
                rate=self._make_jets(jets_v,z_electric,i,j)
                weak=kelvin_voigt_thickness_weak(state,rate,self.material,
                    theory=self.theory,radius_m=self.R,thickness_m=self.h,
                    eta_s=eta,n_gauss=self.n_gauss)
                coefficients[:,:,i,j]=weak.mechanical.T
                power+=self.W[i,j]*weak.power_w_per_m2
        for k,(_,dx,ds) in enumerate(MECHANICAL_DERIVATIVES[:self.n_derivatives]):
            force+=_assemble_adjoint(coefficients[k],self.W,self.dxp[dx],self.dsp[ds])
        if not np.isfinite(force).all() or not math.isfinite(power):
            raise ValueError("INVALID_KELVIN_VOIGT_GLOBAL_DISSIPATION")
        return KelvinVoigtGlobal(force,float(power))

    def mass_proportional_damping_action(self,velocity,*,density_kg_m3,
                                          mass_damping_rate_per_s,mechanical=None):
        """Consistent mechanical C*v, C=alpha_M*M, alpha_M measured in 1/s.

        Uses the very same full CUF/director mass operator as acceleration;
        there is no electric inertia and no approximation of micro.eta_s.
        """
        rate=float(mass_damping_rate_per_s)
        if not math.isfinite(rate) or rate<0:
            raise ValueError("mass damping rate must be finite nonnegative [1/s]")
        return rate*self.mass_action(velocity,density_kg_m3=density_kg_m3,
                                     mechanical=mechanical)

    def mass_action(self,acceleration,*,density_kg_m3,mechanical=None,velocity=None):
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
            if mechanical is None:
                raise ValueError("KL exact nonlinear inertia requires current mechanical state")
            q=_check_fields(mechanical,accel.shape,"KL current displacement")
            v=np.zeros_like(q) if velocity is None else _check_fields(
                velocity,accel.shape,"KL current velocity")
            return kl_inertial_weak_action(
                q,v,accel,Dx=self.ops.Dx,Ds=self.ops.Ds,
                wx=self.ops.wx,ws=self.ops.ws,radius_m=self.R,
                thickness_m=self.h,density_kg_m3=density)
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
         "surface_trapezoid","surface_sbp21","surface_sbp21_csr",
         "periodic_face_gradient","axial_fd_gradient",
         "MECHANICAL_DERIVATIVES","ELECTRICAL_DERIVATIVES")

