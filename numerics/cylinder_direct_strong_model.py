"""Native field-to-equation assembly for experimental direct-strong MKR.

KL, FSDT, Reddy and CUF TE1--TE4 own their field preparation and coefficient
assembly here. Only local material, kinematic and inertial laws are shared.
No CylinderFunctional, SurfaceOperators, MVI solver or reference variational
assembly is imported. This is not automatic production/GUI dispatch.
"""
from functools import partial
import math
import numpy as np

from flexo_gradient_g3d import G3DMaterial
from cylinder_kinematic_models import normalized_model
from cylinder_direct_strong_fdm import DirectStrongGrid, direct_strong_jet_balance
from cylinder_mkr_field_jets import MKRFieldJets, MECHANICAL_JETS, ELECTRIC_JETS


THEORIES=("KL", "FSDT", "REDDY", "CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")


def _state(value, shape, label):
    a=np.asarray(value,float)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError(f"{label}: finite shape {shape} required")
    return a


def _nonnegative(value, label):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(label+": finite nonnegative number required")
    v=float(value)
    if not math.isfinite(v) or v<0:
        raise ValueError(label+": finite nonnegative number required")
    return v


class DirectStrongShell:
    """MKR-owned constitutive-jet producer and direct bulk/end balance.

    Uses the same normalized physical thickness basis as the local kernels.
    Mechanical fields are (nfield,nx,ns), potential fields (P+1,nx,ns).
    Velocity/acceleration, when supplied, belong to the mechanical fields
    only. Electric inertia is never added.

    This preserves the existing trapezoid-route kinematic stencils. It does
    not establish their high-order boundary convergence or prove equivalence
    to the independent primal/dual linear MKR or a complete source §40 scheme.
    """
    def __init__(self, theory, material, *, grid, thickness_m, n_gauss=12,
                 geometry_model="GREEN_LAGRANGE"):
        if not isinstance(theory,str) or theory.upper() not in THEORIES:
            raise ValueError("theory must be KL, FSDT, REDDY or CUF_TE1..4")
        if not isinstance(material,G3DMaterial):
            raise TypeError("G3DMaterial required")
        if not isinstance(grid,DirectStrongGrid):
            raise TypeError("DirectStrongGrid required")
        if isinstance(thickness_m,(bool,np.bool_)):
            raise ValueError("require R>h/2>0")
        h=float(thickness_m)
        if not math.isfinite(h) or not grid.radius_m>h/2>0:
            raise ValueError("require R>h/2>0")
        if type(n_gauss) is not int or n_gauss<2:
            raise ValueError("n_gauss must be an integer >=2")
        self.theory=theory.upper()
        self.material=material
        self.grid=grid
        self.thickness_m=h
        self.n_gauss=n_gauss
        self.geometry_model=normalized_model(geometry_model)
        self.electrical_on=material.kappa_f_per_m>0
        self.n_mechanical=(3 if self.theory=="KL" else
                           5 if self.theory in ("FSDT","REDDY") else
                           3*(int(self.theory[-1])+1))
        self.derivative_orders=MECHANICAL_JETS[:6 if self.theory.startswith("CUF") else 10]
        self.field_jets=MKRFieldJets(grid)
        params=dict(radius_m=grid.radius_m,thickness_m=h,n_gauss=n_gauss,
                    geometry_model=self.geometry_model)
        if self.theory.startswith("CUF"):
            from cuf_g3d_weak import CUFSurfaceJets, thickness_weak_coefficients
            self._jet_type=CUFSurfaceJets
        elif self.theory in ("FSDT","REDDY"):
            from cylinder_director_g3d import DirectorSurfaceJets, thickness_weak_coefficients
            self._jet_type=DirectorSurfaceJets
            params["theory"]=self.theory
        else:
            from cylinder_kl_g3d_weak import KLSurfaceJets, thickness_weak_coefficients
            self._jet_type=KLSurfaceJets
        self._local=partial(thickness_weak_coefficients,material=material,**params)

    @property
    def shape(self):
        return (self.n_mechanical,self.grid.nx,self.grid.ns)

    @property
    def area_weights(self):
        wx=np.full(self.grid.nx,self.grid.dx)
        wx[[0,-1]]*=.5
        return np.broadcast_to(wx[:,None]*self.grid.ds,
                               (self.grid.nx,self.grid.ns)).copy()

    def _point_jets(self, mder, eder, i, j):
        values=[a[:,i,j] for a in mder]
        if self.theory.startswith("CUF"):
            values=[a.reshape(3,int(self.theory[-1])+1) for a in values]
        return self._jet_type(*values,*(a[:,i,j] for a in eder))

    def coefficients(self, mechanical, electrical=None, *, velocity=None,
                     acceleration=None, density_kg_m3=None, eta_s=0.,
                     mass_damping_rate_per_s=0.):
        """Complete unweighted continuum jet densities on MKR's own fields.

        Includes the conservative electromechanical law, Kelvin--Voigt
        mechanics, consistent inertia and mass damping. Does not step time
        or solve a field problem. The caller supplies an instantaneous state.
        """
        q=_state(mechanical,self.shape,"MKR displacement")
        nx,ns=self.grid.nx,self.grid.ns
        if self.electrical_on:
            p=np.asarray(electrical,float)
            if p.ndim!=3 or p.shape[1:]!=(nx,ns) or not 1<=len(p)<=5:
                raise ValueError("active potential requires shape (P+1,nx,ns), P=0..4")
            p=_state(p,p.shape,"MKR potential")
        else:
            if electrical is not None:
                raise ValueError("electric field is OFF")
            p=np.zeros((1,nx,ns))
        eta=_nonnegative(eta_s,"viscosity")
        alpha=_nonnegative(mass_damping_rate_per_s,"mass damping")
        dynamic=(velocity is not None or acceleration is not None or
                 density_kg_m3 is not None or eta>0 or alpha>0)
        if dynamic and (velocity is None or acceleration is None or density_kg_m3 is None):
            raise ValueError("dynamic MKR requires velocity, acceleration and density together")
        mder=self.field_jets.prepare(q,self.derivative_orders)
        eder=self.field_jets.prepare(p,ELECTRIC_JETS)
        mc=np.zeros((len(mder),*self.shape))
        ec=np.zeros((3,*p.shape)) if self.electrical_on else None
        if dynamic:
            v=_state(velocity,self.shape,"MKR velocity")
            a=_state(acceleration,self.shape,"MKR acceleration")
            rho=_nonnegative(density_kg_m3,"density")
            if rho==0:
                raise ValueError("density must be positive")
            vd=self.field_jets.prepare(v,self.derivative_orders)
            ad=self.field_jets.prepare(a,self.derivative_orders)
            zero_elec=tuple(np.zeros((1,nx,ns)) for _ in ELECTRIC_JETS)
            if eta:
                from cylinder_kelvin_voigt import kelvin_voigt_thickness_weak
            if self.theory in ("FSDT","REDDY"):
                from cylinder_director_g3d import kinetic_weak_coefficients
                inertia=partial(kinetic_weak_coefficients,theory=self.theory,
                    radius_m=self.grid.radius_m,thickness_m=self.thickness_m,
                    density_kg_m3=rho,n_gauss=self.n_gauss)
        for i in range(nx):
            for j in range(ns):
                local=self._local(self._point_jets(mder,eder,i,j))
                mc[:,:,i,j]=local.mechanical.reshape(self.n_mechanical,len(mder)).T
                if ec is not None:
                    ec[:,:,i,j]=local.electrical.T
                if dynamic and eta:
                    visc=kelvin_voigt_thickness_weak(
                        self._point_jets(mder,zero_elec,i,j),
                        self._point_jets(vd,zero_elec,i,j),self.material,
                        theory=self.theory,radius_m=self.grid.radius_m,
                        thickness_m=self.thickness_m,eta_s=eta,
                        n_gauss=self.n_gauss,geometry_model=self.geometry_model)
                    mc[:,:,i,j]+=visc.mechanical.T
                if dynamic and self.theory in ("FSDT","REDDY"):
                    mc[:3,:,i,j]+=inertia(self._point_jets(ad,zero_elec,i,j)).T
                    if alpha:
                        mc[:3,:,i,j]+=alpha*inertia(self._point_jets(vd,zero_elec,i,j)).T
        if dynamic and self.theory.startswith("CUF"):
            from cuf_g3d_weak import mass_matrix_te
            n=int(self.theory[-1])+1
            mass=mass_matrix_te(radius_m=self.grid.radius_m,
                thickness_m=self.thickness_m,rho_kg_m3=rho,te_order=n-1)
            # Nodal inertia: no face averaging of primary velocities.
            acceleration_field=(a+alpha*v).reshape(3,n,nx,ns)
            mc[0]+=np.einsum("ij,cjkl->cikl",mass,acceleration_field).reshape(self.shape)
        elif dynamic and self.theory=="KL":
            from cylinder_kl_exact_inertia import kl_inertial_coefficients_from_jets
            params=dict(radius_m=self.grid.radius_m,thickness_m=self.thickness_m,
                        density_kg_m3=rho,geometry_model=self.geometry_model)
            mc[:3]+=kl_inertial_coefficients_from_jets(mder[:3],vd[:3],ad[:3],**params)
            if alpha:
                # M(q)*v: zero velocity in this evaluation excludes the
                # gyroscopic velocity-square term from Rayleigh damping.
                zeros=tuple(np.zeros_like(q) for _ in range(3))
                mc[:3]+=alpha*kl_inertial_coefficients_from_jets(mder[:3],zeros,vd[:3],**params)
        if not np.isfinite(mc).all() or (ec is not None and not np.isfinite(ec).all()):
            raise ValueError("nonfinite MKR constitutive coefficients")
        return mc,ec

    def balance(self, mechanical, electrical=None, *, mechanical_load=None,
                electrical_charge=None, velocity=None, acceleration=None,
                density_kg_m3=None, eta_s=0., mass_damping_rate_per_s=0.):
        mc,ec=self.coefficients(mechanical,electrical,velocity=velocity,
            acceleration=acceleration,density_kg_m3=density_kg_m3,
            eta_s=eta_s,mass_damping_rate_per_s=mass_damping_rate_per_s)
        return direct_strong_jet_balance(mc,self.derivative_orders,self.grid,ec,
            mechanical_load=mechanical_load,electrical_charge=electrical_charge)

    def solve(self, mechanical, electrical=None, *, axial_orders, mechanical_ends,
              mechanical_state_scale, mechanical_equation_scale,
              electrical_ends=None, electric_gauge=None,
              electrical_state_scale=None, electrical_equation_scale=None,
              mechanical_load=None, electrical_charge=None, settings=None):
        """Opt-in stationary solve with explicit field-specific physical ends.

        All seven theories use their native MKR preparation. Differential
        orders, end conditions and dimensional scales remain caller inputs;
        the maximum stored jet is not used to guess physical orders.
        """
        from cylinder_direct_strong_boundary import assemble_physical_end_rows
        from cylinder_direct_strong_newton import (
            StrongNewtonSettings, solve_strong_nonlinear_equations,
        )
        def equation(q,p):
            balance=self.balance(q,p,mechanical_load=mechanical_load,
                                 electrical_charge=electrical_charge)
            rows=assemble_physical_end_rows(balance,q,self.grid,
                axial_orders=axial_orders,mechanical_ends=mechanical_ends,
                electrical=p,electrical_ends=electrical_ends,
                gauge_closed_elsewhere=electric_gauge is not None)
            return rows.mechanical,rows.electrical
        return solve_strong_nonlinear_equations(equation,mechanical,electrical,
            mechanical_state_scale=mechanical_state_scale,
            mechanical_equation_scale=mechanical_equation_scale,
            electrical_state_scale=electrical_state_scale,
            electrical_equation_scale=electrical_equation_scale,
            electric_gauge=electric_gauge,
            settings=StrongNewtonSettings() if settings is None else settings)


__all__=("DirectStrongShell", "THEORIES")
