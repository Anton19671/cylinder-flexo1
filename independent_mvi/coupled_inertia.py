"""Independent MVI mechanical inertia on its OWN SBP/Fourier surface grid.

Seven physical theories. Local mass/kinematic laws are shared continuum
physics, but neither MKR spatial derivatives nor reference variational
nodal assembly is used here. This is an inertial VIRTUAL-WORK action:
no artificial kinetic potential is minimized, no electric inertia exists.

CUF TE1..TE4 use exact thickness mass moments, including odd curvature
moments. FSDT/Reddy use local kinetic weak coefficients with first-order
velocity jets; Reddy's gradient-dependent mass is retained. KL uses the
exact director acceleration (including quadratic velocity terms for GL).
All mechanical nodal actions are quadrature-weighted in joule units.
"""
from __future__ import annotations

import math
import numpy as np

from independent_mvi.coupled_surface import NativeCoupledMVISurface
from cuf_g3d_weak import mass_matrix_te
from cylinder_director_g3d import (DirectorSurfaceJets,
                                    kinetic_weak_coefficients)
from cylinder_kl_exact_inertia import kl_inertial_coefficients_from_jets


def _field(surface,field,label):
    value=np.asarray(field,dtype=float)
    expected=(surface.n_mechanical,surface.grid.nx,surface.grid.ns)
    if value.shape!=expected or not np.isfinite(value).all():
        raise ValueError(label+": finite mechanical grid field required")
    return value


def _derivatives(field,grid):
    return (field,
            np.einsum("ij,cjk->cik",grid.dx,field,optimize=True),
            np.einsum("jk,cik->cij",grid.ds,field,optimize=True))


def _transpose_weak(grid,coeff):
    """Own MVI weighted transpose, for variations [q,q_x,q_s]."""
    W=grid.weights
    c=np.asarray(coeff,float)
    expected=(3,c.shape[1],grid.nx,grid.ns)
    if c.shape!=expected or not np.isfinite(c).all():
        raise ValueError("MVI_INERTIA_INVALID_LOCAL_WEAK_COEFFICIENTS")
    return (c[0]*W[None,:,:]
         +np.einsum("ip,cij->cpj",grid.dx,c[1]*W[None,:,:],
                    optimize=True)
         +np.einsum("jp,cij->cip",grid.ds,c[2]*W[None,:,:],
                    optimize=True))


class NativeCoupledMVIInertia:
    """Physical mass action, independent of the MKR spatial method.

    The optional damping rate is proportional to the same physical mass
    operator, not a replacement for Kelvin--Voigt constitutive damping.
    """
    def __init__(self,surface:NativeCoupledMVISurface, *,
                 density_kg_m3, mass_damping_rate_per_s=0.):
        if not isinstance(surface,NativeCoupledMVISurface):
            raise TypeError("NativeCoupledMVISurface required")
        for name,value in (("density",density_kg_m3),
                           ("mass damping",mass_damping_rate_per_s)):
            if isinstance(value,(bool,np.bool_)):
                raise ValueError(name+": finite physical parameter required")
            try:value=float(value)
            except (ValueError,TypeError,OverflowError) as exc:
                raise ValueError(name+": finite physical parameter required") from exc
            if not math.isfinite(value) or (
                    value<=0. if name=="density" else value<0.):
                raise ValueError(name+": invalid physical parameter")
            setattr(self,"density_kg_m3" if name=="density" else
                    "mass_damping_rate_per_s",value)
        self.surface=surface
        self._mass_te=(
            mass_matrix_te(radius_m=surface.grid.radius_m,
                           thickness_m=surface.thickness_m,
                           rho_kg_m3=self.density_kg_m3,
                           te_order=int(surface.theory[-1]))
            if surface.theory.startswith("CUF") else None)

    def _director_coefficients(self,accelerations):
        model=self.surface
        fields=_derivatives(accelerations,model.grid)
        out=np.zeros((3,model.n_mechanical,
                      model.grid.nx,model.grid.ns))
        zeros=np.zeros(5)
        potential=np.zeros(1)
        for i in range(model.grid.nx):
            for j in range(model.grid.ns):
                args=[fields[0][:,i,j],fields[1][:,i,j],
                      fields[2][:,i,j]]
                jets=DirectorSurfaceJets(
                    *args,zeros,zeros,zeros,zeros,zeros,zeros,zeros,
                    potential,potential,potential)
                c=kinetic_weak_coefficients(
                    jets,theory=model.theory,
                    radius_m=model.grid.radius_m,
                    thickness_m=model.thickness_m,
                    density_kg_m3=self.density_kg_m3,
                    n_gauss=model.n_gauss)
                out[:,:,i,j]=np.asarray(c,float).T
        return out

    def apply(self,mechanical,velocity,acceleration):
        """Return (n_mechanical,nx,ns) inertial + mass-damping nodal work.

        All arguments are mechanical fields, never X/Y factor coordinates.
        The local kinematic coefficients are evaluated using only this
        MVI method's independent axial/circumferential derivative matrices.
        """
        model=self.surface
        q=_field(model,mechanical,"mechanical")
        v=_field(model,velocity,"velocity")
        a=_field(model,acceleration,"acceleration")
        rate=self.mass_damping_rate_per_s
        if self._mass_te is not None:
            m=int(model.theory[-1])+1
            effective=(a+rate*v).reshape(3,m,model.grid.nx,model.grid.ns)
            nodal=np.einsum("ij,cjkl->cikl",self._mass_te,effective,
                            optimize=True).reshape(q.shape)
            nodal=nodal*model.grid.weights[None,:,:]
        elif model.theory in ("FSDT","REDDY"):
            coefficients=self._director_coefficients(a+rate*v)
            nodal=_transpose_weak(model.grid,coefficients)
        elif model.theory=="KL":
            qj=_derivatives(q,model.grid)
            vj=_derivatives(v,model.grid)
            aj=_derivatives(a,model.grid)
            coeff=kl_inertial_coefficients_from_jets(
                qj,vj,aj,
                radius_m=model.grid.radius_m,
                thickness_m=model.thickness_m,
                density_kg_m3=self.density_kg_m3,
                geometry_model=model.geometry_model)
            nodal=_transpose_weak(model.grid,coeff)
            if rate:
                z=np.zeros_like(v)
                damping=kl_inertial_coefficients_from_jets(
                    qj,_derivatives(z,model.grid),vj,
                    radius_m=model.grid.radius_m,
                    thickness_m=model.thickness_m,
                    density_kg_m3=self.density_kg_m3,
                    geometry_model=model.geometry_model)
                nodal+=rate*_transpose_weak(model.grid,damping)
        else:
            raise ValueError("MVI_INERTIA_UNSUPPORTED_THEORY")
        if nodal.shape!=q.shape or not np.isfinite(nodal).all():
            raise ValueError("MVI_INERTIA_NONFINITE_MECHANICAL_WORK")
        return nodal


__all__=("NativeCoupledMVIInertia",)
