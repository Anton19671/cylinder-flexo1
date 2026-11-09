"""Independent nonlinear MVI Kelvin-Voigt virtual-work assembly.

Own SBP42/Fourier spatial derivatives and weighted weak adjoints. The local
material law/through-thickness constitutive physics is shared with MKR, but
the field discretization and surface assembly are strictly independent.
No MKR/FDM/legacy variational numerical driver is imported.

This method returns generalized MECHANICAL nodal work in the same units as
independent_mvi.coupled_surface. No electric viscosity or electric inertia.
"""
from __future__ import annotations

import math
import numpy as np

from independent_mvi.coupled_surface import (
    NativeCoupledMVISurface, MECHANICAL_MULTI,
)
from cylinder_kelvin_voigt import kelvin_voigt_thickness_weak


class NativeCoupledMVIViscosity:
    """Exact local KV virtual work with independent MVI surface adjoints."""

    def __init__(self, surface: NativeCoupledMVISurface, *, eta_s):
        if not isinstance(surface, NativeCoupledMVISurface):
            raise TypeError("NativeCoupledMVISurface required")
        if isinstance(eta_s,(bool,np.bool_)):
            raise ValueError("MVI_KV_INVALID_TIME")
        try:
            eta=float(eta_s)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError("MVI_KV_INVALID_TIME") from exc
        if not math.isfinite(eta) or eta<0:
            raise ValueError("MVI_KV_INVALID_TIME")
        self.surface=surface
        self.eta_s=eta

    def _jets(self, mechanical, i, j):
        surface=self.surface
        count=surface.n_mechanical
        shape=(count,surface.grid.nx,surface.grid.ns)
        f=np.asarray(mechanical,dtype=float)
        if f.shape!=shape or not np.isfinite(f).all():
            raise ValueError("MVI_KV_NONFINITE_MECHANICAL_GRID")
        orders=surface.mechanical_multi
        values=[]
        for ax,ang in orders:
            # Direct independent MVI differentiation in physical x,s:
            # derivatives are applied in the right factor order.
            derivative=np.einsum(
                "ip,cpq,jq->cij",surface._dx_power[ax],f,
                surface._ds_power[ang],optimize=True)
            field=derivative[:,i,j]
            if surface.theory.startswith("CUF_TE"):
                field=field.reshape(3,int(surface.theory[-1])+1)
            values.append(field)
        # Viscous constitutive law does not depend on potential, but
        # each local jets container still has explicit electric slots.
        zeros=np.zeros(max(1,surface.n_electric))
        return surface._jet_type(*values,zeros,zeros,zeros)

    def apply(self,mechanical,velocity):
        """Return viscosity nodal virtual work, without altering input arrays."""
        surface=self.surface
        shape=(surface.n_mechanical,surface.grid.nx,surface.grid.ns)
        q=np.asarray(mechanical,dtype=float)
        v=np.asarray(velocity,dtype=float)
        if (q.shape!=shape or v.shape!=shape or
                not np.isfinite(q).all() or not np.isfinite(v).all()):
            raise ValueError("MVI_KV_REQUIRES_FINITE_Q_V_FIELDS")
        if self.eta_s==0:
            return np.zeros(shape,dtype=float)
        local=np.zeros((surface.n_mechanical,len(surface.mechanical_multi),
                        surface.grid.nx,surface.grid.ns),dtype=float)
        for i in range(surface.grid.nx):
            for j in range(surface.grid.ns):
                qjets=self._jets(q,i,j)
                vjets=self._jets(v,i,j)
                result=kelvin_voigt_thickness_weak(
                    qjets,vjets,surface.material,
                    theory=surface.theory,
                    radius_m=surface.grid.radius_m,
                    thickness_m=surface.thickness_m,
                    eta_s=self.eta_s,
                    n_gauss=surface.n_gauss,
                    geometry_model=surface.geometry_model)
                coefficients=np.asarray(result.coefficients,dtype=float)
                expected=(surface.n_mechanical,len(surface.mechanical_multi))
                if coefficients.shape!=expected or not np.isfinite(coefficients).all():
                    raise ValueError("MVI_KV_LOCAL_WEAK_SHAPE_MISMATCH")
                local[:,:,i,j]=coefficients
        nodal=np.zeros(shape)
        area=surface.grid.weights
        for idx,(ax,ang) in enumerate(surface.mechanical_multi):
            nodal+=np.einsum(
                "ip,cij,jq->cpq",surface._dx_power[ax],
                local[:,idx]*area[None,:,:],surface._ds_power[ang],
                optimize=True)
        if not np.isfinite(nodal).all():
            raise ValueError("MVI_KV_NONFINITE_NODAL_WORK")
        return nodal


__all__=("NativeCoupledMVIViscosity",)
