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

    def _surface_jets(self, field):
        """Build each own MVI spatial derivative ONCE per nonlinear evaluation."""
        surface=self.surface
        return tuple(np.einsum(
            "ip,cpq,jq->cij",surface._dx_power[ax],field,
            surface._ds_power[ang],optimize=True)
            for ax,ang in surface.mechanical_multi)

    def _jets(self, derivative_fields, i, j):
        surface=self.surface
        values=[]
        for derivative in derivative_fields:
            field=derivative[:,i,j]
            if surface.theory.startswith("CUF_TE"):
                field=field.reshape(3,int(surface.theory[-1])+1)
            values.append(field)
        # Viscous constitutive law does not depend on potential, but
        # each local jets container still has explicit electric slots.
        zeros=np.zeros(max(1,surface.n_electric))
        return surface._jet_type(*values,zeros,zeros,zeros)

    def local_weak_coefficients(self,mechanical,velocity):
        """Return unweighted local Kelvin-Voigt coefficients, not a force.

        Shape (n_mechanical,n_mechanical_jets,nx,ns), indexed by exactly
        the same own MVI jet families as the conservative virtual work.
        Reused by apply() and the discrete dynamic boundary-work audit.
        """
        surface=self.surface
        shape=(surface.n_mechanical,surface.grid.nx,surface.grid.ns)
        q=np.asarray(mechanical,dtype=float)
        v=np.asarray(velocity,dtype=float)
        if (q.shape!=shape or v.shape!=shape or
                not np.isfinite(q).all() or not np.isfinite(v).all()):
            raise ValueError("MVI_KV_REQUIRES_FINITE_Q_V_FIELDS")
        local=np.zeros((surface.n_mechanical,len(surface.mechanical_multi),
                        surface.grid.nx,surface.grid.ns),dtype=float)
        if self.eta_s==0:
            return local
        qderivatives=self._surface_jets(q)
        vderivatives=self._surface_jets(v)
        for i in range(surface.grid.nx):
            for j in range(surface.grid.ns):
                qjets=self._jets(qderivatives,i,j)
                vjets=self._jets(vderivatives,i,j)
                result=kelvin_voigt_thickness_weak(
                    qjets,vjets,surface.material,
                    theory=surface.theory,
                    radius_m=surface.grid.radius_m,
                    thickness_m=surface.thickness_m,
                    eta_s=self.eta_s,
                    n_gauss=surface.n_gauss,
                    geometry_model=surface.geometry_model)
                coefficients=np.asarray(result.mechanical,dtype=float)
                expected=(surface.n_mechanical,len(surface.mechanical_multi))
                if coefficients.shape!=expected or not np.isfinite(coefficients).all():
                    raise ValueError("MVI_KV_LOCAL_WEAK_SHAPE_MISMATCH")
                local[:,:,i,j]=coefficients
        return local

    def apply(self,mechanical,velocity):
        """Return own MVI viscosity nodal work via the SAME local jets."""
        surface=self.surface
        local=self.local_weak_coefficients(mechanical,velocity)
        nodal=np.zeros((surface.n_mechanical,
                        surface.grid.nx,surface.grid.ns))
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
