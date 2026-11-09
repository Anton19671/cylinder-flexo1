"""Full dynamic discrete Green audit of native independent coupled MVI.

Every term uses EXACTLY the local weak coefficients already assembled by
the independent MVI spatial method: conservative/electric enthalpy,
mechanical inertia plus mass-rate damping, and Kelvin-Voigt viscosity.
No second dynamic constitutive law, MKR spatial discretization, SVD,
independent physical time solve or boundary enforcement is introduced.

The exact algebraic Green decomposition describes the selected discrete
variational work; it does NOT certify physical microfree face/edge
conditions, radial electric insulation, floating electrical gauge, or
continuum boundary convergence. Inertia and Kelvin-Voigt carry no
electric variation and are never assigned an electrical flux.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from independent_mvi.coupled_discrete_green import (
    MVIDiscreteGreenIdentity,coupled_mvi_discrete_green_identity,
)
from independent_mvi.coupled_surface import (
    CoupledMVIWork,CoupledMVIFactors,NativeCoupledMVISurface,
)
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_viscosity import NativeCoupledMVIViscosity


@dataclass(frozen=True)
class MVIDynamicGreenAudit:
    conservative: MVIDiscreteGreenIdentity
    inertia: MVIDiscreteGreenIdentity
    kelvin_voigt: MVIDiscreteGreenIdentity
    total: MVIDiscreteGreenIdentity
    mechanical_nodal_work_error: float
    electric_nodal_work_error: float
    status: str="MVI_DYNAMIC_EXACT_DISCRETE_GREEN_DIAGNOSTIC"


def _nodal_from_coefficients(surface,coefficients):
    """OWN weak transpose of all independent x/s surface jets."""
    expected=(surface.n_fields,len(surface.multi),
              surface.grid.nx,surface.grid.ns)
    if np.shape(coefficients)!=expected:
        raise ValueError("MVI_DYNAMIC_GREEN_COEFFICIENT_SHAPE_INVALID")
    coeff=np.asarray(coefficients,dtype=float)
    if not np.isfinite(coeff).all():
        raise ValueError("MVI_DYNAMIC_GREEN_NONFINITE_COEFFICIENT")
    nodal=np.zeros(surface.shape)
    for k,(ax,ang) in enumerate(surface.multi):
        if np.any(coeff[:,k]):
            nodal+=np.einsum(
                "ip,cij,jq->cpq",surface._dx_power[ax],
                coeff[:,k]*surface.grid.weights[None,:,:],
                surface._ds_power[ang],optimize=True)
    if not np.isfinite(nodal).all():
        raise ValueError("MVI_DYNAMIC_GREEN_NONFINITE_NODAL")
    return nodal


def _work_from_coefficients(surface,factors,coefficients):
    nodal=_nodal_from_coefficients(surface,coefficients)
    fields=surface.reconstruct(factors)
    return CoupledMVIWork(
        0.,fields[:surface.n_mechanical],
        fields[surface.n_mechanical:] if surface.electrical_on else None,
        np.einsum("cij,crj->cri",nodal,factors.y,optimize=True),
        np.einsum("cij,cri->crj",nodal,factors.x,optimize=True),
        factors.rank,nodal_gradient=nodal,coefficient_jets=coefficients)


def audit_mvi_dynamic_discrete_green(
        surface:NativeCoupledMVISurface,
        factors:CoupledMVIFactors, velocity,acceleration,
        *, inertia:NativeCoupledMVIInertia,
        viscosity:NativeCoupledMVIViscosity,
        mechanical_load=None,free_charge=None,
        relative_tolerance=2.e-9):
    """Audit exact local dynamic work and discrete end/volume split.

    No material/G3D law is rederived here. Fail closed if the mass or
    Kelvin-Voigt work from its own local coefficients differs from the
    actual corresponding method-owned nodal force. This also checks
    the absence of mechanical-rate forces in electrical equations.
    """
    if not isinstance(surface,NativeCoupledMVISurface):
        raise TypeError("MVI_DYNAMIC_GREEN_REQUIRES_OWN_SURFACE")
    surface._check_factors(factors)
    if (not isinstance(inertia,NativeCoupledMVIInertia) or
            inertia.surface is not surface or
            not isinstance(viscosity,NativeCoupledMVIViscosity) or
            viscosity.surface is not surface):
        raise TypeError("MVI_DYNAMIC_GREEN_REQUIRES_MATCHING_PHYSICAL_OPERATORS")
    shape=(surface.n_mechanical,surface.grid.nx,surface.grid.ns)
    v=np.asarray(velocity,dtype=float)
    a=np.asarray(acceleration,dtype=float)
    if (v.shape!=shape or a.shape!=shape or
            not np.isfinite(v).all() or not np.isfinite(a).all()):
        raise ValueError("MVI_DYNAMIC_GREEN_REQUIRES_FINITE_MECHANICAL_RATES")
    conservative=surface.weak_work(
        factors,mechanical_load=mechanical_load,free_charge=free_charge)
    if (conservative.coefficient_jets is None or
            conservative.nodal_gradient is None):
        raise ValueError("MVI_DYNAMIC_GREEN_REQUIRES_PHYSICAL_WEAK_COEFFICIENTS")
    physical=surface.reconstruct(factors)[:surface.n_mechanical]
    ci=inertia.local_weak_coefficients(physical,v,a)
    ck=viscosity.local_weak_coefficients(physical,v)
    expected=(surface.n_mechanical,len(surface.mechanical_multi),
              surface.grid.nx,surface.grid.ns)
    if np.shape(ci)!=expected or np.shape(ck)!=expected:
        raise ValueError("MVI_DYNAMIC_GREEN_LOCAL_MODEL_JET_MISMATCH")
    inertia_coeff=np.zeros_like(conservative.coefficient_jets)
    viscosity_coeff=np.zeros_like(conservative.coefficient_jets)
    inertia_coeff[:surface.n_mechanical,:len(surface.mechanical_multi)]=ci
    viscosity_coeff[:surface.n_mechanical,:len(surface.mechanical_multi)]=ck
    inertia_work=_work_from_coefficients(surface,factors,inertia_coeff)
    viscosity_work=_work_from_coefficients(surface,factors,viscosity_coeff)

    # Independent route's real nodal force is computed using its ordinary
    # apply() functions, so a fake correct coefficient map alone cannot
    # cause the audit to pass after an assembly error.
    actual_inertia=inertia.apply(physical,v,a)
    actual_kv=viscosity.apply(physical,v)
    mechanical_error=max(
        float(np.max(np.abs(actual_inertia-
                            inertia_work.nodal_gradient[:surface.n_mechanical]),
                     initial=0.)),
        float(np.max(np.abs(actual_kv-
                            viscosity_work.nodal_gradient[:surface.n_mechanical]),
                     initial=0.)))
    scale=max(1.,float(np.max(np.abs(actual_inertia),initial=0.)),
              float(np.max(np.abs(actual_kv),initial=0.)),
              float(np.max(np.abs(
                  conservative.nodal_gradient[:surface.n_mechanical]),initial=0.)))
    if mechanical_error>relative_tolerance*scale:
        raise ValueError("MVI_DYNAMIC_GREEN_MASS_OR_KV_FORCE_MISMATCH")
    electrical_error=max(
        float(np.max(np.abs(inertia_work.nodal_gradient[
            surface.n_mechanical:]),initial=0.)),
        float(np.max(np.abs(viscosity_work.nodal_gradient[
            surface.n_mechanical:]),initial=0.)))
    if electrical_error!=0.:
        raise ValueError("MVI_DYNAMIC_GREEN_ELECTRIC_MASS_OR_VISCOSITY_FORBIDDEN")

    total_coeff=np.asarray(conservative.coefficient_jets).copy()
    total_coeff+=inertia_coeff
    total_coeff+=viscosity_coeff
    total_work=_work_from_coefficients(surface,factors,total_coeff)
    expected_nodal=np.asarray(conservative.nodal_gradient).copy()
    expected_nodal[:surface.n_mechanical]+=actual_inertia+actual_kv
    consistency=float(np.max(np.abs(expected_nodal-
                                   total_work.nodal_gradient),initial=0.))
    full_scale=max(1.,float(np.max(np.abs(expected_nodal),initial=0.)),
                   float(np.max(np.abs(total_work.nodal_gradient),initial=0.)))
    if consistency>relative_tolerance*full_scale:
        raise ValueError("MVI_DYNAMIC_GREEN_TOTAL_VIRTUAL_WORK_MISMATCH")

    return MVIDynamicGreenAudit(
        coupled_mvi_discrete_green_identity(
            surface,conservative,relative_tolerance=relative_tolerance),
        coupled_mvi_discrete_green_identity(
            surface,inertia_work,relative_tolerance=relative_tolerance),
        coupled_mvi_discrete_green_identity(
            surface,viscosity_work,relative_tolerance=relative_tolerance),
        coupled_mvi_discrete_green_identity(
            surface,total_work,relative_tolerance=relative_tolerance),
        mechanical_error,electrical_error)


__all__=("MVIDynamicGreenAudit","audit_mvi_dynamic_discrete_green")
