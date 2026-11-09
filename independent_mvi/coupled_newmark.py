"""Independent coupled MVI implicit Newmark, fixed rank 1..8.

This route uses independent_mvi.coupled_surface (own SBP42/Fourier),
coupled_inertia (exact local physical mass laws), and OWN Y/X reduced
Newton halfsteps. No reference FDM/MKR, full-grid Newton, SVD, rank
growth, reference dynamic residual, or shared time integrator.

Experimental boundary subset: immobile parallel mechanical ends and
grounded electric modes. The caller supplies consistent q0,v0,a0,phi0.
This module does not certify higher-gradient natural actions, moving
supports, or a physical initial-acceleration solve. Kelvin-Voigt
constitutive force is assembled by independent_mvi.coupled_viscosity,
separately from mass-proportional damping. No CASE/GUI dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram, _check_end_factors, _scales,
)
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_viscosity import NativeCoupledMVIViscosity
from independent_mvi.coupled_static import (
    CoupledMVISweepSettings, CoupledMVISweepResult,
    solve_coupled_mvi_static,
)
from independent_mvi.coupled_surface import (
    CoupledMVIFactors, CoupledMVIWork, NativeCoupledMVISurface,
)


def _mechanical(surface,a,label):
    v=np.asarray(a,dtype=float)
    expected=(surface.n_mechanical,surface.grid.nx,surface.grid.ns)
    if v.shape!=expected or not np.isfinite(v).all():
        raise ValueError(label+": finite mechanical grid field required")
    return v.copy()


def _positive(value,label,zero_ok=False):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(label+": finite physical value required")
    v=float(value)
    if not math.isfinite(v) or (v<0 if zero_ok else v<=0):
        raise ValueError(label+": finite physical value required")
    return v


@dataclass(frozen=True)
class CoupledMVIDynamicState:
    time_s: float
    factors: CoupledMVIFactors
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self):
        t=_positive(self.time_s,"time",zero_ok=True)
        if not isinstance(self.factors,CoupledMVIFactors):
            raise TypeError("coupled fixed-rank factors required")
        n,nx,ns=self.factors.x.shape[0],self.factors.x.shape[2],self.factors.y.shape[2]
        for name in ("velocity","acceleration"):
            value=np.asarray(getattr(self,name),float)
            if (value.ndim!=3 or value.shape[1:]!=(nx,ns) or
                    not 1<=value.shape[0]<=n or not np.isfinite(value).all()):
                raise ValueError(name+": finite mechanical grid field required")
            copy=value.copy();copy.setflags(write=False)
            object.__setattr__(self,name,copy)
        if self.velocity.shape!=self.acceleration.shape:
            raise ValueError("velocity and acceleration shape mismatch")
        object.__setattr__(self,"time_s",t)


@dataclass(frozen=True)
class CoupledMVIDynamicStep:
    state: CoupledMVIDynamicState | None
    solver: CoupledMVISweepResult
    attempted_time_s: float
    status: str


class NativeCoupledMVINewmark:
    def __init__(self,surface:NativeCoupledMVISurface,
                 inertia:NativeCoupledMVIInertia, *,
                 program:CoupledMVIEndProgram,beta=.25,gamma=.5,
                 stationary_tolerance=1.e-10, eta_s=0.):
        if not isinstance(surface,NativeCoupledMVISurface):
            raise TypeError("NativeCoupledMVISurface required")
        if not isinstance(inertia,NativeCoupledMVIInertia) or inertia.surface is not surface:
            raise TypeError("method-owned inertia must match surface")
        if not isinstance(program,CoupledMVIEndProgram):
            raise TypeError("explicit MVI end program required")
        program.validate(surface)
        self.beta=_positive(beta,"Newmark beta")
        self.gamma=_positive(gamma,"Newmark gamma")
        if not (.5<=self.gamma<=1. and
                self.beta>=.25*(self.gamma+.5)**2):
            raise ValueError("MVI_NEWMARK_UNSTABLE_PARAMETER_PAIR")
        self.stationary_tolerance=_positive(stationary_tolerance,
                                             "stationary boundary tolerance")
        self.surface=surface
        self.inertia=inertia
        self.viscosity=NativeCoupledMVIViscosity(surface,eta_s=eta_s)
        self.program=program

    def _validate_state(self,state):
        if not isinstance(state,CoupledMVIDynamicState):
            raise TypeError("CoupledMVIDynamicState required")
        surf=self.surface
        _check_end_factors(surf,state.factors,self.program)
        q=surf.reconstruct(state.factors)
        v=_mechanical(surf,state.velocity,"velocity")
        a=_mechanical(surf,state.acceleration,"acceleration")
        if len(q)!=surf.n_fields:
            raise ValueError("MVI_DYNAMIC_FIELD_COUNT_MISMATCH")
        free=self.program.free_axial_indices(surf)
        for c in range(surf.n_mechanical):
            blocked=np.ones(surf.grid.nx,dtype=bool)
            blocked[free[c]]=False
            for field in (v,a):
                if np.max(np.abs(field[c,blocked,:]),initial=0.)>self.stationary_tolerance:
                    raise ValueError("MVI_DYNAMIC_MOVING_SUPPORT_NOT_IMPLEMENTED")
        return q,v,a

    def advance(self,state:CoupledMVIDynamicState,dt_s, *,
                field_scales,energy_scale_j,
                mechanical_load=None,free_charge=None,
                settings=CoupledMVISweepSettings(),
                on_iteration=None,cancelled=None):
        """One implicit step, accepted ONLY on full nodal stationarity.

        Inertia is the actual derivative of kinetic energy; all mechanical
        unknowns are changed through fixed-rank factors. Quasi-static
        electric modes remain in the SAME nonlinear Y/X iteration.
        """
        surf=self.surface
        q,v0,a0=self._validate_state(state)
        dt=_positive(dt_s,"Newmark step")
        target_time=state.time_s+dt
        if not math.isfinite(target_time) or target_time<=state.time_s:
            raise ValueError("MVI_DYNAMIC_UNREPRESENTABLE_TIME")
        if not isinstance(settings,CoupledMVISweepSettings):
            raise TypeError("CoupledMVISweepSettings required")
        _scales(surf,field_scales,energy_scale_j)
        mechanical0=q[:surf.n_mechanical]
        q_predictor=(mechanical0+dt*v0+
                     (.5-self.beta)*dt**2*a0)
        v_predictor=v0+(1.-self.gamma)*dt*a0
        if not np.isfinite(q_predictor).all() or not np.isfinite(v_predictor).all():
            raise ValueError("MVI_DYNAMIC_NONFINITE_PREDICTOR")
        for c,ids in enumerate(self.program.free_axial_indices(surf)[:surf.n_mechanical]):
            blocked=np.ones(surf.grid.nx,bool);blocked[ids]=False
            if (np.max(np.abs(q_predictor[c,blocked,:]),initial=0.)>
                    self.stationary_tolerance):
                raise ValueError("MVI_DYNAMIC_INCOMPATIBLE_BOUNDARY_PREDICTOR")

        def effective_work(factors):
            work=surf.weak_work(factors,mechanical_load=mechanical_load,
                                free_charge=free_charge)
            qm=work.mechanical
            acceleration=(qm-q_predictor)/(self.beta*dt**2)
            velocity=v_predictor+self.gamma*dt*acceleration
            inertia_nodal=self.inertia.apply(qm,velocity,acceleration)
            full=np.asarray(work.nodal_gradient,float).copy()
            full[:surf.n_mechanical]+=inertia_nodal
            if self.viscosity.eta_s:
                full[:surf.n_mechanical]+=self.viscosity.apply(qm,velocity)
            # Construct all-field projection explicitly. There is no
            # SVD, MKR or full-grid Newton update.
            gx=np.einsum("cij,crj->cri",full,factors.y,optimize=True)
            gy=np.einsum("cij,cri->crj",full,factors.x,optimize=True)
            if not (np.isfinite(gx).all() and np.isfinite(gy).all()):
                raise ValueError("MVI_DYNAMIC_NONFINITE_EFFECTIVE_WORK")
            return CoupledMVIWork(work.energy_j,qm,work.electrical,
                                  gx,gy,work.rank,
                                  nodal_gradient=full)

        result=solve_coupled_mvi_static(
            surf,state.factors,program=self.program,
            field_scales=field_scales,energy_scale_j=energy_scale_j,
            settings=settings,mechanical_load=mechanical_load,
            free_charge=free_charge,work_evaluator=effective_work,
            on_iteration=on_iteration,cancelled=cancelled)
        if not result.converged:
            return CoupledMVIDynamicStep(None,result,target_time,
                                          "MVI_NEWMARK_"+result.status)
        next_q=surf.reconstruct(result.factors)[:surf.n_mechanical]
        next_a=(next_q-q_predictor)/(self.beta*dt**2)
        next_v=v_predictor+self.gamma*dt*next_a
        next_state=CoupledMVIDynamicState(
            target_time,result.factors,next_v,next_a)
        self._validate_state(next_state)
        return CoupledMVIDynamicStep(
            next_state,result,target_time,"MVI_NEWMARK_CONVERGED")


__all__=("CoupledMVIDynamicState","CoupledMVIDynamicStep",
         "NativeCoupledMVINewmark")
