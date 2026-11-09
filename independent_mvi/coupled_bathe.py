"""Own two-stage implicit Bathe dynamics of fixed-rank coupled MVI.

Independent of MKR temporal/spatial drivers and legacy variational
dynamic MVI. Each stage solves its own nonlinear stationary effective
virtual-work equations in the separated Y/X directions. Both stages
require the full admissible nodal first-variation gate to pass.

No global FDM Newton, SVD, electric inertia, artificial rank growth,
production CASE dispatch or loaded physical test is introduced.
The same explicit stationary-end and electric-grounding subset as the
new coupled MVI static/Newmark branch applies. Kelvin-Voigt rate
constitutive laws, moving supports and higher-order microfree end
closure are not certified and must not be silently accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from independent_mvi.coupled_halfstep import _scales
from independent_mvi.coupled_newmark import (
    CoupledMVIDynamicState, NativeCoupledMVINewmark, _positive,
)
from independent_mvi.coupled_static import (
    CoupledMVISweepSettings, CoupledMVISweepResult,
    solve_coupled_mvi_static,
)
from independent_mvi.coupled_surface import CoupledMVIWork


BATHE_GAMMA=2.-math.sqrt(2.)


@dataclass(frozen=True)
class CoupledMVIBatheStep:
    state: CoupledMVIDynamicState | None
    first_stage: CoupledMVIDynamicState | None
    first_solve: CoupledMVISweepResult
    final_solve: CoupledMVISweepResult | None
    attempted_time_s: float
    status: str


class NativeCoupledMVIBathe:
    """Two different implicit time formulas; no call to Newmark.advance."""

    def __init__(self, configuration: NativeCoupledMVINewmark):
        if not isinstance(configuration,NativeCoupledMVINewmark):
            raise TypeError("explicit native coupled MVI physical configuration required")
        self.configuration=configuration

    def _solve_stage(self, factors, *, predictor,velocity_predictor,
                     acceleration_factor, velocity_factor, time_s,
                     field_scales,energy_scale_j,mechanical_load,free_charge,
                     settings,on_iteration,cancelled):
        config=self.configuration
        surf=config.surface
        def evaluate(f):
            work=surf.weak_work(
                f,mechanical_load=mechanical_load,free_charge=free_charge)
            q=work.mechanical
            a=acceleration_factor*(q-predictor)
            v=velocity_predictor+velocity_factor*a
            nodal=work.nodal_gradient.copy()
            nodal[:surf.n_mechanical]+=config.inertia.apply(q,v,a)
            gx=np.einsum("cij,crj->cri",nodal,f.y,optimize=True)
            gy=np.einsum("cij,cri->crj",nodal,f.x,optimize=True)
            if not (np.isfinite(gx).all() and np.isfinite(gy).all()):
                raise ValueError("MVI_BATHE_NONFINITE_STAGE_WORK")
            return CoupledMVIWork(work.energy_j,q,work.electrical,
                                  gx,gy,f.rank,nodal_gradient=nodal)
        solution=solve_coupled_mvi_static(
            surf,factors,program=config.program,
            field_scales=field_scales,energy_scale_j=energy_scale_j,
            mechanical_load=mechanical_load,free_charge=free_charge,
            settings=settings,on_iteration=on_iteration,cancelled=cancelled,
            work_evaluator=evaluate)
        if not solution.converged:
            return None,solution
        q=surf.reconstruct(solution.factors)[:surf.n_mechanical]
        a=acceleration_factor*(q-predictor)
        v=velocity_predictor+velocity_factor*a
        state=CoupledMVIDynamicState(time_s,solution.factors,v,a)
        config._validate_state(state)
        return state,solution

    def advance(self,state:CoupledMVIDynamicState,dt_s, *,
                field_scales,energy_scale_j,loads_at=None,
                settings=CoupledMVISweepSettings(),
                on_iteration=None,cancelled=None):
        config=self.configuration
        surf=config.surface
        q,v,a=config._validate_state(state)
        dt=_positive(dt_s,"Bathe time step")
        t1=state.time_s+BATHE_GAMMA*dt
        t2=state.time_s+dt
        if not math.isfinite(t2) or not state.time_s<t1<t2:
            raise ValueError("MVI_BATHE_UNREPRESENTABLE_TIME")
        if loads_at is not None and not callable(loads_at):
            raise TypeError("Bathe loads_at must be callable")
        if not isinstance(settings,CoupledMVISweepSettings):
            raise TypeError("CoupledMVISweepSettings required")
        _scales(surf,field_scales,energy_scale_j)

        def loads(t):
            if loads_at is None:
                return None,None
            pair=loads_at(t)
            if not isinstance(pair,(tuple,list)) or len(pair)!=2:
                raise ValueError("Bathe loads_at must provide mechanical/electric pair")
            return pair[0],pair[1]

        def callback(stage):
            return None if on_iteration is None else (
                lambda event:on_iteration(dict(stage=stage,**event)))

        delta=BATHE_GAMMA*dt
        initial_q=q[:surf.n_mechanical]
        predictor1=initial_q+delta*v+delta**2*a/4.
        velocity1=v+delta*a/2.
        load1,charge1=loads(t1)
        first,first_result=self._solve_stage(
            state.factors,predictor=predictor1,
            velocity_predictor=velocity1,
            acceleration_factor=4./delta**2,velocity_factor=delta/2.,
            time_s=t1,field_scales=field_scales,
            energy_scale_j=energy_scale_j,mechanical_load=load1,
            free_charge=charge1,settings=settings,
            on_iteration=callback("TRAPEZOID"),
            cancelled=cancelled)
        if first is None:
            return CoupledMVIBatheStep(
                None,None,first_result,None,t2,
                "MVI_BATHE_FIRST_"+first_result.status)

        c1=(1.-BATHE_GAMMA)/(BATHE_GAMMA*dt)
        c2=-1./((1.-BATHE_GAMMA)*BATHE_GAMMA*dt)
        c3=(2.-BATHE_GAMMA)/((1.-BATHE_GAMMA)*dt)
        first_q=surf.reconstruct(first.factors)[:surf.n_mechanical]
        displacement_history=c1*initial_q+c2*first_q
        velocity_history=c1*v+c2*first.velocity
        predictor2=-(c3*displacement_history+velocity_history)/c3**2
        velocity2=-velocity_history/c3
        load2,charge2=loads(t2)
        final,final_result=self._solve_stage(
            first.factors,predictor=predictor2,
            velocity_predictor=velocity2,
            acceleration_factor=c3**2,velocity_factor=1./c3,
            time_s=t2,field_scales=field_scales,
            energy_scale_j=energy_scale_j,mechanical_load=load2,
            free_charge=charge2,settings=settings,
            on_iteration=callback("THREE_POINT"),
            cancelled=cancelled)
        if final is None:
            return CoupledMVIBatheStep(
                None,first,first_result,final_result,t2,
                "MVI_BATHE_SECOND_"+final_result.status)
        final_q=surf.reconstruct(final.factors)[:surf.n_mechanical]
        identities=(
            (first_q,initial_q+delta*(v+first.velocity)/2.),
            (first.velocity,v+delta*(a+first.acceleration)/2.),
            (final.velocity,c1*initial_q+c2*first_q+c3*final_q),
            (final.acceleration,c1*v+c2*first.velocity+c3*final.velocity),
        )
        for lhs,rhs in identities:
            error=float(np.max(np.abs(lhs-rhs),initial=0.))
            scale=max(1.,float(np.max(np.abs(lhs),initial=0.)),
                         float(np.max(np.abs(rhs),initial=0.)))
            if error>1.e-9*scale:
                raise ValueError("MVI_BATHE_THREE_POINT_IDENTITY_FAILURE")
        return CoupledMVIBatheStep(
            final,first,first_result,final_result,t2,
            "MVI_BATHE_CONVERGED")


__all__=("BATHE_GAMMA","CoupledMVIBatheStep","NativeCoupledMVIBathe")
