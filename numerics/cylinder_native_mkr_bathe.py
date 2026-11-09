"""Independent native MKR two-substep Bathe time integration (research only).

Uses only native strong shell bulk/end equations and the independent strong
Newton solver. No variational MVI/FDM time driver or spatial callback is used.
No time history, loaded calculation, GUI or CASE run occurs on import.

Bathe gamma=2-sqrt(2), first implicit trapezoidal substep followed by
three-point backward differential relations at the full time point.
This integrator is second order, not the Gauss fourth-order integrator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from cylinder_direct_strong_boundary import assemble_physical_end_rows
from cylinder_direct_strong_newton import (
    StrongNewtonSettings, solve_strong_nonlinear_equations,
)
from cylinder_native_mkr_newmark import (
    NativeNewmarkState, NativeStrongNewmark, _finite_number, _field,
)


BATHE_GAMMA=2.-math.sqrt(2.)


@dataclass(frozen=True)
class NativeBatheResult:
    first: NativeNewmarkState | None
    state: NativeNewmarkState | None
    first_nonlinear: object
    final_nonlinear: object | None
    attempted_time_s: float
    status: str


class NativeStrongBathe:
    """Two independent nonlinear full-field MKR solves per physical step.

    The NativeStrongNewmark *configuration* provides the already validated
    physical end program, density, material damping and optional electric
    gauge. Its time-stepping method is NOT called; both Bathe stages use their
    OWN predictor, acceleration and velocity identities. All unknowns
    (including the potential) are solved simultaneously in each stage.
    """

    def __init__(self, engine: NativeStrongNewmark):
        if not isinstance(engine, NativeStrongNewmark):
            raise TypeError("NativeStrongNewmark configuration required")
        self.engine=engine

    def _stage(self, initial, *, t_next, q_predictor, v_predictor, kappa,
               velocity_accel_factor, mechanical_load, electrical_charge,
               qguess, pguess, mechanical_state_scale,
               mechanical_equation_scale, electrical_state_scale,
               electrical_equation_scale, settings):
        engine=self.engine

        def kinematics(q):
            a=kappa*(q-q_predictor)
            v=v_predictor+velocity_accel_factor*a
            if not np.isfinite(a).all() or not np.isfinite(v).all():
                raise ValueError("nonfinite native Bathe mechanical state")
            return v,a

        def equations(q,p):
            v,a=kinematics(q)
            balance=engine.model.balance(q,p,mechanical_load=mechanical_load,
                electrical_charge=electrical_charge,
                velocity=v,acceleration=a,density_kg_m3=engine.density_kg_m3,
                eta_s=engine.eta_s,
                mass_damping_rate_per_s=engine.mass_damping_rate_per_s)
            rows=assemble_physical_end_rows(balance,q,engine.model.grid,
                axial_orders=engine.axial_orders,mechanical_ends=engine.mechanical_ends,
                electrical=p,electrical_ends=engine.electrical_ends,
                gauge_closed_elsewhere=engine.electric_gauge is not None)
            return rows.mechanical,rows.electrical

        result=solve_strong_nonlinear_equations(
            equations,qguess,pguess,
            mechanical_state_scale=mechanical_state_scale,
            mechanical_equation_scale=mechanical_equation_scale,
            electrical_state_scale=electrical_state_scale,
            electrical_equation_scale=electrical_equation_scale,
            electric_gauge=engine.electric_gauge,settings=settings)
        if not result.converged:
            return None,result
        v,a=kinematics(result.mechanical)
        return NativeNewmarkState(t_next,result.mechanical,v,a,result.electrical),result

    def advance(self,state: NativeNewmarkState,dt_s,*,loads_at=None,
                mechanical_state_scale,mechanical_equation_scale,
                electrical_state_scale=None,electrical_equation_scale=None,
                settings=StrongNewtonSettings()):
        """Do one complete Bathe step; do not accept an incomplete second stage.

        loads_at(t) -> (generalized mechanical load, generalized free charge).
        Its time arguments are t+gamma*dt and t+dt. Without a callback, use
        constant loads supplied to the engine at construction.
        """
        engine=self.engine
        if not isinstance(state,NativeNewmarkState):
            raise TypeError("NativeNewmarkState required")
        q=_field(state.mechanical,engine.model.shape,"mechanical")
        v=_field(state.velocity,engine.model.shape,"velocity")
        a=_field(state.acceleration,engine.model.shape,"acceleration")
        if engine.model.electrical_on:
            p=engine._electric_field(state.electrical,"electric potential")
        elif state.electrical is not None:
            raise ValueError("electric potential when material electric field is OFF")
        else:
            p=None
        h=_finite_number(dt_s,"Bathe step",positive=True)
        t1=state.time_s+BATHE_GAMMA*h
        t2=state.time_s+h
        if not math.isfinite(t2) or t1<=state.time_s or t2<=t1:
            raise ValueError("unrepresentable Bathe substep times")
        if loads_at is not None and not callable(loads_at):
            raise TypeError("loads_at must be callable")

        def stage_load(time):
            if loads_at is None:
                return engine.mechanical_load,engine.electrical_charge
            provided=loads_at(time)
            if not isinstance(provided,(tuple,list)) or len(provided)!=2:
                raise ValueError("Bathe loads_at must return mechanical/electric pair")
            qm=provided[0]
            charge=provided[1]
            if qm is not None:
                qm=_field(qm,engine.model.shape,"Bathe mechanical load").copy()
            if charge is not None:
                charge=engine._electric_field(charge,"Bathe electric free charge")
            engine._check_insulated_charge(charge)
            return qm,charge

        delta=BATHE_GAMMA*h
        load1,charge1=stage_load(t1)
        qp1=q+delta*v+.25*delta**2*a
        vp1=v+.5*delta*a
        if not np.isfinite(qp1).all() or not np.isfinite(vp1).all():
            raise ValueError("nonfinite Bathe first predictor")
        first,first_result=self._stage(
            state,t_next=t1,q_predictor=qp1,v_predictor=vp1,
            kappa=4./delta**2,velocity_accel_factor=.5*delta,
            mechanical_load=load1,electrical_charge=charge1,
            qguess=qp1,pguess=p,mechanical_state_scale=mechanical_state_scale,
            mechanical_equation_scale=mechanical_equation_scale,
            electrical_state_scale=electrical_state_scale,
            electrical_equation_scale=electrical_equation_scale,settings=settings)
        if first is None:
            return NativeBatheResult(None,None,first_result,None,t2,
                                     "FIRST_STAGE_"+first_result.status)

        load2,charge2=stage_load(t2)
        c1=(1.-BATHE_GAMMA)/(BATHE_GAMMA*h)
        c2=-1./((1.-BATHE_GAMMA)*BATHE_GAMMA*h)
        c3=(2.-BATHE_GAMMA)/((1.-BATHE_GAMMA)*h)
        displacement_history=c1*q+c2*first.mechanical
        velocity_history=c1*v+c2*first.velocity
        qp2=-(c3*displacement_history+velocity_history)/c3**2
        vp2=-velocity_history/c3
        if not np.isfinite(qp2).all() or not np.isfinite(vp2).all():
            raise ValueError("nonfinite Bathe second predictor")
        final,final_result=self._stage(
            first,t_next=t2,q_predictor=qp2,v_predictor=vp2,
            kappa=c3**2,velocity_accel_factor=1./c3,
            mechanical_load=load2,electrical_charge=charge2,
            qguess=first.mechanical,pguess=first.electrical,
            mechanical_state_scale=mechanical_state_scale,
            mechanical_equation_scale=mechanical_equation_scale,
            electrical_state_scale=electrical_state_scale,
            electrical_equation_scale=electrical_equation_scale,settings=settings)
        if final is None:
            return NativeBatheResult(first,None,first_result,final_result,t2,
                                     "SECOND_STAGE_"+final_result.status)
        # Check original Bathe three-point equations independently of the
        # effective Newmark predictor algebra.
        pairs=(
            (first.mechanical,q+.5*delta*(v+first.velocity)),
            (first.velocity,v+.5*delta*(a+first.acceleration)),
            (final.velocity,c1*q+c2*first.mechanical+c3*final.mechanical),
            (final.acceleration,c1*v+c2*first.velocity+c3*final.velocity),
        )
        for got,want in pairs:
            error=float(np.max(np.abs(got-want)))
            scale=max(1.,float(np.max(np.abs(got))),float(np.max(np.abs(want))))
            if error>1.e-9*scale:
                raise ValueError("native Bathe three-point kinematics are inconsistent")
        return NativeBatheResult(first,final,first_result,final_result,t2,
                                 "CONVERGED")


__all__=("BATHE_GAMMA","NativeBatheResult","NativeStrongBathe")
