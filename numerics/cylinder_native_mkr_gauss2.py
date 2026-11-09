"""Research-only Gauss-Legendre IRK4 on native cylindrical strong MKR fields.

Both implicit collocation stages are solved in ONE coupled strong-Newton
system, without MVI, a variational spatial residual or a shared time driver.
Each stage includes its own mechanical/electrical end equations and free
charge. The method advances (q,v) by the two-stage Gauss RK tableau.

The electric equations are quasistatic. If a floating-potential gauge is
present, one electric equation per stage is replaced by that stage's volume
mean constraint. After convergence, BOTH omitted original equations are
checked in their physical scales; a compatibility failure rejects the step.
This is not a model-independent guarantee of a well-posed electric pencil.

At the end of a step, acceleration and electric potential are recomputed
by the native strong instantaneous balance with q_end,v_end fixed.
No physical shell computation is launched by importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, isfinite

import numpy as np

from cylinder_direct_strong_boundary import assemble_physical_end_rows
from cylinder_direct_strong_newton import (
    StrongNewtonResult, StrongNewtonSettings, _positive_scale,
    solve_strong_nonlinear_equations,
)
from cylinder_native_mkr_initial import (
    _verify_fixed_initial_supports, initialize_consistent_native_mkr,
)
from cylinder_native_mkr_newmark import (
    NativeNewmarkState, NativeStrongNewmark, _field, _finite_number,
)

GAUSS_C = np.array(((3.-sqrt(3.))/6., (3.+sqrt(3.))/6.))
GAUSS_A = np.array(((.25, .25-sqrt(3.)/6.),
                    (.25+sqrt(3.)/6., .25)))
GAUSS_B = np.array((.5, .5))
GAUSS_A2_INV = np.linalg.inv(GAUSS_A @ GAUSS_A)
for _matrix in (GAUSS_C,GAUSS_A,GAUSS_B,GAUSS_A2_INV):
    _matrix.setflags(write=False)


def gauss_stage_kinematics(q_stages, q0, v0, dt_s):
    """Two-stage collocation kinematics; q_stages[0:2] are unknown fields."""
    h=_finite_number(dt_s,"Gauss step",positive=True)
    base=np.asarray(q0,float)
    velocity=_field(v0,base.shape,"initial velocity")
    stages=np.asarray(q_stages,float)
    if stages.shape!=(2,*base.shape) or not np.isfinite(stages).all():
        raise ValueError("two finite mechanical collocation fields required")
    rhs=stages-base[None]-h*GAUSS_C[:,None,None,None]*velocity[None]
    acceleration=np.einsum("ij,j...->i...",GAUSS_A2_INV,rhs,
                           optimize=True)/h**2
    stage_velocities=velocity[None]+h*np.einsum(
        "ij,j...->i...",GAUSS_A,acceleration,optimize=True)
    if not (np.isfinite(acceleration).all() and
            np.isfinite(stage_velocities).all()):
        raise ValueError("nonfinite Gauss stage acceleration/velocity")
    return stage_velocities,acceleration


def gauss_end_kinematics(q0, v0, stage_velocities, stage_accelerations, dt_s):
    """Exact Runge-Kutta endpoint quadrature for q and v."""
    h=_finite_number(dt_s,"Gauss step",positive=True)
    q0=np.asarray(q0,float)
    v0=_field(v0,q0.shape,"velocity")
    v=np.asarray(stage_velocities,float)
    a=np.asarray(stage_accelerations,float)
    if v.shape!=(2,*q0.shape) or a.shape!=v.shape or not (
            np.isfinite(v).all() and np.isfinite(a).all()):
        raise ValueError("two finite Gauss stage velocity/acceleration fields required")
    return (q0+h*np.einsum("i,i...->...",GAUSS_B,v),
            v0+h*np.einsum("i,i...->...",GAUSS_B,a))


@dataclass(frozen=True)
class NativeStrongGauss2Result:
    """No accepted endpoint is provided if any required solve fails."""
    state: NativeNewmarkState | None
    stages: tuple[NativeNewmarkState, ...]
    stage_nonlinear: StrongNewtonResult
    endpoint_nonlinear: StrongNewtonResult | None
    attempted_time_s: float
    status: str


class NativeStrongGauss2:
    """Opt-in two-stage coupled Gauss IRK4 with native strong MKR equations."""

    def __init__(self, engine: NativeStrongNewmark):
        if not isinstance(engine,NativeStrongNewmark):
            raise TypeError("NativeStrongNewmark physical configuration required")
        self.engine=engine

    def advance(self, state: NativeNewmarkState, dt_s, *,
                mechanical_state_scale, mechanical_equation_scale,
                electrical_state_scale=None, electrical_equation_scale=None,
                loads_at=None, settings=StrongNewtonSettings()):
        if not isinstance(state,NativeNewmarkState):
            raise TypeError("NativeNewmarkState required")
        if not isinstance(settings,StrongNewtonSettings):
            raise TypeError("StrongNewtonSettings required")
        h=_finite_number(dt_s,"Gauss step",positive=True)
        engine=self.engine
        shape=engine.model.shape
        q0=_field(state.mechanical,shape,"initial displacement")
        v0=_field(state.velocity,shape,"initial velocity")
        a0=_field(state.acceleration,shape,"initial acceleration")
        _verify_fixed_initial_supports(engine,q0,v0,absolute_tolerance=1.e-9)
        p0=(engine._electric_field(state.electrical,"initial potential")
            if engine.model.electrical_on else None)
        if not engine.model.electrical_on and state.electrical is not None:
            raise ValueError("electrical state supplied when OFF")
        t_end=state.time_s+h
        ts=state.time_s+h*GAUSS_C
        if not isfinite(t_end) or not (state.time_s<ts[0]<ts[1]<t_end):
            raise ValueError("unrepresentable Gauss physical stage times")
        if loads_at is not None and not callable(loads_at):
            raise TypeError("loads_at must be a physical-time load callback")

        def get_load(t):
            if loads_at is None:
                return engine.mechanical_load,engine.electrical_charge
            supplied=loads_at(float(t))
            if not isinstance(supplied,(tuple,list)) or len(supplied)!=2:
                raise ValueError("Gauss loads_at must return mechanical/charge pair")
            lm=(None if supplied[0] is None else
                _field(supplied[0],shape,"stage generalized mechanical load").copy())
            le=(None if supplied[1] is None else
                engine._electric_field(supplied[1],"stage free charge"))
            engine._check_insulated_charge(le)
            return lm,le

        forces=(get_load(ts[0]),get_load(ts[1]))
        # Surface field count is unchanged by collocation: the first axis
        # merely packs TWO independent, simultaneously solved RK stages.
        qguess=np.stack((
            q0+h*GAUSS_C[i]*v0+.5*(h*GAUSS_C[i])**2*a0
            for i in range(2)),axis=0)
        qscale=_positive_scale(mechanical_state_scale,shape,"mechanical state")
        qforce=_positive_scale(mechanical_equation_scale,shape,"mechanical equation")
        packed_q_guess=np.concatenate((qguess[0],qguess[1]),axis=0)
        packed_q_scale=np.concatenate((qscale,qscale),axis=0)
        packed_force=np.concatenate((qforce,qforce),axis=0)

        if p0 is None:
            pshape=None
            packed_p_guess=packed_p_scale=packed_eforce=None
            electric_force_scale=None
        else:
            pshape=p0.shape
            escale=_positive_scale(electrical_state_scale,pshape,"electric state")
            electric_force_scale=_positive_scale(
                electrical_equation_scale,pshape,"electric equation")
            packed_p_guess=np.concatenate((p0,p0),axis=0)
            packed_p_scale=np.concatenate((escale,escale),axis=0)
            packed_eforce=np.concatenate(
                (electric_force_scale,electric_force_scale),axis=0)

        def unpack(qpacked,ppacked):
            qp=_field(qpacked,(2*shape[0],*shape[1:]),"two-stage displacement")
            qs=qp.reshape(2,*shape)
            if pshape is None:
                if ppacked is not None:
                    raise ValueError("electrical stages required to be OFF")
                ps=None
            else:
                pp=_field(ppacked,(2*pshape[0],*pshape[1:]),"two-stage electric")
                ps=pp.reshape(2,*pshape)
            return qs,ps

        def stage_physics(qs,ps,vs,accelerations,*,substitute_gauge):
            rows_m=[]
            rows_e=[]
            for i in range(2):
                p=None if ps is None else ps[i]
                mechanical_load,charge=forces[i]
                balance=engine.model.balance(qs[i],p,
                    velocity=vs[i],acceleration=accelerations[i],
                    density_kg_m3=engine.density_kg_m3,
                    eta_s=engine.eta_s,
                    mass_damping_rate_per_s=engine.mass_damping_rate_per_s,
                    mechanical_load=mechanical_load,electrical_charge=charge)
                rows=assemble_physical_end_rows(balance,qs[i],engine.model.grid,
                    axial_orders=engine.axial_orders,
                    mechanical_ends=engine.mechanical_ends,
                    electrical=p,electrical_ends=engine.electrical_ends,
                    gauge_closed_elsewhere=engine.electric_gauge is not None)
                rows_m.append(rows.mechanical)
                if rows.electrical is not None:
                    e=rows.electrical.copy()
                    if substitute_gauge and engine.electric_gauge is not None:
                        gauge=engine.electric_gauge
                        scaled_mean=(float(np.sum(gauge.weights*p))-
                                     gauge.target_v)/gauge.potential_scale_v
                        # Replace one equation, then verify its ORIGINAL
                        # physical equation after the nonlinear solve.
                        e[0,0,0]=scaled_mean*electric_force_scale[0,0,0]
                    rows_e.append(e)
            return (np.concatenate(rows_m,axis=0),
                    None if pshape is None else np.concatenate(rows_e,axis=0))

        def equation(qpacked,ppacked):
            qs,ps=unpack(qpacked,ppacked)
            v,a=gauss_stage_kinematics(qs,q0,v0,h)
            return stage_physics(qs,ps,v,a,substitute_gauge=True)

        stage_solution=solve_strong_nonlinear_equations(
            equation,packed_q_guess,packed_p_guess,
            mechanical_state_scale=packed_q_scale,
            mechanical_equation_scale=packed_force,
            electrical_state_scale=packed_p_scale,
            electrical_equation_scale=packed_eforce,
            settings=settings)
        if not stage_solution.converged:
            return NativeStrongGauss2Result(None,(),stage_solution,None,t_end,
                                            "GAUSS_STAGES_"+stage_solution.status)

        qs,ps=unpack(stage_solution.mechanical,stage_solution.electrical)
        vs,accelerations=gauss_stage_kinematics(qs,q0,v0,h)
        if engine.electric_gauge is not None:
            _,original_e=stage_physics(qs,ps,vs,accelerations,
                                       substitute_gauge=False)
            gauges=[]
            for i in range(2):
                gauge=engine.electric_gauge
                gauge_error=abs(float(np.sum(gauge.weights*ps[i]))-
                                gauge.target_v)/gauge.potential_scale_v
                original_error=abs(float(original_e.reshape(2,*pshape)[i,0,0,0])
                                   /electric_force_scale[0,0,0])
                gauges.append(max(gauge_error,original_error))
            error=max(gauges)
            if error>settings.scaled_absolute_tolerance:
                failed=StrongNewtonResult(
                    stage_solution.mechanical,stage_solution.electrical,
                    stage_solution.iterations,False,
                    "ELECTRIC_GAUGE_COMPATIBILITY_FAILED",
                    max(stage_solution.scaled_equation_norm,error))
                return NativeStrongGauss2Result(None,(),failed,None,t_end,
                                                failed.status)

        stages=tuple(NativeNewmarkState(float(ts[i]),qs[i],vs[i],
                     accelerations[i],None if ps is None else ps[i])
                     for i in range(2))
        qend,vend=gauss_end_kinematics(q0,v0,vs,accelerations,h)
        # A collocation stage acceleration is NOT automatically the physical
        # endpoint acceleration. Solve for a_end and phi_end at (qend,vend).
        end_load,end_charge=get_load(t_end)
        endpoint=initialize_consistent_native_mkr(
            engine,qend,vend,None if ps is None else ps[1],
            time_s=t_end,acceleration_guess=accelerations[1],
            mechanical_state_scale=mechanical_state_scale,
            mechanical_equation_scale=mechanical_equation_scale,
            electrical_state_scale=electrical_state_scale,
            electrical_equation_scale=electrical_equation_scale,
            mechanical_load=end_load,electrical_charge=end_charge,
            settings=settings)
        if endpoint.state is None:
            return NativeStrongGauss2Result(
                None,stages,stage_solution,endpoint.nonlinear,t_end,
                "GAUSS_ENDPOINT_"+endpoint.nonlinear.status)
        return NativeStrongGauss2Result(endpoint.state,stages,stage_solution,
                                        endpoint.nonlinear,t_end,"CONVERGED")


__all__=("GAUSS_C","GAUSS_A","GAUSS_B","gauss_stage_kinematics",
         "gauss_end_kinematics","NativeStrongGauss2Result","NativeStrongGauss2")
