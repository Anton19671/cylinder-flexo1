"""Two-stage Gauss-Legendre collocation of independent coupled MVI.

All stage mechanical/electric unknowns are the OWN rank-R separated
X/Y factors. The two stage accelerations/velocities are recovered from
the simultaneous 2x2 Gauss collocation identities, so each stage's
mechanical equations depends on BOTH physical stage displacements.
Alternating block stage solves iterate to joint stationarity; neither
MKR nor any shared variational dynamic driver is imported.

The full endpoint must be representable by rank R without SVD, and its
quasi-static electric potential and physical acceleration must be closed
by independent MVI equations. Failure of either stage, joint check,
rank fit or endpoint closure REJECTS the whole step. No acceptance from
one individually stationary stage is allowed.

Research only: high-order G3D/Gz natural conditions and floating
electric gauge remain uncertified; stationary supported ends and
grounded electrodes are the explicit supported subset. Not GUI-wired.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from independent_mvi.coupled_halfstep import _scales, _check_end_factors
from independent_mvi.coupled_newmark import (
    CoupledMVIDynamicState, NativeCoupledMVINewmark, _positive,
)
from independent_mvi.coupled_static import (
    CoupledMVISweepSettings, CoupledMVISweepResult,
    _projected_norms, solve_coupled_mvi_static,
)
from independent_mvi.coupled_surface import CoupledMVIWork, CoupledMVIFactors
from independent_mvi.coupled_rank_endpoint import fit_mvi_endpoint_field
from independent_mvi.coupled_initial import close_mvi_instantaneous_state


SQRT3=math.sqrt(3.)
GAUSS_C=np.array(((3.-SQRT3)/6.,(3.+SQRT3)/6.))
GAUSS_A=np.array(((.25,.25-SQRT3/6.),
                  (.25+SQRT3/6.,.25)))
GAUSS_B=np.array((.5,.5))
GAUSS_A2_INV=np.linalg.inv(GAUSS_A@GAUSS_A)
for _value in (GAUSS_C,GAUSS_A,GAUSS_B,GAUSS_A2_INV):
    _value.setflags(write=False)


def mvi_gauss_stage_kinematics(q_stages,q0,v0,dt_s):
    dt=_positive(dt_s,"MVI Gauss step")
    q0=np.asarray(q0,float)
    v0=np.asarray(v0,float)
    qs=np.asarray(q_stages,float)
    if (q0.shape!=v0.shape or qs.shape!=(2,*q0.shape) or
            not np.isfinite(q0).all() or not np.isfinite(v0).all() or
            not np.isfinite(qs).all()):
        raise ValueError("MVI_GAUSS_INVALID_MECHANICAL_STAGES")
    rhs=qs-q0[None]-dt*GAUSS_C.reshape(2,1,1,1)*v0[None]
    a=np.einsum("ij,j...->i...",GAUSS_A2_INV,rhs,
                optimize=True)/dt**2
    v=v0[None]+dt*np.einsum("ij,j...->i...",GAUSS_A,a,
                            optimize=True)
    if not np.isfinite(a).all() or not np.isfinite(v).all():
        raise ValueError("MVI_GAUSS_NONFINITE_STAGE_KINEMATICS")
    return v,a


def mvi_gauss_endpoint(q0,v0,velocity,acceleration,dt_s):
    dt=_positive(dt_s,"MVI Gauss step")
    v=np.asarray(velocity,float)
    a=np.asarray(acceleration,float)
    q0=np.asarray(q0,float)
    v0=np.asarray(v0,float)
    if (v.shape!=(2,*q0.shape) or a.shape!=v.shape or
            q0.shape!=v0.shape or not np.isfinite(v).all() or
            not np.isfinite(a).all()):
        raise ValueError("MVI_GAUSS_ENDPOINT_KINEMATICS_INVALID")
    return (q0+dt*np.einsum("i,i...->...",GAUSS_B,v,optimize=True),
            v0+dt*np.einsum("i,i...->...",GAUSS_B,a,optimize=True))


@dataclass(frozen=True)
class CoupledMVIGaussStep:
    state: CoupledMVIDynamicState | None
    stages: tuple[CoupledMVIDynamicState,...]
    stage_solvers: tuple[CoupledMVISweepResult,...]
    attempted_time_s: float
    status: str
    joint_nodal_norm: float=float("inf")
    endpoint_rank_error: float=float("inf")


def _relax_gauss_fixed_rank_stage(surface,old,proposed,weight,program):
    """Damp the second GAUSS stage without changing its exact rank.

    When old and proposed circumferential spans agree, the physical
    interpolation is EXACT in the proposed Y span, using only Gram/LU
    solves (no SVD or rank growth). Otherwise interpolate the actual
    X/Y coordinates; the final joint full-work check still gates
    acceptance of that candidate.
    """
    if not 0<weight<=1 or not math.isfinite(weight):
        raise ValueError("MVI_GAUSS_INVALID_STAGE_RELAXATION")
    if old.rank!=proposed.rank:
        raise ValueError("MVI_GAUSS_STAGE_RANK_CHANGED")
    if weight==1:
        return proposed
    x=proposed.x.copy()
    y=proposed.y.copy()
    for c in range(surface.n_fields):
        previous=np.einsum("ri,rj->ij",old.x[c],old.y[c],
                           optimize=True)
        gram=proposed.y[c]@proposed.y[c].T
        exact=False
        if np.isfinite(gram).all():
            try:
                if np.linalg.cond(gram,p=1)<1.e11:
                    # solve returns (rank,nx) directly; transposing
                    # it would turn the rank-1 case into (nx,1) and
                    # break fixed-rank X/Y stage continuation.
                    prior_x=np.linalg.solve(
                        gram,proposed.y[c]@previous.T)
                    defect=np.linalg.norm(prior_x.T@proposed.y[c]-previous)
                    scale=max(np.linalg.norm(previous),1.e-30)
                    exact=bool(defect<=1.e-10*scale)
            except np.linalg.LinAlgError:
                pass
        if exact:
            x[c]=(1.-weight)*prior_x+weight*proposed.x[c]
        else:
            # Two arbitrary rank-R surfaces need not have rank-R sum.
            # Coordinate continuation stays exactly rank-R and makes no
            # claim that the physical field is a convex interpolation.
            x[c]=(1.-weight)*old.x[c]+weight*proposed.x[c]
            y[c]=(1.-weight)*old.y[c]+weight*proposed.y[c]
    relaxed=CoupledMVIFactors(x,y)
    _check_end_factors(surface,relaxed,program)
    if not np.isfinite(surface.reconstruct(relaxed)).all():
        raise ValueError("MVI_GAUSS_NONFINITE_RELAXED_STAGE")
    return relaxed


class NativeCoupledMVIGauss2:
    def __init__(self,configuration: NativeCoupledMVINewmark):
        if not isinstance(configuration,NativeCoupledMVINewmark):
            raise TypeError("NativeCoupledMVINewmark configuration required")
        self.configuration=configuration

    def advance(self,state:CoupledMVIDynamicState,dt_s, *,
                field_scales,energy_scale_j,loads_at=None,
                settings=CoupledMVISweepSettings(),
                max_stage_sweeps=36,
                stage_coupling_relaxation=.25,
                stage_field_tolerance=1.e-8,
                endpoint_rank_tolerance=1.e-9,
                max_acceleration_coordinates=256,
                acceleration_options=None,
                on_iteration=None,cancelled=None):
        config=self.configuration
        surface=config.surface
        if acceleration_options is None:
            acceleration_options={}
        if not isinstance(acceleration_options,dict):
            raise ValueError("MVI_GAUSS_ACCELERATION_OPTIONS_INVALID")
        q,v0,a0=config._validate_state(state)
        dt=_positive(dt_s,"MVI Gauss step")
        times=state.time_s+dt*GAUSS_C
        t_end=state.time_s+dt
        if not math.isfinite(t_end) or not (
                state.time_s<times[0]<times[1]<t_end):
            raise ValueError("MVI_GAUSS_UNREPRESENTABLE_STAGE_TIMES")
        if not isinstance(settings,CoupledMVISweepSettings):
            raise TypeError("CoupledMVISweepSettings required")
        if type(max_stage_sweeps) is not int or max_stage_sweeps<1:
            raise ValueError("MVI_GAUSS_STAGE_SWEEP_LIMIT_INVALID")
        if (isinstance(stage_coupling_relaxation,(bool,np.bool_)) or
                not isinstance(stage_coupling_relaxation,(int,float)) or
                not math.isfinite(stage_coupling_relaxation) or
                not 0<stage_coupling_relaxation<=1):
            raise ValueError("MVI_GAUSS_INVALID_STAGE_RELAXATION")
        for label,value in (("stage field tolerance",stage_field_tolerance),
                            ("rank tolerance",endpoint_rank_tolerance)):
            if isinstance(value,(bool,np.bool_)) or not (
                    math.isfinite(value) and 0<value<1):
                raise ValueError("MVI_GAUSS_"+label.upper().replace(" ","_")+"_INVALID")
        _scales(surface,field_scales,energy_scale_j)
        if loads_at is not None and not callable(loads_at):
            raise TypeError("MVI Gauss loads_at must be callable")

        def loads(time):
            if loads_at is None:
                return None,None
            answer=loads_at(float(time))
            if not isinstance(answer,(tuple,list)) or len(answer)!=2:
                raise ValueError("MVI_GAUSS_LOADS_REQUIRE_MECHANICAL_ELECTRIC_PAIR")
            return answer[0],answer[1]

        forcing=(loads(times[0]),loads(times[1]))
        mechanical0=q[:surface.n_mechanical]
        factors=[state.factors,state.factors]
        last_solvers=()
        final_norm=float("inf")
        converged=False

        def mechanical_fields(pair):
            return np.stack([
                surface.reconstruct(f)[:surface.n_mechanical]
                for f in pair],axis=0)

        def effective(index,trial,pair):
            all_f=list(pair);all_f[index]=trial
            velocities,accelerations=mvi_gauss_stage_kinematics(
                mechanical_fields(all_f),mechanical0,v0,dt)
            load,charge=forcing[index]
            work=surface.weak_work(trial,mechanical_load=load,
                                   free_charge=charge)
            local=work.mechanical
            nodal=np.asarray(work.nodal_gradient,float).copy()
            nodal[:surface.n_mechanical]+=config.inertia.apply(
                local,velocities[index],accelerations[index])
            if config.viscosity.eta_s:
                nodal[:surface.n_mechanical]+=config.viscosity.apply(
                    local,velocities[index])
            if not np.isfinite(nodal).all():
                raise ValueError("MVI_GAUSS_NONFINITE_COUPLED_STAGE_WORK")
            return CoupledMVIWork(
                work.energy_j,local,work.electrical,
                np.einsum("cij,crj->cri",nodal,trial.y,optimize=True),
                np.einsum("cij,cri->crj",nodal,trial.x,optimize=True),
                trial.rank,nodal_gradient=nodal)

        def result(status,stage_states=(),rank_error=float("inf")):
            return CoupledMVIGaussStep(
                None,tuple(stage_states),tuple(last_solvers),
                t_end,status,final_norm,rank_error)

        for sweep in range(1,max_stage_sweeps+1):
            if cancelled is not None and cancelled():
                return result("MVI_GAUSS_CANCELLED")
            previous=mechanical_fields(factors)
            solved=[]
            for index in (0,1):
                def stage_work(trial,index=index):
                    return effective(index,trial,factors)
                load,charge=forcing[index]
                stage=solve_coupled_mvi_static(
                    surface,factors[index],program=config.program,
                    field_scales=field_scales,energy_scale_j=energy_scale_j,
                    mechanical_load=load,free_charge=charge,
                    settings=settings,work_evaluator=stage_work,
                    cancelled=cancelled,
                    on_iteration=(None if on_iteration is None else
                        lambda entry,index=index,sweep=sweep:
                        on_iteration(dict(gauss_sweep=sweep,
                                          gauss_stage=index+1,**entry))))
                solved.append(stage)
                if not stage.converged:
                    last_solvers=tuple(solved)
                    return result("MVI_GAUSS_STAGE_"+str(index+1)+"_"+stage.status)
                if index==1:
                    # Strongly inertial two-stage Gauss A^2 inverse has
                    # off-diagonal GS amplification exactly -3. The
                    # direct sequential stage fixed point can DIVERGE
                    # while each frozen-stage Newton solve converges.
                    # One-quarter damping gives 1+(-3-1)/4=0 in this
                    # limiting case. The exact Gauss tableau and final
                    # *joint* physical stationarity gate are unchanged.
                    factors[index]=_relax_gauss_fixed_rank_stage(
                        surface,factors[index],stage.factors,
                        stage_coupling_relaxation,config.program)
                else:
                    factors[index]=stage.factors
            last_solvers=tuple(solved)
            current=mechanical_fields(factors)
            change=float(np.linalg.norm(current-previous)/
                max(np.linalg.norm(current),np.linalg.norm(previous),1.e-30))
            nodals=[]
            for index in (0,1):
                work=lambda f,index=index: effective(index,f,factors)
                m,e,full=_projected_norms(
                    surface,factors[index],config.program,
                    field_scales,energy_scale_j,*forcing[index],
                    work_evaluator=work)
                nodals.append(max(m,e,full))
            final_norm=max(nodals)
            if on_iteration is not None:
                on_iteration(dict(gauss_sweep=sweep,
                                  gauss_joint_nodal=final_norm,
                                  gauss_stage_change=change,
                                  gauss_coupling_relaxation=stage_coupling_relaxation))
            if (change<=stage_field_tolerance and
                    final_norm<=settings.halfstep.tolerance_scaled):
                converged=True
                break
        if not converged:
            return result("MVI_GAUSS_JOINT_STAGES_NOT_CONVERGED")

        vs,accs=mvi_gauss_stage_kinematics(
            mechanical_fields(factors),mechanical0,v0,dt)
        stage_states=tuple(
            CoupledMVIDynamicState(float(times[i]),factors[i],vs[i],accs[i])
            for i in (0,1))
        for stage in stage_states:
            config._validate_state(stage)
        q_end,v_end=mvi_gauss_endpoint(mechanical0,v0,vs,accs,dt)
        # The last stage potential only seeds the endpoint. Refit the
        # mechanical endpoint at EXACTLY rank R, without SVD or truncation.
        endpoint_target=surface.reconstruct(factors[-1]).copy()
        endpoint_target[:surface.n_mechanical]=q_end
        fit=fit_mvi_endpoint_field(
            surface,factors[-1],endpoint_target,config.program,
            relative_tolerance=endpoint_rank_tolerance)
        if not fit.accepted:
            return result("MVI_GAUSS_"+fit.status,stage_states,
                          fit.relative_error)
        end_load,end_charge=loads(t_end)
        endpoint=close_mvi_instantaneous_state(
            config,fit.factors,v_end,time_s=t_end,
            field_scales=field_scales,energy_scale_j=energy_scale_j,
            mechanical_load=end_load,free_charge=end_charge,
            electric_settings=settings,
            max_acceleration_coordinates=max_acceleration_coordinates,
            **acceleration_options,
            on_iteration=on_iteration,cancelled=cancelled)
        if endpoint.state is None:
            return result("MVI_GAUSS_ENDPOINT_"+endpoint.status,
                          stage_states,fit.relative_error)
        return CoupledMVIGaussStep(
            endpoint.state,stage_states,last_solvers,
            t_end,"MVI_GAUSS_CONVERGED",final_norm,
            fit.relative_error)


__all__=("GAUSS_A","GAUSS_B","GAUSS_C","GAUSS_A2_INV",
         "mvi_gauss_stage_kinematics","mvi_gauss_endpoint",
         "CoupledMVIGaussStep","NativeCoupledMVIGauss2")
