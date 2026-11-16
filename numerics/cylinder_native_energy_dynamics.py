"""Energy-consistent native FDM time drivers: Newmark, Bathe, Gauss IRK4.

These time integrators use the SAME native-FDM discrete first variation as
NativeEnergyFDM and the SAME exact affine end/gauge constraints as
NativeEnergyEndSpace. No MVI, reference FDM/SBP, strong-form spatial stencil,
penalty supports, auto physical order or silent full-step acceptance.

Mechanical positions are affine in native free coordinates; velocities and
accelerations lie in the corresponding homogeneous essential-trace space.
The electrical problem remains quasistatic at every stage. Natural end
loads are incorporated by virtual work within the end-space projection.

Research status: algebraic and virtual-work verification is required before
physical production claims. No time step is executed on import.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from cylinder_native_energy_fdm import (
    NativeEnergyEndSpace, NativeEnergyFDM,
)
from cylinder_direct_strong_newton import (
    StrongNewtonSettings, solve_strong_nonlinear_equations,
)

BATHE_GAMMA=2.-math.sqrt(2.)
_GAUSS_C=np.array(((3.-math.sqrt(3.))/6.,(3.+math.sqrt(3.))/6.))
_GAUSS_A=np.array(((.25,.25-math.sqrt(3.)/6.),
                   (.25+math.sqrt(3.)/6.,.25)))
_GAUSS_A2_INV=np.linalg.inv(_GAUSS_A@_GAUSS_A)
_GAUSS_B=np.array((.5,.5))
for _m in (_GAUSS_C,_GAUSS_A,_GAUSS_A2_INV,_GAUSS_B):
    _m.setflags(write=False)


def _finite_positive(value,name):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(name+": finite positive number required")
    try:
        out=float(value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(name+": finite positive number required") from exc
    if not math.isfinite(out) or out<=0:
        raise ValueError(name+": finite positive number required")
    return out


def _finite_nonnegative(value,name):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(name+": finite nonnegative number required")
    try:
        out=float(value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(name+": finite nonnegative number required") from exc
    if not math.isfinite(out) or out<0:
        raise ValueError(name+": finite nonnegative number required")
    return out


def _finite_field(field,shape,name):
    a=np.asarray(field,float)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError(name+": finite shape "+str(shape)+" required")
    return a


@dataclass(frozen=True)
class NativeEnergyState:
    time_s: float
    mechanical: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    electrical: np.ndarray | None = None

    def __post_init__(self):
        t=_finite_nonnegative(self.time_s,"state time")
        q=np.asarray(self.mechanical,float)
        if q.ndim!=3 or min(q.shape)<1 or not np.isfinite(q).all():
            raise ValueError("NATIVE_ENERGY_STATE_SHAPE_INVALID")
        object.__setattr__(self,"time_s",t)
        for name,value,shape in (
            ("mechanical",self.mechanical,q.shape),
            ("velocity",self.velocity,q.shape),
            ("acceleration",self.acceleration,q.shape)):
            out=_finite_field(value,shape,name).copy()
            out.setflags(write=False)
            object.__setattr__(self,name,out)
        if self.electrical is not None:
            p=np.asarray(self.electrical,float)
            if p.ndim!=3 or p.shape[1:]!=q.shape[1:] or not np.isfinite(p).all():
                raise ValueError("NATIVE_ENERGY_STATE_ELECTRIC_SHAPE_INVALID")
            p=p.copy()
            p.setflags(write=False)
            object.__setattr__(self,"electrical",p)


@dataclass(frozen=True)
class NativeEnergyTimeResult:
    state: NativeEnergyState | None
    nonlinear: object
    status: str
    first_nonlinear: object | None = None
    endpoint_nonlinear: object | None = None
    attempted_time_s: float | None = None


class NativeEnergyDynamics:
    """Three native energy-FDM integrators on one constrained field chart."""

    def __init__(self, space: NativeEnergyEndSpace, *, density_kg_m3,
                 eta_s=0.,mass_damping_rate_per_s=0.,
                 beta=.25,gamma=.5):
        if not isinstance(space,NativeEnergyEndSpace):
            raise TypeError("NativeEnergyEndSpace required")
        if not isinstance(space.engine,NativeEnergyFDM):
            raise TypeError("native energy FDM model required")
        self.space=space
        self.engine=space.engine
        self.model=space.engine.model
        self.density=_finite_positive(density_kg_m3,"density")
        self.eta=_finite_nonnegative(eta_s,"viscosity")
        self.mass_damping=_finite_nonnegative(
            mass_damping_rate_per_s,"mass damping")
        self.beta=_finite_positive(beta,"Newmark beta")
        self.gamma=_finite_positive(gamma,"Newmark gamma")
        if not (.5<=self.gamma<=1. and
                self.beta>=.25*(self.gamma+.5)**2):
            raise ValueError("NATIVE_ENERGY_NEWMARK_PARAMETER_PAIR_UNSTABLE")
        self.nm=sum(space.counts[:space.nmech])
        self.ne=space.reduced_size-self.nm
        if self.ne<0:
            raise ValueError("NATIVE_ENERGY_REDUCED_CHART_INVALID")

    def _project_q(self,q,p,*,electric_guess=False):
        packed=self.space.encode(q,p)
        restored_q,restored_p=self.space.decode(packed)
        self._check_close(restored_q,q,"mechanical essential traces")
        # Electrical data passed as an initial *guess* may not yet obey
        # the gauge or imposed potential. They are projected exactly.
        # By contrast, an already accepted physical state must obey both.
        if p is not None and not electric_guess:
            self._check_close(restored_p,p,"electric essential/gauge traces")
        return packed[:self.nm],packed[self.nm:]

    @staticmethod
    def _check_close(actual,wanted,label):
        # A support may be specified in physical units; a uniform absolute
        # tolerance below is intentionally NOT relaxed by the bulk force.
        error=float(np.max(np.abs(actual-wanted)))
        if error>1.e-9:
            raise ValueError("NATIVE_ENERGY_INCOMPATIBLE_"+label.upper().replace(" ","_")+
                             f": max_abs_defect={error:.6e}")

    def _mechanical_lift(self,freedom):
        z=np.asarray(freedom,float)
        if z.shape!=(self.nm,) or not np.isfinite(z).all():
            raise ValueError("NATIVE_ENERGY_VELOCITY_ACCELERATION_COORDS_INVALID")
        fields=[]
        at=0
        for subspace,n in zip(self.space.mechanical_spaces,
                              self.space.counts[:self.space.nmech]):
            block=z[at:at+n].reshape(subspace.nfree,self.space.ns)
            fields.append(subspace.null@block)
            at+=n
        return np.stack(fields)

    def _project_motion(self,field,label):
        a=_finite_field(field,self.model.shape,label)
        coords=np.concatenate([
            (subspace.null.T@a[k]).ravel()
            for k,subspace in enumerate(self.space.mechanical_spaces)])
        self._check_close(self._mechanical_lift(coords),a,label+" essential trace")
        return coords

    def _coordinates(self,state):
        if not isinstance(state,NativeEnergyState):
            raise TypeError("NativeEnergyState required")
        if state.electrical is None and self.model.electrical_on:
            raise ValueError("NATIVE_ENERGY_MISSING_ELECTRIC_STATE")
        q,e=self._project_q(
            _finite_field(state.mechanical,self.model.shape,"displacement"),
            state.electrical)
        v=self._project_motion(state.velocity,"velocity")
        a=self._project_motion(state.acceleration,"acceleration")
        return q,v,a,e

    def _electric_guess(self,guess):
        if not self.model.electrical_on:
            if guess is not None:
                raise ValueError("NATIVE_ENERGY_ELECTRICITY_OFF")
            return None
        if guess is None:
            return np.zeros((len(self.space.electrical_spaces),
                             *self.model.shape[1:]))
        return _finite_field(guess,(
            len(self.space.electrical_spaces),*self.model.shape[1:]),
            "electric guess")

    def _loads(self,loads_at,time_s):
        if loads_at is None:
            self.space.check_electric_source(None)
            return None,None
        if not callable(loads_at):
            raise TypeError("loads_at must be a physical-time load callback")
        loads=loads_at(float(time_s))
        if not isinstance(loads,(tuple,list)) or len(loads)!=2:
            raise ValueError("loads_at must return mechanical and electric charge")
        q=loads[0]
        e=loads[1]
        if q is not None:
            q=_finite_field(q,self.model.shape,"mechanical load").copy()
        if e is not None:
            if not self.model.electrical_on:
                raise ValueError("NATIVE_ENERGY_ELECTRICITY_OFF")
            e=_finite_field(e,(
                len(self.space.electrical_spaces),*self.model.shape[1:]),
                "free electric charge").copy()
        self.space.check_electric_source(e)
        return q,e

    def _work(self,q,p,v,a,loads):
        return self.engine.balance(
            q,p,velocity=v,acceleration=a,
            density_kg_m3=self.density,eta_s=self.eta,
            mass_damping_rate_per_s=self.mass_damping,
            mechanical_load=loads[0],electrical_charge=loads[1])

    def _linear_solve(self,initial,equations,*,state_scale,equation_scale,
                      settings):
        if not isinstance(settings,StrongNewtonSettings):
            raise TypeError("StrongNewtonSettings required")
        init=np.asarray(initial,float)
        if init.ndim!=1 or not np.isfinite(init).all():
            raise ValueError("NATIVE_ENERGY_INITIAL_COORDS_INVALID")
        n=len(init)
        def _scale(arg,label):
            a=np.asarray(arg,float)
            if a.ndim==0:
                return float(a)
            if a.shape!=(n,):
                raise ValueError(label+": scalar or full reduced coordinate scale required")
            return a.reshape(1,1,n)
        def wrapped(y,not_electric):
            if not_electric is not None:
                raise ValueError("NATIVE_ENERGY_REDUCED_ELECTRIC_PACKING_INVALID")
            residual=np.asarray(equations(y.ravel()),float)
            if residual.shape!=(n,) or not np.isfinite(residual).all():
                raise ValueError("NATIVE_ENERGY_REDUCED_RESIDUAL_INVALID")
            return residual.reshape(1,1,n),None
        return solve_strong_nonlinear_equations(
            wrapped,init.reshape(1,1,n),
            mechanical_state_scale=_scale(state_scale,"reduced state scale"),
            mechanical_equation_scale=_scale(equation_scale,
                                              "reduced equation scale"),
            settings=settings)

    def _state(self,time_s,y_mechanical,v_mechanical,a_mechanical,y_electric):
        q,p=self.space.decode(np.concatenate((y_mechanical,y_electric)))
        return NativeEnergyState(
            time_s,q,self._mechanical_lift(v_mechanical),
            self._mechanical_lift(a_mechanical),p)

    def initialize(self,mechanical,velocity,electrical_guess=None, *,
                   time_s=0.,acceleration_guess=None,loads_at=None,
                   reduced_state_scale,reduced_equation_scale,
                   settings=StrongNewtonSettings()):
        """Solve simultaneously for native reduced initial a0 and phi0."""
        t=_finite_nonnegative(time_s,"initial time")
        p0=self._electric_guess(electrical_guess)
        q0,e0=self._project_q(
            _finite_field(mechanical,self.model.shape,"initial displacement"),p0,
            electric_guess=True)
        v0=self._project_motion(velocity,"initial velocity")
        a0=(np.zeros(self.nm) if acceleration_guess is None else
            self._project_motion(acceleration_guess,"initial acceleration guess"))
        loads=self._loads(loads_at,t)
        def equations(y):
            trial_a=y[:self.nm]
            trial_e=y[self.nm:]
            q,p=self.space.decode(np.concatenate((q0,trial_e)))
            v=self._mechanical_lift(v0)
            a=self._mechanical_lift(trial_a)
            return self.space.residual(self._work(q,p,v,a,loads))
        result=self._linear_solve(
            np.concatenate((a0,e0)),equations,
            state_scale=reduced_state_scale,
            equation_scale=reduced_equation_scale,settings=settings)
        if not result.converged:
            return NativeEnergyTimeResult(
                None,result,"INITIAL_"+result.status,attempted_time_s=t)
        y=result.mechanical.ravel()
        return NativeEnergyTimeResult(
            self._state(t,q0,v0,y[:self.nm],y[self.nm:]),
            result,"CONVERGED",attempted_time_s=t)

    def _step_solve(self,qguess,eguess,kinematics,loads,t_next,
                    *,reduced_state_scale,reduced_equation_scale,settings):
        def residual(y):
            qfree=y[:self.nm]
            efree=y[self.nm:]
            q,p=self.space.decode(y)
            vfree,afree=kinematics(qfree)
            return self.space.residual(
                self._work(q,p,self._mechanical_lift(vfree),
                           self._mechanical_lift(afree),loads))
        result=self._linear_solve(
            np.concatenate((qguess,eguess)),residual,
            state_scale=reduced_state_scale,
            equation_scale=reduced_equation_scale,settings=settings)
        if not result.converged:
            return None,result
        y=result.mechanical.ravel()
        qnew=y[:self.nm]
        vnew,anew=kinematics(qnew)
        return self._state(t_next,qnew,vnew,anew,y[self.nm:]),result

    def newmark(self,state:NativeEnergyState,dt_s, *,
                loads_at=None,reduced_state_scale,
                reduced_equation_scale,settings=StrongNewtonSettings()):
        q0,v0,a0,e0=self._coordinates(state)
        h=_finite_positive(dt_s,"Newmark step")
        h2=h*h
        den=self.beta*h2
        if not np.isfinite(den) or den<1./np.finfo(float).max:
            raise ValueError("NATIVE_ENERGY_UNREPRESENTABLE_NEWMARK_STEP")
        t=state.time_s+h
        if not math.isfinite(t) or t<=state.time_s:
            raise ValueError("NATIVE_ENERGY_NONADVANCING_TIME")
        qp=q0+h*v0+(.5-self.beta)*h2*a0
        vp=v0+(1.-self.gamma)*h*a0
        loads=self._loads(loads_at,t)
        def kinematics(q):
            a=(q-qp)/den
            v=vp+self.gamma*h*a
            return v,a
        accepted,result=self._step_solve(
            qp,e0,kinematics,loads,t,
            reduced_state_scale=reduced_state_scale,
            reduced_equation_scale=reduced_equation_scale,settings=settings)
        return NativeEnergyTimeResult(
            accepted,result,"CONVERGED" if accepted is not None else
            "NEWMARK_"+result.status,attempted_time_s=t)

    def bathe(self,state:NativeEnergyState,dt_s, *,
              loads_at=None,reduced_state_scale,
              reduced_equation_scale,settings=StrongNewtonSettings()):
        q0,v0,a0,e0=self._coordinates(state)
        h=_finite_positive(dt_s,"Bathe step")
        t2=state.time_s+h
        delta=BATHE_GAMMA*h
        t1=state.time_s+delta
        if (not math.isfinite(t2) or
                not state.time_s<t1<t2 or
                not math.isfinite(delta*delta) or
                delta*delta<4./np.finfo(float).max):
            raise ValueError("NATIVE_ENERGY_UNREPRESENTABLE_BATHE_STEP")
        k1=4./(delta*delta)
        qp1=q0+delta*v0+.25*delta*delta*a0
        vp1=v0+.5*delta*a0
        first,first_solver=self._step_solve(
            qp1,e0,lambda q:(vp1+.5*delta*k1*(q-qp1),
                              k1*(q-qp1)),
            self._loads(loads_at,t1),t1,
            reduced_state_scale=reduced_state_scale,
            reduced_equation_scale=reduced_equation_scale,settings=settings)
        if first is None:
            return NativeEnergyTimeResult(
                None,first_solver,"BATHE_FIRST_"+first_solver.status,
                first_nonlinear=first_solver,attempted_time_s=t2)
        q1,v1,a1,e1=self._coordinates(first)
        c1=(1.-BATHE_GAMMA)/delta
        c2=-1./((1.-BATHE_GAMMA)*delta)
        c3=(2.-BATHE_GAMMA)/((1.-BATHE_GAMMA)*h)
        k2=c3*c3
        if not math.isfinite(k2) or k2<=0:
            raise ValueError("NATIVE_ENERGY_UNREPRESENTABLE_BATHE_SECOND_STAGE")
        qp2=-(c3*(c1*q0+c2*q1)+(c1*v0+c2*v1))/k2
        vp2=-(c1*v0+c2*v1)/c3
        second,second_solver=self._step_solve(
            qp2,e1,lambda q:(vp2+(k2/c3)*(q-qp2),
                             k2*(q-qp2)),
            self._loads(loads_at,t2),t2,
            reduced_state_scale=reduced_state_scale,
            reduced_equation_scale=reduced_equation_scale,settings=settings)
        if second is None:
            return NativeEnergyTimeResult(
                None,second_solver,"BATHE_SECOND_"+second_solver.status,
                first_nonlinear=first_solver,attempted_time_s=t2)
        return NativeEnergyTimeResult(
            second,second_solver,"CONVERGED",
            first_nonlinear=first_solver,attempted_time_s=t2)

    def gauss(self,state:NativeEnergyState,dt_s, *,
              loads_at=None,reduced_state_scale,
              reduced_equation_scale,settings=StrongNewtonSettings()):
        q0,v0,a0,e0=self._coordinates(state)
        h=_finite_positive(dt_s,"Gauss step")
        h2=h*h
        if (not math.isfinite(h2) or
                h2<float(np.max(np.abs(_GAUSS_A2_INV)))
                /np.finfo(float).max):
            raise ValueError("NATIVE_ENERGY_UNREPRESENTABLE_GAUSS_STEP")
        times=state.time_s+h*_GAUSS_C
        t2=state.time_s+h
        if not (math.isfinite(t2) and state.time_s<times[0]<times[1]<t2):
            raise ValueError("NATIVE_ENERGY_NONADVANCING_GAUSS_TIME")
        forces=tuple(self._loads(loads_at,t) for t in times)
        n=self.space.reduced_size
        qguess=np.stack([
            q0+h*_GAUSS_C[i]*v0+.5*(h*_GAUSS_C[i])**2*a0
            for i in range(2)])
        eguess=np.stack((e0,e0))
        initial=np.concatenate((
            np.concatenate((qguess[0],eguess[0])),
            np.concatenate((qguess[1],eguess[1]))))
        def stage_kinematics(qst):
            rhs=qst-q0[None,:]-h*_GAUSS_C[:,None]*v0[None,:]
            a=(_GAUSS_A2_INV@rhs)/h2
            v=v0[None,:]+h*_GAUSS_A@a
            return v,a
        def equations(y):
            free=y.reshape(2,n)
            qs=np.stack([free[k,:self.nm] for k in range(2)])
            vv,aa=stage_kinematics(qs)
            work=[]
            for k in range(2):
                q,p=self.space.decode(free[k])
                residual=self.space.residual(
                    self._work(q,p,self._mechanical_lift(vv[k]),
                               self._mechanical_lift(aa[k]),forces[k]))
                work.append(residual)
            return np.concatenate(work)
        scale=np.asarray(reduced_state_scale,float)
        force_scale=np.asarray(reduced_equation_scale,float)
        if scale.ndim:
            if scale.shape!=(n,):
                raise ValueError("GAUSS_REDUCED_SCALE_INVALID")
            scale=np.tile(scale,2)
        if force_scale.ndim:
            if force_scale.shape!=(n,):
                raise ValueError("GAUSS_REDUCED_FORCE_SCALE_INVALID")
            force_scale=np.tile(force_scale,2)
        result=self._linear_solve(
            initial,equations,state_scale=scale,
            equation_scale=force_scale,settings=settings)
        if not result.converged:
            return NativeEnergyTimeResult(
                None,result,"GAUSS_STAGES_"+result.status,
                first_nonlinear=result,attempted_time_s=t2)
        free=result.mechanical.ravel().reshape(2,n)
        qst=np.stack([free[k,:self.nm] for k in range(2)])
        vv,aa=stage_kinematics(qst)
        qend=q0+h*np.einsum("i,ij->j",_GAUSS_B,vv)
        vend=v0+h*np.einsum("i,ij->j",_GAUSS_B,aa)
        endpoint=self.initialize(
            self.space.decode(np.concatenate((qend,free[-1,self.nm:])))[0],
            self._mechanical_lift(vend),
            self.space.decode(free[-1])[1],
            time_s=t2,acceleration_guess=self._mechanical_lift(aa[-1]),
            loads_at=loads_at,
            reduced_state_scale=reduced_state_scale,
            reduced_equation_scale=reduced_equation_scale,settings=settings)
        if endpoint.state is None:
            return NativeEnergyTimeResult(
                None,endpoint.nonlinear,
                "GAUSS_ENDPOINT_"+endpoint.nonlinear.status,
                first_nonlinear=result,
                endpoint_nonlinear=endpoint.nonlinear,attempted_time_s=t2)
        return NativeEnergyTimeResult(
            endpoint.state,endpoint.nonlinear,"CONVERGED",
            first_nonlinear=result,
            endpoint_nonlinear=endpoint.nonlinear,attempted_time_s=t2)


__all__=("NativeEnergyState","NativeEnergyTimeResult",
         "NativeEnergyDynamics","BATHE_GAMMA")
