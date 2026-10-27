"""Independent two-stage Gauss--Legendre IRK4 for second-order electro-mechanics.

Standalone small-grid reference time integrator. It does not switch spatial
methods or silently assert that MVI stage/end rank constraints are satisfied.
The caller supplies the fully assembled equations at both Gauss stage times,
including algebraic electric equations and every physical boundary condition.
The production MVI/Newmark driver is not modified by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve, get_lapack_funcs

SQRT3 = math.sqrt(3.)
GAUSS2_C = np.array([(3.-SQRT3)/6., (3.+SQRT3)/6.])
GAUSS2_A = np.array([[.25, .25-SQRT3/6.],
                     [.25+SQRT3/6., .25]])
GAUSS2_B = np.array([.5, .5])


@dataclass(frozen=True)
class Gauss2Settings:
    dt_s: float
    max_iterations: int = 12
    max_unknowns: int = 256
    difference_step: float = 1.e-6
    scaled_equation_tolerance: float = 1.e-10
    max_line_search_halvings: int = 12

    def __post_init__(self):
        if (isinstance(self.dt_s, (bool, np.bool_)) or
                not math.isfinite(float(self.dt_s)) or float(self.dt_s) <= 0.):
            raise ValueError("Gauss2 dt_s must be finite and positive")
        object.__setattr__(self, "dt_s", float(self.dt_s))
        for name in ("max_iterations", "max_unknowns", "max_line_search_halvings"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "max_line_search_halvings" else 1):
                raise ValueError(name+": invalid integer")
        for name in ("difference_step", "scaled_equation_tolerance"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or
                    not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(name+": positive finite value required")
            object.__setattr__(self, name, float(value))
        if self.difference_step >= .1:
            raise ValueError("difference_step must be below 0.1")


@dataclass(frozen=True)
class Gauss2State:
    time_s: float
    mechanical: np.ndarray
    velocity: np.ndarray
    electrical: np.ndarray | None = None

    def __post_init__(self):
        if isinstance(self.time_s, (bool, np.bool_)) or not math.isfinite(float(self.time_s)):
            raise ValueError("finite physical time required")
        q = np.asarray(self.mechanical, float)
        v = np.asarray(self.velocity, float)
        if (q.ndim != 3 or not q.size or v.shape != q.shape or
                not np.isfinite(q).all() or not np.isfinite(v).all()):
            raise ValueError("finite (component,nx,ns) displacement and velocity required")
        p = None if self.electrical is None else np.asarray(self.electrical, float)
        if (p is not None and (p.ndim != 3 or p.shape[1:] != q.shape[1:] or
                               not p.size or not np.isfinite(p).all())):
            raise ValueError("electric modes must match mechanical surface grid")
        object.__setattr__(self, "mechanical", q.copy())
        object.__setattr__(self, "velocity", v.copy())
        object.__setattr__(self, "electrical", None if p is None else p.copy())
        object.__setattr__(self, "time_s", float(self.time_s))


@dataclass(frozen=True)
class Gauss2Stage:
    time_s: float
    mechanical: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    electrical: np.ndarray | None


@dataclass(frozen=True)
class Gauss2Result:
    final: Gauss2State
    stages: tuple[Gauss2Stage, Gauss2Stage]
    iterations: int
    method: str = "GAUSS_LEGENDRE_2_STAGE_ORDER_4_EXPERIMENTAL_FDM_REFERENCE"


def _scale(value, shape, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(label+": positive finite scale required")
    a = np.asarray(value, float)
    if a.ndim == 0:
        a = np.full(shape, float(a))
    if a.shape != shape or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError(label+": finite positive scale with full field shape required")
    return a


def gauss2_stage_kinematics(q0, v0, acceleration, dt_s):
    """Exact collocation identities for two stage acceleration fields.

    Q_i=q0+dt*c_i*v0+dt^2*(A@A)_ij*a_j;
    V_i=v0+dt*A_ij*a_j. No spatial rank constraint on V_i or a_i.
    """
    q0=np.asarray(q0,float)
    v0=np.asarray(v0,float)
    aa=np.asarray(acceleration,float)
    if (q0.ndim != 3 or q0.size == 0 or v0.shape != q0.shape or
            aa.shape != (2,)+q0.shape or not np.isfinite(q0).all() or
            not np.isfinite(v0).all() or not np.isfinite(aa).all() or
            isinstance(dt_s,(bool,np.bool_)) or not math.isfinite(float(dt_s)) or dt_s<=0):
        raise ValueError("invalid Gauss2 stage kinematics")
    dt=float(dt_s)
    vstage=v0[None,...] + dt*np.einsum("ij,j...->i...", GAUSS2_A, aa)
    qstage=q0[None,...] + dt*GAUSS2_C[:,None,None,None]*v0[None,...] + dt*dt*np.einsum(
        "ij,j...->i...", GAUSS2_A@GAUSS2_A, aa)
    return qstage,vstage


def gauss2_step(initial, settings, *, equations, mechanical_state_scale,
                mechanical_equation_scale, electrical_state_scale=None,
                electrical_equation_scale=None, acceleration_guess=None,
                end_electrical_solver=None):
    """Solve BOTH Gauss stages simultaneously, including algebraic electric modes.

    equations(q_i,v_i,a_i,p_i,t_i)->(mechanical_eq,electrical_eq_or_None).
    The callback must include physical forces, electrical saddle equations,
    end conditions, essential supports and gauge constraints. It is called
    at each interior stage time, never only at t_(n+1).

    No fallback to Newmark, no ad-hoc step correction and no MVI assertion.
    A separate endpoint electric solve is mandatory with electric physics ON.
    """
    if not isinstance(initial, Gauss2State) or not isinstance(settings, Gauss2Settings):
        raise TypeError("Gauss2State and Gauss2Settings required")
    if not callable(equations):
        raise TypeError("coupled stage equations callback required")
    q0,v0,p0=initial.mechanical,initial.velocity,initial.electrical
    qshape=q0.shape
    pshape=None if p0 is None else p0.shape
    dt=settings.dt_s
    qsc=_scale(mechanical_state_scale,qshape,"mechanical displacement")
    qfc=_scale(mechanical_equation_scale,qshape,"mechanical equations")
    if (pshape is None)!=(electrical_state_scale is None) or (
            pshape is None)!=(electrical_equation_scale is None):
        raise ValueError("electric state and its two scales must agree")
    if pshape is None:
        if end_electrical_solver is not None:
            raise ValueError("end electric solver supplied with electric physics OFF")
    else:
        if not callable(end_electrical_solver):
            raise ValueError("electric ON requires separate endpoint algebraic solver")
        psc=_scale(electrical_state_scale,pshape,"electric potential")
        pfc=_scale(electrical_equation_scale,pshape,"electric equations")
    # Unknown stage acceleration has physical dimension q / dt**2.
    asc=qsc/dt**2
    scales=np.concatenate(([asc.ravel(),asc.ravel()] +
        ([] if pshape is None else [psc.ravel(),psc.ravel()])))
    eqscales=np.concatenate(([qfc.ravel(),qfc.ravel()] +
        ([] if pshape is None else [pfc.ravel(),pfc.ravel()])))
    n=scales.size
    if n>settings.max_unknowns:
        raise ValueError("GAUSS2_REFERENCE_UNKNOWN_LIMIT: reduce grid or raise explicit limit")
    guess=np.zeros((2,)+qshape) if acceleration_guess is None else np.asarray(acceleration_guess,float)
    if guess.shape!=(2,)+qshape or not np.isfinite(guess).all():
        raise ValueError("acceleration_guess must supply two full stage fields")
    z=np.concatenate(([guess[0].ravel(),guess[1].ravel()] +
                      ([] if pshape is None else [p0.ravel(),p0.ravel()]))) / scales
    stage_times=initial.time_s+dt*GAUSS2_C
    if not np.isfinite(stage_times).all() or stage_times[0]<=initial.time_s:
        raise ValueError("physical time step cannot be represented")

    def unpack(u):
        if u.shape!=(n,) or not np.isfinite(u).all():
            raise ValueError("nonfinite Gauss2 trial state")
        raw=u*scales
        size=q0.size
        acceleration=np.stack((raw[:size].reshape(qshape),raw[size:2*size].reshape(qshape)))
        electric=None if pshape is None else np.stack((
            raw[2*size:2*size+p0.size].reshape(pshape),raw[2*size+p0.size:].reshape(pshape)))
        qq,vv=gauss2_stage_kinematics(q0,v0,acceleration,dt)
        return qq,vv,acceleration,electric

    def evaluate(u):
        qq,vv,aa,pp=unpack(u)
        parts=[]
        eparts=[]
        for i in range(2):
            out=equations(qq[i].copy(),vv[i].copy(),aa[i].copy(),
                          None if pp is None else pp[i].copy(),float(stage_times[i]))
            if not isinstance(out,tuple) or len(out)!=2:
                raise TypeError("stage equations must return (mechanical,electrical)")
            rm=np.asarray(out[0],float)
            if rm.shape!=qshape or not np.isfinite(rm).all():
                raise ValueError("invalid mechanical stage equations")
            parts.append(rm.ravel())
            if pshape is None:
                if out[1] is not None:
                    raise ValueError("electric equations supplied while electric OFF")
            else:
                ep=np.asarray(out[1],float)
                if ep.shape!=pshape or not np.isfinite(ep).all():
                    raise ValueError("invalid algebraic electric stage equations")
                eparts.append(ep.ravel())
        return np.concatenate(parts+eparts)/eqscales

    fun=evaluate(z)
    norm=float(np.max(np.abs(fun)))
    iteration=0
    for k in range(settings.max_iterations+1):
        if norm<=settings.scaled_equation_tolerance:
            break
        if k==settings.max_iterations:
            raise ValueError("GAUSS2_STAGE_SOLVE_NOT_COMPLETED")
        tangent=np.empty((n,n),float)
        for j in range(n):
            delta=settings.difference_step*(1.+abs(z[j]))
            unit=np.zeros(n);unit[j]=delta
            tangent[:,j]=(evaluate(z+unit)-evaluate(z-unit))/(2.*delta)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error",LinAlgWarning)
                lu,piv=lu_factor(tangent,check_finite=False)
            rcond,info=get_lapack_funcs("gecon",(tangent,))(lu,float(np.linalg.norm(tangent,ord=1)))
            if info!=0 or not np.isfinite(rcond) or rcond<=np.finfo(float).eps:
                raise ValueError("ill-conditioned stage tangent")
            step=lu_solve((lu,piv),-fun,check_finite=False)
        except (LinAlgWarning,ValueError) as exc:
            raise ValueError("GAUSS2_STAGE_TANGENT_SINGULAR") from exc
        if not np.isfinite(step).all():
            raise ValueError("GAUSS2_STAGE_TANGENT_SINGULAR")
        accepted=False
        for halve in range(settings.max_line_search_halvings+1):
            proposal=z+2.**(-halve)*step
            try:
                new_fun=evaluate(proposal)
            except (ValueError,FloatingPointError,OverflowError):
                continue
            new_norm=float(np.max(np.abs(new_fun)))
            if new_norm<norm or new_norm<=settings.scaled_equation_tolerance:
                z,fun,norm=proposal,new_fun,new_norm
                accepted=True
                break
        if not accepted:
            raise ValueError("GAUSS2_STAGE_LINE_SEARCH_FAILED")
        iteration=k+1
    qq,vv,aa,pp=unpack(z)
    q_end=q0+dt*np.einsum("i,i...->...",GAUSS2_B,vv)
    v_end=v0+dt*np.einsum("i,i...->...",GAUSS2_B,aa)
    if not np.isfinite(q_end).all() or not np.isfinite(v_end).all():
        raise ValueError("GAUSS2_NONFINITE_FINAL_MECHANICS")
    t_end=initial.time_s+dt
    if pshape is None:
        p_end=None
    else:
        guess=pp[-1].copy()
        p_end=np.asarray(end_electrical_solver(q_end.copy(),v_end.copy(),t_end,guess),float)
        if p_end.shape!=pshape or not np.isfinite(p_end).all():
            raise ValueError("GAUSS2_ELECTRIC_ENDPOINT_SOLVE_INVALID")
    final=Gauss2State(t_end,q_end,v_end,p_end)
    stages=tuple(Gauss2Stage(float(stage_times[i]),qq[i].copy(),vv[i].copy(),
                             aa[i].copy(),None if pp is None else pp[i].copy())
                 for i in range(2))
    return Gauss2Result(final,stages,iteration)


__all__=("GAUSS2_A","GAUSS2_B","GAUSS2_C","Gauss2Settings","Gauss2State",
         "Gauss2Stage","Gauss2Result","gauss2_stage_kinematics","gauss2_step")
