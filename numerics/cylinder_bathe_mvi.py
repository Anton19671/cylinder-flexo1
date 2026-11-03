"""Opt-in Bathe two-substep integrator for the existing variational MVI/FDM.

This is NOT Gauss IRK4 and NOT fourth order. Bathe with gamma=2-sqrt(2)
has two implicit displacement endpoints, each obtained from the original
variational physical residual with frozen-partner Y -> X spatial MVI.

No extra nodal endpoint, SVD/Gram, rank truncation, FDM substitution, or
new non-variational PDE equations are introduced.

For the first substep use ordinary average-acceleration Newmark on
delta=gamma*h. For the second substep, exact algebraic rewrites of Bathe's
three-point velocity/acceleration identities give effective Newmark
predictors at t+h, using that SAME positive delta and beta=1/4,gamma=1/2.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from cylinder_variational_dynamic import solve_dynamic_mvi,solve_dynamic_fdm

BATHE_GAMMA=2.0-math.sqrt(2.0)
BATHE_KAPPA_FACTOR=4.0/(BATHE_GAMMA*BATHE_GAMMA)


@dataclass(frozen=True)
class BatheCoefficients:
    h: float
    gamma: float
    delta: float
    c1: float
    c2: float
    c3: float
    kappa: float


@dataclass(frozen=True)
class BatheStepResult:
    first: object
    final: object
    time_first_s: float
    time_final_s: float
    coefficient: BatheCoefficients
    method: str = "BATHE_TWO_SUBSTEP_VARIATIONAL_MVI_OR_FDM_ORDER2"


def bathe_coefficients(h):
    if isinstance(h,(bool,np.bool_)) or not np.isfinite(float(h)) or float(h)<=0:
        raise ValueError("BATHE_TIME_STEP_MUST_BE_POSITIVE")
    h=float(h)
    g=BATHE_GAMMA
    c1=(1.-g)/(g*h)
    c2=-1./((1.-g)*g*h)
    c3=(2.-g)/((1.-g)*h)
    kappa=4./(g*g*h*h)
    if not math.isclose(kappa,c3*c3,rel_tol=3e-15):
        raise ValueError("BATHE_EFFECTIVE_OPERATORS_DIFFER")
    if abs(c1+c2+c3)>1e-10*max(abs(c1),abs(c2),abs(c3)):
        raise ValueError("BATHE_THREE_POINT_COEFFICIENTS_INVALID")
    return BatheCoefficients(h,g,g*h,c1,c2,c3,kappa)


def first_predictors(q0,v0,a0,coefficients):
    """Average acceleration trapezoidal rule at t+gamma*h."""
    q0,v0,a0=(_match_field("Q",q0,q0),_match_field("V",v0,q0),
              _match_field("A",a0,q0))
    d=coefficients.delta
    return q0+d*v0+.25*d*d*a0,v0+.5*d*a0


def _match_field(label,raw,template):
    value=np.asarray(raw,float)
    if value.shape!=np.shape(template) or value.ndim!=3 or not np.isfinite(value).all():
        raise ValueError("BATHE_INVALID_"+label)
    return value


def second_predictors(q0,v0,qgamma,vgamma,coefficients):
    """Exactly encode 3-point BDF second substep as a Newmark residual.

    v_end = c1*q0 + c2*qgamma + c3*q_end
    a_end = c1*v0 + c2*vgamma + c3*v_end

    The existing physical variational MVI solver internally uses
    a_end = kappa*(q_end-q_pred) and
    v_end = v_pred + (delta/2)*a_end.

    Because kappa=c3**2 and delta/2=1/c3, choose
    q_pred = -(c3*(c1*q0+c2*qgamma)+c1*v0+c2*vgamma)/kappa
    v_pred = -(c1*v0+c2*vgamma)/c3.
    """
    q=_match_field("Q",q0,q0)
    v=_match_field("V",v0,q)
    qg=_match_field("QGAMMA",qgamma,q)
    vg=_match_field("VGAMMA",vgamma,q)
    c=coefficients
    displacement_history=c.c1*q+c.c2*qg
    velocity_history=c.c1*v+c.c2*vg
    q_pred=-(c.c3*displacement_history+velocity_history)/c.kappa
    v_pred=-velocity_history/c.c3
    return q_pred,v_pred


def audit_bathe_kinematics(q0,v0,a0,first,final,coefficients):
    """Recheck original Bathe formulas, not the rewritten predictors.

    The audit uses physical fields and fails closed. It is deliberately
    separate from the rank-reconstruction and physical boundary audits.
    """
    c=coefficients
    q0=_match_field("Q",q0,q0)
    v0=_match_field("V",v0,q0)
    a0=_match_field("A",a0,q0)
    qg=_match_field("QGAMMA",first.mechanical,q0)
    vg=_match_field("VGAMMA",first.velocity,q0)
    ag=_match_field("AGAMMA",first.acceleration,q0)
    qe=_match_field("QEND",final.mechanical,q0)
    ve=_match_field("VEND",final.velocity,q0)
    ae=_match_field("AEND",final.acceleration,q0)
    expected_qg=q0+.5*c.delta*(v0+vg)
    expected_vg=v0+.5*c.delta*(a0+ag)
    expected_ve=c.c1*q0+c.c2*qg+c.c3*qe
    expected_ae=c.c1*v0+c.c2*vg+c.c3*ve
    errors={}
    for name,got,want in (
        ("first_displacement",qg,expected_qg),
        ("first_velocity",vg,expected_vg),
        ("end_velocity",ve,expected_ve),
        ("end_acceleration",ae,expected_ae)):
        absolute=float(np.max(abs(got-want),initial=0.))
        relative=absolute/max(float(np.max(abs(got),initial=0.)),
                              float(np.max(abs(want),initial=0.)),1.e-30)
        errors[name]={"absolute":absolute,"relative":relative}
        if relative>3.e-8 and absolute>3.e-13:
            raise ValueError("BATHE_KINEMATICS_DRIFT_"+name.upper())
    return errors


def solve_bathe_variational_step(problem,q0,p0,v0,a0, *,
        time_s,h,density_kg_m3,loads_at,method="MVI",
        settings=None,mvi_settings=None,previous_mvi_result=None,
        mass_damping_rate_per_s=0.,eta_s=0.,on_iteration=None):
    """Two independently accepted physical variational solves per time step.

    The same C-C/other prescribed boundary traces must be imposed by the
    existing physical spatial solver. No algorithmic modification of its
    variational residual or X/Y directions is made.
    """
    if getattr(problem,"kind",None)!="DYNAMIC":
        raise ValueError("BATHE_REQUIRES_PHYSICAL_DYNAMIC_PROBLEM")
    if method not in ("MVI","FDM") or not callable(loads_at):
        raise ValueError("BATHE_REQUIRES_EXPLICIT_MVI_OR_FDM_AND_LOADS")
    q=_match_field("Q",q0,q0)
    v=_match_field("V",v0,q)
    a=_match_field("A",a0,q)
    if not np.isfinite(float(time_s)) or not np.isfinite(float(density_kg_m3)) or density_kg_m3<=0:
        raise ValueError("BATHE_INVALID_PHYSICAL_TIME_OR_DENSITY")
    c=bathe_coefficients(h)
    solve=solve_dynamic_mvi if method=="MVI" else solve_dynamic_fdm
    def evaluate_load(t):
        out=loads_at(t)
        if not isinstance(out,tuple) or len(out)!=2:
            raise ValueError("BATHE_LOADS_REQUIRE_DEAD_LOAD_AND_FREE_CHARGE")
        return out
    def run(mechanical,electrical,qp,vp,t,stage,previous):
        dead,charge=evaluate_load(t)
        kw={"q_predictor":qp,"velocity_predictor":vp,
            "time_s":t,"dt_s":c.delta,"beta":.25,"gamma":.5,
            "density_kg_m3":float(density_kg_m3),
            "dead_load":dead,"free_charge":charge,
            "settings":settings,
            "mass_damping_rate_per_s":mass_damping_rate_per_s,
            "eta_s":eta_s}
        if on_iteration is not None:
            def callback(row):
                on_iteration({"bathe_substep":stage,**row})
            kw["on_iteration"]=callback
        if method=="MVI":
            kw.update(mvi_settings=mvi_settings,previous_mvi_result=previous)
        result=solve(problem,mechanical,electrical,**kw)
        if not result.converged or result.status not in (
                "CONVERGED","CONVERGED_FULL","CONVERGED_PROJECTED"):
            raise RuntimeError("BATHE_"+stage+"_SPATIAL_NOT_CONVERGED_"+result.status)
        return result
    p1,v1=first_predictors(q,v,a,c)
    t1=float(time_s)+c.delta
    r1=run(q,p0,p1,v1,t1,"TRAPEZOIDAL",previous_mvi_result)
    p2,v2=second_predictors(q,v,r1.mechanical,r1.velocity,c)
    t2=float(time_s)+c.h
    r2=run(r1.mechanical,r1.electrical,p2,v2,t2,"THREE_POINT",r1 if method=="MVI" else None)
    audit_bathe_kinematics(q,v,a,r1,r2,c)
    return BatheStepResult(r1,r2,t1,t2,c)


__all__=("BATHE_GAMMA","BATHE_KAPPA_FACTOR","BatheCoefficients",
    "BatheStepResult","bathe_coefficients","first_predictors",
    "second_predictors","audit_bathe_kinematics","solve_bathe_variational_step")
