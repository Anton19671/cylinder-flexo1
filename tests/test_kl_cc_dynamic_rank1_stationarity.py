"""Regression: a stationary KL Newmark MVI state must survive next Y/X sweep.

The first forced Newmark step at t=0.001 s is an independently solved
physical shell problem, not a precomputed/artificial MVI result.
"""
import json
from pathlib import Path
import numpy as np
from cylinder_reference_case import prepare_reference_case
from cylinder_variational_dynamic import solve_dynamic_fdm,solve_dynamic_mvi
from cylinder_variational_mvi import StaticMVISettings
from cylinder_dynamic_loads import normalize_time_law

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"config/verification/kl_cc_dynamic"

def prep(suffix):
    path=CASES/("KL_CC_9x9_HARMONIC25HZ_"+suffix+".json")
    obj=json.loads(path.read_text(encoding="utf-8"))
    return prepare_reference_case(obj,base_dir=path.parent),obj

def solve_first(p,method):
    dyn=p.dynamics
    dt=dyn.settings.dt_s
    beta=dyn.settings.beta
    gamma=dyn.settings.gamma
    a0=p.initial_acceleration
    qpred=p.mechanical+dt*dt*(0.5-beta)*a0
    vpred=dt*(1-gamma)*a0
    kwargs=dict(q_predictor=qpred,velocity_predictor=vpred,
                time_s=dt,dt_s=dt,beta=beta,gamma=gamma,
                density_kg_m3=dyn.density_kg_m3,
                dead_load=dyn.time_law.value(dt)*p.dead_load,
                settings=p.settings)
    if method=="MVI":
        return solve_dynamic_mvi(p.problem,p.mechanical,p.electrical,
                                 mvi_settings=StaticMVISettings(rank=1,max_sweeps=4),**kwargs)
    return solve_dynamic_fdm(p.problem,p.mechanical,p.electrical,**kwargs)

def test_first_forced_newmark_step_fixed_rank_one():
    m,_=prep("MVI1")
    f,_=prep("MKR_FDM")
    assert m.problem.functional.theory=="KL"
    assert m.problem.functional.geometry_model=="LINEAR"
    assert m.problem.functional.ops.shape == (9,9)
    actual=solve_first(m,"MVI")
    control=solve_first(f,"FDM")
    assert actual.converged,actual.status
    assert actual.status=="CONVERGED_FULL",actual.status
    assert control.converged,control.status
    assert actual.sweeps<=4
    relative=np.linalg.norm(actual.mechanical-control.mechanical)/np.linalg.norm(control.mechanical)
    print("MVI1 first Newmark step",actual.status,"sweeps",actual.sweeps,
          "relative displacement difference",relative,flush=True)
    assert relative<1e-6
    for field in ("mechanical","velocity","acceleration"):
        a=getattr(actual,field)
        b=getattr(control,field)
        assert np.linalg.norm(a-b) <= 1e-6*max(np.linalg.norm(b),1e-16),field
    assert np.max(np.abs(actual.mechanical[:,[0,-1],:]))<1e-10
    sweeps=[r for r in actual.history if r.get("event")=="sweep"]
    assert len(sweeps)>=2
    assert sweeps[1]["residual_scaled"] < 1e-11
