"""Physical regression for KL uniform C-C static rank-1 MVI across sweeps.

A frozen-partner space that already satisfies KL clamping must not interpret
roundoff in C*T as new constraints and destroy the stationary field.
"""
import copy
import json
from pathlib import Path
import numpy as np
from cylinder_reference_case import prepare_reference_case
from cylinder_variational_mvi import solve_static_mvi, StaticMVISettings
from cylinder_variational_fdm import solve_static_fdm

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"config/verification/kl_cc_uniform_static"

def load(name):
    p=CASES/(name+".json")
    c=json.loads(p.read_text(encoding="utf-8"))
    return prepare_reference_case(c,base_dir=p.parent)

def test_kl_cc_rank1_stationarity_survives_second_sweep():
    m=load("KL_CC_9x9_100KPA_MVI1")
    r=solve_static_mvi(m.problem,m.mechanical,m.electrical,
        dead_load=m.dead_load,settings=m.settings,
        mvi_settings=StaticMVISettings(rank=1,max_sweeps=3))
    assert r.converged, "rank-1 must not drift after first stationary sweep: "+r.status
    assert r.status=="CONVERGED_FULL",r.status
    assert r.sweeps <= 3
    assert np.max(np.abs(r.mechanical[:,[0,-1],:]))<1.e-11
    sweeps=[x for x in r.history if x.get("event")=="sweep"]
    assert len(sweeps)>=3
    assert abs(sweeps[2]["potential_j"]-sweeps[1]["potential_j"]) <= 1.e-8 * max(1.,abs(sweeps[1]["potential_j"]))
    assert sweeps[2]["sweep_field_change_relative"] < 1.e-6

    f=load("KL_CC_9x9_100KPA_MKR_FDM")
    reference=solve_static_fdm(f.problem,f.mechanical,f.electrical,
        dead_load=f.dead_load,settings=f.settings)
    assert reference.converged,reference.status
    delta=np.linalg.norm(r.mechanical-reference.mechanical)
    scale=np.linalg.norm(reference.mechanical)
    assert delta <= 1.e-6*scale, (delta, scale)

def test_identical_physics_half_pressure_rank1_preflight():
    for pressure in (50000,100000):
        name="KL_CC_9x9_"+str(pressure//1000)+"KPA_MVI1"
        p=CASES/(name+".json")
        c=json.loads(p.read_text(encoding="utf-8"))
        prepared=prepare_reference_case(copy.deepcopy(c),base_dir=p.parent)
        assert prepared.mvi_settings.rank==1
        assert prepared.problem.functional.theory=="KL"
        assert prepared.problem.functional.geometry_model=="LINEAR"
        assert np.allclose(prepared.dead_load[2],-pressure)
