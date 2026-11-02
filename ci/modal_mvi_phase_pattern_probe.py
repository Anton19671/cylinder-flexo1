#!/usr/bin/env python3
"""Forensic KL MVI-1: componentwise Fourier phase versus uniform phase.

Uses only actual variational KL shell operators and mass-orthogonality from
six separately computed MVI-1 low modes. Never copies FDM eigenvectors into
MVI and never substitutes a full-grid eigensolver into MVI.
"""
from __future__ import annotations

import json
import traceback
import numpy as np
from cylinder_paths import activate
activate()
from modal_mvi_spectrum_audit import physical_case, make_case
from cylinder_modal_reference import solve_modal_case, _modal_spec
from cylinder_modal_mvi import _SeparatedModalProblem, _initial_factors, _mass_orthogonal_null
from cylinder_variational_mvi import _normalize_partner


def field_phase_diagnostics(modes, n=4):
    field=np.asarray(modes[6],float)
    ns=field.shape[-1]
    angle=2*np.pi*n*np.arange(ns)/ns
    cos=np.cos(angle);sin=np.sin(angle)
    results=[]
    for c in range(field.shape[0]):
        ac=field[c]@cos
        asi=field[c]@sin
        normcos=float(np.linalg.norm(ac))
        normsin=float(np.linalg.norm(asi))
        results.append({
            "component":c,
            "cos_norm":normcos,"sin_norm":normsin,
            "real_projection_axial_cos":ac.tolist(),
            "real_projection_axial_sin":asi.tolist()})
    return results


def trial(op,previous,pattern,n,seed=7):
    rank=1
    x,y=_initial_factors(op.n_components,rank,op.nx,op.ns,seed)
    phase=2*np.pi*n*np.arange(op.ns)/op.ns
    cos=np.cos(phase)/np.linalg.norm(np.cos(phase))
    sin=np.sin(phase)/np.linalg.norm(np.sin(phase))
    for c in range(op.n_components):
        y[c,0]=(sin if pattern=="all_sin" or
                (pattern=="v_sin" and c%3==1) or
                (pattern=="v_cos" and c%3!=1) else cos)
    x,y=_normalize_partner(x,y,"X")
    bm,be=op.projected_bases(x,y,"X")
    constraints=(np.stack([bm.T@op.previous_mass(q) for q in previous])
                 if previous else np.zeros((0,bm.shape[1])))
    z=_mass_orthogonal_null(constraints)
    if not z.shape[1]:
        raise ValueError("no X trial directions")
    rng=np.random.default_rng(54011+103*seed+31*rank)
    q=bm@(z@rng.normal(size=z.shape[1]))
    q/=np.sqrt(q@op.mass(q))
    x,y=op.fit(x,y,"X",q,None)
    before=float(np.sqrt(op.eigenvalue(q,None))/(2*np.pi))
    qx,px,_,_=op.halfstep(x,y,"X",previous,current_q=q)
    x,y=op.fit(x,y,"X",qx,px)
    after=float(np.sqrt(op.eigenvalue(qx,px))/(2*np.pi))
    x,y=_normalize_partner(x,y,"Y")
    qy,py,_,_=op.halfstep(x,y,"Y",previous,current_q=qx)
    after_y=float(np.sqrt(op.eigenvalue(qy,py))/(2*np.pi))
    return {"pattern":pattern,"n":n,"num_x_free":z.shape[1],
            "before_x_hz":before,"after_x_hz":after,
            "after_y_hz":after_y,
            "after_y_component_ranks":[int(np.linalg.matrix_rank(
                qy.reshape(op.shape)[c],tol=1e-10)) for c in range(op.n_components)]}


def main():
    report={"case":"KL C-S 9x9 SBP42", "pattern_probe":[]}
    base=physical_case()
    try:
        print("[PHASEPROBE] START FDM",flush=True)
        fcase=make_case(base,"FDM")
        full=solve_modal_case(fcase)
        report["fdm_frequencies_hz"]=[float(v) for v in full["frequencies_hz"]]
        report["fdm_n4_phase_components"]=field_phase_diagnostics(
            full["mechanical_modes"])
        print("[PHASEPROBE] FDM 7/8 "+
              json.dumps(report["fdm_frequencies_hz"][-2:]),flush=True)
        print("[PHASEPROBE] START MVI-1 first six",flush=True)
        case=make_case(base,"MVI",1)
        case["modal"]["n_modes"]=6
        mvi=solve_modal_case(case)
        report["mvi_first_six"]=[float(v) for v in mvi["frequencies_hz"]]
        print("[PHASEPROBE] SIX "+json.dumps(report["mvi_first_six"]),flush=True)
        prepared,density,_=_modal_spec(case)
        op=_SeparatedModalProblem(prepared,density,1)
        prev=[np.asarray(q,float).ravel() for q in mvi["mechanical_modes"]]
        for n in (3,4):
            for pat in ("all_cos","all_sin","v_sin","v_cos"):
                try:
                    p=trial(op,prev,pat,n)
                    report["pattern_probe"].append(p)
                    print("[PHASEPROBE] TRIAL "+json.dumps(p),flush=True)
                except Exception as exc:
                    p={"pattern":pat,"n":n,"error":repr(exc)}
                    report["pattern_probe"].append(p)
                    print("[PHASEPROBE] ERROR "+json.dumps(p),flush=True)
    except Exception as exc:
        report["harness_error"]=traceback.format_exc(limit=9)
        print("[PHASEPROBE] HARNESS "+str(exc),flush=True)
    finally:
        with open("modal_mvi_phase_pattern_probe_result.json","w") as f:
            json.dump(report,f,indent=2)
    return 0 if report.get("pattern_probe") and not report.get("harness_error") else 1


if __name__=="__main__":
    raise SystemExit(main())
