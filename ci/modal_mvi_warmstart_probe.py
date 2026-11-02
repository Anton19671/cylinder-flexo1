#!/usr/bin/env python3
"""Forensic test of an X-conditioned start inside genuine modal MVI.

No physical matrices, eigenmodes or FDM fields are substituted into the MVI.
The ordinary Y/X modal solver is called on an unchanged 9x9 KL C-S case.
Only initialization for a pure harmonic partner is experimentally replaced
with one EXACT frozen-partner X Ritz halfstep; normal Y/X sweeps follow.
This script MUST NOT be interpreted as proof of global spectrum completeness.
"""
from __future__ import annotations
import json
import sys
import time
import traceback
import numpy as np

from cylinder_paths import activate
activate()
from cylinder_modal_reference import solve_modal_case
import cylinder_modal_mvi as modal_mvi
from modal_mvi_spectrum_audit import physical_case, make_case

REFERENCE=(275.2660957256125,275.26609572889816,
           293.138248794931,293.1382487961865,
           347.4950111733973,347.49501117889577,
           398.6031918744296,398.60319187688526)

def run(rank):
    original = modal_mvi._SeparatedModalProblem.admissible_start
    probe={"rank":rank,"preconditioning":[],"status":"UNKNOWN"}
    def warm_start(self, mode, previous, *, harmonic_seed=None,
                   harmonic_phase="cos"):
        x,y,q,p=original(
            self,mode,previous,harmonic_seed=harmonic_seed,
            harmonic_phase=harmonic_phase)
        if harmonic_seed is None:
            return x,y,q,p
        try:
            # The EXACT current Y-frozen, BC-constrained X eigenproblem,
            # with the same already completed modes mass-deflated.
            qm,pm,ritz,_=self.halfstep(
                x,y,"X",previous,current_q=q)
            mass=float(np.sqrt(qm@self.mass(qm)))
            if not np.isfinite(mass) or mass<=0.:
                raise ValueError("WARMSTART_INVALID_MODAL_MASS")
            qm=qm/mass
            pm=None if pm is None else pm/mass
            x,y=self.fit(x,y,"X",qm,pm)
            if abs(float(qm@self.mass(qm))-1.)>1.e-8:
                raise ValueError("WARMSTART_MASS_NORMALIZATION")
            probe["preconditioning"].append({
                "harmonic":int(harmonic_seed),
                "phase":harmonic_phase,"prior":len(previous),
                "frequency_hz":float(
                    np.sqrt(self.eigenvalue(qm,pm))/(2.*np.pi))})
            return x,y,qm,pm
        except (ValueError,np.linalg.LinAlgError) as exc:
            probe["preconditioning"].append({
                "harmonic":int(harmonic_seed),
                "phase":harmonic_phase,"prior":len(previous),
                "error":str(exc)[:160]})
            raise
    modal_mvi._SeparatedModalProblem.admissible_start=warm_start
    case=make_case(physical_case(),"MVI",rank)
    events=[]
    def event(**kw):
        if kw.get("stage") in (
            "MVI_CANDIDATE_SELECTED","MVI_CANDIDATE_REJECTED"):
            if int(kw.get("mode",0))>=7:
                events.append({
                    k:kw.get(k) for k in (
                        "stage","mode","attempt","seed","reason",
                        "frequency_hz","converged","attempts")})
        if kw.get("stage")=="MVI_MODE_DONE":
            print("[WARMSTART] MODE "+json.dumps({
                "rank":rank,"mode":kw["mode"],
                "frequency_hz":kw["frequency_hz"]}),flush=True)
    started=time.monotonic()
    try:
        result=solve_modal_case(case,detail_callback=event)
        freqs=np.asarray(result["frequencies_hz"],dtype=float).tolist()
        max_error=max(abs(a-b)/b for a,b in zip(freqs,REFERENCE))
        probe.update({
            "status":"SOLVED",
            "frequencies_hz":freqs,
            "relative_max_index_error":float(max_error),
            "last_mode_audit":(result.get("mvi_multistart_audit") or [])[-2:],
            "xy_dimensions":result.get("mvi_final_xy_trial_dimensions"),
            "frequency_matches_full_FDM_8":bool(max_error<1.e-5)})
    except Exception as exc:
        probe.update({"status":"FAILED","error":repr(exc),
                      "traceback":traceback.format_exc(limit=9)})
    finally:
        modal_mvi._SeparatedModalProblem.admissible_start=original
    probe["events"]=events
    probe["elapsed_s"]=round(time.monotonic()-started,2)
    print("[WARMSTART] RESULT "+json.dumps(probe,ensure_ascii=False),flush=True)
    return probe

def main():
    import os
    ranks=[int(x) for x in os.environ.get("MODAL_PROBE_RANKS","1,2").split(",")]
    output={"description":"X-frozen variational warm start only; not production solver",
            "runs":{}}
    for rank in ranks:
        output["runs"][f"MVI_{rank}"]=run(rank)
        with open("modal_mvi_warmstart_probe_result.json","w",encoding="utf-8") as f:
            json.dump(output,f,indent=2,ensure_ascii=False)
    return 0 if all(v.get("status")=="SOLVED"
                    for v in output["runs"].values()) else 1

if __name__=="__main__":
    raise SystemExit(main())
