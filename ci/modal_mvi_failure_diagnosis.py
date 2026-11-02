#!/usr/bin/env python3
"""Targeted forensic modal MVI: failed rank-2 trace and seed sensitivity.
Diagnostic only: no stiffness, mass, BC or solver source changes.
"""
from __future__ import annotations

import json
import traceback
import numpy as np
from scipy.linalg import qr

from cylinder_paths import activate
activate()
import cylinder_modal_mvi as mvi
from cylinder_modal_reference import solve_modal_case
from cylinder_boundary_spatial import separated_profile_directions
from cylinder_variational_mvi import reconstruct_profiles
from modal_mvi_spectrum_audit import physical_case,make_case

ORIGINAL_HALFSTEP = mvi._SeparatedModalProblem.halfstep
ORIGINAL_START = mvi._SeparatedModalProblem.admissible_start

def project_error(b,old):
    if b.shape[1] == 0:return 1.,0,[]
    frame, triangular, _ = qr(b,mode="economic",pivoting=True,check_finite=False)
    diag=np.abs(np.diag(triangular))
    dim=int(np.count_nonzero(diag>1e-12*float(np.max(diag,initial=0.))))
    error=float(np.linalg.norm(old-frame[:,:dim]@(frame[:,:dim].T@old))/
                np.linalg.norm(old))
    return error,dim,[float(diag[j]) for j in range(min(8,len(diag)))]

def main():
    report={"experiments":[],"solver_unchanged":True}
    orig=mvi._SeparatedModalProblem.halfstep
    def debug(self,x,y,axis,previous,*,current_q=None,on_stage=None):
        try:return orig(self,x,y,axis,previous,
                        current_q=current_q,on_stage=on_stage)
        except ValueError as exc:
            if current_q is None:raise
            old=np.asarray(current_q,float).ravel()
            raw=separated_profile_directions(
                x[:self.shape[0]],y[:self.shape[0]],axis)
            bm,_=self.projected_bases(x,y,axis)
            reconstructed=(reconstruct_profiles(x,y)*self.scales.reshape(
                self.n_components,self.nx,self.ns)).ravel()
            rawerr,rawrank,rawdiag=project_error(raw,old)
            berror,brank,bdiag=project_error(bm,old)
            residual=self.cm@old
            # Singular directions of C*T and the boundary residual.
            constr=self.cm@raw
            cr,rr,_=qr(constr.T,mode="economic",pivoting=True,
                       check_finite=False)
            cd=np.abs(np.diag(rr))
            data={
                "type":"FROZEN_PARTNER_DIAG","reason":repr(exc),
                "axis":axis,"rank":self.rank,"previous_modes":len(previous),
                "raw_dim":int(raw.shape[1]),"raw_rank_qr":rawrank,
                "projected_dim":int(bm.shape[1]),"projected_rank":brank,
                "raw_error":rawerr,"projected_error":berror,
                "current_vs_reconstructed":float(
                    np.linalg.norm(old-reconstructed[:len(old)])/
                    np.linalg.norm(old)),
                "trace_inf":float(np.max(np.abs(residual),initial=0.)),
                "C_times_T_max":float(np.max(np.abs(constr),initial=0.)),
                "C_times_T_qr_diag_max":float(np.max(cd,initial=0.)),
                "C_times_T_qr_diag_min_nonzero":float(np.min(
                    cd[cd>0.],initial=0.)),
                "raw_qr_first_diag":rawdiag,
                "basis_qr_first_diag":bdiag
            }
            report.setdefault("failure_diagnostics",[]).append(data)
            print("[FORENSIC] DIAG "+json.dumps(data),flush=True)
            raise
    mvi._SeparatedModalProblem.halfstep=debug

    for rank,offset in ((2,0),(4,0),(4,31),(4,101),(1,31),(1,101)):
        name=f"MVI_{rank}_SEEDOFFSET_{offset}"
        start=ORIGINAL_START
        if offset:
            def other_start(self,mode,previous,_offset=offset,**kwargs):
                # Forward the expanded MVI seed interface without changing
                # operator, rank, boundary conditions or Y/X halfsteps.
                return start(self,mode+_offset,previous,**kwargs)
            mvi._SeparatedModalProblem.admissible_start=other_start
        else:
            mvi._SeparatedModalProblem.admissible_start=ORIGINAL_START
        completed=[]
        def evt(**e):
            if e.get("stage")=="MVI_MODE_DONE":
                completed.append([int(e["mode"]),float(e["frequency_hz"])])
        print("[FORENSIC] BEGIN "+name,flush=True)
        try:
            out=solve_modal_case(make_case(physical_case(),"MVI",rank),
                                 detail_callback=evt)
            record={"name":name,"status":"SOLVED","frequencies_hz":[
                float(x) for x in out["frequencies_hz"]],
                "sweeps":out.get("mvi_sweeps"),
                "xy_dims":out.get("mvi_final_xy_trial_dimensions")}
        except Exception as exc:
            record={"name":name,"status":"FAILED","error":repr(exc),
                    "completed":completed,
                    "traceback":traceback.format_exc(limit=5)}
        report["experiments"].append(record)
        print("[FORENSIC] END "+name+" "+json.dumps({
            k:v for k,v in record.items() if k not in ("xy_dims","traceback")}),
            flush=True)
        with open("modal_mvi_failure_diagnosis_result.json","w",
                  encoding="utf-8") as f:
            json.dump(report,f,indent=2,ensure_ascii=False)
    print("[FORENSIC] SUMMARY "+json.dumps({
        "results":[{k:v for k,v in o.items()
                    if k in ("name","status","error","frequencies_hz")}
                   for o in report["experiments"]],
        "failure_diagnostics":report.get("failure_diagnostics",[])}),flush=True)
    # A reproduction of the existing bug is an expected diagnostic result.
    return 0

if __name__=="__main__":
    raise SystemExit(main())
