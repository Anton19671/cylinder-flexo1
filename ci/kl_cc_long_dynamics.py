#!/usr/bin/env python3
"""Execute physical two-stage Gauss IRK4 rank-one history or audit saved shell dynamics.

No PDE solve is performed in report mode. No fallback or fabricated fields.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"config/verification/kl_cc_dynamic_long"
OUT=CASES/"output"
CASES_NAMES=("FDM_COARSE","MVI1_COARSE","MVI4_COARSE","FDM_FINE","MVI1_FINE")
FIELDS=("mechanical","velocity","acceleration")

def run_gauss():
    from cylinder_gauss2_case import run_gauss2_mvi_case
    p=CASES/"MVI1_COARSE.json"
    result=run_gauss2_mvi_case(p,output_dir=OUT/"GAUSS2_MVI1_COARSE")
    if result.accepted_steps != 10 or result.summary.get("status")!="CONVERGED":
        raise SystemExit("IRK4_NOT_ALL_PHYSICAL_STEPS_ACCEPTED")
    print("GAUSS_IRK4_PHYSICAL_10_STEPS_ACCEPTED")

def read_newmark(name):
    folder=OUT/name
    p=folder/"FINAL_RUN_SUMMARY.json"
    if not p.is_file():
        return {"status":"MISSING_SUMMARY","states":{}}
    try:
        summary=json.loads(p.read_text(encoding="utf-8"))
        if summary.get("converged") is not True or summary.get("status")!="CONVERGED":
            return {"status":"NOT_CONVERGED","summary":summary,"states":{}}
        expected=10 if name.endswith("COARSE") else 20
        if summary.get("requested_steps")!=expected or summary.get("accepted_steps")!=expected:
            return {"status":"INCOMPLETE","summary":summary,"states":{}}
        dt=.002 if name.endswith("COARSE") else .001
        result={}
        for step in summary.get("steps",[]):
            idx=step.get("index")
            if type(idx) is not int or idx<1 or idx>expected or step.get("converged") is not True:
                return {"status":"UNACCEPTED_STEP","summary":summary,"states":{}}
            if step.get("state_file")!=f"REFERENCE_DYNAMIC_STEP_{idx:04d}.npz":
                return {"status":"STEP_FILENAME_MISMATCH","states":{}}
            with np.load(folder/step["state_file"],allow_pickle=False) as z:
                record={key:np.array(z[key],dtype=float) for key in FIELDS}
                time=float(z["time_s"])
            if any(record[key].shape!=(3,9,9) or not np.isfinite(record[key]).all()
                   for key in FIELDS) or abs(time-idx*dt)>1e-11:
                return {"status":"BAD_STATE_OR_TIME","states":{}}
            record["time_s"]=time
            result[idx]=record
        if len(result)!=expected:
            return {"status":"MISSING_STEP","states":{}}
        return {"status":"ACCEPTED","summary":summary,"states":result}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {"status":"INVALID:"+str(exc),"states":{}}

def summarize():
    runs={label:read_newmark(label) for label in CASES_NAMES}
    gaussfolder=OUT/"GAUSS2_MVI1_COARSE"
    gauss_status="MISSING_SUMMARY"
    gauss_data={}
    if (gaussfolder/"GAUSS2_MVI_HISTORY.json").is_file():
        try:
            g=json.loads((gaussfolder/"GAUSS2_MVI_HISTORY.json").read_text(encoding="utf-8"))
            if (g.get("status")=="CONVERGED" and g.get("n_steps_accepted")==10
                    and g.get("n_steps_requested")==10):
                from cylinder_newmark_gauss2_comparison import compare_newmark_gauss2_mvi
                # This checks physical CASE digests, exact Gauss collocation,
                # initial states and rank-preserving endpoint reconstruction.
                cmp=compare_newmark_gauss2_mvi(OUT/"MVI1_COARSE",gaussfolder)
                gauss_status="ACCEPTED"
                for row in g["steps"]:
                    idx=row["index"]
                    with np.load(gaussfolder/row["state_file"],allow_pickle=False) as z:
                        gauss_data[idx]={"mechanical":z["mechanical"].copy(),
                                         "velocity":z["velocity"].copy(),
                                         "time_s":float(z["time_s"])}
                (CASES/"NEWMARK_GAUSS_IRK4_DETAILED.json").write_text(
                    json.dumps(cmp,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            else:gauss_status="INCOMPLETE_OR_FAILED"
        except (OSError,KeyError,ValueError,TypeError) as exc:
            gauss_status="INVALID:"+str(exc)
    columns=["method","integrator","dt_s","time_s","w_midspan_mean_m",
             "w_max_abs_m","velocity_max_abs_m_per_s","acceleration_max_abs_m_per_s2",
             "delta_displacement_percent_vs_fdm_coarse","delta_displacement_percent_vs_fdm_fine",
             "delta_displacement_percent_vs_gauss","max_clamped_end_m"]
    rows=[]
    def rel(a,b):
        den=np.linalg.norm(b.ravel())
        return 100.*np.linalg.norm((a-b).ravel())/den if den>0 else None
    for label in CASES_NAMES:
        data=runs[label]
        if data["status"]!="ACCEPTED":continue
        dt=.002 if label.endswith("COARSE") else .001
        for idx,state in data["states"].items():
            t=state["time_s"]
            w=state["mechanical"][2]
            refc=runs["FDM_COARSE"]["states"].get(round(t/.002)) if runs["FDM_COARSE"]["status"]=="ACCEPTED" and abs(round(t/.002)*.002-t)<1e-10 else None
            reff=runs["FDM_FINE"]["states"].get(round(t/.001)) if runs["FDM_FINE"]["status"]=="ACCEPTED" else None
            gauss=gauss_data.get(round(t/.002)) if abs(round(t/.002)*.002-t)<1e-10 else None
            rows.append(dict(method=label,integrator="NEWMARK",dt_s=dt,time_s=t,
                             w_midspan_mean_m=float(np.mean(w[4])),
                             w_max_abs_m=float(np.max(abs(w))),
                             velocity_max_abs_m_per_s=float(np.max(abs(state["velocity"]))),
                             acceleration_max_abs_m_per_s2=float(np.max(abs(state["acceleration"]))),
                             delta_displacement_percent_vs_fdm_coarse="" if refc is None else rel(state["mechanical"],refc["mechanical"]),
                             delta_displacement_percent_vs_fdm_fine="" if reff is None else rel(state["mechanical"],reff["mechanical"]),
                             delta_displacement_percent_vs_gauss="" if gauss is None else rel(state["mechanical"],gauss["mechanical"]),
                             max_clamped_end_m=float(np.max(abs(state["mechanical"][:,[0,-1],:])))))
    for idx,s in gauss_data.items():
        rows.append(dict(method="MVI1_GAUSS2",integrator="GAUSS_LEGENDRE_IRK4",
                         dt_s=.002,time_s=s["time_s"],
                         w_midspan_mean_m=float(np.mean(s["mechanical"][2,4,:])),
                         w_max_abs_m=float(np.max(abs(s["mechanical"][2]))),
                         velocity_max_abs_m_per_s=float(np.max(abs(s["velocity"]))),
                         acceleration_max_abs_m_per_s2="",
                         delta_displacement_percent_vs_fdm_coarse="" if runs["FDM_COARSE"]["status"]!="ACCEPTED" else rel(s["mechanical"],runs["FDM_COARSE"]["states"][idx]["mechanical"]),
                         delta_displacement_percent_vs_fdm_fine="" if runs["FDM_FINE"]["status"]!="ACCEPTED" else rel(s["mechanical"],runs["FDM_FINE"]["states"][2*idx]["mechanical"]),
                         delta_displacement_percent_vs_gauss=0.,
                         max_clamped_end_m=float(np.max(abs(s["mechanical"][:,[0,-1],:])))))
    with (CASES/"LONG_DYNAMICS_COMPARISON.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(sorted(rows,key=lambda r:(r["time_s"],r["method"])))
    manifest=dict(physical_problem="KL_C_C_25HZ_100KPA_LINEAR_9X9",
                  final_time_s=.020,time_methods=["NEWMARK","GAUSS_LEGENDRE_IRK4"],
                  cases={k:v["status"] for k,v in runs.items()},
                  gauss_status=gauss_status,
                  is_complete=(all(v["status"]=="ACCEPTED" for v in runs.values()) and gauss_status=="ACCEPTED"),
                  externally_validated=False,spatial_convergence_verified=False,
                  formal_temporal_order_measured=False,rows=len(rows))
    (CASES/"LONG_DYNAMICS_COMPARISON.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)
    if not manifest["is_complete"]:
        raise SystemExit("PHYSICAL_DYNAMICS_INCOMPLETE_OR_FAILED")
    if any(row["max_clamped_end_m"]>1.e-9 for row in rows):
        raise SystemExit("CLAMPED_END_MOTION_IN_ACCEPTED_STATE")
    print("PHYSICAL_NEWMARK_IRK4_COMPARISON_ACCEPTED",len(rows),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("gauss","report"))
    args=ap.parse_args()
    if args.command=="gauss":run_gauss()
    else:summarize()
