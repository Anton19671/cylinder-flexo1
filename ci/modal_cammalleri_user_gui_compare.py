#!/usr/bin/env python3
"""Fail-closed compare of actual GUI FDM and MVI results, including full mass-MAC."""
import json,sys
from pathlib import Path
import numpy as np
from cylinder_paths import activate
activate()
from cylinder_modal_reference import _modal_spec
from modal_mvi_case_comparison import compare_modal_results
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"modal-cammalleri-10-report"
ART=ROOT/"modal-cammalleri-10-input"
NAMES=["FDM","MVI_1","MVI_1_AUTO"]+[f"MVI_{i}" for i in range(2,9)]
def main():
    summary={"cases":{},"comparisons":{},"issues":[]}
    states={}
    for n in NAMES:
        try:
            report=json.loads((ART/n/"case_check.json").read_text(encoding="utf-8"))
            summary["cases"][n]=report
            if report["status"]!="COMPLETED_12_MODES":
                summary["issues"].append(n+":RUN_NOT_COMPLETED")
                continue
            with np.load(ART/n/"computed_modes.npz",allow_pickle=False) as d:
                states[n]=(np.asarray(d["frequencies_hz"],float),
                           np.asarray(d["mechanical_modes"],float))
        except Exception as exc:
            summary["cases"][n]={"status":"MISSING","error":repr(exc)}
            summary["issues"].append(n+":MISSING_RUN_ARTIFACT")
    if "FDM" in states:
        try:
            p=ROOT/"config/verification/cammalleri2023/user_13x21/CAMMALLERI2023_CUF_TE1_13x21_MKR_FDM.json"
            model=json.loads(p.read_text(encoding="utf-8"))
            prepared,rho,_=_modal_spec(model)
            for name in NAMES[1:]:
                if name not in states:continue
                result=compare_modal_results(states["FDM"][0],states[name][0],
                    states["FDM"][1],states[name][1],prepared.problem.functional,rho)
                summary["comparisons"][name]=result
                print("[USER-CAMMALLERI] COMPARISON "+name+" "+json.dumps(result),flush=True)
                if result["status"]!="PASS":summary["issues"].append(name+":FREQUENCY_OR_MODE_MISMATCH")
        except Exception as exc:summary["issues"].append("COMPARISON_FAILED:"+repr(exc))
    else:summary["issues"].append("FDM_NOT_AVAILABLE")
    if len(states)!=10:summary["issues"].append("INCOMPLETE_TEN_CASE_MATRIX")
    summary["status"]="PASS" if not summary["issues"] else "FAIL"
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"comparison.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[USER-CAMMALLERI] SUMMARY "+json.dumps({
        "status":summary["status"],"issues":summary["issues"],
        "wall_seconds":{k:v.get("wall_seconds") for k,v in summary["cases"].items()}
    },ensure_ascii=False),flush=True)
    return int(bool(summary["issues"]))
if __name__=="__main__":sys.exit(main())
