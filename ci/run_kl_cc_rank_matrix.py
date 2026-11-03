"""Independent real KL C-C dynamic MVI 1..8 / FDM control matrix runner.

A distinct physical solver invocation per rank with untouched strict acceptance.
This creates new JSON CASEs from exactly one verified physical baseline.
"""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"config/verification/kl_cc_dynamic_long"
OUT=ROOT/"config/verification/kl_cc_mvi_rank_matrix"


def make_case(rank,integrator):
    if rank not in range(0,9) or integrator not in ("newmark","gauss"):
        raise ValueError("UNSUPPORTED_RANK_OR_INTEGRATOR")
    original=BASE/("FDM_COARSE.json" if rank==0 else "MVI1_COARSE.json")
    case=json.loads(original.read_text(encoding="utf-8"))
    if integrator=="gauss" and rank==0:
        raise ValueError("GAUSS_FDM_NOT_RUN_IN_RANK_FIXED_MVI_MATRIX")
    if rank:
        case["rank"]=rank
        case["method"]="MVI"
        case["reference_mvi"]={"max_sweeps":12}
    else:
        case["method"]="FDM"
        case.pop("rank",None)
        case.pop("reference_mvi",None)
    case["dynamics"]["time_step_s"]=0.002
    case["dynamics"]["n_steps"]=1 if integrator=="gauss" else 5
    name=("GAUSS2_MVI"+str(rank)) if integrator=="gauss" else ("NEWMARK_FDM" if rank==0 else "NEWMARK_MVI"+str(rank))
    case["output_dir"]="output/"+name
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/(name+".json")
    if path.exists():
        raise ValueError("REFUSE_OVERWRITE_CASE")
    path.write_text(json.dumps(case,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return path,name


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--integrator",choices=("newmark","gauss"),required=True)
    ap.add_argument("--rank",type=int,required=True)
    args=ap.parse_args()
    path,name=make_case(args.rank,args.integrator)
    print("PHYSICAL_CASE",path,"METHOD",name,flush=True)
    if args.integrator=="newmark":
        from cylinder_reference_case import run_reference_case
        # Public dispatcher normally supplied by RUN_FINAL_CASE; use subprocess
        # to keep execution exactly identical to established CASE semantics.
        import subprocess,sys
        proc=subprocess.run([sys.executable,str(ROOT/"RUN_FINAL_CASE.py"),str(path)],
                            check=False)
        if proc.returncode:
            raise SystemExit(proc.returncode)
        folder=OUT/"output"/name
        summary=json.loads((folder/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
        n=5
        if (not summary.get("converged") or summary.get("status")!="CONVERGED"
                or summary.get("accepted_steps")!=n or len(summary.get("steps",[]))!=n):
            raise SystemExit("PHYSICAL_NEWMARK_INCOMPLETE")
        if args.rank:
            if summary.get("method")!="MVI" or summary.get("rank")!=args.rank:
                raise SystemExit("MVI_RANK_MISMATCH")
            if any(step.get("status") not in ("CONVERGED_FULL","CONVERGED_PROJECTED")
                   or not step.get("converged") for step in summary["steps"]):
                raise SystemExit("MVI_STEP_NOT_ACCEPTED")
        print("FIVE_REAL_NEWMARK_STEPS_ACCEPTED",name,flush=True)
    else:
        from cylinder_gauss2_case import run_gauss2_mvi_case
        result=run_gauss2_mvi_case(path,output_dir=OUT/"output"/name)
        if result.summary.get("status")!="CONVERGED" or result.accepted_steps!=1:
            raise SystemExit("PHYSICAL_GAUSS_STEP_NOT_ACCEPTED")
        print("ONE_REAL_IRK4_STEP_ACCEPTED",name,flush=True)


if __name__=="__main__":
    main()
