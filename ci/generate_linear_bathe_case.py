#!/usr/bin/env python3
"""Generate reproducible LINEAR Bathe physical CASEs for every shell theory.

Only *verification data* are constrained to HOOKE+LINEAR.
The opt-in Bathe solver and its nonlinear/electric pathways are untouched.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"config/verification/kl_cc_bathe/BATHE_MVI1.json"
THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")

def make_case(theory,method,rank,steps,dt_s,output_folder):
    if theory not in THEORIES or method not in ("FDM","MVI"):
        raise ValueError("BATHE_LINEAR_UNSUPPORTED_THEORY_OR_METHOD")
    if method=="MVI" and (type(rank) is not int or not 1<=rank<=8):
        raise ValueError("BATHE_MVI_RANK_OUT_OF_RANGE")
    if method=="FDM" and rank is not None:
        raise ValueError("BATHE_FDM_CANNOT_HAVE_MVI_RANK")
    if type(steps) is not int or steps<1 or not (0<dt_s<=.005):
        raise ValueError("BATHE_CONTROL_INVALID_TIME_GRID")
    case=json.loads(BASE.read_text(encoding="utf-8"))
    case["theory"]=theory if not theory.startswith("CUF_") else "CUF"
    if theory.startswith("CUF_"):
        case["cuf_model"]=theory.split("_")[1]
    else:
        case.pop("cuf_model",None)
    case["quadrature"]["n_gauss"]=(2*int(theory[-1])+1
        if theory.startswith("CUF_TE") else 7 if theory=="REDDY" else 3)
    case["dynamics"]["time_step_s"]=dt_s
    case["dynamics"]["n_steps"]=steps
    case["dynamics"]["integrator"]="BATHE"
    case["method"]=method
    if method=="MVI":
        case["rank"]=rank
        case["mvi_grid"]=case.get("mvi_grid") or {"nx":9,"ns":9,
            "operator_scheme":"SBP42_FOURIER_ODD"}
        case.pop("fdm_grid",None)
        case["fdm_backend"]="DENSE_REFERENCE"
        case["reference_newton"]["max_free_dofs"]=256
    else:
        case.pop("rank",None)
        case.pop("reference_mvi",None)
        case.pop("mvi_grid",None)
        case["fdm_grid"]={"nx":9,"ns":9,"operator_scheme":"SBP42_FOURIER_ODD_CSR"}
        # In TE4: 15*81=1215 physical unknowns, so the original dense
        # FDM cap (512) cannot be used. Matrix-free is a REAL full-nodal FDM
        # with its own Newton-Krylov solve, not an MVI fallback.
        case["fdm_backend"]="MATRIX_FREE_GMRES"
        case["reference_newton"]={
           "relative_tolerance":1.e-7,
           "absolute_tolerance_scaled":1.e-12,
           "tangent_step_scaled":2.e-6,
           "max_iterations":12,
           "max_total_dofs":10000,
           "max_surface_nodes":256,
           "max_constraint_rows":3000,
           # The original 0.015 Krylov stop changed the saved physical
           # fields at the precision used by the cross-method audit.
           # Dedicated FSDT/Reddy/CUF_TE2 runs established that 1e-4
           # resolves the dominant field differences on this fixed grid.
           "krylov_rtol":1.e-4,
           "krylov_restart":50,
           "krylov_max_matvecs":300,
           "krylov_preconditioner":"NONE"
        }
    case["output_dir"]="output/"+output_folder
    if (case["geometry_model"]!="LINEAR" or
        case["mode"]!="LINEAR_ELASTIC" or case["material_model"]!="HOOKE"):
        raise ValueError("BATHE_THIS_MATRIX_IS_LINEAR_ONLY")
    return case

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--theory",required=True,choices=THEORIES)
    parser.add_argument("--method",required=True,choices=("FDM","MVI"))
    parser.add_argument("--rank",type=int)
    parser.add_argument("--steps",type=int,default=1)
    parser.add_argument("--dt",type=float,default=.002)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    name=args.theory+"_"+args.method+(
        "" if args.method=="FDM" else str(args.rank))
    base=Path(args.output)
    base.mkdir(parents=True,exist_ok=True)
    case=make_case(args.theory,args.method,args.rank,args.steps,args.dt,name)
    path=base/(name+".json")
    if path.exists():raise ValueError("BATHE_GENERATOR_REFUSES_OVERWRITE")
    path.write_text(json.dumps(case,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8")
    print("BATHE_LINEAR_CASE",path,flush=True)

if __name__=="__main__":
    main()
