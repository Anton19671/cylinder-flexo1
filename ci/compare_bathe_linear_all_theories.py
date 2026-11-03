#!/usr/bin/env python3
"""Compare REAL independently computed Bathe FDM and MVI1 across all theories.

No physical shell solve, no nodal fallback to FDM, no rank truncation.
Each theory has one full Bathe step, with two stage checkpoints.
The artifacts must originate from the separate method jobs of
GitHub Actions run SOURCE_RUN_ID (default: archived run 41645787449);
absent/failed results are NOT filled.
"""
from __future__ import annotations
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
PHYSICAL_KEYS=("theory","scale_model","mode","geometry_model","material_model",
 "material","geometry","quadrature","boundary_conditions",
 "load_definition","dynamics","calculation_kind","units")
BASE=Path(__file__).resolve().parents[1]/"config/verification/linear_bathe_all_theories"
GAMMA=2-math.sqrt(2)
SOURCE_RUN=int(os.environ.get("SOURCE_RUN_ID","41645787449"))


def nmechanical(theory):
    if theory=="KL":return 3
    if theory in ("FSDT","REDDY"):return 5
    return 3*(int(theory[-1])+1)


def read_case(theory,method):
    name=theory+"_"+method+("1" if method=="MVI" else "")
    case_path=BASE/(name+".json")
    output=BASE/"output"/name
    assert case_path.is_file(),f"CASE_ABSENT: {name}"
    assert (output/"FINAL_RUN_SUMMARY.json").is_file(),f"PHYSICAL_OUTPUT_ABSENT: {name}"
    case=json.loads(case_path.read_text(encoding="utf-8"))
    summary=json.loads((output/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    theory_internal=theory if not theory.startswith("CUF_") else "CUF"
    assert case["theory"]==theory_internal,name
    if theory.startswith("CUF_"):assert case["cuf_model"]==theory.split("_")[1],name
    expected_theory=theory
    assert summary["theory"]==expected_theory,name
    assert summary["method"]==method and case["method"]==method,name
    assert summary["integrator"]=="BATHE" and summary["integrator_order"]==2,name
    assert summary["converged"] is True and summary["status"]=="CONVERGED",name
    assert summary["accepted_steps"]==summary["requested_steps"]==1,name
    assert len(summary["steps"])==1,name
    if method=="MVI":
        assert case["rank"]==summary["rank"]==1,name
        grid=case["mvi_grid"]
        assert grid["operator_scheme"]=="SBP42_FOURIER_ODD",name
    else:
        assert "rank" not in case and summary["rank"] is None,name
        assert case["fdm_backend"]=="MATRIX_FREE_GMRES",name
        grid=case["fdm_grid"]
        assert grid["operator_scheme"]=="SBP42_FOURIER_ODD_CSR",name
    assert grid["nx"]==grid["ns"]==9,name
    assert case["geometry_model"]=="LINEAR" and case["mode"]=="LINEAR_ELASTIC",name
    assert case["material_model"]=="HOOKE",name
    stages={}
    for stage,filename,when in (
        ("GAMMA","BATHE_SUBSTEP_0001_GAMMA.npz",GAMMA*.002),
        ("END","REFERENCE_DYNAMIC_STEP_0001.npz",.002)
    ):
        path=output/filename
        assert path.is_file(),f"MISSING_PHYSICAL_BATHE_SUBSTEP:{name}:{stage}"
        with np.load(path,allow_pickle=False) as saved:
            fields={k:np.array(saved[k],float) for k in
                ("mechanical","velocity","acceleration")}
            t=float(saved["time_s"])
            factor=float(saved["load_factor"])
            if method=="MVI":
                x=np.array(saved["x_profiles"],float)
                y=np.array(saved["y_profiles"],float)
                scales=np.array(saved["state_scales"],float)
                assert x.shape==(nmechanical(theory),1,9),name
                assert y.shape==(nmechanical(theory),1,9),name
                reconstructed=np.einsum("crx,crs->cxs",x,y)*scales
                assert np.max(np.abs(reconstructed-fields["mechanical"]))<1e-11,name
        assert all(a.shape==(nmechanical(theory),9,9) and np.isfinite(a).all()
                   for a in fields.values()),name
        assert abs(t-when)<=1e-12,f"WRONG_PHYSICAL_TIME:{name}:{stage}"
        assert abs(factor-math.sin(2*math.pi*25*t))<=1e-12,name
        stages[stage]=(t,fields)
    assert summary["steps"][0]["status"] in (
        "CONVERGED","CONVERGED_FULL","CONVERGED_PROJECTED")
    assert summary["steps"][0]["first_substep_status"] in (
        "CONVERGED","CONVERGED_FULL","CONVERGED_PROJECTED")
    return case,summary,stages


def main():
    rows=[]
    for theory in THEORIES:
        cases={method:read_case(theory,method) for method in ("FDM","MVI")}
        for key in PHYSICAL_KEYS:
            if key=="theory":
                # Original CASE CUF model has the same "CUF" prefix and
                # extra cuf_model checked in read_case above.
                pass
            if cases["FDM"][0][key]!=cases["MVI"][0][key]:
                raise ValueError(f"PHYSICAL_MVI_FDM_PASSPORT_DIFFERENCE:{theory}:{key}")
        for stage in ("GAMMA","END"):
            tm,actual=cases["MVI"][2][stage]
            tf,reference=cases["FDM"][2][stage]
            assert abs(tm-tf)<1e-12
            row=dict(theory=theory,method="MVI1_vs_FDM",stage=stage,
                     time_s=tm,components=nmechanical(theory))
            for field in ("mechanical","velocity","acceleration"):
                delta=float(np.linalg.norm(actual[field]-reference[field]))
                ref=float(np.linalg.norm(reference[field]))
                row["difference_"+field+"_percent"]=(
                    100*delta/ref if ref else (0. if delta==0 else math.inf))
            radial=2*(int(theory[-1])+1) if theory.startswith("CUF_") else 2
            row["mvi_midspan_radial_w0_m"]=float(np.mean(actual["mechanical"][radial,4]))
            row["fdm_midspan_radial_w0_m"]=float(np.mean(reference["mechanical"][radial,4]))
            rows.append(row)
            print("BATHE_LINEAR_ACTUAL_FIELD_DIFF",row,flush=True)
    dest=BASE/"BATHE_LINEAR_SEVEN_THEORIES_MVI1_VS_FDM.csv"
    with dest.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (BASE/"BATHE_LINEAR_SEVEN_THEORIES_MVI1_VS_FDM.json").write_text(
        json.dumps({"source_physical_run":SOURCE_RUN,"n_theories":7,
            "physical_substep_comparisons":len(rows),
            "status":"EVALUATED_WITHOUT_FALLBACK",
            "time_convergence_for_all_theories_verified":False,
            "rank_convergence_for_all_theories_verified":False,
            "data":rows},indent=2)+"\n",encoding="utf-8")
    # A strict regression gate: even one failed theory is reported and
    # never silently reclassified as a successful physical verification.
    problems=[(r["theory"],r["stage"],r["difference_mechanical_percent"])
        for r in rows if r["difference_mechanical_percent"]>1e-3 or
        r["difference_velocity_percent"]>1e-2 or
        r["difference_acceleration_percent"]>1e-2]
    if problems:
        raise ValueError("BATHE_LINEAR_SEVEN_THEORIES_MVI_FDM_MISMATCH: "+str(problems))
    print("BATHE_LINEAR_ALL_SEVEN_THEORIES_FIRST_TWO_SUBSTEPS_AGREE",flush=True)

if __name__=="__main__":
    main()
