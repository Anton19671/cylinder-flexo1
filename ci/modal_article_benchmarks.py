#!/usr/bin/env python3
"""Published-frequency audit: Rawat 2016 Table 2; Cammalleri 2023 Tables A2/16.

Mode matching uses circumferential n, never nearest frequency alone.
Matching axial m=1 remains provisional until verified from mode fields.
"""
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from cylinder_paths import activate
activate()

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"config/verification/rawat2016/modal_linear"

def save(stem, obj):
    path=ROOT/("modal_article_"+stem+"_result.json")
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("[LITERATURE] "+stem+" "+json.dumps(obj,ensure_ascii=False),flush=True)

def match(frequencies, wavenumbers):
    if len(frequencies)!=len(wavenumbers):
        raise ValueError("frequency and harmonic arrays differ")
    groups={}
    for f,n in zip(frequencies,wavenumbers):
        f=float(f);n=int(n)
        if not np.isfinite(f) or f<=0: raise ValueError("invalid frequency")
        groups[n]=min(f,groups.get(n,float("inf")))
    return groups

def run_rawat():
    report={"article":"Rawat et al. (2016), Table 2, C-C, L/R=2, h/R=1/20",
        "doi":"10.16943/ptinsa/2016/48426",
        "geometry_m":{"L":2.0,"R":1.0,"h":0.05},
        "method":["FDM","MVI_1"],"theory":"CUF_TE1","grid":[13,13],
        "matched_by":"circumferential n, candidate m=1",
        "axial_index_verified":False,"mesh_convergence_verified":False,
        "status":"ERROR"}
    try:
        records=list(csv.DictReader((BASE/"RAWAT2016_TABLE2_CC_M1_REFERENCE.csv").open(encoding="utf-8",newline="")))
        ref={int(r["n"]):float(r["f_hz"]) for r in records
             if int(r["m"])==1 and int(r["n"]) in (2,5)}
        if set(ref)!={2,5}: raise ValueError("missing published n=2 or n=5")
        audit=json.loads((ROOT/"modal_mvi_rawat_te1_cc_audit_result.json").read_text(encoding="utf-8"))
        report["spectrum_audit_completed"]=audit.get("completed")
        report["comparisons"]={}
        issues=[]
        for method in ("FDM","MVI_1"):
            run=audit.get("runs",{}).get(method)
            if run is None:
                issues.append(method+":NOT_COMPUTED");continue
            harmonics=run["wavenumbers"]
            groups=match(run["frequencies_hz"],harmonics)
            rows=[]
            for n,published in sorted(ref.items()):
                f=groups.get(n)
                err=None if f is None else (f-published)/published*100
                rows.append({"m":1,"n":n,"published_hz":published,
                             "computed_hz":f,"difference_percent":err})
                if f is None: issues.append(method+":MISSING_N"+str(n))
                elif abs(err)>5:issues.append(method+":DIFFERENCE_GT_5_PERCENT_N"+str(n))
            report["comparisons"][method]={"frequencies_hz":run["frequencies_hz"],
                "wavenumbers":harmonics,"rows":rows}
        report["issues"]=issues
        report["status"]="PRELIMINARY_MATCH" if not issues and audit.get("completed") else "DISAGREEMENT_OR_INCOMPLETE"
    except Exception as exc: report["error"]=repr(exc)
    finally: save("rawat2016",report)
    return report["status"]=="PRELIMINARY_MATCH"

def run_cammalleri():
    from cylinder_modal_reference import solve_modal_case
    from modal_mvi_rawat_te1_cc_audit import harmonic_numbers
    published={5:(605.,581.),6:(555.,538.),7:(611.,600.)}
    report={"article":"Cammalleri & Castellano (2023), C-C, Tables A2 and 16",
        "doi":"10.3390/designs7060122",
        "published_approximation":"Donnell-Mushtari",
        "published_independent_FEM":"ANSYS SHELL181",
        "computed_theory":"KL","method":"FDM",
        "geometry_m":{"L":0.305,"R":0.076,"h":0.000254},
        "material":{"E_Pa":207e9,"nu":0.3,"rho_kg_m3":7833},
        "grid":[13,21],"operator":"SBP42_FOURIER_ODD_CSR",
        "axial_index_verified":False,"mesh_convergence_verified":False,
        "matched_by":"circumferential n, lowest retained branch candidate m=1",
        "status":"ERROR"}
    started=time.monotonic()
    try:
        source=BASE/"fdm_13x13/RAWAT2016_KL_13x13_MODAL_FDM.json"
        case=json.loads(source.read_text(encoding="utf-8"))
        case["geometry"]={"L":0.305,"R":0.076,"h":0.000254}
        case["material"]={"E0":207e9,"nu0":0.3}
        case["modal"]["density_kg_m3"]=7833
        case["modal"]["n_modes"]=12
        case["fdm_grid"]={"nx":13,"ns":21,"operator_scheme":"SBP42_FOURIER_ODD_CSR"}
        case["output_dir"]="OUTPUT_CAMMALLERI2023_CC_KL_FDM_13x21"
        print("[LITERATURE] START Cammalleri2023 C-C KL FDM 13x21",flush=True)
        result=solve_modal_case(case)
        freqs=np.asarray(result["frequencies_hz"],float)
        modes=np.asarray(result["mechanical_modes"],float)
        if len(freqs)!=12 or len(modes)!=12:
            raise ValueError("fewer than 12 computed modes")
        ns=result.get("circumferential_wavenumbers")
        if ns is None or len(ns)!=len(freqs):ns=harmonic_numbers(modes)
        groups=match(freqs,ns)
        report["frequencies_hz"]=freqs.tolist()
        report["wavenumbers"]=list(map(int,ns))
        report["global_spectrum_completeness_verified"]=result.get("global_spectrum_completeness_verified")
        issues=[];rows=[]
        for n,(theory,fem) in published.items():
            f=groups.get(n)
            fem_diff=None if f is None else 100*(f-fem)/fem
            rows.append({"m":1,"n":n,"published_model_hz":theory,
                "published_ANSYS_hz":fem,"computed_KL_FDM_hz":f,
                "model_diff_percent":None if f is None else 100*(f-theory)/theory,
                "ANSYS_diff_percent":fem_diff})
            if f is None:issues.append("MISSING_N"+str(n))
            elif abs(fem_diff)>10: issues.append("DIFFERENCE_GT_10_PERCENT_N"+str(n))
        report["rows"]=rows;report["issues"]=issues
        report["status"]="PRELIMINARY_MATCH" if not issues else "DISAGREEMENT_OR_INCOMPLETE"
    except Exception as exc: report["error"]=repr(exc)
    finally:
        report["elapsed_s"]=round(time.monotonic()-started,3)
        save("cammalleri2023",report)
    return report["status"]=="PRELIMINARY_MATCH"

if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ("rawat","cammalleri"):
        raise SystemExit("Usage: modal_article_benchmarks.py rawat|cammalleri")
    success=run_rawat() if sys.argv[1]=="rawat" else run_cammalleri()
    raise SystemExit(0 if success else 1)
