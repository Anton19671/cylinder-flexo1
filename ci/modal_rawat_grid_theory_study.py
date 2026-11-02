#!/usr/bin/env python3
"""Rawat 2016 C-C h/R=1/20, H/R=2: TE1/TE4 x 13/17 FDM convergence.

For each n compare its first computed Fourier-sector branch with Table 2.
A mode indexed '1' within a sector is NOT automatically validated as axial
wave number m=1. Preserve full fields for subsequent modal-shape checks.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from cylinder_paths import activate
activate()
from cylinder_modal_reference import solve_modal_case
from modal_mvi_rawat_te1_cc_audit import harmonic_numbers
from modal_literature_diagnostics import axial_harmonic_signature

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/verification/rawat2016/modal_linear"

def run(theory, grid):
    filename=f"RAWAT2016_{theory}_{grid}x{grid}_MODAL_FDM.json"
    if grid==21 and theory!="CUF_TE4":
        raise ValueError("Only CUF TE4 is scheduled on 21x21")
    source=(BASE/"mvi_21x21/RAWAT2016_CUF_TE4_21x21_MODAL_FDM_SBP42.json"
            if grid==21 else
            (BASE/"fdm_13x13"/filename if grid==13 else BASE/filename))
    case=json.loads(source.read_text(encoding="utf-8"))
    if case["method"] != "FDM": raise ValueError("Not FDM")
    original_scheme=case["fdm_grid"]["operator_scheme"]
    case["fdm_grid"]={"nx":grid,"ns":grid,
                       "operator_scheme":"SBP42_FOURIER_ODD_CSR"}
    case["output_dir"]=f"OUTPUT_RAWAT2016_{theory}_SBP42_{grid}x{grid}"
    case["modal"]["n_modes"]=10
    report={"source":"Rawat Matsagar Nagpal 2016 Table 2 C-C",
            "theory":theory,"grid":[grid,grid],"case":str(source.relative_to(ROOT)),
            "operator_scheme":"SBP42_FOURIER_ODD_CSR",
            "original_case_scheme":original_scheme,
            "axial_m1_verified":False,"status":"ERROR"}
    prefix=f"modal_rawat_{theory}_{grid}x{grid}"
    started=time.monotonic()
    try:
        print(f"[RAWAT-GRID] START {theory} {grid}x{grid}",flush=True)
        result=solve_modal_case(case)
        freq=np.asarray(result["frequencies_hz"],float)
        modes=np.asarray(result["mechanical_modes"],float)
        if len(freq)!=10 or len(modes)!=10 or not np.isfinite(freq).all() or not np.isfinite(modes).all():
            raise ValueError("Invalid or incomplete ten-mode spectrum")
        harmonics=result.get("circumferential_wavenumbers")
        if harmonics is None: harmonics=harmonic_numbers(modes)
        with (BASE/"RAWAT2016_TABLE2_CC_M1_REFERENCE.csv").open(encoding="utf-8",newline="") as fh:
            reference={int(r["n"]):float(r["f_hz"]) for r in csv.DictReader(fh) if int(r["m"])==1}
        scan=result.get("harmonic_family_scan") or []
        rows=[]
        for n in (2,3,4,5):
            sector=[x for x in scan if int(x["n"])==n and int(x.get("axial_mode_index",0))==1]
            saved=[float(f) for f,k in zip(freq,harmonics) if int(k)==n]
            value=(float(sector[0]["frequency_hz"]) if sector else (min(saved) if saved else None))
            index=next((i for i,k in enumerate(harmonics) if int(k)==n),None)
            radial_component=2*(int(theory[-1])+1)
            signature=(None if index is None else axial_harmonic_signature(
                modes[index,radial_component],int(n)))
            rows.append({"circumferential_n":n,"published_m1_hz":reference[n],
                         "first_sector_branch_hz":value,
                         "difference_percent":None if value is None else 100*(value-reference[n])/reference[n],
                         "source":"full_sector_scan" if sector else ("saved_modes" if saved else "missing"),
                         "radial_axial_signature":signature,
                         "axial_m1_verified":False})
        np.savez_compressed(ROOT/(prefix+"_fields.npz"),mechanical_modes=modes,
                            frequencies_hz=freq,circumferential_wavenumbers=np.asarray(harmonics,int))
        report.update(status="COMPUTED_PENDING_AXIAL_AND_MESH_VALIDATION",
            rows=rows, frequencies_hz=freq.tolist(), wavenumbers=list(map(int,harmonics)),
            complete=result.get("global_spectrum_completeness_verified"))
    except Exception as exc:
        report["error"]=repr(exc)
        raise
    finally:
        report["elapsed_s"]=round(time.monotonic()-started,3)
        (ROOT/(prefix+"_report.json")).write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print("[RAWAT-GRID] "+json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("theory",choices=["CUF_TE1","CUF_TE4"])
    ap.add_argument("grid",type=int,choices=[13,17,21])
    args=ap.parse_args()
    run(args.theory,args.grid)
