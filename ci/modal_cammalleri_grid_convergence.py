#!/usr/bin/env python3
"""Cammalleri & Castellano 2023, C-C KL: fixed-SBP42 mesh study.

A computation/status check, NOT independent validation of axial m=1:
the same variational KL operator is used at each mesh. Save all mode fields
and the radial axial-node diagnostic, so modal branches can be inspected.
The strict Fourier K/M eigenpair certificate remains active in the solver.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

from cylinder_paths import activate
activate()
from cylinder_modal_reference import solve_modal_case
from modal_literature_diagnostics import axial_harmonic_signature

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/verification/rawat2016/modal_linear"
SOURCE = BASE / "fdm_13x13/RAWAT2016_KL_13x13_MODAL_FDM.json"
OUT = ROOT / "modal_cammalleri2023_grid_convergence.json"
PUBLISHED = {5: {"donnell_hz":605., "ansys_hz":581.},
             6: {"donnell_hz":555., "ansys_hz":538.},
             7: {"donnell_hz":611., "ansys_hz":600.}}
GRIDS = ((13,21), (17,31), (21,41))

def _save(report):
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print("[CAMMALLERI-GRID] "+json.dumps(report,ensure_ascii=False),flush=True)

def main():
    base=json.loads(SOURCE.read_text(encoding="utf-8"))
    report={"source":"Cammalleri & Castellano 2023, Designs 7, 122; Tables A2 and 16",
            "doi":"10.3390/designs7060122",
            "method":"FDM","theory":"KL","boundary":"C-C",
            "model":"LINEAR",
            "operator_scheme":"SBP42_FOURIER_ODD_CSR",
            "published":PUBLISHED,"grids":[],
            "same_discrete_operator_family":True,
            "axial_m1_independently_verified":False,
            "independent_literature_validation":False,
            "mesh_convergence_verified":False}
    for nx,ns in GRIDS:
        case=json.loads(json.dumps(base))
        case["geometry"]={"L":.305,"R":.076,"h":.000254}
        case["material"]={"E0":207e9,"nu0":.3}
        case["modal"]["density_kg_m3"]=7833
        case["modal"]["n_modes"]=12
        case["fdm_grid"]={"nx":nx,"ns":ns,
                           "operator_scheme":"SBP42_FOURIER_ODD_CSR"}
        case["output_dir"]=f"OUTPUT_CAMMALLERI_KL_SBP42_{nx}x{ns}"
        data={"grid":[nx,ns],"status":"ERROR"}
        started=time.monotonic()
        try:
            print(f"[CAMMALLERI-GRID] START {nx}x{ns}",flush=True)
            solved=solve_modal_case(case)
            freqs=np.asarray(solved["frequencies_hz"],float)
            fields=np.asarray(solved["mechanical_modes"],float)
            harmonics=solved.get("circumferential_wavenumbers")
            if harmonics is None:
                raise ValueError("CAMMALLERI_MISSING_HARMONIC_LABELS")
            harmonics=np.asarray(harmonics,dtype=int)
            if (freqs.shape!=(12,) or
                    fields.shape!=(12,3,nx,ns) or
                    harmonics.shape!=(12,) or
                    not np.isfinite(freqs).all() or
                    np.any(freqs<=0) or
                    not np.isfinite(fields).all() or
                    np.any(np.diff(freqs)<-1.e-8)):
                raise ValueError("CAMMALLERI_INVALID_TWELVE_EIGENMODES")
            if not solved.get("global_spectrum_completeness_verified",False):
                raise ValueError("CAMMALLERI_UNVERIFIED_GLOBAL_SPECTRUM")
            rows=[]
            for n,published in PUBLISHED.items():
                hits=np.flatnonzero(harmonics==n)
                if not len(hits):
                    raise ValueError(f"CAMMALLERI_MISSING_RETAINED_HARMONIC_{n}")
                index=int(hits[0])
                f=float(freqs[index])
                shape=axial_harmonic_signature(fields[index,2],int(n))
                rows.append({
                    "n":n,"mode_index_in_first_twelve":index+1,
                    "computed_frequency_hz":f,
                    "donnell_reference_hz":published["donnell_hz"],
                    "ansys_reference_hz":published["ansys_hz"],
                    "donnell_difference_percent":100.*(f/published["donnell_hz"]-1.),
                    "ansys_difference_percent":100.*(f/published["ansys_hz"]-1.),
                    "radial_axial_diagnostic":shape,
                    "axial_m1_verified":False,
                })
            name=f"modal_cammalleri2023_KL_{nx}x{ns}_fields.npz"
            np.savez_compressed(ROOT/name,frequencies_hz=freqs,
                                circumferential_wavenumbers=harmonics,
                                mechanical_modes=fields)
            data.update({
                "status":"COMPUTED_AXIAL_M_UNVERIFIED",
                "frequencies_hz":freqs.tolist(),
                "circumferential_wavenumbers":harmonics.tolist(),
                "global_spectrum_completeness_verified":True,
                "rows":rows,
                "modes_artifact":name,
            })
        except Exception as exc:
            data["error"]=repr(exc)
        finally:
            data["elapsed_s"]=round(time.monotonic()-started,3)
            report["grids"].append(data)
            _save(report)
    if all(g["status"]=="COMPUTED_AXIAL_M_UNVERIFIED" for g in report["grids"]):
        report["frequency_changes_percent"]={
            str(n):[
                100.*(report["grids"][i]["rows"][j]["computed_frequency_hz"]/
                       report["grids"][i-1]["rows"][j]["computed_frequency_hz"]-1.)
                for i in (1,2)]
            for j,n in enumerate(PUBLISHED)}
        report["diagnostic_completed"]=True
    else:
        report["diagnostic_completed"]=False
    _save(report)
    return 0 if report["diagnostic_completed"] else 1

if __name__=="__main__":
    sys.exit(main())
