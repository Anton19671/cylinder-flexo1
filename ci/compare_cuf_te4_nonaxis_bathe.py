#!/usr/bin/env python3
"""Independent fieldwise comparison of one *real* CUF TE4 Bathe patch step.

All outputs come from the three solver-produced NPZ arrays, without
interpolation, synthetic fields, or changes to the weak form. An absent or
unaccepted solver stage is a hard error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_archive(directory: Path, method: str, rank: int | None):
    base=directory.resolve()
    summary=json.loads((base/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    case=json.loads((base/"REFERENCE_CASE.json").read_text(encoding="utf-8"))
    required={"method":method,"theory":"CUF_TE4","integrator":"BATHE",
              "rank":rank,"accepted_steps":1,"requested_steps":1,
              "converged":True,"status":"CONVERGED"}
    for key,value in required.items():
        if summary.get(key)!=value:
            raise ValueError(f"UNACCEPTED_OR_WRONG_CUF_TE4_CASE:{base}:{key}")
    if (case["geometry_model"]!="LINEAR" or
            case["load_definition"]["type"]!="PATCH_RADIAL_FACE" or
            case["load_definition"]["full_circumference"] is not False):
        raise ValueError("NONAXISYMMETRIC_LOAD_MISSING:"+method)
    arrays=[]
    for fname in ("BATHE_SUBSTEP_0001_GAMMA.npz","REFERENCE_DYNAMIC_STEP_0001.npz"):
        with np.load(base/fname,allow_pickle=False) as data:
            fields={}
            for key in ("mechanical","velocity","acceleration"):
                q=np.asarray(data[key],float)
                if q.shape!=(15,9,9) or not np.isfinite(q).all():
                    raise ValueError("NONAXIS_INVALID_PHYSICAL_FIELD:"+fname+":"+key)
                fields[key]=q
            fields["time_s"]=float(data["time_s"])
            if method=="MVI":
                x=np.asarray(data["x_profiles"],float)
                y=np.asarray(data["y_profiles"],float)
                scales=np.asarray(data["state_scales"],float)
                if (x.shape!=(15,rank,9) or y.shape!=(15,rank,9) or
                        scales.shape!=(15,9,9)):
                    raise ValueError("NONAXIS_BAD_MVI_RANK_FACTORS")
                represented=np.einsum("crx,crs->cxs",x,y)*scales
                if not np.allclose(represented,fields["mechanical"],
                                   rtol=1.e-10,atol=1.e-13):
                    raise ValueError("NONAXIS_MVI_RECONSTRUCTION_FAILED")
            arrays.append(fields)
    expected=[(2-math.sqrt(2))*.002,.002]
    for record,t in zip(arrays,expected):
        if abs(record["time_s"]-t)>1.e-12:
            raise ValueError("NONAXIS_STAGES_NOT_SAME_PHYSICAL_TIME")
    return case,arrays


def main():
    parser=argparse.ArgumentParser()
    for name in ("fdm","mvi1","mvi8"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    results={}
    cases={}
    for name,method,rank in (("fdm","FDM",None),("mvi1","MVI",1),("mvi8","MVI",8)):
        cases[name],results[name]=load_archive(getattr(args,name),method,rank)
    comparable=("geometry","material","boundary_conditions",
                "load_definition","dynamics","quadrature",
                "scale_model","geometry_model","theory","cuf_model","mode","units")
    for key in comparable:
        expected=cases["fdm"].get(key)
        # The restart switch only changes how work is scheduled, never time.
        if key=="dynamics":
            expected=dict(expected)
            expected.pop("bathe_resume_accepted",None)
        for name in ("mvi1","mvi8"):
            actual=cases[name].get(key)
            if key=="dynamics":
                actual=dict(actual)
                actual.pop("bathe_resume_accepted",None)
            if actual!=expected:
                raise ValueError("NONAXIS_UNMATCHED_PHYSICAL_CASE:"+name+":"+key)
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for stage in range(2):
        for name in ("mvi1","mvi8"):
            reference=results["fdm"][stage]
            computed=results[name][stage]
            for field in ("mechanical","velocity","acceleration"):
                a=reference[field]
                b=computed[field]
                for component in range(15):
                    error=float(np.max(np.abs(b[component]-a[component])))
                    scale=float(np.max(np.abs(a[component])))
                    rows.append({
                        "stage":("GAMMA" if stage==0 else "END"),
                        "time_s":reference["time_s"],
                        "method":name.upper(),
                        "quantity":field,
                        "cuf_component":component,
                        "max_abs_difference_SI":error,
                        "reference_peak_abs_SI":scale,
                        "peak_normalized_percent":(
                            100*error/scale if scale>0 else float("nan"))
                    })
    with (output/"CUF_TE4_PATCH_ALL_COMPONENT_FIELD_COMPARISONS.csv").open(
            "w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    physics={}
    for stage,name in enumerate(("GAMMA","END")):
        for method,values in results.items():
            q=values[stage]["mechanical"]
            w=q[10]
            singular=np.linalg.svd(w,compute_uv=False)
            total=float(np.linalg.norm(singular))
            fraction=(float(np.linalg.norm(singular[1:])/total)
                      if total>0 else float("nan"))
            fft=np.fft.rfft(w,axis=1)
            angular=float(np.linalg.norm(fft[:,1:])/
                          max(float(np.linalg.norm(fft)),1.e-300))
            physics[f"{name}_{method}"]={
                "time_s":values[stage]["time_s"],
                "center_w_m":float(w[4,4]),
                "max_abs_w_m":float(np.max(np.abs(w))),
                "spatial_rank1_remainder_fraction":fraction,
                "nonaxisymmetric_fourier_fraction":angular}
    (output/"CUF_TE4_PATCH_PHYSICAL_RANK_AUDIT.json").write_text(
        json.dumps({"source":"independent actual accepted NPZ states",
            "no_synthetic_data":True,"physical":physics,
            "methods":["MKR/FDM","MVI-1","MVI-8"],"stages":["GAMMA","END"],
            "fieldwise_csv":"CUF_TE4_PATCH_ALL_COMPONENT_FIELD_COMPARISONS.csv"},
            indent=2)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout="constrained")
    for ax,(label,method) in zip(axes,(
            ("MKR / FDM","fdm"),("MVI-1","mvi1"),("MVI-8","mvi8"))):
        w=results[method][1]["mechanical"][10]
        art=ax.imshow(w,origin="lower",aspect="auto",cmap="coolwarm",
                      extent=[0,2*math.pi,0,2])
        ax.set(xlabel="Circumferential angle (rad)",ylabel="Axial x (m)",
               title=f"{label}: W0 at t=2 ms (m)")
        fig.colorbar(art,ax=ax,fraction=.05)
    fig.savefig(output/"CUF_TE4_PATCH_ACTUAL_THREE_FIELDS.png",dpi=175)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for method,name,mark in (
        ("fdm","MKR / FDM","o"),("mvi1","MVI-1","s"),("mvi8","MVI-8","^")):
        w=results[method][1]["mechanical"][10]
        ax.plot(np.linspace(0,2*math.pi,9),w[4],
                marker=mark,label=name)
    ax.set(xlabel="Circumferential angle (rad)",
           ylabel="Midsurface radial displacement W0 at x=L/2 (m)",
           title="Actual non-axisymmetric response, t=2 ms")
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout()
    fig.savefig(output/"CUF_TE4_PATCH_ANGULAR_SECTIONS.png",dpi=175)
    plt.close(fig)
    print("CUF_TE4_PATCH_FULL_FIELD_COMPARISON_COMPLETE",
          json.dumps(physics),flush=True)


if __name__=="__main__":
    main()
