#!/usr/bin/env python3
"""Plot accepted, measured CUF TE4 Bathe states; never synthesize a trajectory."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    folder=args.output.resolve()
    summary=json.loads((folder/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    case=json.loads((folder/"REFERENCE_CASE.json").read_text(encoding="utf-8"))
    if (summary.get("theory")!="CUF_TE4" or summary.get("method") not in ("FDM","MVI") or
            summary.get("integrator")!="BATHE"):
        raise ValueError("TE4_EXPORT_INCORRECT_PHYSICAL_PASSPORT")
    accepted=summary["accepted_steps"]
    expected=summary["requested_steps"]
    if (type(accepted) is not int or type(expected) is not int or
            not 0<=accepted<=expected or expected!=120 or
            len(summary["steps"])!=accepted):
        raise ValueError("TE4_EXPORT_ACCEPTED_HISTORY_COUNT_INVALID")
    if accepted==0:
        print("TE4_EXPORT_NO_ACCEPTED_STATES",flush=True)
        return
    if (case["dynamics"]["n_steps"]!=120 or case["dynamics"]["time_step_s"]!=.002 or
            case["load_definition"]["q"]!=100000 or
            case["dynamics"]["time_law"]["frequency_hz"]!=25):
        raise ValueError("TE4_EXPORT_CASE_PHYSICS_CHANGED")
    method=summary["method"]
    rank=summary.get("rank")
    rows=[]
    gamma=2-math.sqrt(2)
    for step in range(accepted+1):
        stages=(("INITIAL","REFERENCE_DYNAMIC_INITIAL.npz"),) if step==0 else (
            ("GAMMA",f"BATHE_SUBSTEP_{step:04d}_GAMMA.npz"),
            ("END",f"REFERENCE_DYNAMIC_STEP_{step:04d}.npz"))
        for stage,filename in stages:
            with np.load(folder/filename,allow_pickle=False) as saved:
                t=float(saved["time_s"])
                q=np.asarray(saved["mechanical"],dtype=float)
                v=np.asarray(saved["velocity"],dtype=float)
                a=np.asarray(saved["acceleration"],dtype=float)
                target=(0. if step==0 else
                    (step-1+gamma)*.002 if stage=="GAMMA" else step*.002)
                if (q.shape!=(15,9,9) or v.shape!=q.shape or a.shape!=q.shape or
                        any(not np.isfinite(x).all() for x in (q,v,a)) or
                        abs(t-target)>1e-12):
                    raise ValueError("TE4_EXPORT_INVALID_SAVED_TIME_OR_FIELD:"+filename)
                factor=0. if stage=="INITIAL" else float(saved["load_factor"])
                if abs(factor-math.sin(2*math.pi*25*t))>1e-11:
                    raise ValueError("TE4_EXPORT_INVALID_SAVED_PRESSURE:"+filename)
                w=q[10:15,4,4]
                rows.append(dict(step=step,stage=stage,time_s=t,time_ms=t*1000,
                    pressure_inward_kpa=100*factor,
                    w_middle_center_mm=float(w[0]*1000),
                    w_inner_center_mm=float(np.dot(w,[1,-1,1,-1,1])*1000),
                    w_outer_center_mm=float(np.sum(w)*1000),
                    radial_velocity_center_mm_s=float(v[10,4,4]*1000),
                    radial_acceleration_center_mm_s2=float(a[10,4,4]*1000)))
    if len(rows)!=2*accepted+1:
        raise ValueError("TE4_EXPORT_MISSING_PHYSICAL_STAGES")
    def write_csv(name,selection):
        with (folder/name).open("w",newline="",encoding="utf-8") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selection)
    write_csv("CUF_TE4_ACCEPTED_ALL_STAGES.csv",rows)
    write_csv("CUF_TE4_ACCEPTED_FULL_STEPS.csv",
        [row for row in rows if row["stage"] in ("INITIAL","END")])
    complete=(accepted==expected and summary["status"]=="CONVERGED" and
              summary["converged"] is True)
    (folder/"CUF_TE4_PLOT_PASSPORT.json").write_text(json.dumps({
        "method":method,"rank":rank,"accepted_steps":accepted,
        "requested_steps":expected,"full_history_complete":complete,
        "physical_duration_s":rows[-1]["time_s"],
        "frequency_hz":25,"pressure_amplitude_kpa":100,
        "thickness_basis":"F_k(z)=(2z/h)^k",
        "physical_only":True,
        "note":"No graph is evidence of convergence without independent verification."
    },indent=2)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":11,"axes.spines.right":False,
                         "axes.spines.top":False,"axes.grid":True,"grid.alpha":.2})
    x=[r["time_ms"] for r in rows]
    suffix="" if complete else " — accepted steps only"
    name=f"CUF TE4 {method}"+(f"-{rank}" if method=="MVI" else "")
    charts=(
        ("PRESSURE", "pressure_inward_kpa", "Signed inward pressure (kPa)"),
        ("DEFLECTION","w_middle_center_mm","Midsurface radial displacement at center (mm)"),
        ("VELOCITY","radial_velocity_center_mm_s","Radial velocity at center (mm/s)"),
        ("ACCELERATION","radial_acceleration_center_mm_s2",
         "Radial acceleration at center (mm/s²)"))
    for title,key,ylabel in charts:
        fig,ax=plt.subplots(figsize=(9.5,5))
        ax.plot(x,[row[key] for row in rows],lw=2,color="#1763a6")
        if title=="DEFLECTION":
            ax.plot(x,[r["w_inner_center_mm"] for r in rows],
                    lw=1.6,ls="--",color="#237a67",label="Inner face")
            ax.plot(x,[r["w_outer_center_mm"] for r in rows],
                    lw=1.6,ls=":",color="#584eac",label="Outer face")
            ax.legend(frameon=False)
        ax.set(xlabel="Time (ms)",ylabel=ylabel,
               title=f"{name}: real {title.lower()} response"+suffix)
        fig.tight_layout()
        fig.savefig(folder/f"CUF_TE4_{title}_TIME_COLOR.png",dpi=200)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    ax.plot([row["w_middle_center_mm"] for row in rows],
            [row["pressure_inward_kpa"] for row in rows],color="#c44536",lw=2)
    ax.set(xlabel="Midsurface radial displacement at center (mm)",
           ylabel="Signed inward pressure (kPa)",
           title=f"{name}: actual dynamic load-deflection"+suffix)
    fig.tight_layout()
    fig.savefig(folder/"CUF_TE4_LOAD_DEFLECTION_COLOR.png",dpi=200)
    plt.close(fig)
    print("CUF_TE4_REAL_PLOT_EXPORT",method,rank,accepted,"/",expected,
          "rows",len(rows),"complete",complete,flush=True)

if __name__=="__main__":
    main()
