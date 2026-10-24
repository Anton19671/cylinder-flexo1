#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import csv,json,sys,zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from FINAL_POSTPROCESS import process_output, _state_files, _state_from_npz, _summary
import final_units as funits


def main(series_dir):
    root=Path(series_dir).resolve()
    graphs=root/"GRAPHS"
    graphs.mkdir(parents=True,exist_ok=True)
    rows=[]
    displacement_unit="length unit"
    for rank in range(1,9):
        out=root/f"MVI_{rank}"/"OUT"
        if not out.exists():
            continue
        process_output(out,make_zip=False,fast=True)
        units=(_summary(out).get("units") or {})
        displacement_unit=funits.unit_labels(str(units.get("system","N_MM_MPA")))["displacement"]
        states=_state_files(out)
        if not states:
            continue
        st=_state_from_npz(states[-1])
        w=np.asarray(st.get("w"),float)
        sm=root/f"MVI_{rank}_SUMMARY.json"
        sj=json.loads(sm.read_text(encoding="utf-8")) if sm.exists() else {}
        rows.append({
            "rank":rank,
            "converged":bool(sj.get("converged",False)),
            "status":sj.get("status"),
            "w_abs_max":float(np.max(np.abs(w))),
            "w_min":float(np.min(w)),
            "w_max":float(np.max(w)),
        })

    if rows:
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        ax.plot([r["rank"] for r in rows],[r["w_abs_max"] for r in rows],marker="o")
        ax.set_xlabel("MVI rank N [-]")
        ax.set_ylabel(f"|w|max [{displacement_unit}]")
        ax.set_title("MVI-1…8 rank convergence")
        ax.grid(True,alpha=0.3)
        fig.tight_layout()
        fig.savefig(graphs/"MVI_RANK_WMAX.png",dpi=160)
        plt.close(fig)

    with (root/"MVI_RANK_RESULTS.csv").open("w",newline="",encoding="utf-8") as f:
        fieldnames=["rank [-]","MVI projected equations converged [0/1]","status [-]",f"w_abs_max [{displacement_unit}]",f"w_min [{displacement_unit}]",f"w_max [{displacement_unit}]"]
        wr=csv.writer(f); wr.writerow(fieldnames)
        for r in rows:wr.writerow([r["rank"],int(r["converged"]),r["status"],r["w_abs_max"],r["w_min"],r["w_max"]])
    (root/"MVI_RANK_RESULTS.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")

    zpath=root/"NUMERIC_RESULTS_MVI_1_8.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p==zpath or graphs in p.parents:
                continue
            if p.suffix.lower() in (".npz",".json",".csv",".txt",".xlsx"):
                zf.write(p,arcname=p.relative_to(root))
    print(f"MVI SERIES POSTPROCESS PASS: ranks={len(rows)}; graphs={graphs}; zip={zpath}")


if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("Usage: python FINAL_MVI_SERIES_POSTPROCESS.py MVI_1_8_SERIES_DIR")
    main(sys.argv[1])
