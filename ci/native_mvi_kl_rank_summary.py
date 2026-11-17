"""Summarize EXACTLY 32 independently saved loaded native MVI KL cases.

Never interprets projected equilibrium as full and never interpolates a
missing rank. Errors remain visible with exact CASE/commit provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RANKS=tuple(range(1,9))
GRADIENTS=("G3D","Gz")
KINDS=("STATIC","DYNAMIC")


def summarize(source,target):
    source=Path(source)
    target=Path(target)
    target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for gradient in GRADIENTS:
        for kind in KINDS:
            for rank in RANKS:
                path=source/"rank_cases"/f"{gradient}_{kind}_R{rank}"/"RANK_EVIDENCE.json"
                if not path.is_file():
                    rows.append(dict(gradient=gradient,kind=kind,rank=rank,
                                     status="MISSING",quality="NOT_MEASURED",
                                     residual=None,radial_max_m=None,
                                     potential_max_v=None,sha=None,
                                     error="Evidence missing"))
                    continue
                record=json.loads(path.read_text(encoding="utf-8"))
                if (record.get("requested_rank")!=rank or
                        record.get("gradient_model")!=gradient or
                        record.get("kind")!=kind):
                    raise ValueError("RANK_EVIDENCE_IDENTITIES_CHANGED")
                quality=record.get("solution_quality")
                if record.get("status")=="ACCEPTED_RESEARCH_ONLY":
                    if quality not in ("FULL_NODAL_EQUILIBRIUM",
                                       "PROJECTED_EQUILIBRIUM",
                                       "PROJECTED_WITH_UNREPORTED_STEPS",
                                       "UNREPORTED_EQUILIBRIUM"):
                        raise ValueError("RANK_EQUILIBRIUM_QUALITY_MISSING")
                    if (quality=="FULL_NODAL_EQUILIBRIUM" and
                            record.get("full_nodal_equilibrium_satisfied") is not True):
                        raise ValueError("RANK_FALSE_FULL_BALANCE")
                rows.append(dict(
                    gradient=gradient,kind=kind,rank=rank,
                    status=record["status"],quality=quality or "NOT_MEASURED",
                    residual=record.get("full_nodal_norm"),
                    radial_max_m=record.get("radial_max_abs_m"),
                    potential_max_v=record.get("potential_max_abs_v"),
                    sha=record.get("git_sha"),
                    error=record.get("error")))
    revisions={v["sha"] for v in rows if v["sha"] is not None}
    if len(revisions)>1:
        raise ValueError("RANK_MIXED_SOURCE_REVISIONS")
    with (target/"RANKS_1_TO_8.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    accepted=sum(r["status"]=="ACCEPTED_RESEARCH_ONLY" for r in rows)
    full=sum(r["status"]=="ACCEPTED_RESEARCH_ONLY" and
             r["quality"]=="FULL_NODAL_EQUILIBRIUM" for r in rows)
    projected=sum(r["status"]=="ACCEPTED_RESEARCH_ONLY" and
             r["quality"].startswith("PROJECTED") for r in rows)
    report=dict(expected=32,accepted=accepted,full=full,projected=projected,
                incomplete=32-accepted,revision=next(iter(revisions),None),
                external_accuracy_certified=False,
                higher_order_boundaries_certified=False,
                rank_matrix=rows,
                status="ALL_32_RECORDED_RESEARCH_ONLY" if accepted==32
                else "INCOMPLETE_WITH_FAILING_RANKS")
    (target/"RANKS_1_TO_8.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("REAL_MVI_32_RANK_REPORT",json.dumps({
        k:report[k] for k in ("expected","accepted","full",
                             "projected","incomplete","status")}),flush=True)
    return report


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--source",required=True)
    cli.add_argument("--output",required=True)
    args=cli.parse_args()
    summarize(args.source,args.output)
