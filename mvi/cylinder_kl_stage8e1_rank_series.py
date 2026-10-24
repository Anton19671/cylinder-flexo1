#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8E-1: independent KL MVI-rank batch driver.

One explicit profile bank is supplied:
    X_bank shape (3,max_rank,nx)
    Y_bank shape (3,max_rank,ns)

For requested MVI-N:
    X = X_bank[:, :N, :]
    Y = Y_bank[:, :N, :]

No seed terms are generated.
No rank is inferred.
No rank-to-rank state transfer occurs.
Each rank is an independent CASE and runs through the already verified
Stage8D-3 launcher.

A failure at one rank is recorded but does not trigger retries or prevent
other explicitly requested independent ranks from running.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np

import cylinder_kl_stage8d3_launcher as la


class KLRankSeriesError(RuntimeError):
    pass


@dataclass
class KLRankRun:
    rank: int
    converged: bool
    status: str
    accepted_steps: int
    requested_steps: int
    failed_step_index: object
    failed_step_status: object
    case_json: str
    result_dir: str


@dataclass
class KLRankSeriesResult:
    requested_ranks: list
    runs: list
    output_root: str

    @property
    def all_converged(self):
        return all(x.converged for x in self.runs)

    def by_rank(self,rank):
        rr=int(rank)
        for x in self.runs:
            if x.rank==rr:
                return x
        raise KeyError(rr)


def _resolve(config_path,value):
    p=Path(str(value))
    if not p.is_absolute():
        p=config_path.parent/p
    return p.resolve()


def _read_json(path,what):
    p=Path(path)
    try:
        obj=json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise KLRankSeriesError(f"cannot read {what}: {p}") from exc
    if not isinstance(obj,dict):
        raise KLRankSeriesError(f"{what} root must be a JSON object")
    return obj,p.resolve()


def _required(obj,key,context):
    if key not in obj or obj[key] is None:
        raise KLRankSeriesError(f"missing required field: {context}.{key}")
    return obj[key]


def _load_profile_bank(path):
    p=Path(path)
    if not p.exists():
        raise KLRankSeriesError(f"profile_bank_npz does not exist: {p}")
    try:
        with np.load(p,allow_pickle=False) as z:
            if "X" not in z.files or "Y" not in z.files:
                raise KLRankSeriesError("profile bank must contain X and Y")
            X=np.asarray(z["X"],float)
            Y=np.asarray(z["Y"],float)
    except KLRankSeriesError:
        raise
    except Exception as exc:
        raise KLRankSeriesError(f"cannot read profile bank: {p}") from exc

    if X.ndim!=3 or Y.ndim!=3:
        raise KLRankSeriesError("profile-bank X/Y must be 3D")
    if X.shape[0]!=3 or Y.shape[0]!=3:
        raise KLRankSeriesError("profile-bank first dimension must be 3")
    if X.shape[1]!=Y.shape[1] or X.shape[1]<1:
        raise KLRankSeriesError("profile-bank rank dimension mismatch")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise KLRankSeriesError("profile bank contains non-finite values")
    return X,Y


def _validate_ranks(values,max_rank):
    if not isinstance(values,list) or len(values)<1:
        raise KLRankSeriesError("ranks must be a non-empty JSON list")
    out=[]
    for x in values:
        try:
            r=int(x)
        except Exception as exc:
            raise KLRankSeriesError("each rank must be an integer") from exc
        if float(x)!=float(r):
            raise KLRankSeriesError("each rank must be integer-valued")
        if r<1 or r>max_rank:
            raise KLRankSeriesError(
                f"requested rank {r} outside profile-bank range 1..{max_rank}"
            )
        if r in out:
            raise KLRankSeriesError(f"duplicate requested rank {r}")
        out.append(r)
    return out


def _write_summary(result):
    out=Path(result.output_root)
    records=[]
    for rr in result.runs:
        records.append({
            "rank":rr.rank,
            "converged":rr.converged,
            "status":rr.status,
            "accepted_steps":rr.accepted_steps,
            "requested_steps":rr.requested_steps,
            "failed_step_index":rr.failed_step_index,
            "failed_step_status":rr.failed_step_status,
            "case_json":rr.case_json,
            "result_dir":rr.result_dir,
        })

    (out/"RANK_SERIES_SUMMARY.json").write_text(
        json.dumps({
            "requested_ranks":result.requested_ranks,
            "all_converged":result.all_converged,
            "runs":records,
        },indent=2,ensure_ascii=False),
        encoding="utf-8"
    )

    with (out/"RANK_SERIES_SUMMARY.csv").open(
        "w",newline="",encoding="utf-8"
    ) as f:
        w=csv.DictWriter(
            f,
            fieldnames=[
                "rank","converged","status","accepted_steps",
                "requested_steps","failed_step_index","failed_step_status",
                "case_json","result_dir"
            ]
        )
        w.writeheader()
        w.writerows(records)


def run_kl_rank_series(config_json):
    cfg,cfg_path=_read_json(config_json,"rank-series config")

    base_case_path=_resolve(
        cfg_path,_required(cfg,"base_case_json","RANK_SERIES")
    )
    profile_bank_path=_resolve(
        cfg_path,_required(cfg,"profile_bank_npz","RANK_SERIES")
    )
    output_root=_resolve(
        cfg_path,_required(cfg,"output_root","RANK_SERIES")
    )

    base_case,_=_read_json(base_case_path,"base KL CASE")
    Xbank,Ybank=_load_profile_bank(profile_bank_path)

    max_rank=Xbank.shape[1]
    ranks=_validate_ranks(
        _required(cfg,"ranks","RANK_SERIES"),max_rank
    )

    # Grid-size compatibility is checked before creating any rank CASE.
    gj=_required(base_case,"grid","BASE_CASE")
    if not isinstance(gj,dict):
        raise KLRankSeriesError("BASE_CASE.grid must be an object")
    nx=int(_required(gj,"nx","BASE_CASE.grid"))
    ns=int(_required(gj,"ns","BASE_CASE.grid"))
    if Xbank.shape[2]!=nx:
        raise KLRankSeriesError(
            f"profile-bank X nx={Xbank.shape[2]}, base CASE nx={nx}"
        )
    if Ybank.shape[2]!=ns:
        raise KLRankSeriesError(
            f"profile-bank Y ns={Ybank.shape[2]}, base CASE ns={ns}"
        )

    output_root.mkdir(parents=True,exist_ok=True)
    runs=[]

    for rank in ranks:
        rank_dir=output_root/f"MVI_{rank}"
        rank_dir.mkdir(parents=True,exist_ok=True)

        profile_file=rank_dir/f"INITIAL_PROFILES_MVI_{rank}.npz"
        np.savez_compressed(
            profile_file,
            X=Xbank[:,:rank,:].copy(),
            Y=Ybank[:,:rank,:].copy(),
        )

        case=json.loads(json.dumps(base_case))
        case["rank"]=rank
        case["initial_profiles_npz"]=str(profile_file.resolve())
        case["output_dir"]=str((rank_dir/"RESULTS").resolve())

        case_file=rank_dir/f"CASE_MVI_{rank}.json"
        case_file.write_text(
            json.dumps(case,indent=2,ensure_ascii=False),
            encoding="utf-8"
        )

        try:
            rr=la.run_kl_case(case_file)
            run=KLRankRun(
                rank=rank,
                converged=bool(rr.converged),
                status=str(rr.status),
                accepted_steps=len(rr.accepted_steps),
                requested_steps=int(rr.requested_steps),
                failed_step_index=rr.failed_step_index,
                failed_step_status=rr.failed_step_status,
                case_json=str(case_file.resolve()),
                result_dir=str((rank_dir/"RESULTS").resolve()),
            )
        except Exception as exc:
            run=KLRankRun(
                rank=rank,
                converged=False,
                status=f"CASE_ERROR:{type(exc).__name__}:{exc}",
                accepted_steps=0,
                requested_steps=0,
                failed_step_index=None,
                failed_step_status=None,
                case_json=str(case_file.resolve()),
                result_dir=str((rank_dir/"RESULTS").resolve()),
            )
        runs.append(run)

    result=KLRankSeriesResult(
        requested_ranks=list(ranks),
        runs=runs,
        output_root=str(output_root),
    )
    _write_summary(result)
    return result
