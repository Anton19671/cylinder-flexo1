#!/usr/bin/env python3
import json,sys
from cylinder_final_stage11a_dispatcher import run_mvi_rank_series
if len(sys.argv)!=2:
    raise SystemExit("Usage: python RUN_MVI_1_8.py FINAL_CASE.json")
try:
    r=run_mvi_rank_series(sys.argv[1],range(1,9))
except Exception as exc:
    print(f"ERROR: {type(exc).__name__}: {exc}")
    raise SystemExit(3)
print(json.dumps(r,indent=2,ensure_ascii=False))
raise SystemExit(0 if r.get("all_converged",False) else 2)
