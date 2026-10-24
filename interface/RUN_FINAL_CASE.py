#!/usr/bin/env python3
import json,sys
from cylinder_final_stage11a_dispatcher import run_final_case

if len(sys.argv)!=2:
    raise SystemExit("Usage: python RUN_FINAL_CASE.py FINAL_CASE.json")

try:
    result=run_final_case(sys.argv[1])
except Exception as exc:
    # stdout is consumed by the Windows GUI pipe.  Keep it ASCII-only so a
    # locale-dependent child stdout encoding cannot corrupt the journal.
    message=str(exc).encode("ascii","backslashreplace").decode("ascii")
    print(f"ERROR: {type(exc).__name__}: {message}")
    raise SystemExit(3)

# The complete UTF-8 JSON is written by the dispatcher into the run folder.
# Keep the live journal concise and ASCII-only; do not dump Russian metadata
# through a Windows text pipe.
print(
    "RESULT:"
    f" status={result.get('status','UNKNOWN')}"
    f" converged={bool(result.get('converged',False))}"
    f" accepted_steps={result.get('accepted_steps','?')}"
    f" failed_step={result.get('failed_step_index')}"
    f" failed_code={result.get('failed_step_code')}"
    f" failed_status={result.get('failed_step_status')}"
    f" failed_diagnostic={result.get('failed_step_diagnostic_ru')}"
)
raise SystemExit(0 if result.get("converged",False) else 2)
