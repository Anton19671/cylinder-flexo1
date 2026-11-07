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

# The public CLI must not report a successful independent run as failed.
# Independent method runners historically returned status=CONVERGED without
# the legacy reference-only converged flag. Prefer explicit booleans and
# otherwise accept only a complete, committed dynamic history.
def _confirmed_success(summary):
    flag=summary.get("converged")
    if type(flag) is bool:
        return flag
    if summary.get("status")!="CONVERGED":
        return False
    if summary.get("calculation_kind")=="STATIC":
        return summary.get("solver_backend") in (
            "INDEPENDENT_MKR","INDEPENDENT_MVI")
    expected=summary.get("requested_steps")
    accepted=summary.get("accepted_steps")
    return (type(expected) is int and expected>0 and
            type(accepted) is int and accepted==expected)

confirmed_success=_confirmed_success(result)

# The complete UTF-8 JSON is written by the dispatcher into the run folder.
# Keep the live journal concise and ASCII-only; do not dump Russian metadata
# through a Windows text pipe.
print(
    "RESULT:"
    f" status={result.get('status','UNKNOWN')}"
    f" converged={confirmed_success}"
    f" accepted_steps={result.get('accepted_steps','?')}"
    f" failed_step={result.get('failed_step_index')}"
    f" failed_code={result.get('failed_step_code')}"
    f" failed_status={result.get('failed_step_status')}"
    f" failed_diagnostic={result.get('failed_step_diagnostic_ru')}"
)
raise SystemExit(0 if confirmed_success else 2)
