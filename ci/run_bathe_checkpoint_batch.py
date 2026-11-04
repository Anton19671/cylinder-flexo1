#!/usr/bin/env python3
"""Execute only accepted physical Bathe steps in a bounded CI batch.

This is an orchestration entry point. The CASE physics/time grid remains
unchanged, and every batch is validated by the existing physical acceptance
checks and by fail-closed checkpoint restoration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cylinder_case_io import load_case_file
from cylinder_reference_case import prepare_reference_case
from cylinder_bathe_case import run_bathe_case


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("case",type=Path)
    parser.add_argument("--new-steps",type=int,required=True)
    args=parser.parse_args()
    if not 1<=args.new_steps<=120:
        raise ValueError("BATHE_BATCH_STEPS_OUTSIDE_1_TO_120")
    case_path=args.case.resolve()
    case=load_case_file(case_path)
    dynamics=case.get("dynamics")
    if (not isinstance(dynamics,dict) or
            str(dynamics.get("integrator","")).upper()!="BATHE"):
        raise ValueError("BATHE_BATCH_REQUIRES_BATHE_CASE")
    # Execution-only switch; never alter pressure, time law, mesh or physics.
    if dynamics.get("bathe_resume_accepted") not in (None,True):
        raise ValueError("BATHE_BATCH_RESUME_DISABLED")
    dynamics["bathe_resume_accepted"]=True
    case_path.write_text(json.dumps(case,ensure_ascii=False,indent=2)+"\n",
                         encoding="utf-8")
    prepared=prepare_reference_case(case,base_dir=case_path.parent)
    summary=run_bathe_case(case,case_path,prepared=prepared,
                           max_new_steps=args.new_steps)
    accepted=summary["accepted_steps"]
    requested=summary["requested_steps"]
    if not (1<=accepted<=requested and isinstance(accepted,int)):
        raise ValueError("BATHE_BATCH_NO_ACCEPTED_PHYSICAL_STEPS")
    print("BATHE_BATCH_PHYSICAL_STATES_SAVED",
          "accepted",accepted,"requested",requested,
          "full_history_complete",summary["converged"],flush=True)


if __name__=="__main__":
    main()
