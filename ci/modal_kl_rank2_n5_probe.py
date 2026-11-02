#!/usr/bin/env python3
"""Test whether extra harmonic starts recover KL rank-two n=5 modes."""
import runpy
import sys

from cylinder_paths import activate
activate()
import cylinder_modal_mvi

original_seed_plan = cylinder_modal_mvi._modal_seed_plan


def extended_seed_plan(operator, mode, requested):
    starts = original_seed_plan(operator, mode, requested)
    for n in range(5, min(operator.ns // 2, 6) + 1):
        for phase in ("cos", "sin"):
            starts.append({"kind": "harmonic_partner",
                           "index": int(mode + 4001 + 17*n + (7 if phase == "sin" else 0)),
                           "n": n, "phase": phase})
    return starts


cylinder_modal_mvi._modal_seed_plan = extended_seed_plan
sys.argv = ["ci/modal_mvi_all_ranks_theories_audit.py", "KL", "2"]
runpy.run_path(sys.argv[0], run_name="__main__")
