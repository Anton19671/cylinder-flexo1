#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression guard for explicit KL S1/S2/S3/S4 support semantics."""
from __future__ import annotations
import final_boundary_registry as bc
import FINAL_POSTPROCESS as post


def run():
    expected={
        "PINNED_S1":("NATURAL","GEOMETRIC","GEOMETRIC","NATURAL"),
        "PINNED_S2":("GEOMETRIC","NATURAL","GEOMETRIC","NATURAL"),
        "SIMPLY_SUPPORTED":("GEOMETRIC","GEOMETRIC","GEOMETRIC","NATURAL"),
        "PINNED_S4":("NATURAL","NATURAL","GEOMETRIC","NATURAL"),
    }
    for code,kinds in expected.items():
        got=bc.kl_kinds(code)
        if got!=kinds:
            raise AssertionError(f"{code}: {got} != {kinds}")
    if bc.resolved_preset_code("SIMPLY_SUPPORTED","KL")!="PINNED_S3":
        raise AssertionError("KL SIMPLY_SUPPORTED must resolve to PINNED_S3")
    captions=bc.gui_caption_map("KL")
    required={"PINNED_S1","PINNED_S2","SIMPLY_SUPPORTED","PINNED_S4"}
    if not required.issubset(set(captions.values())):
        raise AssertionError("KL GUI does not expose all S1/S2/S3/S4 support choices")
    summary={
        "theory":"KL","geometry_model":"GREEN_LAGRANGE","method":"FDM",
        "boundary_conditions":{"left":"SIMPLY_SUPPORTED","right":"PINNED_S4"},
    }
    if post._resolved_bc_pair(summary)!=("PINNED_S3","PINNED_S4"):
        raise AssertionError("postprocess support caption is not exact")
    title=post._run_identity_caption(summary)
    if "PINNED_S3–PINNED_S4" not in title:
        raise AssertionError(f"wrong postprocess identity: {title}")
    print("BC_HINGE_VARIANTS_SELF_CHECK: PASS")
    return True


if __name__=="__main__":
    run()
