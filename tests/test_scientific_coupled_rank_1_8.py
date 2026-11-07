"""Safe branch generation for *actual* MVI 1..8 versus FDM TE1 flexo.

No mechanical or electrical solve is advanced in these tests.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
from cylinder_paths import activate
activate()

from cylinder_reference_case import prepare_reference_case
from ci.scientific_coupled_rank_1_8 import (
    catalog,build_case,compare,STUDY,NX,NS,RANKS,
)
from ci.scientific_coupled_convergence import physical_identity


def test_eight_distinct_fixed_rank_cases_and_one_matched_fdm_case():
    rows=catalog()
    assert len(rows)==9 and len({r["id"] for r in rows})==9
    assert [r["rank"] for r in rows[1:]]==list(RANKS)
    assert rows[0]["method"]=="FDM"
    assert all(r["method"]=="MVI" for r in rows[1:])
    assert len({r["study"] for r in rows})==1
    assert rows[0]["study"]==STUDY
    assert min(NX,NS)>=max(RANKS)==8


@pytest.mark.parametrize("entry",catalog(),ids=lambda e:e["id"])
def test_rank_1_to_8_case_is_admissible_without_solving(entry,tmp_path):
    output=tmp_path/entry["id"]
    case=build_case(entry,output)
    grid=case["fdm_grid" if entry["method"]=="FDM" else "mvi_grid"]
    assert (grid["nx"],grid["ns"])==(8,8)
    assert grid["operator_scheme"]=="LEGACY_REFERENCE"
    assert case["geometry_model"]=="GREEN_LAGRANGE"
    assert case["micro"]["enable_flexoelectric"] is True
    assert case["micro"]["gradient_model"]=="Gz"
    assert case["micro"]["potential_order"]==1
    assert case["load_definition"]["type"]=="PATCH_RADIAL_FACE"
    assert case["load_definition"]["face"]=="OUTER"
    assert case["reference_newton"]["max_free_dofs"]==512
    prepared=prepare_reference_case(case,base_dir=tmp_path)
    assert prepared.problem.functional.theory=="CUF_TE1"
    assert prepared.electrical.shape==(2,8,8)
    assert prepared.method==entry["method"]
    if entry["method"]=="MVI":
        assert case["rank"]==entry["rank"]
        assert prepared.mvi_settings.rank==entry["rank"]
    assert not output.exists()


def test_same_continuum_physics_all_nine_rank_cases(tmp_path):
    hashes={physical_identity(build_case(row,tmp_path/row["id"]))
            for row in catalog()}
    assert len(hashes)==1


def test_no_archives_cannot_claim_rank_comparison(tmp_path):
    with pytest.raises(RuntimeError,match="INCOMPLETE_REAL_CASES"):
        compare(tmp_path,tmp_path/"report")
    report=json.loads(
        (tmp_path/"report"/"SCIENTIFIC_MVI_RANKS_1_TO_8.json").read_text())
    assert len(report["cases"])==9
    assert report["all_eight_ranks_compared"] is False
    assert report["scientific_accuracy_certified"] is False
    assert report["independent_coupled_spatial_discretizations_certified"] is False
    assert all(x["status"]=="MISSING_FAILED_OR_INVALID"
               for x in report["cases"])
