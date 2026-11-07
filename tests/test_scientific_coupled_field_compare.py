"""Numerical field comparator's input integrity, no physical solver runs."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from cylinder_paths import activate
activate()
from ci.scientific_coupled_convergence import catalog,case_from_entry,digest,physical_identity
from ci.scientific_coupled_field_compare import metric,matching_fields,compare_rank_fields


def _fields(relative_adjustment=0.):
    x=np.linspace(0,1.2,5)
    s=np.arange(5)*2*np.pi*2/5
    z=np.linspace(-.1,.1,11)
    a=np.arange(150.,dtype=float).reshape(6,5,5)/150.
    return {
        "x_m":x,"s_m":s,"z_m":z,
        "mechanical":a*(1.+relative_adjustment),
        "electrical":np.ones((2,5,5))*(1.+relative_adjustment)*1.e-7,
        "face_outer_radial_displacement_m":np.ones((5,5))*.003*(
            1.+relative_adjustment),
        "face_outer_electric_outward_induction_c_per_m2":
            np.ones((5,5))*1.e-9*(1.+relative_adjustment),
        "stress6_pa":np.ones((11,5,5,6))*.2*(1.+relative_adjustment),
        "hyperstress6x3_pa_m":np.ones((11,5,5,6,3))*.05*(
            1.+relative_adjustment),
        "induction3_c_per_m2":np.ones((11,5,5,3))*1.e-9*(
            1.+relative_adjustment),
    }


def _fake_saved(root,entry,alter=0.):
    folder=root/entry["id"]
    (folder/"RESULTS").mkdir(parents=True)
    case=case_from_entry(entry,folder)
    cf=folder/"CASE.json"
    cf.write_text(json.dumps(case),encoding="utf-8")
    state="REFERENCE_STEP_0001.npz"
    statefile=folder/"RESULTS"/state
    np.savez_compressed(statefile,**_fields(alter))
    record={
        "case_id":entry["id"],"status":"ACCEPTED",
        "case_sha256":digest(cf),
        "physics_sha256":physical_identity(case),
        "git_sha":"fixture-verified",
        "method":entry["method"],
        "state_file":state,
        "state_sha256":digest(statefile),
        "solution_class":("RANK_CONSTRAINED_ONLY" if entry["method"]=="MVI"
                          else "FULL_SPACE_STATIONARY"),
        "full_space_stationarity_confirmed":entry["method"]=="FDM",
    }
    (folder/"CONVERGENCE_EVIDENCE.json").write_text(
        json.dumps(record),encoding="utf-8")
    return folder,record


def test_full_field_norm_reports_real_relative_difference_and_zero_reference():
    a=np.array([1.,2.,3.])
    b=a*1.1
    m=metric(a,b)
    assert m["relative_l2"]==pytest.approx(.1)
    assert m["relative_to_reference_peak"]==pytest.approx(.1)
    assert m["max_abs_difference"]==pytest.approx(.3)
    z=metric(np.zeros(3),np.zeros(3))
    assert z["reference_identically_zero"] is True
    assert z["relative_l2"] is None
    assert z["relative_to_reference_peak"] is None


def test_missing_nonfinite_and_incompatible_grid_fail_closed():
    a=_fields();b=_fields()
    b["z_m"]=np.linspace(-.2,.2,11)
    with pytest.raises(ValueError,match="DIFFERENT_GRID"):
        matching_fields(a,b)
    b=_fields();b.pop("electrical")
    with pytest.raises(ValueError,match="FIELD_MISSING"):
        matching_fields(a,b)
    b=_fields();b["stress6_pa"][-1,2,3,1]=np.nan
    with pytest.raises(ValueError,match="SHAPE_OR_FINITE"):
        matching_fields(a,b)


def test_real_archive_pasport_and_sha_and_same_grid_comparison(tmp_path):
    full=[x for x in catalog("full") if x["study"]=="TE1_RANK"]
    fdm=next(x for x in full if x["method"]=="FDM")
    mvi=next(x for x in full if x["method"]=="MVI" and x["rank"]==1)
    folder, _=_fake_saved(tmp_path,fdm)
    mfolder,_=_fake_saved(tmp_path,mvi,alter=.02)
    output=tmp_path/"tables"
    result=compare_rank_fields(tmp_path,output)
    assert len(result["ranks"])==1
    assert result["ranks"][0]["solution_class"]=="RANK_CONSTRAINED_ONLY"
    assert result["all_ranks_full_space_stationary"] is False
    assert result["ranks"][0]["measured_field_differences"][
        "mechanical"]["relative_l2"]==pytest.approx(.02)
    assert (output/"SCIENTIFIC_RANK_FULL_FIELDS.csv").is_file()
    assert (output/"SCIENTIFIC_RANK_FULL_FIELDS.json").is_file()
    # A corrupt NPZ must not be accepted because a JSON record says ACCEPTED.
    with (mfolder/"RESULTS"/"REFERENCE_STEP_0001.npz").open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError,match="HASH_INCONSISTENT"):
        compare_rank_fields(tmp_path,tmp_path/"tables2")
    # A falsified physical model in CASE invalidates the archive itself.
    mcase=mfolder/"CASE.json"
    config=json.loads(mcase.read_text())
    config["micro"]["ell_m"]*=2
    mcase.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="CASE_HASH_INCONSISTENT"):
        compare_rank_fields(tmp_path,tmp_path/"tables3")


def test_mixed_physical_problem_cannot_be_compared(tmp_path):
    entries=[e for e in catalog("full") if e["study"]=="TE1_RANK"]
    fdm=next(e for e in entries if e["method"]=="FDM")
    mvi=next(e for e in entries if e["method"]=="MVI" and e["rank"]==1)
    _fake_saved(tmp_path,fdm)
    folder,record=_fake_saved(tmp_path,mvi,alter=.04)
    # Even when CASE itself is correctly checksummed, a different physical
    # length cannot be silently used in an FDM/MVI comparison.
    case=folder/"CASE.json"
    content=json.loads(case.read_text())
    content["geometry"]["L"]*=1.1
    case.write_text(json.dumps(content))
    record["case_sha256"]=digest(case)
    record["physics_sha256"]=physical_identity(content)
    (folder/"CONVERGENCE_EVIDENCE.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="MISMATCHED_PHYSICS_OR_REVISION"):
        compare_rank_fields(tmp_path,tmp_path/"tables")
