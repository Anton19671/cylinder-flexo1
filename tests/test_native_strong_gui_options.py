"""GUI numerical choices preserve the explicit native CASE and physical data."""
import json
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import cylinder_native_research_gui as gui
from independent_mkr import run_coupled_research as native
from test_native_coupled_mkr_case_dispatch import make_case

def strong_case(tmp_path):
    path,case=make_case(tmp_path,theory="KL",kind="STATIC",electric=True)
    case["grid"]["nx"]=21
    q=np.zeros((3,21,5));p=np.zeros((1,21,5))
    np.savez_compressed(tmp_path/"initial.npz",mechanical=q,electrical=p)
    np.savez_compressed(tmp_path/"loads.npz",mechanical_load=q,electrical_charge=p)
    path.write_text(json.dumps(case))
    return path,case

@pytest.mark.parametrize("scheme,axisymmetric",(
    ("LEGACY_TRAPEZOID",False),("MATCHED_QUARTIC_V1",False),
    ("MATCHED_QUARTIC_SAT_V1",False),("MATCHED_QUARTIC_SAT_V1",True)))
def test_gui_choice_passes_actual_native_case_preflight(tmp_path,scheme,axisymmetric):
    path,case=strong_case(tmp_path);original=deepcopy(case)
    changed=gui.updated_strong_options(case,scheme,axisymmetric)
    assert case==original
    assert json.loads(path.read_text())==original
    expected=deepcopy(original)
    expected["grid"]["axial_scheme"]=scheme
    expected["solver"]["axisymmetric"]=axisymmetric
    assert changed==expected
    path.write_text(json.dumps(changed))
    result=gui.validate_final_case(path,dry_run=True)
    assert result["axial_scheme"]==scheme
    assert result["axisymmetric"] is axisymmetric
    assert result["physical_steps_computed"]==0
    assert not (tmp_path/"result").exists()
    assert scheme in gui._strong_options_description(result)

@pytest.mark.parametrize("schema",(
    "INDEPENDENT_ENERGY_FDM_COUPLED_RESEARCH_V1","INDEPENDENT_MVI_COUPLED_RESEARCH_V1"))
def test_gui_never_changes_other_solver_method(tmp_path,schema):
    _,case=strong_case(tmp_path);case["schema"]=schema
    original=deepcopy(case)
    with pytest.raises(ValueError,match="strong-FDM"):
        gui.updated_strong_options(case,"MATCHED_QUARTIC_SAT_V1",True)
    assert case==original

@pytest.mark.parametrize("problem",("small_grid","floating","axis_without_sat"))
def test_invalid_scheme_rejected_without_changing_case(tmp_path,problem):
    _,case=strong_case(tmp_path);scheme="MATCHED_QUARTIC_SAT_V1"
    if problem=="small_grid":case["grid"]["nx"]=19
    if problem=="floating":case["boundary"]["electric"]="INSULATED_FLOATING"
    if problem=="axis_without_sat":scheme="LEGACY_TRAPEZOID"
    original=deepcopy(case)
    with pytest.raises(ValueError):
        gui.updated_strong_options(case,scheme,True)
    assert case==original

def test_applying_changes_editor_not_file(tmp_path):
    path,case=strong_case(tmp_path);writes=[];logs=[];states=[]
    window=SimpleNamespace(
        _require_idle=lambda:None,_edited=lambda:case,
        strong_scheme=SimpleNamespace(get=lambda:"MATCHED_QUARTIC_SAT_V1"),
        strong_axisymmetric=SimpleNamespace(get=lambda:True),
        editor=SimpleNamespace(delete=lambda *a:None,insert=lambda *a:writes.append(a[1])),
        state_description=SimpleNamespace(set=states.append),_log=logs.append)
    gui.NativeResearchWindow.apply_strong_options(window)
    assert json.loads(writes[0])["solver"]["axisymmetric"] is True
    assert json.loads(path.read_text())==case
    assert "сохраните" in states[0]

def test_cannot_apply_options_during_running_calculation(tmp_path,monkeypatch):
    errors=[]
    def reject():raise ValueError("running")
    window=SimpleNamespace(_require_idle=reject)
    monkeypatch.setattr(gui.messagebox,"showerror",lambda *a,**k:errors.append(a))
    gui.NativeResearchWindow.apply_strong_options(window)
    assert errors and "running" in errors[0][1]
