"""Regression for audit V-4/V-5/V-12, without expensive scientific solves."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from cylinder_director_g3d import DirectorSurfaceJets, director_point
from cylinder_director_rotation_audit import director_rotation_diagnostic
from cylinder_mvi_essential_trace_guard import require_homogeneous_mvi_traces
from cylinder_reference_case import prepare_reference_case, _write_dynamic_history
from cylinder_reference_gui import reference_locking_message

@pytest.mark.parametrize("theory",["FSDT","REDDY"])
@pytest.mark.parametrize("theta",[0.,.05,.15,.5])
def test_spurious_midplane_estrain(theory,theta):
    names=("q","qx","qs","qxx","qxs","qss","qxxx","qxxs","qxss","qsss")
    data={name:np.zeros(5) for name in names}
    data["q"][3]=theta
    data.update(psi=np.zeros(1),psix=np.zeros(1),psis=np.zeros(1))
    p=director_point(DirectorSurfaceJets(**data),theory=theory,z_m=0.,radius_m=2.,thickness_m=.2)
    q=np.zeros((5,4,5));q[3]=theta
    d=director_rotation_diagnostic(theory,q)
    assert p.strain6[2]==pytest.approx(.5*theta**2,abs=1e-14)
    assert d["spurious_Ezz_midplane_max"]==pytest.approx(.5*theta**2)
    assert d["advisory_limit_exceeded"]==(theta>.1)

@pytest.mark.parametrize("theory",["KL","CUF_TE1","CUF_TE4"])
def test_other_theories_not_misclassified(theory):
    assert director_rotation_diagnostic(theory)["director_type"]=="NOT_APPLICABLE"

@pytest.mark.parametrize("support,values,reject",[
    ("CLAMPED",[.2,0,0],True),("SYMMETRY",[0,.2,0],False),
    ("SIMPLY_SUPPORTED",[.2,0,0],True),("FREE",[.2,0,0],False)])
def test_guard_essential_mechanics_only(support,values,reject):
    bc={"mechanical":{"left":{"support":support,"prescribed":values}}}
    if reject:
        with pytest.raises(ValueError,match="MVI_NONZERO_ESSENTIAL_TRACE_UNSUPPORTED"):
            require_homogeneous_mvi_traces(bc,theory="KL",n_mechanical=3)
    else: assert require_homogeneous_mvi_traces(bc,theory="KL",n_mechanical=3)

@pytest.mark.parametrize("site",["inner","outer","left","right"])
def test_guard_nonzero_voltage(site):
    with pytest.raises(ValueError,match="MVI_NONZERO_ESSENTIAL_TRACE_UNSUPPORTED"):
        require_homogeneous_mvi_traces(
            {"electrical":{site:{"mode":"POTENTIAL","value":.3}}},
            theory="CUF_TE1",n_mechanical=6)

def test_guard_allows_charge_and_zero_voltage():
    assert require_homogeneous_mvi_traces(
        {"electrical":{"inner":{"mode":"POTENTIAL","value":0.},
                       "outer":{"mode":"CHARGE","value":1.}}},
        theory="CUF_TE1",n_mechanical=6)

def test_mvi_preflight_before_output(tmp_path):
    c=json.loads((Path(__file__).resolve().parent.parent/
        "config/CASE_REFERENCE_CUF_G3D_STATIC.json").read_text())
    c["method"]="MVI";c["rank"]=1;c["mvi_grid"]=c.pop("fdm_grid")
    c["output_dir"]=str(tmp_path/"unused")
    c["boundary_conditions"]["variational"]["mechanical"]["left"]["prescribed"]=[.2,0,0,0,0,0]
    with pytest.raises(ValueError,match="MVI_NONZERO_ESSENTIAL_TRACE_UNSUPPORTED"):
        prepare_reference_case(c,base_dir=tmp_path)
    assert not (tmp_path/"unused").exists()

def test_gui_warning_for_rotation():
    assert "V-4" in reference_locking_message({"theory":"FSDT","material":{"nu0":.3}})

def test_dynamic_history_csv_marks_rejected_step(tmp_path):
    dest=tmp_path/"history.csv"
    _write_dynamic_history(dest,[{"index":1,"time_s":.1,"status":"CONVERGED","converged":True},
                                 {"index":2,"time_s":.2,"status":"MAX_ITERATIONS","converged":False}])
    with dest.open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    assert [r["converged"] for r in rows]==["True","False"]
    assert [r["status"] for r in rows]==["CONVERGED","MAX_ITERATIONS"]
