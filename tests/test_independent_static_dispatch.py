"""Independent STATIC public dispatch and method-owned output.

All execution is mocked, no physical shell solve is performed.
"""
from __future__ import annotations
import json
import numpy as np
import pytest

from cylinder_paths import activate
activate()
from ci.scientific_complete_method_matrix import (
    THEORIES,independent_static_template)
from cylinder_final_stage11a_dispatcher import (
    run_final_case,validate_final_case)
from independent_mkr import run_static as mk
from independent_mvi import run_static as mv
from independent_mkr import run_models as md
from independent_mvi import run_models as vd


@pytest.mark.parametrize("method",("MKR","MVI"))
@pytest.mark.parametrize("theory",THEORIES)
def test_preflight_all_seven_static_theories(method,theory,tmp_path):
    case=independent_static_template(theory,method,
                                     1 if method=="MVI" else None)
    case["output_dir"]=str(tmp_path/"not-created")
    path=tmp_path/"case.json"
    path.write_text(json.dumps(case))
    result=validate_final_case(path)
    assert result["valid"] and result["validated_only"]
    assert result["calculation_kind"]=="STATIC"
    assert result["solver_backend"]=="INDEPENDENT_"+method
    assert result["physical_steps_computed"]==0
    assert not (tmp_path/"not-created").exists()


@pytest.mark.parametrize("method",("MKR","MVI"))
def test_public_static_dispatch_never_calls_opposite_or_dynamic(
        method,monkeypatch,tmp_path):
    case=independent_static_template("CUF_TE4",method,
                                     2 if method=="MVI" else None)
    path=tmp_path/"case.json"
    path.write_text(json.dumps(case))
    chosen=mk if method=="MKR" else mv
    events=[]
    def own_run(path,*,validate_only):
        events.append((path,validate_only))
        return {"method":chosen.METHOD,"status":"CONVERGED",
                "converged":True,"calculation_kind":"STATIC"}
    def wrong(*a,**kw):
        raise AssertionError("CROSS_METHOD_OR_DYNAMIC_ROUTE_USED")
    monkeypatch.setattr(chosen,"run",own_run)
    monkeypatch.setattr(mv if method=="MKR" else mk,"run",wrong)
    monkeypatch.setattr(md,"run",wrong)
    monkeypatch.setattr(vd,"run",wrong)
    result=run_final_case(path)
    assert result["converged"] is True
    assert result["solver_backend"]=="INDEPENDENT_"+method
    assert result["calculation_kind"]=="STATIC"
    assert events==[(path.resolve(),False)]


@pytest.mark.parametrize("method",("MKR","MVI"))
def test_static_rejects_foreign_schema_dynamic_and_micro(method,tmp_path):
    doc=independent_static_template("CUF_TE1",method,
                                     1 if method=="MVI" else None)
    path=tmp_path/"test.json"
    path.write_text(json.dumps(doc))
    selected=mk if method=="MKR" else mv
    opposite=mv if method=="MKR" else mk
    with pytest.raises(ValueError,match="CASE_KEYS_INVALID"):
        opposite.load_case(path)
    with pytest.raises(ValueError,match="CASE_KEYS_INVALID"):
        (md if method=="MKR" else vd).load_case(path)
    doc["physics"]["gradient_model"]="G3D"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="PHYSICS_KEYS_INVALID"):
        selected.load_case(path)


@pytest.mark.parametrize("method",("MKR","MVI"))
def test_method_specific_stationary_outputs_are_atomic(
        method,tmp_path,monkeypatch):
    case=independent_static_template("CUF_TE3",method,
                                    2 if method=="MVI" else None)
    folder=tmp_path/"method_result"
    case["output_dir"]=str(folder)
    path=tmp_path/"case.json"
    path.write_text(json.dumps(case))
    chosen=mk if method=="MKR" else mv
    calls=[]
    class FakeSolver:
        def __init__(self,physics,*args):
            calls.append("construct")
            self.n=12
            self.case=physics
        def solve_static(self,factor=1.,**kwargs):
            calls.append("own_static")
            q=np.zeros((self.n,self.case.nx,self.case.ns))
            q[8,self.case.nx//2,self.case.ns//2]=1.e-5*factor
            if method=="MKR":
                return q
            X=np.zeros((self.n,2,self.case.nx))
            Y=np.zeros((self.n,2,self.case.ns))
            X[8,0,self.case.nx//2]=1.e-5*factor
            Y[8,0,self.case.ns//2]=1.
            return q,(X,Y),((1,"X"),(1,"Y"))
    monkeypatch.setattr(chosen,"IndependentStrongModel"
                        if method=="MKR" else "IndependentVariationalModel",
                        FakeSolver)
    result=chosen.run(path,validate_only=False)
    assert result["converged"] and result["status"]=="CONVERGED"
    assert calls==["construct","own_static"]
    assert (folder/f"{method}_STATIC_SUMMARY.json").is_file()
    assert (folder/f"{method}_STATIC_OWN_CASE.json").is_file()
    if method=="MKR":
        assert (folder/"MKR_STATIC_FIELD.npz").is_file()
        assert not list(folder.glob("MVI*"))
    else:
        assert (folder/"MVI_R2_STATIC_FIELD.npz").is_file()
        assert (folder/"MVI_R2_STATIC_FACTORS.npz").is_file()
        assert not list(folder.glob("MKR*"))


def test_success_booleans_present_in_both_dynamic_method_runners():
    import inspect
    assert 'summary["converged"]=True' in inspect.getsource(md.run)
    assert 'summary["converged"]=True' in inspect.getsource(vd.run)
