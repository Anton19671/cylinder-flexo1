"""Evidence-preserving REAL physical coupled-shell pilot and optional 112-case sweep.

No synthetic/fallback solution, no alteration of the interactive GUI.
The coupled FDM and MVI still share the physical weak kernel; this is
real-solution smoke verification, NOT independent spatial validation.
"""
import argparse
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import traceback
import numpy as np

from cylinder_paths import activate
activate()
from cylinder_reference_case import prepare_reference_case
from cylinder_final_stage11a_dispatcher import run_final_case

ROOT=Path(__file__).resolve().parents[1]
THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
GRADIENTS=("G3D","Gz")
PILOT=(
 ("KL","G3D",True,"STATIC","FDM"),
 ("FSDT","Gz",False,"STATIC","MVI"),
 ("REDDY","G3D",False,"DYNAMIC","FDM"),
 ("CUF_TE1","Gz",True,"DYNAMIC","MVI"),
 ("CUF_TE2","G3D",True,"STATIC","FDM"),
 ("CUF_TE3","Gz",False,"DYNAMIC","FDM"),
 ("CUF_TE4","G3D",True,"STATIC","FDM"),
 ("CUF_TE4","Gz",True,"DYNAMIC","MVI"),
)

def catalog(full=False):
    combinations=(
        itertools.product(THEORIES,GRADIENTS,(False,True),
                          ("STATIC","DYNAMIC"),("FDM","MVI"))
        if full else PILOT)
    output=[]
    for theory,grad,on,kind,method in combinations:
        ident="_".join((theory,grad.upper(),"ON" if on else "OFF",kind,method))
        output.append(dict(id=ident,theory=theory,gradient=grad,
                           on=on,kind=kind,method=method))
    assert len({entry["id"] for entry in output})==len(output)
    return output

def build_case(spec,folder):
    source=("CASE_REFERENCE_CUF_G3D_STATIC.json" if spec["kind"]=="STATIC"
            else "CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json")
    c=json.loads((ROOT/"config"/source).read_text(encoding="utf-8"))
    theory=spec["theory"];method=spec["method"]
    c["theory"]="CUF" if theory.startswith("CUF_") else theory
    if theory.startswith("CUF_"):
        c["cuf_model"]=theory[-3:]
    else:
        c.pop("cuf_model",None)
    c["quadrature"]["n_gauss"]=(2*int(theory[-1])+1
        if theory.startswith("CUF_") else 7 if theory=="REDDY" else 3)
    c["method"]=method
    c["micro"]["gradient_model"]=spec["gradient"]
    c["micro"]["enable_flexoelectric"]=spec["on"]
    if not spec["on"]:
        c["micro"]["potential_order"]=0
        c["boundary_conditions"]["variational"]["electrical"]={}
    # Reference face and constitutive passport are explicit.
    # KL offset-face dead pressure is not represented by its current nodal
    # kinematics. Do not bypass the validated restriction.
    c["load_definition"].update(face="MIDDLE" if theory=="KL" else "OUTER",
                                kinematics="DEAD_REFERENCE",q=.01)
    if method=="FDM":
        for name in ("mvi_grid","rank","reference_mvi"):
            c.pop(name,None)
        c["fdm_grid"]={"nx":4,"ns":4}
    else:
        c.pop("fdm_grid",None)
        c["mvi_grid"]={"nx":4,"ns":4}
        c["rank"]=1
        c["reference_mvi"]={"max_sweeps":30}
    if spec["kind"]=="DYNAMIC":
        c["dynamics"]["n_steps"]=1
        c["dynamics"]["time_step_s"]=.01
        c["dynamics"].pop("integrator",None)
        c["load_definition"].pop("load_factors",None)
        c["postprocess"]={"dynamic_analytics":False}
    else:
        c["load_definition"]["load_factors"]=[1.]
        c.pop("postprocess",None)
    c["output_dir"]=str((folder/"RESULTS").resolve())
    return c

def digest(path):
    hash=sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            hash.update(part)
    return hash.hexdigest()

def assert_physical_output(case,folder):
    output=folder/"RESULTS"
    summary=json.loads((output/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    theory=("CUF_"+case["cuf_model"] if case["theory"]=="CUF" else case["theory"])
    expected=dict(theory=theory,method=case["method"],
        calculation_kind=case["calculation_kind"],
        gradient_model=case["micro"]["gradient_model"].upper(),
        electrical_enabled=case["micro"]["enable_flexoelectric"],
        active_strain_gradient_components=(
            18 if case["micro"]["gradient_model"].upper()=="G3D" else 6),
        converged=True,accepted_steps=1,status="CONVERGED")
    for key,value in expected.items():
        if summary.get(key)!=value:
            raise AssertionError("PHYSICAL_SUMMARY_MISMATCH_"+key)
    steps=summary["steps"]
    if len(steps)!=1 or not steps[0].get("converged"):
        raise AssertionError("NO_COMPLETE_REAL_PHYSICAL_STEP")
    dynamic=case["calculation_kind"]=="DYNAMIC"
    expected_file=("REFERENCE_DYNAMIC_STEP_0001.npz" if dynamic
                   else "REFERENCE_STEP_0001.npz")
    if steps[0].get("state_file")!=expected_file:
        raise AssertionError("PHYSICAL_STATE_NAME_MISMATCH")
    if dynamic:
        if summary["requested_steps"]!=1 or summary["electric_inertia"] is not False:
            raise AssertionError("DYNAMIC_INERTIA_OR_STEP_INVALID")
        if not (output/"REFERENCE_DYNAMIC_INITIAL.npz").exists():
            raise AssertionError("INITIAL_STATE_MISSING")
    state=output/expected_file
    with np.load(state,allow_pickle=False) as saved:
        if "mechanical" not in saved:
            raise AssertionError("MECHANICAL_STATE_MISSING")
        q=saved["mechanical"]
        if q.ndim!=3 or q.shape[1:]!=(4,4):
            raise AssertionError("MECHANICAL_STATE_SHAPE_INVALID")
        on=case["micro"]["enable_flexoelectric"]
        if ("electrical" in saved.files)!=on:
            raise AssertionError("ELECTRICAL_STATE_SWITCH_INVALID")
        if dynamic:
            if not {"velocity","acceleration","time_s"}.issubset(saved.files):
                raise AssertionError("DYNAMIC_STATE_MISSING")
            if saved["velocity"].shape!=q.shape or saved["acceleration"].shape!=q.shape:
                raise AssertionError("DYNAMIC_STATE_SHAPE_INVALID")
            if not np.isclose(float(saved["time_s"]),.01,rtol=0,atol=1e-12):
                raise AssertionError("DYNAMIC_TIME_INCORRECT")
        for key in saved.files:
            array=saved[key]
            if np.issubdtype(array.dtype,np.number) and not np.isfinite(array).all():
                raise AssertionError("NONFINITE_SAVED_FIELD_"+key)
        if case["method"]=="MVI":
            if not {"x_profiles","y_profiles","state_scales"}.issubset(saved.files):
                raise AssertionError("SAVED_SEPARATED_FACTORS_MISSING")
            x,y=saved["x_profiles"],saved["y_profiles"]
            field=np.einsum("krx,krs->kxs",x,y,optimize=True)
            target=np.concatenate((q,saved["electrical"])) if on else q
            np.testing.assert_allclose(saved["state_scales"]*field,
                                       target,rtol=2e-8,atol=2e-11)
    return {"saved_state":expected_file,"state_sha256":digest(state),
            "physical_state_accepted":True,
            "external_physical_accuracy_verified":False,
            "coupled_spatial_methods_independent":False}

def run(spec,folder):
    folder=folder.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("SCIENTIFIC_CASE_OUTPUT_NOT_EMPTY")
    folder.mkdir(parents=True,exist_ok=True)
    case=build_case(spec,folder)
    path=folder/"CASE.json"
    path.write_text(json.dumps(case,indent=2)+"\n",encoding="utf-8")
    evidence=dict(id=spec["id"],input_sha256=digest(path),
        code_revision=os.getenv("GITHUB_SHA","UNSPECIFIED"),
        reference="Krysko-Zhong corrected 2026-10-24",
        load_face=case["load_definition"]["face"],
        status="STARTED",scientific_scope="REAL_ONE_STEP_SMOKE")
    marker=folder/"SCIENTIFIC_EVIDENCE.json"
    def save():
        marker.write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    save()
    try:
        preflight=prepare_reference_case(case,base_dir=folder)
        if preflight.method!=spec["method"] or preflight.calculation_kind!=spec["kind"]:
            raise AssertionError("PREFLIGHT_DISPATCH_MISMATCH")
        print("REAL_PHYSICAL_CASE_BEGIN",spec["id"],flush=True)
        run_final_case(path)
        evidence["physical_state"]=assert_physical_output(case,folder)
        evidence["status"]="REAL_STEP_ACCEPTED"
        save()
        print("REAL_PHYSICAL_CASE_ACCEPTED",spec["id"],flush=True)
    except BaseException as exc:
        evidence.update(status="FAILED",error_type=type(exc).__name__,
                        error_message=str(exc))
        save()
        (folder/"FAILURE_TRACEBACK.txt").write_text(
            traceback.format_exc(),encoding="utf-8")
        raise

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="command",required=True)
    listing=sub.add_parser("matrix")
    listing.add_argument("--scope",choices=("pilot","full"),default="pilot")
    runner=sub.add_parser("run")
    runner.add_argument("--id",required=True)
    runner.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.command=="matrix":
        print(json.dumps({"include":catalog(full=a.scope=="full")},separators=(",",":")))
        return
    found=[spec for spec in catalog(full=True) if spec["id"]==a.id]
    if len(found)!=1:
        raise ValueError("UNKNOWN_SCIENTIFIC_CASE")
    run(found[0],a.output)

if __name__=="__main__":
    main()
