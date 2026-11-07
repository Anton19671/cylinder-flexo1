"""All supported cylinder solver combinations: CASE-only inventory.

No physical load/time step is advanced by this module.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
from cylinder_paths import activate
activate()
from ci.scientific_real_coupled_case import (
    THEORIES, catalog as micro_catalog, build_case as build_micro_case)
from independent_mkr.run_models import SCHEMA as MKR_SCHEMA
from independent_mvi.run_models import SCHEMA as MVI_SCHEMA

GEOMETRIES=("LINEAR","VON_KARMAN","GREEN_LAGRANGE")
KINDS=("STATIC","DYNAMIC")
METHODS=("FDM","MVI")
INTEGRATORS=("BATHE","NEWMARK","GAUSS_IRK4")
ROOT=Path(__file__).resolve().parents[1]
TEMPLATES=ROOT/"config"/"verification"/"independent_all_models"


def macro_catalog():
    return [dict(
        id="_".join(("MACRO",theory,geo,kind,method)),
        family="MACRO_REFERENCE",theory=theory,geometry_model=geo,
        kind=kind,method=method,integrator=(
            "NEWMARK" if kind=="DYNAMIC" else None),
        gradient_model=None,flexoelectric_enabled=False,
        rank=1 if method=="MVI" else None,
        validation_level="CASE_PREFLIGHT_ONLY",
    ) for theory,geo,kind,method in itertools.product(
        THEORIES,GEOMETRIES,KINDS,METHODS)]


def macro_case(row,folder):
    base=build_micro_case(dict(
        theory=row["theory"],gradient="Gz",on=False,
        kind=row["kind"],method=row["method"]),Path(folder))
    base["scale_model"]="MACRO"
    base["geometry_model"]=row["geometry_model"]
    base["mode"]=("LINEAR_ELASTIC" if row["geometry_model"]=="LINEAR"
                  else "NONLINEAR_ELASTIC")
    base.pop("micro",None)
    base["boundary_conditions"]["variational"]["electrical"]={}
    return base


def independent_catalog():
    rows=[]
    for theory,integrator in itertools.product(THEORIES,INTEGRATORS):
        for method,rank in [("MKR",None)]+[("MVI",r) for r in range(1,9)]:
            name=f"INDEPENDENT_{method}_{theory}_{integrator}"
            if rank is not None:name+=f"_R{rank}"
            rows.append(dict(
                id=name,family="INDEPENDENT_LINEAR_MACRO",
                theory=theory,geometry_model="LINEAR",kind="DYNAMIC",
                method=method,integrator=integrator,rank=rank,
                gradient_model=None,flexoelectric_enabled=False,
                validation_level="CASE_PREFLIGHT_ONLY"))
    return rows


def independent_template(theory,integrator,method):
    if (theory not in THEORIES or integrator not in INTEGRATORS or
        method not in ("MKR","MVI")):
        raise ValueError("UNSUPPORTED_INDEPENDENT_TEMPLATE")
    content=json.loads((TEMPLATES/f"{method}_{theory}_{integrator}.json"
                        ).read_text(encoding="utf-8"))
    if (content.get("schema")!=(
            MKR_SCHEMA if method=="MKR" else MVI_SCHEMA) or
        content["physics"]["theory"]!=theory or
        content["time"]["integrator"]!=integrator):
        raise ValueError("INDEPENDENT_TEMPLATE_PASSPORT_MISMATCH")
    return content



def independent_static_catalog():
    """Seven-theory, method-owned LINEAR MACRO stationary CASE-only routes."""
    rows=[]
    for theory in THEORIES:
        for method,rank in (("MKR",None),)+tuple(
                ("MVI",r) for r in range(1,9)):
            ident=f"INDEPENDENT_STATIC_{method}_{theory}"
            if rank is not None:
                ident+=f"_R{rank}"
            rows.append(dict(
                id=ident,family="INDEPENDENT_STATIC_LINEAR_MACRO",
                theory=theory,geometry_model="LINEAR",kind="STATIC",
                method=method,integrator=None,rank=rank,
                gradient_model=None,flexoelectric_enabled=False,
                validation_level="CASE_PREFLIGHT_ONLY"))
    return rows


def independent_static_template(theory,method,rank=None):
    """Re-use an established physical passport, never dynamic computation."""
    if theory not in THEORIES or method not in ("MKR","MVI"):
        raise ValueError("INDEPENDENT_STATIC_TEMPLATE_UNKNOWN")
    if method=="MKR" and rank is not None:
        raise ValueError("INDEPENDENT_STATIC_MKR_HAS_NO_RANK")
    if method=="MVI" and (
            type(rank) is not int or not 1<=rank<=8):
        raise ValueError("INDEPENDENT_STATIC_MVI_RANK_INVALID")
    source=independent_template(theory,"NEWMARK",method)
    doc={
        "schema":f"INDEPENDENT_{method}_STATIC_SEVEN_MODELS_V1",
        "physics":source["physics"].copy(),
        "static":{"load_factor":1.0},
        "output_dir":("output/independent_static/" + method.lower() + "/" + theory.lower() + (f"/rank_{rank}" if method=="MVI" else ""))
    }
    if method=="MVI":
        doc["rank"]=rank
        doc["static"]["max_sweeps"]=60
    return doc


def coupled_optional_time_catalog():
    """Real CASE routes beyond the Newmark pilot, not claims of real solves.

    BATHE is a public variational FDM/MVI CASE dispatch. GAUSS2_IRK4 is
    a distinct opt-in MVI-only CASE API; do not mislabel the standalone
    numerical FDM adapter as an exposed finished CASE runner.
    """
    rows=[]
    for item in micro_catalog(full=True):
        if item["kind"]!="DYNAMIC":
            continue
        for mode in ("BATHE", "GAUSS2_IRK4"):
            if mode=="GAUSS2_IRK4" and item["method"]!="MVI":
                continue
            rows.append(dict(
                id=f"MICRO_{mode}_"+item["id"],
                family=("MICRO_BATHE_CASE" if mode=="BATHE"
                        else "MICRO_GAUSS_MVI_EXPLICIT_CASE"),
                theory=item["theory"],geometry_model="GREEN_LAGRANGE",
                kind="DYNAMIC",method=item["method"],integrator=mode,
                rank=1 if item["method"]=="MVI" else None,
                gradient_model=item["gradient"],
                flexoelectric_enabled=item["on"],
                validation_level="CASE_PREFLIGHT_ONLY",
            ))
    if len(rows)!=84:
        raise ValueError("COUPLED_OPTIONAL_TIME_CASES_INCOMPLETE")
    return rows


def coupled_optional_time_case(row,folder):
    if row["family"] not in (
        "MICRO_BATHE_CASE","MICRO_GAUSS_MVI_EXPLICIT_CASE"):
        raise ValueError("OPTIONAL_COUPLED_TIME_NOT_SUPPORTED")
    entry=dict(theory=row["theory"],gradient=row["gradient_model"],
               on=row["flexoelectric_enabled"],kind="DYNAMIC",
               method=row["method"])
    case=build_micro_case(entry,Path(folder))
    if row["integrator"]=="BATHE":
        case["dynamics"]["integrator"]="BATHE"
    # IRK4 requires a genuine Newmark-compatible physical CASE, then
    # explicit prepare_gauss2_mvi_case, without inventing integrator key.
    elif row["integrator"]!="GAUSS2_IRK4" or row["method"]!="MVI":
        raise ValueError("OPTIONAL_COUPLED_TIME_METHOD_MISMATCH")
    return case


def catalog():
    micro=[dict(
        id="MICRO_"+e["id"],family="MICRO_REFERENCE",
        theory=e["theory"],geometry_model="GREEN_LAGRANGE",
        kind=e["kind"],method=e["method"],gradient_model=e["gradient"],
        flexoelectric_enabled=e["on"],
        integrator="NEWMARK" if e["kind"]=="DYNAMIC" else None,
        rank=1 if e["method"]=="MVI" else None,
        validation_level="CASE_PREFLIGHT_ONLY")
        for e in micro_catalog(full=True)]
    optional=coupled_optional_time_catalog()
    static=independent_static_catalog()
    rows=micro+optional+macro_catalog()+independent_catalog()+static
    if (len(micro),len(optional),len(macro_catalog()),
            len(independent_catalog()),len(static))!=(
            112,84,84,189,63) or len({r["id"] for r in rows})!=532:
        raise ValueError("SCIENTIFIC_METHOD_INVENTORY_INCOMPLETE")
    return rows


def export(folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    entries=catalog()
    data={
        "count":len(entries),
        "family_counts":{"MICRO_REFERENCE":112,
                         "MICRO_BATHE_CASE":56,
                         "MICRO_GAUSS_MVI_EXPLICIT_CASE":28,
                         "MACRO_REFERENCE":84,
                         "INDEPENDENT_LINEAR_MACRO":189,
                         "INDEPENDENT_STATIC_LINEAR_MACRO":63},
        "validation_level":"CASE_PREFLIGHT_ONLY",
        "physical_solutions_computed":False,
        "scientific_accuracy_certified":False,
        "coupled_fdm_mvi_spatial_independence_certified":False,
        "coupled_gauss_fdm_public_case_dispatch_available":False,
        "constraints":[
            "MICRO G3D/Gz requires GREEN_LAGRANGE in this backend",
            "Coupled Bathe uses dedicated DYNAMIC FDM/MVI CASE route",
            "Coupled Gauss2 IRK4 has an opt-in MVI CASE API only",
            "Coupled Gauss full FDM has a numerical step adapter, not a complete public case dispatch",
            "Independent MKR/MVI Bathe/Newmark/Gauss are linear elastic without electrical coupling",
            "Independent MKR/MVI STATIC has its own method-specific stationary solver, no time integrator",
            "A CASE preflight is not a completed physical calculation"
        ],
        "cases":entries,
    }
    (folder/"SCIENTIFIC_ALL_MODEL_METHOD_CASES.json").write_text(
        json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    with (folder/"SCIENTIFIC_ALL_MODEL_METHOD_CASES.csv").open(
            "w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    return data


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    result=export(options.output)
    print("CASE_PREFLIGHT_MATRIX_ONLY",result["family_counts"])
