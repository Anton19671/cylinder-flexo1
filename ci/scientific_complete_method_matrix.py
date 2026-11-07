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
    rows=micro+macro_catalog()+independent_catalog()
    if (len(micro),len(macro_catalog()),len(independent_catalog()))!=(
            112,84,189) or len({r["id"] for r in rows})!=385:
        raise ValueError("SCIENTIFIC_METHOD_INVENTORY_INCOMPLETE")
    return rows


def export(folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    entries=catalog()
    data={
        "count":len(entries),
        "family_counts":{"MICRO_REFERENCE":112,"MACRO_REFERENCE":84,
                         "INDEPENDENT_LINEAR_MACRO":189},
        "validation_level":"CASE_PREFLIGHT_ONLY",
        "physical_solutions_computed":False,
        "scientific_accuracy_certified":False,
        "coupled_fdm_mvi_spatial_independence_certified":False,
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
