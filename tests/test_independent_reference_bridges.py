"""Method-specific physical CASE bridging: exact supported transfer only.

No shell calculations: conversion and safety rejection only.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from independent_mkr.reference_bridge import from_reference_case as into_mkr
from independent_mvi.reference_bridge import from_reference_case as into_mvi
from independent_mkr.gui import inspect_document as inspect_mkr
from independent_mvi.gui import inspect_document as inspect_mvi
from independent_mkr.run_models import load_case as load_mkr
from independent_mvi.run_models import load_case as load_mvi

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"config/verification/kl_cc_bathe/BATHE_MVI1.json"


def _source(method,theory="KL"):
    value=json.loads(SOURCE.read_text(encoding="utf-8"))
    value["method"]=method
    if theory in ("KL","FSDT","REDDY"):
        # At nu=0 the linear classical plane-stress and 3D material
        # coefficient laws agree. At nu!=0 the bridge rejects conversion.
        value["material"]["nu0"]=0.
    value["theory"]="CUF" if theory.startswith("CUF_") else theory
    if theory.startswith("CUF_"):
        value["cuf_model"]=theory[4:]
    else:
        value.pop("cuf_model",None)
    grid=value.pop("mvi_grid")
    if method=="FDM":
        value.pop("reference_mvi",None)
        value.pop("rank",None)
        value["fdm_grid"]={"nx":15,"ns":9,"operator_scheme":"SBP42_FOURIER_ODD_CSR"}
    else:
        value["mvi_grid"]={"nx":15,"ns":9,"operator_scheme":"SBP42_FOURIER_ODD"}
    return value


@pytest.mark.parametrize("method,adapter,inspect,loader",[
    ("FDM",into_mkr,inspect_mkr,load_mkr),
    ("MVI",into_mvi,inspect_mvi,load_mvi),
])
@pytest.mark.parametrize("theory",("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
def test_reference_to_independent_preserves_physics(method,adapter,inspect,loader,theory,tmp_path):
    source=_source(method,theory)
    result=adapter(source)
    assert result["physics"]["theory"]==theory
    assert result["physics"]["length_m"]==2
    assert result["physics"]["radius_m"]==1
    assert result["physics"]["thickness_m"]==.05
    assert result["physics"]["young_pa"]==210e9
    assert result["physics"]["density_kg_m3"]==7800.
    assert result["physics"]["frequency_hz"]==25.
    assert result["physics"]["pressure_pa"]==100000.
    assert result["physics"]["pressure_face"]=="MIDDLE"
    assert result["time"]=={"integrator":"BATHE","dt_s":.002,"steps":2}
    assert result["schema"].startswith("INDEPENDENT_")
    assert inspect(result).theory==theory
    if method=="MVI":assert result["rank"]==1
    else:assert "rank" not in result
    filename=tmp_path/"converted.json"
    filename.write_text(json.dumps(result),encoding="utf-8")
    assert loader(filename)[0].theory==theory
    assert _source(method,theory)==source  # no in-place mutation


@pytest.mark.parametrize("method,adapter",[
    ("FDM",into_mkr),("MVI",into_mvi)
])
@pytest.mark.parametrize("field,value",[
    ("geometry_model","GREEN_LAGRANGE"),
    ("geometry_model","VON_KARMAN"),
    ("scale_model","MICRO"),
    ("calculation_kind","MODAL"),
    ("material_model","J2"),
    ("mode","NONLINEAR_ELASTIC"),
])
def test_unsupported_source_physics_rejected(method,adapter,field,value):
    source=_source(method)
    source[field]=value
    with pytest.raises(ValueError,match="BRIDGE"):
        adapter(source)


@pytest.mark.parametrize("method,adapter",[
    ("FDM",into_mkr),("MVI",into_mvi)
])
def test_cross_method_rejected(method,adapter):
    source=_source("MVI" if method=="FDM" else "FDM")
    with pytest.raises(ValueError,match="OTHER_METHOD"):
        adapter(source)


@pytest.mark.parametrize("method,adapter",[
    ("FDM",into_mkr),("MVI",into_mvi)
])
@pytest.mark.parametrize("kind",[
    "wrong_support","wrong_load","wrong_frequency","wrong_amplitude",
    "nonzero_initial","nonzero_phase","wrong_units","wrong_damping",
    "extra_boundary","electric_trace","nonzero_start","wrong_scheme",
])
def test_non_equivalent_physics_not_silently_converted(method,adapter,kind):
    source=_source(method)
    if kind=="wrong_support":
        source["boundary_conditions"]["variational"]["mechanical"]["left"]["support"]="FREE"
    elif kind=="wrong_load":
        source["load_definition"]["face"]="TOP"
    elif kind=="wrong_frequency":
        source["dynamics"]["time_law"]["kind"]="RAMP"
    elif kind=="wrong_amplitude":
        source["dynamics"]["time_law"]["amplitude"]=2.
    elif kind=="nonzero_initial":
        source["dynamics"]["initial_conditions"]["type"]="NPZ"
    elif kind=="nonzero_phase":
        source["dynamics"]["time_law"]["phase_rad"]=.1
    elif kind=="wrong_units":
        source["units"]["system"]="N_MM_MPA"
    elif kind=="wrong_damping":
        source["dynamics"]["damping"]="RAYLEIGH"
    elif kind=="extra_boundary":
        source["boundary_conditions"]["variational"]["mechanical"]["extra"]="GEOMETRIC"
    elif kind=="electric_trace":
        source["boundary_conditions"]["variational"]["electrical"]={"left":"ZERO"}
    elif kind=="nonzero_start":
        source["dynamics"]["start_time_s"]=.1
    elif kind=="wrong_scheme":
        source["dynamics"]["integrator"]="RK45"
    with pytest.raises(ValueError,match="BRIDGE"):
        adapter(source)


@pytest.mark.parametrize("method,adapter",[
    ("FDM",into_mkr),("MVI",into_mvi)
])
def test_time_method_explicit_mapping(method,adapter):
    source=_source(method)
    source["dynamics"]["integrator"]="GAUSS2_IRK4"
    assert adapter(source)["time"]["integrator"]=="GAUSS_IRK4"
    source["dynamics"]["integrator"]="NEWMARK"
    assert adapter(source)["time"]["integrator"]=="NEWMARK"



@pytest.mark.parametrize("method,adapter",[
    ("FDM",into_mkr),("MVI",into_mvi)
])
@pytest.mark.parametrize("theory",("KL","FSDT","REDDY"))
def test_reference_classical_3d_is_not_silently_mapped_to_plane_stress(
    method,adapter,theory
):
    old=_source(method,theory)
    old["material"]["nu0"]=.3
    with pytest.raises(ValueError,match="CLASSICAL_CONSTITUTIVE_MISMATCH"):
        adapter(old)


@pytest.mark.parametrize("method,adapter",[
    ("FDM",into_mkr),("MVI",into_mvi)
])
@pytest.mark.parametrize("theory",("KL","FSDT","REDDY","CUF_TE4"))
def test_reference_pressure_face_is_retained_at_conversion(
    method,adapter,theory
):
    source=_source(method,theory)
    source["load_definition"]["face"]="OUTER"
    converted=adapter(source)
    assert converted["physics"]["pressure_face"]=="OUTER"
    source["load_definition"]["face"]="MIDDLE"
    assert adapter(source)["physics"]["pressure_face"]=="MIDDLE"
