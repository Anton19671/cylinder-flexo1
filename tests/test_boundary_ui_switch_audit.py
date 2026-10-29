"""V-10 graphical boundary selectors must follow actual solver field ordering."""
import pytest
from cylinder_boundary_ui_model import boundary_ui_model

@pytest.mark.parametrize("theory,order,expected",[
    ("KL",None,("u0","v0","w0")),
    ("FSDT",None,("u0","v0","w0","theta_x","theta_s")),
    ("REDDY",None,("u0","v0","w0","theta_x","theta_s")),
    ("CUF","TE1",("u0","u1","v0","v1","w0","w1")),
    ("CUF","TE2",("u0","u1","u2","v0","v1","v2","w0","w1","w2")),
    ("CUF","TE3",tuple(f"{c}{i}" for c in ("u","v","w") for i in range(4))),
    ("CUF","TE4",tuple(f"{c}{i}" for c in ("u","v","w") for i in range(5)))])
def test_switch_theory_controls(theory,order,expected):
    case={"theory":theory,"cuf_model":order,"scale_model":"MICRO",
          "micro":{"enable_flexoelectric":True}}
    p=boundary_ui_model(case)
    assert p["components"]==expected
    assert p["support_components"]["CLAMPED"]==expected
    assert p["support_components"]["SIMPLY_SUPPORTED"]==(
        "u0","v0","w0")
    assert p["electric_enabled"] is True
    if theory=="CUF":
        assert "FIX_FIRST_AND_SECOND_NORMAL" not in p["higher_choices"]
    else:
        assert "FIX_FIRST_AND_SECOND_NORMAL" in p["higher_choices"]

@pytest.mark.parametrize("scale,flexo,enabled",[
    ("MICRO",True,True),("MICRO",False,False),
    ("MACRO",True,False),("MACRO",False,False)])
def test_electric_controls_switch_with_physics(scale,flexo,enabled):
    p=boundary_ui_model({"theory":"KL","scale_model":scale,
                         "micro":{"enable_flexoelectric":flexo}})
    assert p["electric_enabled"] is enabled
    assert ("CHARGE" in p["electric_modes"]) is enabled
    assert bool(p["electric_sites"]) is enabled

def test_guided_and_symmetry_component_mapping():
    p=boundary_ui_model({"theory":"CUF","cuf_model":"TE2"})
    assert p["support_components"]["GUIDED"]==(
        "v0","v1","v2","w0","w1","w2")
    assert p["support_components"]["SYMMETRY"]==("u0","u1","u2")
    p=boundary_ui_model({"theory":"KL"})
    assert p["support_components"]["SYMMETRY"]==("u0",)

def test_bad_model_rejected():
    with pytest.raises(ValueError):
        boundary_ui_model({"theory":"CUF","cuf_model":"TE5"})
