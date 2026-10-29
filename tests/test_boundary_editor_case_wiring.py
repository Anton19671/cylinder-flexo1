"""GUI/CASE boundary switching contract: seven theories and electrical gating."""
import copy
import pytest
from cylinder_boundary_ui_model import boundary_ui_model, theory_from_case
from cylinder_boundary_editor import merge_boundary_controls, _adapt_zero_vectors
from cylinder_variational_boundary import validate_boundary_passport

@pytest.mark.parametrize("theory,order,n",[
    ("KL",None,3),("FSDT",None,5),("REDDY",None,5),
    ("CUF","TE1",6),("CUF","TE2",9),("CUF","TE3",12),("CUF","TE4",15)])
def test_theory_to_boundary_component_count(theory,order,n):
    c={"theory":theory,"cuf_model":order,"scale_model":"MICRO",
       "micro":{"enable_flexoelectric":True,"gradient_model":"G3D","potential_order":2}}
    info=boundary_ui_model(c)
    assert len(info["components"])==n
    assert info["support_components"]["CLAMPED"]==info["components"]
    assert len(info["electric_components"])==3
    spec={"mechanical":{"left":{"support":"CLAMPED","higher":"NATURAL"},
                         "right":{"support":"SIMPLY_SUPPORTED","higher":"NATURAL"}},
          "electrical":{"inner":{"mode":"POTENTIAL","value":0.}}}
    assert validate_boundary_passport(spec,electrical_enabled=True,theory=theory_from_case(c))

@pytest.mark.parametrize("gradient",["G3D","Gz"])
def test_gradient_model_switch(gradient):
    d=boundary_ui_model({"theory":"CUF","cuf_model":"TE3","scale_model":"MICRO",
                         "micro":{"enable_flexoelectric":True,"gradient_model":gradient,
                                  "potential_order":1}})
    assert d["gradient_model"]==gradient
    assert "CHARGE" in d["electric_modes"]

def test_zero_vector_is_removed_on_theory_switch_but_nonzero_is_never_remapped():
    src={"mechanical":{"left":{"support":"CLAMPED","prescribed":[0.]*6}}}
    out=_adapt_zero_vectors(src,15)
    assert "prescribed" not in out["mechanical"]["left"]
    assert "prescribed" in src["mechanical"]["left"]
    src["mechanical"]["left"]["prescribed"][2]=1.
    with pytest.raises(ValueError,match="15"): _adapt_zero_vectors(src,15)

def test_independent_traction_and_expert_json_keys_survive_controls():
    source={"mechanical":{"left":{"support":"FREE","higher":"NATURAL",
              "traction":[1.,0.,0.],"time_law":{"kind":"CONSTANT","amplitude":1.}}},
              "electrical":{"inner":{"mode":"POTENTIAL","value":0.}}}
    before=copy.deepcopy(source)
    out=merge_boundary_controls(source,
        {"left":{"support":"CLAMPED","higher":"NATURAL"}},
        {"inner":{"mode":"POTENTIAL","value":"0.0"}},
        previous=({"left":{"support":"FREE","higher":"NATURAL"}},
                  {"inner":{"mode":"POTENTIAL","value":"0.0"}}))
    assert out["mechanical"]["left"]["support"]=="CLAMPED"
    assert out["mechanical"]["left"]["traction"]==before["mechanical"]["left"]["traction"]
    assert out["mechanical"]["left"]["time_law"]==before["mechanical"]["left"]["time_law"]
    assert source==before
