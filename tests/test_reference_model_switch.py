"""GUI model switch regression tests; no numerical computations."""
from __future__ import annotations
import copy

import pytest

from cylinder_reference_model_switch import selected_model_case


def fixture():
    return {
        "solver_backend": "VARIATIONAL_REFERENCE",
        "theory": "CUF", "cuf_model": "TE4",
        "calculation_kind": "MODAL", "mode": "LINEAR_ELASTIC",
        "geometry_model": "GREEN_LAGRANGE", "material_model": "HOOKE",
        "method": "FDM", "fdm_grid": {"nx": 5, "ns": 5},
        "units": {"system": "SI_N_M_PA"},
        "geometry": {"L": 2.0, "R": 1.0, "h": 0.05},
        "material": {"E0": 210e9, "nu0": 0.3},
        "modal": {"density_kg_m3": 7800.0, "n_modes": 6},
        "boundary_conditions": {"variational": {"mechanical": {"left": {"support": "CLAMPED"}}}},
    }


def test_model_selection_never_mutates_source():
    original = fixture()
    snapshot = copy.deepcopy(original)
    chosen = selected_model_case(original, theory="FSDT", geometry_model="VON_KARMAN")
    assert chosen["theory"] == "FSDT"
    assert "cuf_model" not in chosen
    assert chosen["geometry_model"] == "VON_KARMAN"
    assert chosen["mode"] == "LINEAR_ELASTIC"
    for key in ("units", "geometry", "material", "boundary_conditions", "fdm_grid"):
        assert chosen[key] == original[key]
    assert original == snapshot


@pytest.mark.parametrize("order", ["TE1", "TE2", "TE3", "TE4"])
def test_cuf_order_switch(order):
    chosen = selected_model_case(fixture(), cuf_model=order)
    assert chosen["cuf_model"] == order


@pytest.mark.parametrize("geometry", ["LINEAR", "VON_KARMAN", "GREEN_LAGRANGE"])
def test_kinematics_in_static_and_modal(geometry):
    modal = selected_model_case(fixture(), geometry_model=geometry)
    assert modal["mode"] == "LINEAR_ELASTIC"
    static = fixture()
    static["calculation_kind"] = "STATIC"
    chosen = selected_model_case(static, geometry_model=geometry)
    assert chosen["mode"] == ("LINEAR_ELASTIC" if geometry == "LINEAR" else "NONLINEAR_ELASTIC")


def test_unknown_model_is_rejected_without_changing_source():
    original = fixture()
    with pytest.raises(ValueError):
        selected_model_case(original, geometry_model="MYSTERY")
    assert original["geometry_model"] == "GREEN_LAGRANGE"


def test_can_restore_cuf_after_switch_to_other_theory():
    chosen = selected_model_case(fixture(), theory="KL")
    chosen = selected_model_case(chosen, theory="CUF", cuf_model="TE3")
    assert chosen["cuf_model"] == "TE3"


def test_modal_preflight_uses_exact_gl_zero_state_tangent(monkeypatch):
    """All three modal labels must reach the same unloaded tangent contract."""
    from types import SimpleNamespace
    import cylinder_modal_reference as modal

    seen=[]
    class DummyPrepared:
        pass
    def fake_prepare(surrogate, *, base_dir=None):
        seen.append(copy.deepcopy(surrogate))
        assert surrogate["geometry_model"] == "GREEN_LAGRANGE"
        assert surrogate["mode"] == "NONLINEAR_ELASTIC"
        assert surrogate["reference_loads"]["load_factors"] == [0.0]
        return DummyPrepared()

    monkeypatch.setattr(modal, "prepare_reference_case", fake_prepare)
    for theory in ("KL", "FSDT", "REDDY", "CUF"):
        for geometry in ("LINEAR", "VON_KARMAN", "GREEN_LAGRANGE"):
            cfg=fixture()
            cfg["theory"]=theory
            if theory != "CUF":
                cfg.pop("cuf_model")
            cfg["geometry_model"]=geometry
            original=copy.deepcopy(cfg)
            prepared,density,modes=modal._modal_spec(cfg)
            assert isinstance(prepared,DummyPrepared)
            assert density == 7800.0 and modes == 6
            assert cfg == original
    assert len(seen)==12
