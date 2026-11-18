"""Reload the production CASE reader without requiring a display server."""
import json
from types import SimpleNamespace
import tkinter as tk
from unittest.mock import Mock

import numpy as np
import pytest

import FINAL_GUI as gui
import final_load_builder as loads


@pytest.mark.parametrize("sector", [None, False, True])
def test_harmonic_reload_preserves_pressure_field(tmp_path, monkeypatch, sector):
    spec = dict(type="harmonic_radial_face", m=1, n=2, q=-3, load_factors=[1],
                x1=.2, x2=.8, edge_width_x=.1, edge_width=.25, c1=0., c2=2.)
    if sector is not None:
        spec.update(full_circumference=sector, c1=0., c2=2.)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(dict(theory="KL", load_definition=spec)))
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kw: str(path))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, detail: pytest.fail(detail))
    app = Mock()
    tcl = tk.Tcl()
    monkeypatch.setattr(tk, "_default_root", tcl)
    variables = {}
    app.vars = variables
    def variable(key, default=""):
        if key not in variables:
            variables[key] = tk.StringVar(tcl, value=default)
        return variables[key]
    app._v.side_effect = variable
    app.load_full_var = tk.BooleanVar(tcl)
    gui.FinalSolverGUI.load_case(app)
    assert variable("load_face").get() == "MIDDLE"
    assert variable("load_pressure_direction").get() == "EXTERNAL"
    app._parse_factors.side_effect = lambda: gui.FinalSolverGUI._parse_factors(app)
    restored = gui.FinalSolverGUI._load_definition(app)
    assert restored["edge_width_c"] == .25
    grid = loads.mvi.CylinderGrid(1., 2., 9, 13)
    original, _, _ = loads._surface_radial_field(spec, grid)
    replayed, _, _ = loads._surface_radial_field(restored, grid)
    np.testing.assert_array_equal(replayed, original)


def test_string_flag_cannot_expand_sector_to_whole_cylinder():
    with pytest.raises(loads.LoadDefinitionError, match="boolean"):
        loads.full_circumference({"type": "PATCH_RADIAL_FACE", "full_circumference": "false"})

def test_point_location_and_layer_bounds_follow_unit_switch():
    tcl = tk.Tcl()
    values = dict(h="10", layer_bounds="-5; 0; 5", load_point_x0="250",
                  load_point_P="12", load_point_theta0="1.2",
                  load_coord="THETA", load_edge_width_c=".4", load_edge_width_x="5")
    fields = {key: tk.StringVar(tcl, value=value) for key, value in values.items()}
    app = SimpleNamespace(vars=fields, _unit_fmt=gui.FinalSolverGUI._unit_fmt)
    gui.FinalSolverGUI._convert_gui_units(app, "N_MM_MPA", "SI_N_M_PA")
    assert fields["load_point_x0"].get() == "0.25"
    assert fields["layer_bounds"].get() == "-0.005, 0, 0.005"
    assert fields["load_edge_width_x"].get() == "0.005"
    for key in ("load_point_P", "load_point_theta0", "load_edge_width_c"):
        assert fields[key].get() == values[key]
    gui.FinalSolverGUI._convert_gui_units(app, "SI_N_M_PA", "N_MM_MPA")
    assert fields["load_point_x0"].get() == "250"
    assert fields["layer_bounds"].get() == "-5, 0, 5"


@pytest.mark.parametrize(("spec","expected"), [
    ({"type":"UNIFORM_RADIAL_FACE","q":100000,"direction":"RADIAL_INWARD"}, "EXTERNAL"),
    ({"type":"POINT_RADIAL_FORCE","force":-4,"direction":"RADIAL_INWARD","s0":.4}, "INTERNAL"),
    ({"type":"LINE_RADIAL_GENERATOR","p_line":-5}, "EXTERNAL"),
])
def test_reference_view_matches_signed_radial_load(spec, expected):
    app = Mock()
    fields = {}
    tcl = tk.Tcl()
    def variable(key, default=""):
        if key not in fields:
            fields[key] = tk.StringVar(tcl, value=default)
        return fields[key]
    app._v.side_effect = variable
    app.load_full_var = tk.BooleanVar(tcl)
    variable("case_path").set("case.json")
    gui.FinalSolverGUI._populate_reference_view(app, {
        "geometry":{"R":2}, "load_definition":dict(spec, surface="MIDDLE")})
    assert variable("load_pressure_direction").get() == expected
    assert variable("load_face").get() == "MIDDLE"
    if spec["type"] == "POINT_RADIAL_FORCE":
        assert float(variable("load_point_P").get()) == 4
        assert float(variable("load_point_theta0").get()) == .2
