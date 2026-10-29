"""Regression contracts for read-only modal FULL export.

No stiffness assembly, eigenvalue solve, or modal calculation is run here.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from cylinder_reference_export import export
from cylinder_modal_full_export import CHARACTERISTICS_XLSX, FIELDS_XLSX
from cylinder_modal_student_export import EXTRA_WORKBOOKS


def test_saved_modal_full_export_creates_tables_and_figures(tmp_path, monkeypatch):
    import cylinder_modal_full_export as full
    from cylinder_modal_diagnostics import inspect_modal_shapes
    mode = np.zeros((1, 6, 5, 5), float)
    mode[0, 4] = np.array([0, 1, 2, 1, 0])[:, None]
    frequencies = np.array([100.], float)
    x = np.linspace(0, 2, 5)
    s = np.arange(5) * 2*np.pi/5
    case = {"geometry": {"L": 2., "R": 1., "h": .1},
            "modal": {"n_modes": 1}, "theory": "CUF", "cuf_model": "TE1"}
    summary = {"solver_backend": "VARIATIONAL_REFERENCE",
               "calculation_kind": "MODAL", "method": "FDM",
               "theory": "CUF_TE1", "status": "MODAL_COMPLETED"}
    (tmp_path/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(summary),encoding="utf-8")
    (tmp_path/"REFERENCE_CASE.json").write_text(json.dumps(case),encoding="utf-8")
    np.savez_compressed(tmp_path/"REFERENCE_MODAL_MODES.npz",
                        frequencies_hz=frequencies, omega_rad_s=2*np.pi*frequencies,
                        omega_squared_rad2_s2=(2*np.pi*frequencies)**2,
                        mechanical_modes=mode, x_m=x, s_m=s)

    def fake_physics(prepared, density, fields, electric, frequency, config, diagnostics):
        shape = (3, 5, 5)
        physical = {"z_m": np.array([-.05, 0, .05]),
                    "displacement3_m": np.zeros((*shape, 3)),
                    "strain6": np.zeros((*shape, 6)),
                    "stress6_pa": np.zeros((*shape, 6)),
                    "quadratic_enthalpy_j_per_m3": np.zeros(shape)}
        physical["displacement3_m"][:, :, :, 2] = fields[0,4]
        rows = [(1,100.,2*np.pi*100.,1.,(2*np.pi*100.)**2,
                 2., "W","BENDING_OR_MIXED",0,0,0,0.,None)]
        return rows, [physical], np.eye(1), np.eye(1), [(1,"W",1.,1.,100.,1.)]
    monkeypatch.setattr(full, "_modal_spec", lambda case, base_dir: (None, 1., 1))
    monkeypatch.setattr(full, "_physical_mode_data", fake_physics)
    target = export(tmp_path,"FULL")
    assert (tmp_path/CHARACTERISTICS_XLSX).is_file()
    assert (tmp_path/FIELDS_XLSX).is_file()
    for filename in EXTRA_WORKBOOKS:
        assert (tmp_path/filename).is_file(), filename
    assert not list(target.rglob("*.png"))


def test_export_missing_modal_npz_fails_explicitly(tmp_path):
    summary = {"solver_backend": "VARIATIONAL_REFERENCE",
               "calculation_kind": "MODAL"}
    (tmp_path/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(summary),encoding="utf-8")
    with pytest.raises(ValueError, match="REFERENCE_MODAL_MODES.npz"):
        export(tmp_path,"FULL")
