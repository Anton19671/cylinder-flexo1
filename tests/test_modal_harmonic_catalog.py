"""Fourier-family output contracts; do not launch the physical solver."""
import csv
import numpy as np
import pytest

from cylinder_modal_harmonic_catalog import (
    CATALOG_COLUMNS, harmonic_family_catalog, write_harmonic_family_catalog)


def _fields():
    theta = 2 * np.pi * np.arange(9) / 9
    x = np.array([0., .5, 1., .5, 0.])
    q = np.zeros((2, 6, 5, 9), dtype=float)
    q[0, 4] = np.outer(x, np.cos(2 * theta))
    q[1, 4] = np.outer(x, np.sin(2 * theta))
    return q


def test_catalog_contains_all_computed_fourier_families_not_just_saved_modes():
    result = {"method": "MVI", "rank": 4, "theory": "CUF_TE1",
              "frequencies_hz": [520., 520.], "mechanical_modes": _fields(),
              "circumferential_wavenumbers": [2, 2],
              "axial_mode_indices": [1, 1],
              "harmonic_family_scan": [
                  {"n": n, "axial_mode_index": 1, "frequency_hz": 500+n*10,
                   "degeneracy": 1 if n==0 else 2,
                   "axial_mode_index_verified": False} for n in range(5)]}
    rows, scope = harmonic_family_catalog(result)
    assert len(rows) == 5
    assert [r["circumferential_n"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[2]["saved_mode_numbers"] == "1,2"
    assert rows[2]["status"] == "COMPUTED_SAVED"
    assert rows[4]["frequency_hz"] == 540
    assert rows[4]["status"] == "COMPUTED_NOT_SAVED"
    assert "NOT_GLOBAL_SPECTRUM" in scope
    assert all(r["axial_index_verified"] is False for r in rows)


def test_fdm_does_not_invent_eigenfrequencies_for_unsolved_harmonics():
    result = {"method": "FDM", "theory": "CUF_TE1",
              "frequencies_hz": [520., 520.], "mechanical_modes": _fields()}
    rows, scope = harmonic_family_catalog(result)
    assert [r["circumferential_n"] for r in rows] == [0,1,2,2,3,4]
    assert all(r["frequency_hz"] == "" for r in rows if r["circumferential_n"] != 2)
    assert rows[2]["saved_mode_numbers"] == "1"
    assert rows[3]["saved_mode_numbers"] == "2"
    assert all(r["axial_candidate_index"] == "" for r in rows)
    assert "RETAINED_MODES" in scope


def test_csv_catalog_roundtrip(tmp_path):
    rows, _ = harmonic_family_catalog({"method": "FDM", "theory": "CUF_TE1",
              "frequencies_hz": [520., 520.], "mechanical_modes": _fields()})
    path=tmp_path/"families.csv"
    write_harmonic_family_catalog(path, rows)
    with path.open(encoding="utf-8-sig",newline="") as f:
        result=list(csv.DictReader(f))
    assert tuple(result[0]) == CATALOG_COLUMNS
    assert result[0]["frequency_hz"] == ""
    assert result[2]["status"] == "SAVED_MODE_NOT_INDEPENDENT_SECTOR_SCAN"


def test_invalid_modal_data_rejected():
    with pytest.raises(ValueError, match="INVALID_EIGENMODES"):
        harmonic_family_catalog({"mechanical_modes": np.full((1,6,5,9),np.nan),
                                 "frequencies_hz":[100.]})


def test_standard_modal_run_writes_coverage_and_does_not_invent_frequencies(tmp_path, monkeypatch):
    from pathlib import Path
    from cylinder_modal_reference import run_modal_case
    import cylinder_modal_reference as ref

    q = _fields()
    output = tmp_path / "result"
    cfg = {
        "output_dir": str(output), "method": "FDM", "theory": "CUF",
        "cuf_model": "TE1", "scale_model": "MACRO", "geometry_model": "LINEAR",
        "material_model": "HOOKE", "geometry": {"L": .305, "R": .076, "h": .000254},
        "material": {"E0": 207e9, "nu0": .3},
        "modal": {"n_modes": 2, "density_kg_m3": 7833},
        "fdm_grid": {"nx": 5, "ns": 9, "operator_scheme": "SBP42_FOURIER_ODD"},
        "quadrature": {"type": "GAUSS", "n_gauss": 9},
        "boundary_conditions": {"variational": {"mechanical": {
            "left": {"support": "CLAMPED"}, "right": {"support": "CLAMPED"}}}},
    }
    result = {"theory": "CUF_TE1", "method": "FDM", "density_kg_m3": 7833,
              "frequencies_hz": np.array([520., 520.]),
              "omega_rad_s": 2*np.pi*np.array([520., 520.]),
              "omega_squared_rad2_s2": (2*np.pi*np.array([520., 520.]))**2,
              "mechanical_modes": q, "electrical_modes": None,
              "n_mechanical_free_dofs": 100,
              "n_electric_free_dofs": 0}
    monkeypatch.setattr(ref, "_effective_modal_case", lambda case: case)
    monkeypatch.setattr(ref, "prepare_modal_case", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref, "solve_modal_case", lambda *args, **kwargs: result)
    casepath = tmp_path / "CASE.json"
    casepath.write_text("{}", encoding="utf-8")
    summary = run_modal_case(cfg,casepath)
    assert summary["harmonic_catalog_scope"].startswith("RETAINED_MODES_")
    assert summary["harmonic_frequency_completeness_verified"] is False
    with (output / "HARMONIC_FAMILY_CATALOG.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["circumferential_n"] == "0"
    assert rows[-1]["circumferential_n"] == "4"
    assert (output / "00_МОДАЛЬНЫЙ_АНАЛИЗ_ЧАСТОТЫ_ФОРМЫ.xlsx").is_file()
