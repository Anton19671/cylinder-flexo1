"""No shell eigenproblem: method dispatch cannot share solver results."""
from types import SimpleNamespace
import numpy as np
import pytest

from cylinder_modal_reference import solve_modal_case


def _prepared(method, rank=None, backend="DENSE_REFERENCE"):
    return SimpleNamespace(method=method, fdm_backend=backend,
                           mvi_settings=(None if rank is None else
                                         SimpleNamespace(rank=rank)))


def test_mvi_dispatch_never_calls_mkr(monkeypatch):
    import cylinder_modal_reference as dispatch
    import cylinder_modal_mvi
    import cylinder_modal_mkr
    monkeypatch.setattr(dispatch, "_modal_spec", lambda *a, **kw:
                        (_prepared("MVI", rank=3), 7850., 1))
    called = []
    def mvi(*a, **kw):
        called.append("MVI")
        return {"method": "MVI", "rank": 3,
                "mvi_modal_algorithm":
                    "fixed_rank_constrained_variational_YX_stationarity",
                "classical_unrestricted_mvi": True,
                "harmonic_sector_restriction": False,
                "mvi_sweeps": [2]}
    monkeypatch.setattr(cylinder_modal_mvi, "solve_modal_mvi", mvi)
    monkeypatch.setattr(cylinder_modal_mkr, "solve_modal_mkr",
                        lambda *a, **kw: pytest.fail("MKR executed for MVI"))
    result = solve_modal_case({"modal": {}})
    assert called == ["MVI"]
    assert result["modal_solver_branch"] == "MVI_FIXED_RANK_YX"
    assert result["rank"] == 3


def test_mkr_dispatch_never_calls_mvi(monkeypatch):
    import cylinder_modal_reference as dispatch
    import cylinder_modal_mvi
    import cylinder_modal_mkr
    monkeypatch.setattr(dispatch, "_modal_spec", lambda *a, **kw:
                        (_prepared("FDM"), 7850., 1))
    monkeypatch.setattr(cylinder_modal_mvi, "solve_modal_mvi",
                        lambda *a, **kw: pytest.fail("MVI executed for MKR"))
    monkeypatch.setattr(cylinder_modal_mkr, "solve_modal_mkr",
                        lambda *a, **kw: {
                            "method": "FDM",
                            "modal_solver_branch": "MKR_FULL_SPACE"})
    result = solve_modal_case({"modal": {}})
    assert result["modal_solver_branch"] == "MKR_FULL_SPACE"


@pytest.mark.parametrize("bad", [
    {"method": "FDM"},
    {"method": "MVI", "rank": 1},
    {"method": "MVI", "rank": 3,
     "mvi_modal_algorithm": "fixed_rank_constrained_variational_YX_stationarity",
     "classical_unrestricted_mvi": True, "harmonic_sector_restriction": False},
    {"method": "MVI", "rank": 3, "mvi_sweeps": [2],
     "mvi_modal_algorithm": "fixed_rank_fourier_restricted_variational_YX_stationarity",
     "classical_unrestricted_mvi": False, "harmonic_sector_restriction": True},
])
def test_mvi_rejects_crosscontaminated_result(monkeypatch, bad):
    import cylinder_modal_reference as dispatch
    import cylinder_modal_mvi
    monkeypatch.setattr(dispatch, "_modal_spec", lambda *a, **kw:
                        (_prepared("MVI", rank=3), 7850., 1))
    monkeypatch.setattr(cylinder_modal_mvi, "solve_modal_mvi",
                        lambda *a, **kw: bad)
    with pytest.raises(ValueError, match="CROSSCONTAMINATION"):
        solve_modal_case({"modal": {}})


def test_mkr_branch_rejects_rank_or_mvi_steps(monkeypatch):
    import cylinder_modal_matrixfree
    from cylinder_modal_mkr import solve_modal_mkr
    def poison(*a, **kw):
        return {"method": "FDM", "rank": 1, "mvi_sweeps": [1]}
    monkeypatch.setattr(cylinder_modal_matrixfree, "solve_modal_matrixfree", poison)
    with pytest.raises(ValueError, match="CROSSCONTAMINATION"):
        solve_modal_mkr(_prepared("FDM", backend="MATRIX_FREE_GMRES"), 7850., 1)
