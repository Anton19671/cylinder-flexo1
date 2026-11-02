"""CUF modal rank-one seed coverage respects thickness coefficient ordering."""
from types import SimpleNamespace

import pytest

from cylinder_modal_mvi import _circumferential_seed_components, _modal_seed_plan


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_cuf_quadrature_seed_covers_every_circumferential_coefficient(degree):
    n = degree + 1
    f = SimpleNamespace(theory=f"CUF_TE{degree}", n_mechanical=3*n)
    assert _circumferential_seed_components(f) == tuple(range(n, 2*n))
    op = SimpleNamespace(f=f, rank=1, n_components=3*n, nx=9, ns=9)
    seeds = _modal_seed_plan(op, 0, 8)
    found = {(s["n"], s["phase"]) for s in seeds
             if s.get("component_phase") == "circumferential_quadrature"}
    assert found == {(n, phase) for n in range(1, 5) for phase in ("cos", "sin")}


def test_cuf_quadrature_starts_do_not_change_higher_rank_plan():
    f = SimpleNamespace(theory="CUF_TE4", n_mechanical=15)
    op = SimpleNamespace(f=f, rank=2, n_components=15, nx=9, ns=9)
    assert all("component_phase" not in s for s in _modal_seed_plan(op, 0, 8))


def test_cuf_component_count_mismatch_is_rejected():
    f = SimpleNamespace(theory="CUF_TE4", n_mechanical=12)
    with pytest.raises(ValueError, match="INVALID_CUF_COMPONENT_ORDER"):
        _circumferential_seed_components(f)
