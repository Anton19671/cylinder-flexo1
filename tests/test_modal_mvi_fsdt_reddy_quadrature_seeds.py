"""FSDT/Reddy rank-one modal seed coverage, without a solver run.

Mechanical field order is (u,v,w,theta_x,theta_s). The circumferential
displacement v and director rotation theta_s need the sine/cosine partner
opposite to w for low-bending shell modes. Initial seeds do not constrain
the subsequent full variational Y/X halfsteps.
"""
from types import SimpleNamespace

import pytest

from cylinder_modal_mvi import (
    _circumferential_seed_components, _modal_seed_plan,
)


def _seeds(theory, *, rank, requested, nx=13, ns=13):
    ncomp = 5 if theory in ("FSDT", "REDDY") else 6
    functional = SimpleNamespace(theory=theory, n_mechanical=ncomp)
    operator = SimpleNamespace(
        f=functional, rank=rank, n_components=ncomp, nx=nx, ns=ns,
    )
    return _modal_seed_plan(operator, mode=6, requested=requested)


@pytest.mark.parametrize("theory", ("FSDT", "REDDY"))
def test_director_quadrature_components_are_v_and_theta_s(theory):
    functional = SimpleNamespace(theory=theory, n_mechanical=5)
    assert _circumferential_seed_components(functional) == (1, 4)


@pytest.mark.parametrize("theory", ("FSDT", "REDDY"))
def test_all_nonzero_harmonics_through_six_retain_both_initial_phase_options(theory):
    seeds = _seeds(theory, rank=1, requested=8)
    for n in range(1, 7):
        for phase in ("cos", "sin"):
            kinds = {
                s.get("component_phase", "uniform")
                for s in seeds
                if s["kind"] == "harmonic_partner"
                and s["n"] == n and s["phase"] == phase
            }
            assert kinds == {"uniform", "circumferential_quadrature"}


@pytest.mark.parametrize("theory", ("FSDT", "REDDY"))
def test_eight_mode_high_harmonic_starts_are_unique_and_not_a_fourier_solver(theory):
    seeds = _seeds(theory, rank=1, requested=8)
    # Every restart must be traceable and deterministic, including the
    # additional phase partners at n=5 and n=6.
    indices = [s["index"] for s in seeds]
    assert len(indices) == len(set(indices))
    assert all(s["kind"] in ("random", "harmonic_partner") for s in seeds)
    assert all(s.get("n", 0) <= 6 for s in seeds)


@pytest.mark.parametrize("theory", ("FSDT", "REDDY"))
def test_rank_two_remains_on_original_seed_bank_without_forced_director_phase(theory):
    seeds = _seeds(theory, rank=2, requested=8)
    assert all("component_phase" not in s for s in seeds)
    for n in (5, 6):
        for phase in ("cos", "sin"):
            assert any(s.get("n") == n and s.get("phase") == phase
                       for s in seeds)


def test_cammalleri_cuf_te1_seed_bank_is_not_changed():
    seeds = _seeds("CUF_TE1", rank=1, requested=12, ns=21)
    for n in (5, 6):
        assert all("component_phase" not in s
                   for s in seeds if s.get("n") == n)
    for n in (7, 8):
        matches = [s for s in seeds if s.get("n") == n]
        assert len(matches) == 2
        assert all(s.get("component_phase") == "circumferential_quadrature"
                   for s in matches)


def test_unknown_director_layout_does_not_silently_change_fields():
    with pytest.raises(ValueError, match="DIRECTOR_COMPONENT_ORDER"):
        _circumferential_seed_components(
            SimpleNamespace(theory="REDDY", n_mechanical=6))
