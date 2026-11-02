"""Cammalleri thin CUF TE1 13x21 modal seed coverage (no calculations)."""
from types import SimpleNamespace
from cylinder_modal_mvi import _modal_seed_plan


def test_requested_twelve_modes_cover_n7_and_n8_both_phases_all_ranks():
    f=SimpleNamespace(theory="CUF_TE1",n_mechanical=6)
    for rank in range(1,9):
        op=SimpleNamespace(nx=13,ns=21,rank=rank,n_components=6,f=f)
        seeds=_modal_seed_plan(op,10,12)
        covered={(row.get("n"),row.get("phase")) for row in seeds
                 if row["kind"]=="harmonic_partner"}
        for n in (5,6,7,8):
            assert (n,"cos") in covered and (n,"sin") in covered


def test_eight_mode_quick_audit_seed_count_unchanged():
    f=SimpleNamespace(theory="CUF_TE1",n_mechanical=6)
    op=SimpleNamespace(nx=13,ns=21,rank=1,n_components=6,f=f)
    short=_modal_seed_plan(op,0,8)
    long=_modal_seed_plan(op,0,12)
    assert all(x.get("n",0)<=6 for x in short)
    assert len(long)==len(short)+4


def test_only_initial_partners_not_a_fourier_solver():
    f=SimpleNamespace(theory="CUF_TE1",n_mechanical=6)
    op=SimpleNamespace(nx=13,ns=21,rank=1,n_components=6,f=f)
    seeds=_modal_seed_plan(op,11,12)
    assert all(seed["kind"] in ("random","harmonic_partner") for seed in seeds)
