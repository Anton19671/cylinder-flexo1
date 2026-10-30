"""Bounded 17x17 Fourier modal MVI regression (no large nodal trial basis)."""
import json
from pathlib import Path
import numpy as np
import pytest
from cylinder_modal_reference import prepare_modal_case, _modal_spec
from cylinder_modal_large_harmonic import mechanical_fixed_trace_mask
from cylinder_modal_harmonic_spectrum import (
    _harmonic_variation_directions, _harmonic_phase_basis)


def _case(nx=17,rank=8):
    root=Path(__file__).resolve().parents[1]
    p=(root/"config/verification/rawat2016/modal_linear/sbp42_parity_check/"
       "RAWAT2016_CUF_TE4_13x13_MODAL_MVI1_SBP42.json")
    case=json.loads(p.read_text(encoding="utf8"))
    case["mvi_grid"].update(nx=nx,ns=nx)
    case["modal"]["n_modes"]=8
    case["rank"]=rank
    return case


@pytest.mark.parametrize("rank",[1,3,8])
def test_preflight_17x17_fourier_mvi_bounded(rank):
    passport=prepare_modal_case(_case(rank=rank))
    assert passport["method"]=="MVI"
    assert passport["rank"]==rank
    assert passport["mechanical_free_dofs"]==3825
    assert passport["operator_scheme"]=="SBP42_FOURIER_ODD"


def test_17x17_constraints_are_exact_simple_zero_clamps():
    case=_case(rank=1)
    prepared,density,n_modes=_modal_spec(case)
    mask=mechanical_fixed_trace_mask(prepared.problem,prepared.mechanical)
    assert mask.shape==(15*17*17,)
    assert int(mask.sum())==510
    per_comp=mask.reshape(15,17,17)
    assert np.all(per_comp[:,0,:])
    assert np.all(per_comp[:,-1,:])
    assert not np.any(per_comp[:,1:-1,:])


def test_17x17_cap_does_not_change_for_general_mvi():
    case=_case(rank=8)
    case["mvi_grid"]["operator_scheme"]="LEGACY_REFERENCE"
    with pytest.raises(ValueError,match="surface node count exceeds"):
        prepare_modal_case(case)


def test_harmonic_basis_17_keeps_exact_sector():
    rng=np.random.default_rng(2026)
    x=rng.normal(size=(15,8,17))
    phases=_harmonic_phase_basis(17,5)
    y=np.repeat(phases[None,:,:],15,axis=0)
    y=np.concatenate([y]*4,axis=1)
    for axis in ("X","Y"):
        b=_harmonic_variation_directions(x,y,axis,5)
        assert b.shape[0]==4335
        assert 0<b.shape[1]<=510
        assert np.isfinite(b).all()
