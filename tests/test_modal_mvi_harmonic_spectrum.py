"""Fourier family search contracts and optional physical CUF TE4 verification."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from cylinder_modal_harmonic_spectrum import (
    _harmonic_phase_basis, _harmonic_variation_directions,
    _phase_quadrature_partner, _pure_harmonic_fraction)
from cylinder_modal_reference import _use_harmonic_mvi_spectrum
from cylinder_boundary_spatial import separated_profile_directions


@pytest.mark.parametrize("n",[0,1,2,3,5,6])
def test_phase_basis(n):
    b=_harmonic_phase_basis(13,n)
    np.testing.assert_allclose(b@b.T,np.eye(len(b)),atol=1e-13)
    with pytest.raises(ValueError,match="INVALID_CIRCUMFERENTIAL_HARMONIC"):
        _harmonic_phase_basis(12,n)


def test_rank8_partner_span_ignores_inactive_angular_directions():
    nx,ns=9,13
    rng=np.random.default_rng(29)
    x=rng.standard_normal((1,8,nx))
    phase=_harmonic_phase_basis(ns,2)
    y=np.stack([phase[i%2] for i in range(8)])[None,:,:]
    for axis,expected in (("X",2*nx),("Y",2*8)):
        b=_harmonic_variation_directions(x,y,axis,2)
        assert b.shape==(nx*ns,expected)
        for j in range(b.shape[1]):
            assert _pure_harmonic_fraction(b[:,j].reshape(1,nx,ns),2)>1-1e-12
    y=np.broadcast_to(_harmonic_phase_basis(ns,0),(1,8,ns)).copy()
    assert _harmonic_variation_directions(x,y,"X",0).shape==(nx*ns,nx)


def test_harmonic_rotated_partner_is_independent():
    nx,ns=11,13
    rng=np.random.default_rng(35)
    basis=_harmonic_phase_basis(ns,5)
    q=rng.normal(size=(2,nx))[:,:,None]*basis[0]+rng.normal(size=(2,nx))[:,:,None]*basis[1]
    companion=_phase_quadrature_partner(q,5)
    np.testing.assert_allclose(np.sum(q*companion),0,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(q),np.linalg.norm(companion),rtol=1e-12)
    assert _pure_harmonic_fraction(companion,5)>1-1e-12


def test_automatic_scan_only_for_uniform_macro_shell():
    case={"method":"MVI","scale_model":"MACRO",
          "mvi_grid":{"nx":13,"ns":13,"operator_scheme":"SBP42_FOURIER_ODD"},
          "boundary_conditions":{"variational":{"mechanical":{
              "left":{"support":"CLAMPED","higher":"NATURAL"},
              "right":{"support":"CLAMPED","higher":"NATURAL"}},"electrical":{}}}}
    assert not _use_harmonic_mvi_spectrum(case,8)
    case["modal"]={"mvi_spectrum_method":"FOURIER_FAMILIES"}
    assert _use_harmonic_mvi_spectrum(case,8)
    assert not _use_harmonic_mvi_spectrum(case,9)
    changed=json.loads(json.dumps(case))
    changed["scale_model"]="MICRO"
    assert not _use_harmonic_mvi_spectrum(changed,8)
    changed=json.loads(json.dumps(case))
    changed["boundary_conditions"]["variational"]["mechanical"]["left"]["angle_mask"]=[0,1]
    assert not _use_harmonic_mvi_spectrum(changed,8)


@pytest.mark.skipif(os.environ.get("CYL_MODAL_PHYSICS_TEST")!="1",
                    reason="set CYL_MODAL_PHYSICS_TEST=1 for full CUF TE4 modal calculation")
@pytest.mark.parametrize("rank",range(1,9))
def test_rawat_first_eight_modes(rank):
    from cylinder_modal_reference import solve_modal_case
    root=Path(__file__).resolve().parents[1]
    source=root/"config/verification/rawat2016/modal_linear/sbp42_parity_check/RAWAT2016_CUF_TE4_13x13_MODAL_MVI1_SBP42.json"
    case=json.loads(source.read_text(encoding="utf-8"))
    case["rank"]=rank
    case["modal"]["n_modes"]=8
    # The Rawat frequency table is a Fourier-family benchmark, never an
    # implicit selection of the unrestricted Y/X modal algorithm.
    case["modal"]["mvi_spectrum_method"]="FOURIER_FAMILIES"
    result=solve_modal_case(case,base_dir=source.parent)
    np.testing.assert_allclose(result["frequencies_hz"],
        [271.09764224,271.09764224,278.28159831,278.28159831,
         344.20139376,344.20139376,352.23732570,352.23732570],
        rtol=0,atol=0.001)
    assert result["circumferential_wavenumbers"]==[3,3,4,4,2,2,5,5]
