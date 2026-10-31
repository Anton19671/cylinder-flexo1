"""Static 21x63 MODAL MVI limits; no physical shell solve."""
import copy
import numpy as np
import pytest
from cylinder_modal_large_harmonic import (
    bounded_harmonic_modal_grid, projected_harmonic_y_basis)
from cylinder_reference_method_switch import selected_case


def _case():
    return {
        "solver_backend":"VARIATIONAL_REFERENCE", "method":"FDM",
        "fdm_backend":"MATRIX_FREE_GMRES", "calculation_kind":"MODAL",
        "scale_model":"MACRO", "theory":"CUF", "cuf_model":"TE4",
        "fdm_grid":{"nx":21,"ns":63,
                    "operator_scheme":"SBP42_FOURIER_ODD_CSR"},
        "reference_newton":{"max_total_dofs":100000,
                            "max_surface_nodes":8192,
                            "max_constraint_rows":20000},
        "modal":{"density_kg_m3":7850., "n_modes":8}}


@pytest.mark.parametrize("rank",range(1,9))
def test_switch_mkr_to_mvi_21x63_preserves_grid_and_rank(rank):
    before=_case()
    untouched=copy.deepcopy(before)
    after=selected_case(before,"MVI",rank)
    assert after["rank"]==rank
    assert after["mvi_grid"]=={"nx":21,"ns":63,
                              "operator_scheme":"SBP42_FOURIER_ODD"}
    assert after["fdm_backend"]=="DENSE_REFERENCE"
    assert after["modal"]==before["modal"]
    assert before==untouched
    inverse=selected_case(after,"MKR")
    assert inverse["fdm_grid"]==before["fdm_grid"]


def test_boundaries_odd_grid_only():
    assert bounded_harmonic_modal_grid(21,63,15)
    assert not bounded_harmonic_modal_grid(21,64,15)
    assert not bounded_harmonic_modal_grid(21,65,15)
    assert not bounded_harmonic_modal_grid(22,63,15)
    assert not bounded_harmonic_modal_grid(21,63,16)


def test_compact_y_space_enforces_both_ends_exactly():
    rng=np.random.default_rng(11)
    x=rng.standard_normal((1,8,21))
    mask=np.zeros((1,21,63),dtype=bool)
    mask[:,0,:]=True
    mask[:,-1,:]=True
    basis=projected_harmonic_y_basis(x,mask.ravel(),
                                      np.ones(mask.size),3)
    assert basis.shape==(mask.size,16)
    assert np.max(np.abs(basis[mask.ravel()]))<1.e-13
    np.testing.assert_allclose(basis.T@basis,np.eye(16),atol=1.e-12)
