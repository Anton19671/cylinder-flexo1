"""Manufactured polynomial symbols; no shell-equation simulations are run."""
import numpy as np
import pytest
from cylinder_coupled_principal_symbol import (
    CoupledJetLayout,MECHANICAL_ORDERS,THEORIES,
    frozen_symbol_from_tangent,finite_difference_local_jet_tangent,
)


def test_theory_field_and_jet_counts():
    counts={"KL":(3,10),"FSDT":(5,10),"REDDY":(5,10),
            "CUF_TE1":(6,6),"CUF_TE2":(9,6),"CUF_TE3":(12,6),"CUF_TE4":(15,6)}
    assert len(THEORIES)==7
    for theory,(nfield,njet) in counts.items():
        layout=CoupledJetLayout.for_theory(theory,5)
        assert layout.n_mechanical==nfield
        assert len(layout.mechanical_orders)==njet
        assert layout.n_electric==5
        assert layout.n_jets==nfield*njet+15


def test_frozen_coupled_saddle_has_hermitian_reciprocal_cross_blocks():
    layout=CoupledJetLayout(1,1,((0,0),(2,0)))
    # L = D*q_xx^2/2 - kappa*phi_x^2/2 + mu*q_xx*phi_x
    h=np.zeros((layout.n_jets,layout.n_jets))
    qm=1
    phix=layout.n_mechanical*len(layout.mechanical_orders)+1
    h[qm,qm]=4.
    h[phix,phix]=-2.
    h[qm,phix]=h[phix,qm]=.8
    symbol=frozen_symbol_from_tangent(layout,h)
    assert symbol.reciprocity_rel_error==0
    p=symbol.evaluate(2.,0.)
    np.testing.assert_allclose(p,p.conj().T,atol=1e-12)
    np.testing.assert_allclose(p[0,0],4.*2.**4)
    np.testing.assert_allclose(p[1,1],-2.*2.**2)
    np.testing.assert_allclose(p[0,1],-1j*.8*2.**3)
    assert symbol.pairwise_highest_degrees(2,0).tolist()==[[4,3],[3,2]]
    # No hand-waving away the odd imaginary off-diagonal term.
    assert 3 in symbol.homogeneous_degrees(2.,0.)
    np.testing.assert_allclose(symbol.evaluate(1.,0.),
                               sum(symbol.homogeneous_degrees(1.,0.).values()))


def test_mixed_derivative_occurs_once_without_spurious_two():
    layout=CoupledJetLayout(1,0,((0,0),(1,1)))
    H=np.diag([0.,7.])
    symbol=frozen_symbol_from_tangent(layout,H)
    # L = 7*q_xs^2/2 => P = 7*kx^2*ks^2, with no factor 2.
    np.testing.assert_allclose(symbol.evaluate(2.,3.)[0,0],7.*2**2*3**2)
    np.testing.assert_allclose(symbol.evaluate(0.,3.)[0,0],0.)


def test_full_linearization_includes_lower_degree_terms():
    layout=CoupledJetLayout(1,0,((0,0),(1,0),(2,0)))
    H=np.diag([9.,5.,3.])
    symbol=frozen_symbol_from_tangent(layout,H)
    terms=symbol.homogeneous_degrees(2,0)
    assert set(terms)=={0,2,4}
    np.testing.assert_allclose(symbol.evaluate(2.,0.)[0,0],9+5*4+3*16)
    np.testing.assert_allclose(symbol.pairwise_highest_degrees(2,0)[0,0],4)


def test_finite_difference_uses_original_local_gradient_without_symmetrizing():
    layout=CoupledJetLayout(1,1,((0,0),(1,0)))
    rng=np.random.default_rng(1234)
    a=rng.normal(size=(layout.n_jets,layout.n_jets))
    H=a+a.T+np.eye(layout.n_jets)*3.
    x0=rng.normal(size=layout.n_jets)
    symbol=finite_difference_local_jet_tangent(
        layout,lambda x:H@x, state=x0,jet_scales=np.linspace(.5,2.,layout.n_jets),
        relative_step=1e-5,reciprocity_tolerance=1e-8)
    np.testing.assert_allclose(symbol.jet_tangent,H,atol=1e-9,rtol=1e-9)
    assert symbol.reciprocity_rel_error<1e-9


def test_reject_nonreciprocal_coupled_tangent_and_invalid_inputs():
    layout=CoupledJetLayout(1,1,((0,0),(2,0)))
    H=np.zeros((layout.n_jets,layout.n_jets))
    H[1,-2]=3
    with pytest.raises(ValueError,match="RECIPROCITY"):
        frozen_symbol_from_tangent(layout,H)
    with pytest.raises(ValueError,match="nonzero"):
        frozen_symbol_from_tangent(layout,np.eye(layout.n_jets)).evaluate(0.,0.)
    with pytest.raises(ValueError,match="unique"):
        CoupledJetLayout(1,0,((0,0),(1,1),(1,1)))
    with pytest.raises(ValueError,match="state and positive"):
        finite_difference_local_jet_tangent(layout,lambda z:z,jet_scales=np.zeros(layout.n_jets))


def test_mechanical_gz_like_and_g3d_like_principal_directions_are_distinct():
    # Manufactured different constitutive Hessians mimic expected suppression
    # of surface-gradient terms in Gz, without claiming a shell certification.
    layout=CoupledJetLayout(1,0,((0,0),(2,0),(1,1),(0,2)))
    h3d=np.diag([0.,2.,3.,4.])
    hz=np.diag([0.,2.,0.,0.])
    p3=frozen_symbol_from_tangent(layout,h3d)
    pz=frozen_symbol_from_tangent(layout,hz)
    assert p3.evaluate(0.,2.)[0,0].real>0
    assert pz.evaluate(0.,2.)[0,0]==0
    assert p3.evaluate(2.,0.)[0,0].real==pz.evaluate(2.,0.)[0,0].real


def test_zero_polynomial_and_nonzero_state_scales_guard():
    layout=CoupledJetLayout(1,0,((0,0),(1,0)))
    symbol=frozen_symbol_from_tangent(layout,np.zeros((layout.n_jets,layout.n_jets)))
    assert np.all(symbol.pairwise_highest_degrees(1.,1.)==-1)
    with pytest.raises(ValueError,match="dimensioned jet_scales"):
        finite_difference_local_jet_tangent(layout,lambda z:z,state=np.ones(layout.n_jets))


def test_directional_audit_never_converts_mixed_symbol_into_boundary_orders():
    layout=CoupledJetLayout(1,1,((0,0),(2,0)))
    h=np.zeros((layout.n_jets,layout.n_jets))
    h[1,1]=4.
    h[1,-2]=h[-2,1]=2.
    h[-2,-2]=-3.
    symbol=frozen_symbol_from_tangent(layout,h)
    report=symbol.directional_coupling_audit()
    assert report["mechanical_electric_coupling"]
    assert report["maximum_observed_degrees"][0,1]==3
    assert report["status"].endswith("NOT_CERTIFIED")
