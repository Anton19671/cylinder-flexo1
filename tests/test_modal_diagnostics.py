"""Modal diagnostic regression contracts; no shell solve is performed."""
import numpy as np
from cylinder_modal_diagnostics import (
    inspect_modal_shapes, recover_modal_displacements)


def test_same_frequency_independent_circumferential_modes():
    ns = 5
    theta = 2*np.pi*np.arange(ns)/ns
    fields = np.zeros((2, 15, 5, ns))
    profile = np.array([0., 1., 2., 1., 0.])
    fields[0, 10] = profile[:, None] * np.cos(theta)
    fields[1, 10] = profile[:, None] * np.sin(theta)
    result = inspect_modal_shapes(
        fields, [522., 522.], "CUF_TE4", operator_scheme="SBP21_PERIODIC_ODD")
    assert result["pairs"][0]["classification"] == "CIRCUMFERENTIAL_SINE_COSINE_DOUBLET"
    assert result["pairs"][0]["euclidean_mac"] < 1e-12


def test_duplicate_vector_is_not_mistaken_for_physical_doublet():
    ns = 5
    fields = np.zeros((2, 15, 5, ns))
    fields[:, 10, 2, :] = np.cos(2*np.pi*np.arange(ns)/ns)
    result = inspect_modal_shapes(fields, [522., 522.], "CUF_TE4")
    assert result["pairs"][0]["classification"] == "POSSIBLE_DUPLICATED_EIGENVECTOR"


def test_axisymmetric_torsion_nx5_is_flagged_without_deleting_modes():
    fields = np.zeros((2, 15, 5, 5))
    fields[0, 5, :, :] = np.array([0., 1., 0., -1., 0.])[:, None]
    fields[1, 5, :, :] = np.array([0., 1., 2., 1., 0.])[:, None]
    result = inspect_modal_shapes(
        fields, [724., 724.], "CUF_TE4", operator_scheme="SBP21_PERIODIC_ODD")
    assert len(result["modes"]) == 2
    assert result["pairs"][0]["classification"] == "CHECK_SBP21_NX5_AXIAL_DISCRETE_DEGENERACY"


def test_cuf_thickness_reconstruction():
    q = np.zeros((1, 15, 5, 5))
    q[:, 10, :, :] = 2.
    q[:, 11, :, :] = 1.
    z, displacements = recover_modal_displacements(q, "CUF_TE4", .05)
    np.testing.assert_allclose(z, [-.025, 0., .025])
    np.testing.assert_allclose(displacements[0, :, 2, 2, 3], [1., 2., 3.])
