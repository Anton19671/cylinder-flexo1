"""Zh-15: same meaning of repeated BC indices in all matrix variants."""
from types import SimpleNamespace
import numpy as np
import pytest
from cylinder_constraint_row import combined_constraint_entries
from cylinder_dynamic_trace import mechanical_trace_matrix, sparse_mechanical_trace_matrix
from cylinder_boundary_spatial import affine_trace_space


def test_duplicates_sum_with_offset():
    index, value = combined_constraint_entries((0, 1, 0, 2, 1), (3, -1, 5, 0, 1), 3, offset=6)
    np.testing.assert_array_equal(index, [6])
    np.testing.assert_array_equal(value, [8])


def test_dense_sparse_trace_matches_for_duplicate_terms():
    rows = [SimpleNamespace(field="mechanical", indices=(0, 0, 2), weights=(1., 2., -3.), prescribed=0.),
            SimpleNamespace(field="mechanical", indices=(1,), weights=(1.,), prescribed=4.),
            SimpleNamespace(field="electrical", indices=(0,), weights=(1.,), prescribed=1.)]
    c, d = mechanical_trace_matrix(rows, 4)
    cs, ds = sparse_mechanical_trace_matrix(rows, 4)
    np.testing.assert_allclose(c, cs.toarray(), atol=1e-15)
    np.testing.assert_allclose(d, ds, atol=1e-15)


@pytest.mark.parametrize("indexes,weights", [((0, 0),(2.,-2.)), ((0,),(float("nan"),))])
def test_zero_or_nonfinite_trace_fails_closed(indexes, weights):
    rows=[SimpleNamespace(field="mechanical",indices=indexes,weights=weights,prescribed=0.)]
    with pytest.raises(ValueError): mechanical_trace_matrix(rows, 3)
    with pytest.raises(ValueError): sparse_mechanical_trace_matrix(rows, 3)


def test_invalid_indices_rejected():
    for indexes in ((-1,), (3,), (True,), (0,2)):
        weights=(1.,) if len(indexes)==1 else (1.,)
        with pytest.raises(ValueError): combined_constraint_entries(indexes, weights, 3)


def test_large_origin_cannot_hide_incompatible_prescriptions():
    c=np.array([[1.,0.],[1.,0.]])
    with pytest.raises(ValueError,match="inconsistent/unreachable"):
        affine_trace_space(c,np.array([0.,1e-6]),np.array([1e12,0.]))


def test_small_roundoff_still_allowed():
    c=np.array([[1.,0.,0.]])
    directions=np.array([[0.],[1.],[0.]])
    a=affine_trace_space(c,np.zeros(1),np.array([1e-16,1.,0.]),directions=directions)
    assert abs(a.defect(a.origin)[0])<1e-14
