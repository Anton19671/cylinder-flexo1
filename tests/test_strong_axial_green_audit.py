"""Algebraic proof-of-defect for the native axial FDM Green identity.

These tests only construct differentiation matrices. They do not integrate
the shell equations or treat a nonzero defect as a certified discretization.
"""
import numpy as np
import pytest

from cylinder_direct_strong_fdm import DirectStrongGrid, direct_coefficient_derivative
from cylinder_mkr_field_jets import MKRFieldJets
from cylinder_strong_axial_green_audit import audit_native_axial_green


def test_axial_green_pair_is_not_an_exact_discrete_adjoint():
    grid=DirectStrongGrid(11,9,1.2,2.)
    audit=audit_native_axial_green(grid)
    assert audit.status=="AXIAL_FIRST_ORDER_GREEN_PAIR_NOT_ADJOINT"
    assert audit.max_absolute_matrix_defect>1.e-8
    assert audit.relative_matrix_defect>1.e-10
    assert not audit.defect_matrix.flags.writeable


def test_green_matrix_defect_equals_actual_work_identity_defect():
    grid=DirectStrongGrid(11,7,1.2,2.)
    audit=audit_native_axial_green(grid)
    rng=np.random.default_rng(20261116)
    q=rng.normal(size=grid.nx)
    a=rng.normal(size=grid.nx)
    qs=np.broadcast_to(q[None,:,None],(1,grid.nx,grid.ns)).copy()
    a_s=np.broadcast_to(a[None,:,None],(1,grid.nx,grid.ns)).copy()
    dx_q=MKRFieldJets(grid).derivative(qs,1,0)[0,:,0]
    dx_a=direct_coefficient_derivative(a_s,1,0,grid)[0,:,0]
    weights=np.full(grid.nx,grid.dx)
    weights[[0,-1]]*=.5
    work_defect=float(
        np.dot(weights,a*dx_q+q*dx_a)-(q[-1]*a[-1]-q[0]*a[0]))
    matrix_defect=float(q @ audit.defect_matrix @ a)
    np.testing.assert_allclose(work_defect,matrix_defect,rtol=1.e-12,atol=1.e-12)


def test_green_audit_rejects_non_grid_and_oversized_diagnostics():
    with pytest.raises(TypeError,match="DirectStrongGrid"):
        audit_native_axial_green(None)
    with pytest.raises(ValueError,match="SIZE_LIMIT"):
        audit_native_axial_green(DirectStrongGrid(65,4,1.2,2.))
