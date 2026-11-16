"""Algebraic proof-of-defect for the native axial FDM Green identity.

These tests only construct differentiation matrices. They do not integrate
the shell equations or treat a nonzero defect as a certified discretization.
"""
import numpy as np
import pytest

from cylinder_direct_strong_fdm import (
    DirectStrongGrid, direct_coefficient_derivative, direct_strong_jet_balance,
)
from cylinder_mkr_field_jets import MKRFieldJets
from cylinder_strong_axial_green_audit import (
    audit_native_axial_green, audit_native_higher_axial_green,
)


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


@pytest.mark.parametrize("nx", [7, 11, 17])
def test_higher_axial_green_operator_pair_is_not_automatically_sbp(nx):
    grid=DirectStrongGrid(nx,5,1.2,2.)
    report=audit_native_higher_axial_green(grid)
    assert report.orders==(1,2,3)
    assert report.status=="HIGHER_AXIAL_GREEN_MISMATCH_NOT_BOUNDARY_CERTIFIED"
    for order,defect,absolute,relative in zip(
            report.orders, report.defect_matrices,
            report.max_absolute_defect, report.relative_defect):
        assert defect.shape==(nx,nx)
        assert not defect.flags.writeable
        assert absolute>1.e-8
        assert relative>1.e-11
        if order==1:
            first=audit_native_axial_green(grid)
            np.testing.assert_allclose(
                defect,first.defect_matrix,rtol=1.e-12,atol=1.e-12)


@pytest.mark.parametrize("order", [1,2,3])
def test_higher_green_matrix_equals_true_strong_bulk_and_end_work(order):
    """Check the *real* strong balance and physical normal trace assembly.

    Independent random q and local coefficient A expose the full adjoint
    defect, including the signs of double/triple tractions on both ends.
    This does not treat nonzero Green defects as valid natural closures.
    """
    from cylinder_direct_strong_boundary import native_kinematic_normal_trace
    grid=DirectStrongGrid(11,5,1.2,2.)
    rng=np.random.default_rng(131+order)
    q=rng.normal(size=(1,grid.nx,grid.ns))
    a=rng.normal(size=(grid.nx,grid.ns))
    coefficients=np.zeros((2,1,grid.nx,grid.ns))
    coefficients[1,0]=a
    balance=direct_strong_jet_balance(
        coefficients,(("q",0,0),("q_ax",order,0)),grid)
    weights=np.full(grid.nx,grid.dx)
    weights[[0,-1]]*=.5
    W=weights[:,None]*grid.ds
    weak=float(np.sum(W*a*MKRFieldJets(grid).derivative(q,order,0)[0]))
    bulk=float(np.sum(W*q[0]*balance.mechanical_bulk[0]))
    edge_work=0.
    for end_idx,side in enumerate(("left","right")):
        node=0 if side=="left" else -1
        end=balance.mechanical_end_displacement[end_idx,0]
        edge_work+=grid.ds*float(np.dot(end,q[0,node]))
        if order>=2:
            slope=balance.mechanical_end_slope[end_idx,0]
            dn=native_kinematic_normal_trace(q,grid,0,1,side)
            edge_work+=grid.ds*float(np.dot(slope,dn))
        if order>=3:
            triple=balance.mechanical_end_curvature[end_idx,0]
            dnn=native_kinematic_normal_trace(q,grid,0,2,side)
            edge_work+=grid.ds*float(np.dot(triple,dnn))
    report=audit_native_higher_axial_green(grid,highest_order=order)
    predicted=float(sum(
        q[0,:,angle]@report.defect_matrices[order-1]@a[:,angle]
        for angle in range(grid.ns))*grid.ds)
    np.testing.assert_allclose(
        weak-bulk-edge_work,predicted,rtol=2.e-10,atol=2.e-8)
    assert abs(predicted)>1.e-6


def test_higher_axial_green_order_and_size_guard():
    grid=DirectStrongGrid(11,4,1.2,2.)
    with pytest.raises(ValueError,match="ORDER_MUST"):
        audit_native_higher_axial_green(grid,highest_order=0)
    with pytest.raises(ValueError,match="ORDER_MUST"):
        audit_native_higher_axial_green(grid,highest_order=True)
    with pytest.raises(TypeError,match="DirectStrongGrid"):
        audit_native_higher_axial_green(None)
    with pytest.raises(ValueError,match="SIZE_LIMIT"):
        audit_native_higher_axial_green(DirectStrongGrid(65,4,1.2,2.))
