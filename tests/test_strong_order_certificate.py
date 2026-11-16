"""Tests of physical-end traces and fail-closed axial order certificate."""
import unittest
import numpy as np
from cylinder_direct_strong_fdm import DirectStrongGrid,DirectStrongBalance
from cylinder_direct_strong_boundary import assemble_physical_end_rows, native_kinematic_normal_trace
from cylinder_mkr_field_jets import MKRFieldJets
from cylinder_strong_order_certificate import certify_decoupled_axial_orders

def bc(kind,v=0.):
    return {"kind":kind,"value":v}

class AlgorithmOrderTraceTests(unittest.TestCase):
    def test_outward_derivatives_sample_true_end_not_adjacent_row(self):
        grid=DirectStrongGrid(11,4,2.,1.)
        x=np.arange(grid.nx)*grid.dx
        q=np.zeros((1,grid.nx,grid.ns))
        q[0]=x[:,None]**3
        zeros=np.zeros_like(q)
        edges=np.zeros((2,1,grid.ns))
        b=DirectStrongBalance(zeros,None,edges,edges,edges,None,
                             tuple(range(3,8)),3)
        # End values must be sampled at x=0/L with the SAME native
        # kinematic stencil as the constitutive jets. A second-order
        # one-sided first derivative is not cubic-exact at the end:
        # D_x(x**3)(0)=-2*dx**2; the LEFT OUTWARD trace is +2*dx**2.
        ends={side:[[
            bc("ESSENTIAL",native_kinematic_normal_trace(q,grid,0,k,side))
            for k in (0,1,2)]]
            for side in ("left","right")}
        result=assemble_physical_end_rows(
            b,q,grid,axial_orders=(3,),mechanical_ends=ends)
        np.testing.assert_allclose(result.mechanical[0,:3],0,atol=1e-9)
        np.testing.assert_allclose(result.mechanical[0,-3:],0,atol=1e-9)
        jets=MKRFieldJets(grid)
        for side,node in (("left",0),("right",-1)):
            for order in (0,1,2):
                exact=jets.derivative(q,order,0)[0,node,:]
                outward=(-1 if side=="left" and order%2 else 1)
                np.testing.assert_allclose(
                    native_kinematic_normal_trace(q,grid,0,order,side),
                    outward*exact)
        left_slope=native_kinematic_normal_trace(q,grid,0,1,"left")
        np.testing.assert_allclose(left_slope,2*grid.dx**2,atol=1e-12)
        # The previous continuum-exact slope request is NOT equivalent
        # to the discrete clamp, and cannot be used as a zero-residual
        # assertion for the native second-order endpoint operator.
        old_ends={side:[list(items[0])] for side,items in ends.items()}
        old_ends["left"][0][1]=bc("ESSENTIAL",0.)
        legacy=assemble_physical_end_rows(
            b,q,grid,axial_orders=(3,),mechanical_ends=old_ends)
        np.testing.assert_allclose(
            legacy.mechanical[0,1,:],2*grid.dx**2,atol=1e-12)

    def test_classification_is_not_guessed_from_shell_name(self):
        matrices={2:np.diag([1.,1.,1.]),4:np.diag([0.,1.,0.]),
                  6:np.diag([0.,0.,1.])}
        c=certify_decoupled_axial_orders(matrices)
        self.assertTrue(c.admissible)
        self.assertEqual(c.orders,(1,2,3))
        mixed=certify_decoupled_axial_orders({2:np.array([[2.,1.],[1.,2.]])})
        self.assertFalse(mixed.admissible)
        self.assertIn("MIXED",mixed.reason)

    def test_orders_are_invariant_under_nonzero_unit_rescaling(self):
        base={2:np.diag([2.,0.]),4:np.diag([0.,3.])}
        expected=certify_decoupled_axial_orders(base)
        for factor in (1.e-20,1.e-8,1.,1.e20):
            got=certify_decoupled_axial_orders(
                {degree:factor*matrix for degree,matrix in base.items()})
            self.assertEqual(got,expected)
        mixed=np.array([[2.,1.],[1.,2.]])
        for factor in (1.e-20,1.,1.e20):
            got=certify_decoupled_axial_orders({2:factor*mixed})
            self.assertFalse(got.admissible)
            self.assertIn("MIXED",got.reason)

    def test_degenerate_orders_blocked(self):
        certificate=certify_decoupled_axial_orders({2:np.diag([1.,0.])})
        self.assertFalse(certificate.admissible)
        self.assertIn("UNCLASSIFIED",certificate.reason)

if __name__=="__main__":
    unittest.main()
