"""Tests of physical-end traces and fail-closed axial order certificate."""
import unittest
import numpy as np
from cylinder_direct_strong_fdm import DirectStrongGrid,DirectStrongBalance
from cylinder_direct_strong_boundary import assemble_physical_end_rows
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
        ends={"left":[[bc("ESSENTIAL",0.),bc("ESSENTIAL",0.),bc("ESSENTIAL",0.)]],
              "right":[[bc("ESSENTIAL",8.),bc("ESSENTIAL",12.),bc("ESSENTIAL",12.)]]}
        result=assemble_physical_end_rows(
            b,q,grid,axial_orders=(3,),mechanical_ends=ends)
        np.testing.assert_allclose(result.mechanical[0,:3],0,atol=1e-9)
        np.testing.assert_allclose(result.mechanical[0,-3:],0,atol=1e-9)

    def test_classification_is_not_guessed_from_shell_name(self):
        matrices={2:np.diag([1.,1.,1.]),4:np.diag([0.,1.,0.]),
                  6:np.diag([0.,0.,1.])}
        c=certify_decoupled_axial_orders(matrices)
        self.assertTrue(c.admissible)
        self.assertEqual(c.orders,(1,2,3))
        mixed=certify_decoupled_axial_orders({2:np.array([[2.,1.],[1.,2.]])})
        self.assertFalse(mixed.admissible)
        self.assertIn("MIXED",mixed.reason)

    def test_degenerate_orders_blocked(self):
        certificate=certify_decoupled_axial_orders({2:np.diag([1.,0.])})
        self.assertFalse(certificate.admissible)
        self.assertIn("UNCLASSIFIED",certificate.reason)

if __name__=="__main__":
    unittest.main()
