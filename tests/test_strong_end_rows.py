"""Analytic row and end-stencil checks for the separate §41 strong FDM route."""
import unittest
import numpy as np

from cylinder_strong_end_rows import (
    CLASSICAL_SECOND_ORDER, FULL_GRADIENT_FOURTH_ORDER,
    axial_row_layout, axial_end_derivatives, replace_strong_end_rows,
)


class StrongEndRowsTests(unittest.TestCase):
    def test_one_sided_stencils_are_exact_for_quadratic_fields(self):
        dx=.2
        x=dx*np.arange(7)
        f=(3*x*x-2*x+5)[:,None]*np.ones((1,3))
        dl,dr,ddl,ddr=axial_end_derivatives(f,dx)
        np.testing.assert_allclose(dl,-2.)
        np.testing.assert_allclose(dr,6*x[-1]-2.)
        np.testing.assert_allclose(ddl,6.)
        np.testing.assert_allclose(ddr,6.)

    def test_fourth_order_places_physical_flux_at_near_end_rows(self):
        nx=7
        pde=np.arange(nx,dtype=float)[None,:,None]*np.ones((2,1,3))
        q=np.ones_like(pde)*4.
        left=np.ones((2,3))*7.
        right=np.ones((2,3))*11.
        rows=replace_strong_end_rows(pde,q,branch=FULL_GRADIENT_FOURTH_ORDER,
                                     higher_left=left,higher_right=right)
        np.testing.assert_array_equal(rows[:,0,:],np.full((2,3),4.))
        np.testing.assert_array_equal(rows[:,1,:],left)
        np.testing.assert_array_equal(rows[:,2:-2,:],pde[:,2:-2,:])
        np.testing.assert_array_equal(rows[:,-2,:],right)
        np.testing.assert_array_equal(rows[:,-1,:],np.full((2,3),4.))
        layout=axial_row_layout(nx,branch=FULL_GRADIENT_FOURTH_ORDER)
        self.assertEqual(len(layout.interior)+4,nx)

    def test_classical_limit_retains_all_interior_pde_rows(self):
        pde=np.arange(6,dtype=float)[None,:,None]*np.ones((1,1,2))
        q=np.zeros_like(pde)
        rows=replace_strong_end_rows(pde,q,branch=CLASSICAL_SECOND_ORDER)
        np.testing.assert_array_equal(rows[:,1:-1,:],pde[:,1:-1,:])
        self.assertEqual(len(axial_row_layout(6,branch=CLASSICAL_SECOND_ORDER).interior)+2,6)
        with self.assertRaisesRegex(ValueError,"no higher traction"):
            replace_strong_end_rows(pde,q,branch=CLASSICAL_SECOND_ORDER,
                                    higher_left=np.zeros((1,2)))

    def test_fourth_order_does_not_invent_missing_constitutive_flux(self):
        r=np.zeros((1,5,2))
        with self.assertRaisesRegex(ValueError,"full constitutive end flux"):
            replace_strong_end_rows(r,r,branch=FULL_GRADIENT_FOURTH_ORDER)


if __name__=="__main__":unittest.main()
