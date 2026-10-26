"""Acceptance norms must be physical nodal, not coordinate-dependent QR values."""
import unittest
import numpy as np
from cylinder_residual_norms import (
    admissible_nodal_metrics, admissible_nodal_inf_norm)


class TestNodalResidualMVI(unittest.TestCase):
    def test_rotation_of_orthonormal_MVI_basis_must_not_change_norm(self):
        basis=np.array([[1.,0.],[0.,1.],[0.,0.]])
        angle=np.pi/4
        rot=np.array([[np.cos(angle),-np.sin(angle)],
                      [np.sin(angle),np.cos(angle)]])
        coordinates=np.array([1.,.3])
        first=admissible_nodal_metrics(basis,coordinates,2)
        second=admissible_nodal_metrics(
            basis@rot,rot.T@coordinates,2)
        self.assertNotAlmostEqual(first["basis_coordinates"],second["basis_coordinates"])
        self.assertAlmostEqual(first["full"],second["full"],places=13)
        self.assertAlmostEqual(second["full"],admissible_nodal_inf_norm(
            basis@rot,rot.T@coordinates),places=13)

    def test_full_norm_can_exceed_both_free_coordinate_maxes(self):
        b=np.array([[1.,1.],[1.,-1.],[0.,0.]])/np.sqrt(2)
        reduced=np.ones(2)
        result=admissible_nodal_metrics(b,reduced,2)
        self.assertAlmostEqual(result["basis_coordinates"],1.)
        self.assertAlmostEqual(result["full"],np.sqrt(2))

    def test_separate_mechanical_and_electric_blocks(self):
        audit=admissible_nodal_metrics(np.eye(5),
            np.array([-.3,.2,1.,-5.,.4]),3)
        self.assertEqual(audit["mechanical"],1.)
        self.assertEqual(audit["electrical"],5.)
        self.assertEqual(audit["full"],5.)

    def test_constrained_directions_excluded_from_admissible_equations(self):
        b=np.array([[1/np.sqrt(2)],[-1/np.sqrt(2)],[0.]])
        m=admissible_nodal_metrics(b,np.array([2.]),2)
        self.assertAlmostEqual(m["full"],np.sqrt(2),places=14)
        self.assertEqual(m["electrical"],0.)

    def test_shape_and_finiteness_fail_closed(self):
        for b,g,n in ((np.ones((3,1)),np.ones(2),2),
                      (np.ones((3,1)),np.array([np.nan]),2),
                      (np.ones((3,1)),np.ones(1),4)):
            with self.subTest(n=n),self.assertRaises(ValueError):
                admissible_nodal_metrics(b,g,n)


if __name__=="__main__":
    unittest.main()
