"""Static/Newmark MVI certificates: exact representation and full residual."""
import unittest
import numpy as np
from cylinder_mvi_acceptance import certify_mvi_solution


def fixture(rank=1,nm=1,ne=1):
    n=nm+ne
    x=np.zeros((n,rank,8));y=np.zeros((n,rank,9))
    x[:,0]=1.;y[:,0]=2.
    q=np.einsum("crx,crs->cxs",x,y)
    return dict(x_profiles=x,y_profiles=y,
        scales=np.ones_like(q),mechanical=q[:nm].copy(),
        electrical=q[nm:].copy() if ne else None,
        full_basis=np.eye(q.size),reduced_residual=np.zeros(q.size),
        projected_residual_scaled=0.,full_tolerance_scaled=1.e-8,
        constraint_defect_scaled=0.,solver_converged=True)


class MVICertificateTest(unittest.TestCase):
    def test_valid_coupled_rank_one(self):
        c=certify_mvi_solution(**fixture())
        self.assertTrue(c["accepted"]);self.assertEqual(c["fixed_rank"],1)
    def test_valid_rank_eight_mechanical_only(self):
        c=certify_mvi_solution(**fixture(rank=8,nm=3,ne=0))
        self.assertTrue(c["accepted"]);self.assertEqual(c["fixed_rank"],8)
    def test_projected_only_is_not_full_equilibrium(self):
        d=fixture();d["reduced_residual"][3]=1.e-3
        c=certify_mvi_solution(**d)
        self.assertTrue(c["projected_equilibrium_satisfied"])
        self.assertFalse(c["full_equilibrium_satisfied"])
        self.assertTrue(c["accepted"])
        self.assertEqual(c["solution_status"],"CONVERGED_PROJECTED")
        self.assertTrue(c["finite_order_control_recommended"])
        self.assertEqual(c["mechanical_residual_scaled"],1.e-3)
    def test_fails_if_reconstruction_drifts(self):
        d=fixture();d["mechanical"][0,1,2]+=1.e-3
        c=certify_mvi_solution(**d)
        self.assertFalse(c["accepted"])
        self.assertFalse(c["profile_reconstruction_satisfied"])
    def test_fails_if_boundary_defect(self):
        d=fixture();d["constraint_defect_scaled"]=1.e-4
        self.assertFalse(certify_mvi_solution(**d)["accepted"])
    def test_inner_halfstep_failure_cannot_be_overridden(self):
        d=fixture();d["solver_converged"]=False
        c=certify_mvi_solution(**d)
        self.assertFalse(c["accepted"])
        self.assertTrue(c["full_equilibrium_satisfied"])
    def test_invalid_and_nonfinite_rejected(self):
        d=fixture();d["y_profiles"]=np.ones((2,2,9))
        with self.assertRaises(ValueError):certify_mvi_solution(**d)
        d=fixture();d["reduced_residual"][0]=np.nan
        with self.assertRaises(ValueError):certify_mvi_solution(**d)


if __name__=="__main__":
    unittest.main()
