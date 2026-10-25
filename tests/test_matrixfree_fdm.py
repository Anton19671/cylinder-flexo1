"""Deferred Newton-Krylov sparse trace checks, not run before full assembly."""
import unittest
import numpy as np
from scipy import sparse
from cylinder_matrixfree_fdm import (
    SparseTraceProjector, MatrixFreeNewtonSettings,
    solve_matrixfree_static_fdm)


class MatrixFreeFDMTests(unittest.TestCase):
    def test_affine_projector_preserves_independent_traces(self):
        n=120
        C=sparse.csr_matrix(([1.,-2.,3.],([0,1,2],[3,41,119])),shape=(3,n))
        projector=SparseTraceProjector(C,np.array([.25,-1.,.3]),max_rows=20)
        rng=np.random.default_rng(8)
        x=rng.normal(size=n)
        u=projector.project_affine(x)
        self.assertLess(np.linalg.norm(projector.defect(u)),1e-12)
        v=projector.project_null(x)
        self.assertLess(np.linalg.norm(projector.c@v),1e-12)
        np.testing.assert_allclose(projector.project_null(v),v,atol=1e-13)

    def test_rank_deficient_bc_fails_closed(self):
        C=sparse.csr_matrix([[1.,0.],[2.,0.]])
        with self.assertRaisesRegex(ValueError,"SPARSE_TRACE_"):
            SparseTraceProjector(C,np.array([0.,0.]),max_rows=4)

    def test_settings_reject_bad_parameters(self):
        with self.assertRaises(ValueError):MatrixFreeNewtonSettings(krylov_rtol=1.)
        with self.assertRaises(ValueError):MatrixFreeNewtonSettings(max_total_dofs=0)

    def test_optional_complete_static_fdm(self):
        from cylinder_functional_assembly import CylinderFunctional,surface_sbp21
        from cylinder_boundary_spatial import BoundarySpatialProblem
        from flexo_gradient_g3d import G3DMaterial
        ops=surface_sbp21(4,5,1.2,2.)
        f=CylinderFunctional("CUF_TE1",G3DMaterial(
            100.,.25,.08,0.,0.,0.),radius_m=2.,thickness_m=.2,
            surface_operators=ops,n_gauss=3)
        q=np.zeros((f.n_mechanical,*ops.shape))
        bc={"mechanical":{"left":{"support":"CLAMPED"},
                          "right":{"support":"CLAMPED"}}}
        problem=BoundarySpatialProblem(f,bc_spec=bc,calculation_kind="STATIC")
        settings=MatrixFreeNewtonSettings(max_iterations=1)
        result=solve_matrixfree_static_fdm(problem,q,settings=settings)
        self.assertTrue(result.converged)
        self.assertLess(result.admissible_residual_scaled,1e-9)


if __name__=="__main__":unittest.main()
