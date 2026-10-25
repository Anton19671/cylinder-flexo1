"""Deferred physics-block ILU regression for the full coupled CSR solver."""
import unittest
import numpy as np
from cylinder_functional_assembly import CylinderFunctional,surface_sbp21_csr
from cylinder_matrixfree_fdm import SparseTraceProjector,MatrixFreeNewtonSettings
from cylinder_krylov_block_preconditioner import PhysicsBlockILU
from flexo_gradient_g3d import G3DMaterial


class PhysicsBlockILUTests(unittest.TestCase):
    def make(self,theory="CUF_TE1",electric=True):
        mat=G3DMaterial(100.,.25,.08,.001 if electric else 0.,
                        -.002 if electric else 0.,1.5 if electric else 0.)
        f=CylinderFunctional(theory,mat,radius_m=2.,thickness_m=.2,
            surface_operators=surface_sbp21_csr(4,5,1.2,2.),n_gauss=3)
        n=f.ops.shape[0]*f.ops.shape[1]
        size=(f.n_mechanical+(2 if electric else 0))*n
        c=np.zeros((1,size));c[0,0]=1.
        projector=SparseTraceProjector(c,[0.],max_rows=3)
        scales=np.ones(size)
        return f,scales,projector

    def test_mechanical_and_coupled_preconditioner(self):
        for electric in (False,True):
            for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE4"):
                with self.subTest(electric=electric,theory=theory):
                    f,s,p=self.make(theory,electric)
                    pre=PhysicsBlockILU(f,s,100.,p,drop_tolerance=1.e-3,
                        fill_factor=5.,max_surface_nodes=100)
                    v=np.random.default_rng(42).normal(size=len(s))
                    result=pre.apply(v)
                    self.assertEqual(result.shape,v.shape)
                    self.assertTrue(np.isfinite(result).all())
                    self.assertEqual(pre.diagnostics.variable_count,len(s))
                    self.assertLess(np.linalg.norm(p.c@result),1.e-9)

    def test_dynamic_mass_shift_and_scaling(self):
        f,s,p=self.make()
        pre=PhysicsBlockILU(f,s,100.,p,drop_tolerance=1.e-3,
            fill_factor=5.,max_surface_nodes=100,
            dynamic={"dt_s":.01,"beta":.25,"gamma":.5,
                     "density_kg_m3":1300.,"eta_s":.005,
                     "mass_damping_rate_per_s":.01})
        self.assertGreater(pre.diagnostics.combined_factor_nnz,0)

    def test_large_grid_limit_fails_closed(self):
        f,s,p=self.make()
        with self.assertRaisesRegex(ValueError,"SURFACE_NODE_SAFETY_LIMIT"):
            PhysicsBlockILU(f,s,100.,p,drop_tolerance=1.e-3,
                fill_factor=5.,max_surface_nodes=5)

    def test_explicit_settings(self):
        cfg=MatrixFreeNewtonSettings(krylov_preconditioner="PHYSICS_BLOCK_ILU")
        self.assertEqual(cfg.krylov_preconditioner,"PHYSICS_BLOCK_ILU")
        with self.assertRaisesRegex(ValueError,"krylov_preconditioner"):
            MatrixFreeNewtonSettings(krylov_preconditioner="AUTO")


if __name__=="__main__":unittest.main()
