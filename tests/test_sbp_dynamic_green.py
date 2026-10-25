"""Deferred full dynamic SBP audit for all seven coupled shell models."""
import unittest
import numpy as np
from cylinder_functional_assembly import CylinderFunctional,surface_sbp21
from flexo_gradient_g3d import G3DMaterial
from cylinder_boundary_spatial import BoundarySpatialProblem


class DynamicSBPGreenAudit(unittest.TestCase):
    def test_full_residual_versus_exact_sbp_split(self):
        ops=surface_sbp21(4,5,1.2,2.)
        rng=np.random.default_rng(442)
        mat=G3DMaterial(100.,.25,.08,.001,-.002,1.5)
        for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
            with self.subTest(theory=theory):
                ngauss=(2*int(theory[-1])+1 if theory.startswith("CUF_TE")
                        else 7)
                f=CylinderFunctional(theory,mat,radius_m=2.,
                    thickness_m=.2,surface_operators=ops,n_gauss=ngauss)
                q=rng.normal(0,1.e-5,(f.n_mechanical,*ops.shape))
                v=rng.normal(0,1.e-4,q.shape)
                a=rng.normal(0,1.e-3,q.shape)
                phi=rng.normal(0,1.e-3,(2,*ops.shape))
                kwargs=dict(velocity=v,acceleration=a,
                            density_kg_m3=1300.,eta_s=.002,
                            mass_damping_rate_per_s=.01)
                split=f.sbp_green_reduction(q,phi,**kwargs)
                problem=BoundarySpatialProblem(f,calculation_kind="DYNAMIC")
                _,r,_=problem.evaluate_effective_residual(
                    q,phi,time_s=.01,**kwargs)
                np.testing.assert_allclose(
                    split.mechanical.reconstructed,r[:q.size].reshape(q.shape),
                    rtol=2e-9,atol=2e-8)
                np.testing.assert_allclose(
                    split.electrical.reconstructed,r[q.size:].reshape(phi.shape),
                    rtol=2e-9,atol=2e-8)
                self.assertLess(split.mechanical.relative_defect,2e-10)

    def test_bad_dynamic_inputs_are_rejected(self):
        ops=surface_sbp21(4,5,1.2,2.)
        f=CylinderFunctional("CUF_TE1",
            G3DMaterial(100.,.25,.08,0.,0.,0.),
            radius_m=2.,thickness_m=.2,surface_operators=ops,n_gauss=3)
        q=np.zeros((f.n_mechanical,*ops.shape))
        with self.assertRaisesRegex(ValueError,"velocity, acceleration and density"):
            f.sbp_green_reduction(q,velocity=q)
        with self.assertRaisesRegex(ValueError,"nonnegative"):
            f.sbp_green_reduction(q,eta_s=-.1)


if __name__=="__main__": unittest.main()
