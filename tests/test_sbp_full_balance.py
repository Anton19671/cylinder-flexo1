"""Deferred integrated full-work audit tests; execute after solver assembly."""
import unittest
import numpy as np

from cylinder_functional_assembly import CylinderFunctional, surface_sbp21
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_sbp_full_balance import audit_full_sbp_work
from flexo_gradient_g3d import G3DMaterial


class FullSBPBalanceTests(unittest.TestCase):
    def test_all_seven_theories_full_external_work_dynamic(self):
        ops=surface_sbp21(4,5,1.2,2.)
        rng=np.random.default_rng(171)
        material=G3DMaterial(90.,.24,.08,.002,-.001,1.1)
        for theory in ("KL","FSDT","REDDY","CUF_TE1",
                       "CUF_TE2","CUF_TE3","CUF_TE4"):
            with self.subTest(theory=theory):
                quad=(2*int(theory[-1])+1 if theory.startswith("CUF_TE")
                      else 7)
                f=CylinderFunctional(theory,material,radius_m=2.,
                    thickness_m=.2,surface_operators=ops,n_gauss=quad)
                q=rng.normal(0.,1.e-5,(f.n_mechanical,*ops.shape))
                v=rng.normal(0.,1.e-4,q.shape)
                a=rng.normal(0.,1.e-3,q.shape)
                phi=rng.normal(0.,.001,(2,*ops.shape))
                load=np.zeros_like(q)
                load[2*(int(theory[-1])+1) if theory.startswith("CUF_TE")
                     else 2]=.003
                mechanical={
                    "left":{"support":"CLAMPED",
                            "higher":"NATURAL"},
                    "right":{"support":"FREE",
                             "traction":[1.e-3]+[0.]*(f.n_mechanical-1),
                             "spring":[.01]+[0.]*(f.n_mechanical-1)},
                }
                bc={"mechanical":mechanical,
                    "electrical":{"inner":{"mode":"POTENTIAL","value":0.}}}
                problem=BoundarySpatialProblem(
                    f,bc_spec=bc,calculation_kind="DYNAMIC")
                ledger=audit_full_sbp_work(
                    problem,q,phi,time_s=.01,dead_load=load,
                    velocity=v,acceleration=a,density_kg_m3=1300.,
                    eta_s=.005,mass_damping_rate_per_s=.03)
                self.assertLessEqual(ledger.mechanical.absolute_defect,
                    ledger.mechanical.absolute_tolerance+
                    ledger.mechanical.relative_tolerance*max(
                        1.,ledger.mechanical.metrics()["independent_full_peak"]))
                self.assertIsNotNone(ledger.electrical)
                np.testing.assert_allclose(
                    ledger.mechanical.reconstructed,
                    ledger.mechanical.independent_full,rtol=2.e-9,atol=2.e-10)

    def test_reject_non_sbp_grid_and_invalid_electric_charge(self):
        from cylinder_functional_assembly import surface_trapezoid
        ops=surface_trapezoid(4,5,1.2,2.)
        f=CylinderFunctional("CUF_TE1",
            G3DMaterial(90.,.24,.08,.0,.0,.0),
            radius_m=2.,thickness_m=.2,surface_operators=ops,n_gauss=3)
        q=np.zeros((f.n_mechanical,*ops.shape))
        p=BoundarySpatialProblem(f)
        with self.assertRaisesRegex(ValueError,"SBP_GREEN_IDENTITY_VIOLATED"):
            audit_full_sbp_work(p,q)


if __name__=="__main__":
    unittest.main()
