"""MVI cold-start must activate nonaxisymmetric rank-one loads (no SVD)."""
import unittest
import numpy as np
from test_variational_fdm import AlgebraicFunctional
from test_variational_dynamic import DynamicFunctional
from cylinder_functional_assembly import surface_trapezoid
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_variational_mvi import (StaticMVISettings,solve_static_mvi,
                                       reconstruct_profiles)
from cylinder_variational_dynamic import solve_dynamic_mvi
from cylinder_variational_fdm import solve_static_fdm,StaticFDMSettings
from cylinder_mvi_cold_start import seed_inactive_mvi_partners


def _problem(dynamic=False):
    f=DynamicFunctional(nx=4,ns=6) if dynamic else AlgebraicFunctional()
    f.ops=surface_trapezoid(4,6,1.2,2.)
    f.W=f.ops.area_weights
    spec={"mechanical":{"left":{"support":"CLAMPED"}},
          "electrical":{"inner":{"mode":"POTENTIAL","value":0.}}}
    kind="DYNAMIC" if dynamic else "STATIC"
    return (BoundarySpatialProblem(f,bc_spec=spec,calculation_kind=kind),
            np.zeros((3,4,6)),np.zeros((2,4,6)))


def _load(shape):
    force=np.zeros(shape)
    # Exact separated rank one with zero circumferential average,
    # with the first axial node clamped (u(x=0)=0).
    force[0]=np.outer(np.array([0.,1.,2.,1.]),
                      np.sin(2*np.pi*np.arange(shape[2])/shape[2]))
    return force


class MVIResidualSeedingTests(unittest.TestCase):
    def test_original_constant_seed_is_blind_to_sine(self):
        q=np.zeros((1,1,4))
        y=np.ones((1,1,6))/np.sqrt(6)
        gradient=_load((3,4,6))[:1]
        self.assertAlmostEqual(float(np.sum(gradient[0]*y[0,0])),0.)
        x,ynew=seed_inactive_mvi_partners(q,y,gradient)
        self.assertGreater(abs(float(np.sum(gradient[0]*ynew[0,0]))),1.)
        np.testing.assert_array_equal(reconstruct_profiles(x,ynew),0.)

    def test_seven_conditions_hold_for_rank_eight_unchanged_fields(self):
        rng=np.random.default_rng(27)
        x=np.zeros((7,8,8))
        y=np.zeros((7,8,9))
        x2,y2=seed_inactive_mvi_partners(x,y,rng.normal(size=(7,8,9)))
        np.testing.assert_array_equal(reconstruct_profiles(x2,y2),0.)
        np.testing.assert_allclose(y2@np.swapaxes(y2,1,2),
                                   np.broadcast_to(np.eye(8),(7,8,8)),
                                   atol=2.e-14)

    def test_static_nonaxisymmetric_rank_one_converges(self):
        problem,q,p=_problem()
        load=_load(q.shape)
        result=solve_static_mvi(problem,q,p,dead_load=load,
                                mvi_settings=StaticMVISettings(rank=1))
        self.assertTrue(result.converged,result.status)
        self.assertLess(result.admissible_residual_scaled,1.e-8)
        self.assertLess(result.constraint_defect_scaled,1.e-10)
        fdm=solve_static_fdm(problem,q,p,dead_load=load)
        self.assertTrue(fdm.converged,fdm.status)
        np.testing.assert_allclose(result.mechanical,fdm.mechanical,
                                   rtol=1.e-6,atol=1.e-9)
        fields=result.scales*(result.lift_scaled+
                              reconstruct_profiles(result.x_profiles,result.y_profiles))
        np.testing.assert_allclose(fields,np.concatenate((result.mechanical,
                                                           result.electrical)),atol=1.e-12)

    def test_dynamic_newmark_nonaxisymmetric_rank_one_converges(self):
        problem,q,p=_problem(dynamic=True)
        force=_load(q.shape)
        result=solve_dynamic_mvi(problem,q,p,q_predictor=q,
            velocity_predictor=np.zeros_like(q),time_s=.1,dt_s=.1,
            beta=.25,gamma=.5,density_kg_m3=1.,dead_load=force,
            mvi_settings=StaticMVISettings(rank=1))
        self.assertTrue(result.converged,result.status)
        self.assertLess(result.admissible_residual_scaled,1.e-8)
        self.assertLess(result.constraint_defect_scaled,1.e-10)
        represented=result.scales*(result.lift_scaled+
                     reconstruct_profiles(result.x_profiles,result.y_profiles))
        np.testing.assert_allclose(represented,
            np.concatenate((result.mechanical,result.electrical)),atol=1.e-12)


if __name__=="__main__":
    unittest.main()
