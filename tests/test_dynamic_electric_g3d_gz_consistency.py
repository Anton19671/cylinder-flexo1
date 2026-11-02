"""Coupled G3D/Gz dynamic electricity contracts at the actual shell-weak level.

These are operator checks without launching time integration or a local
physical run; Newmark time-step convergence remains a separate validation.
"""
import unittest
import numpy as np

from cylinder_functional_assembly import CylinderFunctional, surface_sbp21
from cylinder_boundary_spatial import BoundarySpatialProblem
from flexo_gradient_g3d import G3DMaterial

THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")


class CoupledElectricDynamicConsistency(unittest.TestCase):
    def test_all_fourteen_dynamic_electric_states_are_quasistatic(self):
        ops=surface_sbp21(4,5,1.2,2.)
        rng=np.random.default_rng(20261102)
        for theory in THEORIES:
            gauss=(2*int(theory[-1])+1 if theory.startswith("CUF_TE") else 7)
            for mode in ("G3D","Gz"):
                with self.subTest(theory=theory,gradient=mode):
                    f=CylinderFunctional(theory,
                        G3DMaterial(170.,.28,.08,.015,-.007,3.,
                                    gradient_model=mode),
                        radius_m=2.,thickness_m=.2,
                        surface_operators=ops,n_gauss=gauss)
                    shape=(f.n_mechanical,*ops.shape)
                    q=rng.normal(0,1.e-5,shape)
                    v=rng.normal(0,2.e-5,shape)
                    a=rng.normal(0,3.e-5,shape)
                    phi=rng.normal(0,.001,(2,*ops.shape))
                    p=BoundarySpatialProblem(f,calculation_kind="DYNAMIC")
                    _,static,_=p.evaluate_effective_residual(
                        q,phi,time_s=.02,
                        acceleration=np.zeros_like(q),
                        density_kg_m3=1300.)
                    _,dynamic,_=p.evaluate_effective_residual(
                        q,phi,time_s=.02,velocity=v,acceleration=a,
                        density_kg_m3=1300.,
                        mass_damping_rate_per_s=.03,eta_s=.002)
                    # In the reference electromechanical model, electric
                    # charge is stationary at every mechanical Newmark step:
                    # no artificial electric velocity/mass/viscous term.
                    np.testing.assert_allclose(
                        dynamic[q.size:],static[q.size:],
                        rtol=1.e-11,atol=1.e-10)
                    self.assertGreater(
                        np.max(np.abs(dynamic[:q.size]-static[:q.size])),
                        0.)
                    self.assertTrue(np.isfinite(dynamic).all())

    def test_six_z_gradients_keep_converse_electric_coupling(self):
        from flexo_gradient_g3d import enthalpy_derivatives
        for mode in ("G3D","Gz"):
            with self.subTest(gradient=mode):
                material=G3DMaterial(170.,.28,.08,.015,-.007,3.,
                                     gradient_model=mode)
                e=np.zeros(6)
                g=np.zeros((6,3))
                g[4,2]=.1
                E=np.array((.2,-.1,.15))
                _,hyper,D=enthalpy_derivatives(e,g,E,material)
                self.assertTrue(np.isfinite(hyper).all())
                self.assertTrue(np.isfinite(D).all())
                # A thickness shear gradient can polarize the x direction.
                self.assertNotEqual(D[0],material.kappa_f_per_m*E[0])
                if mode=="Gz":
                    np.testing.assert_array_equal(hyper[:,:2],0.)


if __name__=="__main__":
    unittest.main()
