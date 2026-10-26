"""Finite-volume face-flux checks: standalone and seven-theory weak comparison."""
import unittest
import numpy as np
from cylinder_electric_face_flux import dielectric_face_flux_fdm


class ElectricFaceFluxTests(unittest.TestCase):
    def base(self, potential, **kw):
        return dielectric_face_flux_fdm(
            potential,length_m=1.2,radius_m=2.,
            thickness_m=.2,kappa_f_per_m=1.5,**kw)

    def test_face_flux_is_exact_discrete_enthalpy_derivative(self):
        rng=np.random.default_rng(44)
        phi=rng.normal(size=(5,6))*.2
        direction=rng.normal(size=phi.shape)
        epsilon=1.e-5
        result=self.base(phi)
        positive=self.base(phi+epsilon*direction).potential_energy_j
        negative=self.base(phi-epsilon*direction).potential_energy_j
        self.assertAlmostEqual((positive-negative)/(2*epsilon),
                               float(np.sum(result.residual_c*direction)),
                               places=8)
        self.assertAlmostEqual(result.net_residual_c,0.,places=12)

    def test_even_circumferential_grid_sees_nyquist_mode(self):
        for ns in (4,6,8,12):
            with self.subTest(ns=ns):
                phi=np.broadcast_to((-1.)**np.arange(ns),(5,ns)).copy()
                result=self.base(phi)
                self.assertTrue(result.circumferential_checkerboard_detected)
                self.assertGreater(np.max(np.abs(result.residual_c)),.01)
                np.testing.assert_allclose(
                    self.base(phi+20.).residual_c,result.residual_c,
                    rtol=0.,atol=2.e-14)

    def test_constant_gauge_and_insulated_charge_balance(self):
        phi=np.ones((5,7))*2.
        result=self.base(phi)
        np.testing.assert_allclose(result.residual_c,0.,atol=1.e-14)
        self.assertEqual(result.potential_energy_j,0.)
        with self.assertRaisesRegex(ValueError,"zero total charge"):
            self.base(phi,free_charge_c_per_m2=np.ones_like(phi))

    def test_periodic_zero_net_charge_load(self):
        charge=np.broadcast_to(
            np.sin(2*np.pi*np.arange(7)/7),(5,7)).copy()
        result=self.base(np.zeros_like(charge),
                         free_charge_c_per_m2=charge)
        self.assertAlmostEqual(result.net_residual_c,0.,places=12)
        self.assertGreater(np.max(np.abs(result.residual_c)),0.)

    def test_circumferential_residual_matches_coupled_weak_te0_subproblem(self):
        # This one checks actual coupled G3D/Gz kernels but only in the
        # thickness-constant dielectric limit, q=0 and phi_x=phi_z=0.
        from cylinder_functional_assembly import (
            CylinderFunctional,surface_trapezoid)
        from flexo_gradient_g3d import G3DMaterial
        ns=6
        ops=surface_trapezoid(4,ns,1.2,2.)
        p=np.broadcast_to((-1.)**np.arange(ns),
                           (1,4,ns)).copy()
        independent=self.base(p[0])
        for mode in ("G3D","Gz"):
            for theory in ("KL","FSDT","REDDY","CUF_TE1",
                           "CUF_TE2","CUF_TE3","CUF_TE4"):
                with self.subTest(mode=mode,theory=theory):
                    mat=G3DMaterial(100.,.25,.08,.001,-.002,1.5,
                                    gradient_model=mode)
                    ngauss=(2*int(theory[-1])+1 if theory.startswith("CUF")
                            else 7)
                    f=CylinderFunctional(theory,mat,radius_m=2.,
                        thickness_m=.2,surface_operators=ops,
                        n_gauss=ngauss)
                    q=np.zeros((f.n_mechanical,*ops.shape))
                    result=f.evaluate(q,p,energy_breakdown=False)
                    np.testing.assert_allclose(
                        result.electrical_residual[0],
                        independent.residual_c,rtol=2.e-9,atol=1.e-10)
                    np.testing.assert_allclose(
                        result.total_enthalpy_j,
                        independent.potential_energy_j,
                        rtol=2.e-9,atol=1.e-10)


if __name__=="__main__":
    unittest.main()
