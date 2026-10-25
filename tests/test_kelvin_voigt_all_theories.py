"""Deferred nonlinear Kelvin-Voigt physics tests (run after assembly stage)."""
import unittest
import numpy as np
from flexo_gradient_g3d import G3DMaterial
from cylinder_functional_assembly import CylinderFunctional,surface_trapezoid
from cylinder_kelvin_voigt import viscous_stress_and_power


class KelvinVoigtSevenTheories(unittest.TestCase):
    def setUp(self):
        self.mat=G3DMaterial(120.,.24,.13,.06,-.02,.7)
        self.ops=surface_trapezoid(4,5,1.2,2.)
        self.rng=np.random.default_rng(323)
        self.theories=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")

    def test_constitutive_power_and_shear_factors(self):
        e=np.array([.1,-.2,.05,.04,.08,-.03])
        s,p=viscous_stress_and_power(e,self.mat,.3)
        self.assertAlmostEqual(s[3],.3*self.mat.shear_modulus*e[3])
        self.assertAlmostEqual(p,float(s@e))
        self.assertGreater(p,0.)
        self.assertEqual(viscous_stress_and_power(e,self.mat,0.)[1],0.)
        with self.assertRaisesRegex(ValueError,"nonnegative"):
            viscous_stress_and_power(e,self.mat,-1.)

    def test_seven_theories_positive_power_and_virtual_rate(self):
        for theory in self.theories:
            with self.subTest(theory=theory):
                order=(2*int(theory[-1])+1 if theory.startswith("CUF") else
                       7 if theory=="REDDY" else 4)
                f=CylinderFunctional(theory,self.mat,radius_m=2.,thickness_m=.2,
                    surface_operators=self.ops,n_gauss=order)
                q=self.rng.normal(0,1e-5,(f.n_mechanical,*self.ops.shape))
                v=self.rng.normal(0,2e-5,q.shape)
                d=self.rng.normal(0,2e-5,q.shape)
                eta=.02
                action=f.kelvin_voigt_action(q,v,eta_s=eta)
                self.assertGreaterEqual(action.power_w,0.)
                self.assertEqual(action.mechanical_force.shape,q.shape)
                # For fixed q the Rayleigh potential is quadratic in v.
                eps=5e-4
                up=f.kelvin_voigt_action(q,v+eps*d,eta_s=eta).power_w/2
                down=f.kelvin_voigt_action(q,v-eps*d,eta_s=eta).power_w/2
                exact=float(np.sum(action.mechanical_force*d))
                np.testing.assert_allclose((up-down)/(2*eps),exact,
                    atol=1e-8,rtol=2e-4)
                zero=f.kelvin_voigt_action(q,np.zeros_like(v),eta_s=eta)
                self.assertAlmostEqual(zero.power_w,0.,places=14)
                np.testing.assert_allclose(zero.mechanical_force,0.,atol=1e-13)
                off=f.kelvin_voigt_action(q,v,eta_s=0.)
                np.testing.assert_allclose(off.mechanical_force,0.,atol=0.)
                self.assertEqual(off.power_w,0.)

    def test_no_viscosity_in_static_enthalpy(self):
        f=CylinderFunctional("CUF_TE1",self.mat,radius_m=2.,thickness_m=.2,
            surface_operators=self.ops,n_gauss=4)
        q=np.zeros((f.n_mechanical,*self.ops.shape))
        phi=np.zeros((2,*self.ops.shape))
        first=f.evaluate(q,phi).total_enthalpy_j
        _=f.kelvin_voigt_action(q,np.zeros_like(q),eta_s=.05)
        self.assertEqual(f.evaluate(q,phi).total_enthalpy_j,first)


if __name__=="__main__": unittest.main()
