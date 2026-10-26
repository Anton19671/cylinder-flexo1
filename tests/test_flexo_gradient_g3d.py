"""Regression tests for source-derived macro/micro constitutive contract."""
import unittest
import numpy as np
from flexo_gradient_g3d import (G3DMaterial, enthalpy_parts, enthalpy_derivatives,
    covariant_strain_gradients, gradient_invariants, active_strain_gradients,
    reference_electric_field)
from flexo_scale_policy import numerical_status

class TestG3D(unittest.TestCase):
    def test_27_energy_derivatives(self):
        rng=np.random.default_rng(7124)
        p=G3DMaterial(170,.31,.2,1.5,-.7,3.)
        e=rng.normal(0,.05,6);g=rng.normal(0,.1,(6,3));q=rng.normal(0,.2,3)
        s,m,D=enthalpy_derivatives(e,g,q,p)
        def energy(a,b,c):return enthalpy_parts(a,b,c,p)["total"]
        h=1.e-6
        for i in range(6):
            d=np.zeros(6);d[i]=h
            np.testing.assert_allclose((energy(e+d,g,q)-energy(e-d,g,q))/(2*h),s[i],rtol=1e-7,atol=1e-8)
        for i in range(6):
            for k in range(3):
                d=np.zeros((6,3));d[i,k]=h
                np.testing.assert_allclose((energy(e,g+d,q)-energy(e,g-d,q))/(2*h),m[i,k],rtol=1e-7,atol=1e-8)
        for i in range(3):
            d=np.zeros(3);d[i]=h
            np.testing.assert_allclose((energy(e,g,q+d)-energy(e,g,q-d))/(2*h),-D[i],rtol=1e-7,atol=1e-8)

    def test_rotating_basis(self):
        R=2.;z=.25;C=.12
        es=C/(R+z)+.5*(C/(R+z))**2
        e=np.array([0.,es,0.,0.,0.,0.])
        dz=np.zeros(6);dz[1]=-C/(R+z)**2-C*C/(R+z)**3
        g=covariant_strain_gradients(e,np.zeros(6),np.zeros(6),dz,radius_m=R,z_m=z)
        self.assertAlmostEqual(g[5,1],-2*es/(R+z))
        t,b=gradient_invariants(g)
        self.assertAlmostEqual(t[1],0.)
        self.assertAlmostEqual(b[2],-es/(R+z))
        np.testing.assert_allclose(reference_electric_field(2,3,5,radius_m=R,z_m=z),[-2,-3/(1+z/R),-5])

    def test_zero_gradient_classical_limit(self):
        p=G3DMaterial(120,.25,0.,0.,0.,4.)
        e=np.array([.1,-.07,.02,.03,-.015,.045])
        s,m,D=enthalpy_derivatives(e,np.ones((6,3)),np.array([1.,2.,3.]),p)
        np.testing.assert_array_equal(m,np.zeros((6,3)))
        np.testing.assert_allclose(D,[4,8,12])

    def test_route_blocks_micro(self):
        self.assertEqual(numerical_status({}),"EXISTING_CLASSICAL_ELASTIC")
        case={"scale_model":"MICRO","geometry_model":"GREEN_LAGRANGE",
            "units":{"system":"SI_N_M_PA"},"material":{"E0":1e11,"nu0":.2},
            "micro":{"gradient_model":"G3D","ell_m":1e-9,"mu1_c_per_m":0.,
                "mu2_c_per_m":1e-6,"kappa_f_per_m":1e-10,"rho_kg_m3":2400.,
                "eta_s":0.,"potential_order":2}}
        self.assertEqual(numerical_status(case),"NOT_IMPLEMENTED_COUPLED_G3D")
        case["micro"]["gradient_model"]="Gz"
        self.assertEqual(numerical_status(case),"NOT_IMPLEMENTED_COUPLED_GZ")
        case["micro"]["gradient_model"]="unknown"
        with self.assertRaisesRegex(ValueError,"gradient_model"):
            numerical_status(case)

if __name__=="__main__":unittest.main()
