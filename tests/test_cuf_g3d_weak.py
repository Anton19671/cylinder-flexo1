#!/usr/bin/env python3
"""CUF G3D thickness weak-form source-equation regression."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cylinder_paths import activate
activate()
import unittest
import numpy as np
from flexo_gradient_g3d import G3DMaterial
from cuf_g3d_weak import (CUFSurfaceJets, point_state, te_basis, mass_matrix_te,
    gauge_moments, thickness_enthalpy, thickness_weak_coefficients,
    MECHANICAL_VARIATIONS, ELECTRICAL_VARIATIONS)

GEOMETRY=dict(radius_m=2.2,thickness_m=.8)
MAT=G3DMaterial(120,.24,.16,.65,-.31,3.5)

def jets(n=1,p=1,seed=10):
    r=np.random.default_rng(seed)
    return CUFSurfaceJets(*[r.normal(0,.012,(3,n+1)) for _ in range(6)],
                          *[r.normal(0,.032,p+1) for _ in range(3)])

def delta(j,n=100):
    return jets(j.te_order,j.potential_order,n)

def shift(j,d,h):
    return j.replaced(**{name:getattr(j,name)+h*getattr(d,name)
             for name in (*MECHANICAL_VARIATIONS,*ELECTRICAL_VARIATIONS)})

def weak_dot(a,d):
    return float(sum(np.sum(a.mechanical[...,i]*getattr(d,name))
                     for i,name in enumerate(MECHANICAL_VARIATIONS))
                +sum(np.sum(a.electrical[...,i]*getattr(d,name))
                     for i,name in enumerate(ELECTRICAL_VARIATIONS)))

class TestKryskoZhongCUF(unittest.TestCase):
    def test_gateaux_TE1_to_TE4_separate_electrical_orders(self):
        for n,p in ((1,1),(2,0),(3,2),(4,4)):
            with self.subTest(TE=n,P=p):
                j=jets(n,p,15+n);d=delta(j,31+n)
                a=thickness_weak_coefficients(j,MAT,**GEOMETRY,n_gauss=8)
                e=1e-6
                fd=(thickness_enthalpy(shift(j,d,e),MAT,**GEOMETRY,n_gauss=8)
                    -thickness_enthalpy(shift(j,d,-e),MAT,**GEOMETRY,n_gauss=8))/(2*e)
                np.testing.assert_allclose(weak_dot(a,d),fd,rtol=4e-7,atol=4e-9)
                self.assertEqual(a.mechanical.shape,(3,n+1,6))
                self.assertEqual(a.electrical.shape,(p+1,3))

    def test_dimensionless_basis_mass_and_gauge(self):
        h=GEOMETRY["thickness_m"];R=GEOMETRY["radius_m"]
        f,df,d2=te_basis(.11,h,4)
        self.assertAlmostEqual(f[1],.22/h)
        self.assertAlmostEqual(df[1],2/h)
        self.assertAlmostEqual(d2[4],12*(2/h)**2*(.22/h)**2)
        mass=mass_matrix_te(**GEOMETRY,rho_kg_m3=2000,te_order=4)
        self.assertAlmostEqual(mass[0,1],2000*h*h/(6*R))
        np.testing.assert_allclose(mass[0,:]/2000,gauge_moments(**GEOMETRY,potential_order=4))
        self.assertTrue(np.linalg.eigvalsh(mass).min()>0)

    def test_cylindrical_covariant_terms(self):
        j=jets();zero=j.replaced(**{k:np.zeros_like(getattr(j,k))
                     for k in (*MECHANICAL_VARIATIONS,*ELECTRICAL_VARIATIONS)})
        q=zero.q.copy();q[2,0]=.07;z=.13
        s=point_state(zero.replaced(q=q),z_m=z,**GEOMETRY)
        R=GEOMETRY["radius_m"];e=.07/(R+z)+.5*(.07/(R+z))**2
        self.assertAlmostEqual(s.strain6[1],e)
        self.assertAlmostEqual(s.gradient6x3[5,1],-2*e/(R+z))

    def test_rigid_rotation_is_strain_free(self):
        j=jets();zero=j.replaced(**{k:np.zeros_like(getattr(j,k))
                     for k in (*MECHANICAL_VARIATIONS,*ELECTRICAL_VARIATIONS)})
        R=GEOMETRY["radius_m"];h=GEOMETRY["thickness_m"];ang=.32
        q=zero.q.copy()
        q[1]=[R*np.sin(ang),.5*h*np.sin(ang)]
        q[2]=[R*(np.cos(ang)-1),.5*h*(np.cos(ang)-1)]
        for z in (-.4,0,.31,.4):
            state=point_state(zero.replaced(q=q),z_m=z,**GEOMETRY)
            np.testing.assert_allclose(state.strain6,0,atol=1e-14)
            np.testing.assert_allclose(state.gradient6x3,0,atol=1e-14)
            self.assertAlmostEqual(state.det_f,1,places=12)

    def test_mixed_xs_and_electrical_gauge(self):
        j=jets();d=delta(j,38)
        for field in (*MECHANICAL_VARIATIONS,*ELECTRICAL_VARIATIONS):
            d=d.replaced(**{field:np.zeros_like(getattr(d,field))})
        qxs=d.qxs.copy();qxs[1,0]=1;d=d.replaced(qxs=qxs)
        a=thickness_weak_coefficients(j,MAT,**GEOMETRY,n_gauss=8)
        e=1e-6
        fd=(thickness_enthalpy(shift(j,d,e),MAT,**GEOMETRY,n_gauss=8)-
            thickness_enthalpy(shift(j,d,-e),MAT,**GEOMETRY,n_gauss=8))/(2*e)
        np.testing.assert_allclose(a.mechanical[1,0,4],fd,rtol=4e-7,atol=2e-9)
        q=j.psi.copy();q[0]+=10
        self.assertAlmostEqual(thickness_enthalpy(j,MAT,**GEOMETRY,n_gauss=8),
            thickness_enthalpy(j.replaced(psi=q),MAT,**GEOMETRY,n_gauss=8),places=11)

    def test_classical_limit_and_invalid_jacobian(self):
        j=jets()
        old=G3DMaterial(120,.24,0,0,0,3.5)
        a=thickness_weak_coefficients(j,old,**GEOMETRY,n_gauss=8)
        np.testing.assert_array_equal(a.mechanical[...,3:],0)
        qx=j.qx.copy();qx[0,0]=-2
        zero=j.replaced(**{k:np.zeros_like(getattr(j,k))
                     for k in (*MECHANICAL_VARIATIONS,*ELECTRICAL_VARIATIONS)})
        with self.assertRaisesRegex(ValueError,"INVALID_DEFORMATION_MAPPING"):
            point_state(zero.replaced(qx=qx),z_m=0,**GEOMETRY)

if __name__=="__main__":
    unittest.main()
