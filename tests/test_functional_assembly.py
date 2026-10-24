#!/usr/bin/env python3
"""Reference variational assembler checks; no engineering shell solve."""
import unittest
import numpy as np
from cylinder_functional_assembly import (
    CylinderFunctional, SurfaceOperators, surface_trapezoid,
    periodic_face_gradient, axial_fd_gradient,
)
from flexo_gradient_g3d import G3DMaterial


class FunctionalAssemblyChecks(unittest.TestCase):
    def setUp(self):
        self.ops=surface_trapezoid(4,5,1.2,3.)
        self.kw=dict(radius_m=3., thickness_m=.2,
                     surface_operators=self.ops,n_gauss=2)
        self.rng=np.random.default_rng(72)

    def test_directional_enthalpy_gradient_and_coupling(self):
        material=G3DMaterial(50.,.25,.08,.004,-.002,.9)
        for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE4"):
            with self.subTest(theory=theory):
                f=CylinderFunctional(theory,material,**self.kw)
                q=self.rng.normal(0,2e-4,(f.n_mechanical,*self.ops.shape))
                phi=self.rng.normal(0,.003,(2,*self.ops.shape))
                dq=self.rng.normal(0,1e-4,q.shape)
                dp=self.rng.normal(0,.002,phi.shape)
                val=f.evaluate(q,phi,energy_breakdown=False)
                h=2e-4
                fp=f.evaluate(q+h*dq,phi+h*dp,energy_breakdown=False).total_enthalpy_j
                fm=f.evaluate(q-h*dq,phi-h*dp,energy_breakdown=False).total_enthalpy_j
                exact=float(np.sum(val.mechanical_residual*dq)+
                            np.sum(val.electrical_residual*dp))
                np.testing.assert_allclose((fp-fm)/(2*h),exact,rtol=1e-5,atol=1e-7)

    def test_symmetric_cross_tangent(self):
        material=G3DMaterial(90.,.24,.08,.04,-.025,1.2)
        for theory in ("KL","CUF_TE2","FSDT","REDDY"):
            with self.subTest(theory=theory):
                f=CylinderFunctional(theory,material,**self.kw)
                q=self.rng.normal(0,2e-4,(f.n_mechanical,*self.ops.shape))
                p=self.rng.normal(0,.003,(2,*self.ops.shape))
                dq=self.rng.normal(0,1e-3,q.shape)
                dp=self.rng.normal(0,.004,p.shape)
                h=2e-4
                rp=f.evaluate(q,p+h*dp,energy_breakdown=False)
                rm=f.evaluate(q,p-h*dp,energy_breakdown=False)
                mixed1=float(np.sum((rp.mechanical_residual-rm.mechanical_residual)*dq)/(2*h))
                rp=f.evaluate(q+h*dq,p,energy_breakdown=False)
                rm=f.evaluate(q-h*dq,p,energy_breakdown=False)
                mixed2=float(np.sum((rp.electrical_residual-rm.electrical_residual)*dp)/(2*h))
                np.testing.assert_allclose(mixed1,mixed2,rtol=2e-4,atol=2e-7)

    def test_gradient_without_electrics(self):
        for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE3"):
            with self.subTest(theory=theory):
                mat=G3DMaterial(60.,.21,.08,0,0,0)
                f=CylinderFunctional(theory,mat,**self.kw)
                q=self.rng.normal(0,.002,(f.n_mechanical,*self.ops.shape))
                r=f.evaluate(q)
                self.assertIsNone(r.electrical_residual)
                self.assertGreater(r.energy_parts_j["gradient"],0)
                self.assertEqual(r.energy_parts_j["dielectric"],0)
                self.assertEqual(r.energy_parts_j["flexoelectric"],0)

    def test_periodic_even_checkerboard(self):
        n=6
        g=periodic_face_gradient(n,2*np.pi)
        self.assertGreater(np.linalg.norm(g@((-1.)**np.arange(n))),1)
        bad=np.zeros((n,n))
        for i in range(n):
            bad[i,(i+1)%n]=1
            bad[i,(i-1)%n]=-1
        bad/=2*(2*np.pi/n)
        with self.assertRaisesRegex(ValueError,"spurious null"):
            SurfaceOperators(axial_fd_gradient(5,1.),bad,np.ones(5),np.ones(n))

    def test_gauge_and_insulated_charge(self):
        mat=G3DMaterial(80.,.23,.1,.001,.003,.8)
        f=CylinderFunctional("CUF_TE1",mat,**self.kw)
        q=self.rng.normal(0,.001,(f.n_mechanical,*self.ops.shape))
        p=self.rng.normal(0,.003,(2,*self.ops.shape))
        r=f.evaluate(q,p)
        shift=p.copy();shift[0]+=8.
        shifted=f.evaluate(q,shift)
        np.testing.assert_allclose(r.total_enthalpy_j,shifted.total_enthalpy_j,atol=1e-9)
        self.assertNotEqual(r.electric_gauge_value,shifted.electric_gauge_value)
        self.assertEqual(len(r.augmented_residual()),q.size+p.size+1)
        charge=np.zeros_like(p);charge[0]+=1
        with self.assertRaisesRegex(ValueError,"net charge"):
            f.evaluate(q,p,free_charge=charge)

    def test_consistent_inertia_and_kl_guard(self):
        mat=G3DMaterial(50,.2,.08,0,0,0)
        for theory in ("CUF_TE1","CUF_TE4","FSDT","REDDY"):
            with self.subTest(theory=theory):
                f=CylinderFunctional(theory,mat,**self.kw)
                u=self.rng.normal(0,.003,(f.n_mechanical,*self.ops.shape))
                d=self.rng.normal(0,.002,u.shape)
                force=f.mass_action(u,density_kg_m3=1300)
                h=1e-4
                fp=f.mass_action(u+h*d,density_kg_m3=1300)
                fm=f.mass_action(u-h*d,density_kg_m3=1300)
                fd=(np.sum((u+h*d)*fp)-np.sum((u-h*d)*fm))/(4*h)
                np.testing.assert_allclose(fd,np.sum(force*d),rtol=5e-5,atol=1e-7)
        f=CylinderFunctional("KL",mat,**self.kw)
        with self.assertRaises(NotImplementedError):
            f.mass_action(np.zeros((3,*self.ops.shape)),density_kg_m3=1300)


if __name__=="__main__":
    unittest.main()
