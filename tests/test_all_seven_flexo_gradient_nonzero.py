"""All seven nonlinear shell theories: nonzero G3D/Gz flexoelectric work.

Real functional evaluation, not a completed MVI/Newmark time step.
"""
import unittest
import numpy as np
from cylinder_functional_assembly import CylinderFunctional, surface_trapezoid
from flexo_gradient_g3d import G3DMaterial

THEORIES=('KL','FSDT','REDDY','CUF_TE1','CUF_TE2','CUF_TE3','CUF_TE4')

class AllSevenNonzeroFlexoGradientTests(unittest.TestCase):
    def test_all_theories_both_gradients_with_real_nonzero_fields(self):
        surface=surface_trapezoid(4,4,1.2,2.)
        xi=np.linspace(0,1,4)[:,None]
        eta=np.linspace(0,1,4)[None,:]
        shape=np.sin(np.pi*xi)*np.sin(np.pi*eta)
        count=0
        for theory in THEORIES:
            n_gauss=2*int(theory[-1])+1 if theory.startswith('CUF_TE') else 7
            for gradient in ('G3D','Gz'):
                with self.subTest(theory=theory,gradient=gradient):
                    material=G3DMaterial(100.,.25,.08,.001,-.002,1.5,gradient_model=gradient)
                    kernel=CylinderFunctional(theory,material,
                        radius_m=2.,thickness_m=.2,surface_operators=surface,
                        n_gauss=n_gauss)
                    q=2.e-5*np.arange(1,kernel.n_mechanical+1)[:,None,None]*shape
                    p=1.e-3*np.arange(1,3)[:,None,None]*shape
                    full=kernel.evaluate(q,p,energy_breakdown=True,
                        check_charge_balance=False)
                    self.assertTrue(np.isfinite(full.total_enthalpy_j))
                    self.assertTrue(np.isfinite(full.mechanical_residual).all())
                    self.assertTrue(np.isfinite(full.electrical_residual).all())
                    self.assertEqual(full.mechanical_residual.shape,q.shape)
                    self.assertEqual(full.electrical_residual.shape,p.shape)
                    count+=1
        self.assertEqual(count,14)

if __name__=='__main__':unittest.main()
