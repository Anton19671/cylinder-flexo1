"""Regression for nonlinear electric rank growth in Gauss X/Y MVI.

Pure spatial manufactured fields: not a physical cylindrical-shell run.
"""
import unittest
import numpy as np
from cylinder_gauss2_mvi import _rank_factors, _profile_field


class CubicElectricRankGrowthTests(unittest.TestCase):
    @staticmethod
    def fields():
        x=np.linspace(0.,1.,8)
        y=np.linspace(0.,1.,8)
        a=np.outer(np.sin(1.3*x+0.5),np.cos(0.8*y+0.4))
        b=0.65*np.outer(np.cos(2.1*x+0.9),np.sin(1.65*y+0.6))
        q=(a+b)[None]
        p=0.5*q+2.*q**3
        return q,p

    def test_mechanical_two_electric_six(self):
        q,p=self.fields()
        self.assertEqual(np.linalg.matrix_rank(q[0],tol=1e-10),2)
        self.assertEqual(np.linalg.matrix_rank(p[0],tol=1e-10),6)
        with self.assertRaisesRegex(ValueError,'OUTSIDE_FIXED_RANK'):
            _rank_factors(q,1,2e-10,'Q')
        x,y=_rank_factors(q,2,2e-10,'Q')
        np.testing.assert_allclose(_profile_field(x,y),q,atol=1e-14,rtol=0.)

    def test_electric_rank_not_implicitly_truncated(self):
        q,p=self.fields()
        together=np.concatenate((q,p))
        for rank in range(1,6):
            with self.subTest(rank=rank):
                with self.assertRaisesRegex(ValueError,'OUTSIDE_FIXED_RANK'):
                    _rank_factors(together,rank,2e-10,'ELECTRIC')
        for rank in (6,7,8):
            with self.subTest(rank=rank):
                x,y=_rank_factors(together,rank,2e-10,'ELECTRIC')
                observed=_profile_field(x,y)
                np.testing.assert_allclose(observed,together,atol=2e-14,rtol=0.)

    def test_tiny_electric_component_preserves_rank_gate(self):
        q,p=self.fields()
        for amplitude in (1.,1e-8,1e-13,1e-19):
            with self.subTest(amplitude=amplitude):
                together=np.concatenate((q,amplitude*p))
                with self.assertRaisesRegex(ValueError,'OUTSIDE_FIXED_RANK'):
                    _rank_factors(together,5,2e-10,'ELECTRIC_SMALL')
                x,y=_rank_factors(together,6,2e-10,'ELECTRIC_SMALL')
                observed=_profile_field(x,y)
                np.testing.assert_allclose(observed[0],q[0],atol=2e-14,rtol=0.)
                self.assertLess(
                    np.max(np.abs(observed[1]-amplitude*p[0])) /
                    np.max(np.abs(amplitude*p)),1e-12)


if __name__=='__main__':
    unittest.main()
