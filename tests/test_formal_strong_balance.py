"""Symbolic polynomial checks of the continuum Euler jet algebra."""
import unittest
import numpy as np

from cylinder_formal_strong_balance import formal_coupled_balance


class FormalStrongBalanceTests(unittest.TestCase):
    def test_mixed_mechanical_jet_appears_once_and_electric_signs(self):
        x=np.arange(5,dtype=float)[:,None]
        s=np.arange(7,dtype=float)[None,:]
        mc=np.zeros((4,2,5,7))
        # A00=3, A10=x^2, A01=s^2, A11=2*x*s.
        mc[0]=3.
        mc[1]=x*x
        mc[2]=s*s
        mc[3]=2*x*s
        ec=np.zeros((3,1,5,7))
        ec[0]=4.
        ec[1]=x*x
        ec[2]=s*s

        # Exact polynomial derivative independent of finite differences.
        def derivative(field, ax, az):
            if (ax,az)==(1,0):
                if np.array_equal(field,mc[1]) or np.array_equal(field,ec[1]):
                    return np.broadcast_to(2*x,field.shape).copy()
            if (ax,az)==(0,1):
                if np.array_equal(field,mc[2]) or np.array_equal(field,ec[2]):
                    return np.broadcast_to(2*s,field.shape).copy()
            if (ax,az)==(1,1) and np.array_equal(field,mc[3]):
                return np.full_like(field,2.)
            raise AssertionError("unexpected differentiation request")

        balance=formal_coupled_balance(mc,(
            ("q",0,0),("qx",1,0),("qs",0,1),("qxs",1,1)),
            derivative,ec,mechanical_load=np.ones((2,5,7)),
            electrical_charge=np.ones((1,5,7)))
        expected=4.-2*x-2*s
        np.testing.assert_allclose(balance.mechanical,
                                   np.broadcast_to(expected,(2,5,7)))
        np.testing.assert_allclose(balance.electrical,
                                   np.broadcast_to(5.-2*x-2*s,(1,5,7)))

    def test_missing_electric_model_rejects_charge(self):
        mc=np.zeros((1,1,4,5))
        with self.assertRaisesRegex(ValueError,"charge requires"):
            formal_coupled_balance(mc,(("q",0,0),),
                                   lambda *_:None,electrical_charge=np.zeros((1,4,5)))


if __name__=="__main__":unittest.main()
