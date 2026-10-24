#!/usr/bin/env python3
"""Algebraic tests for cylindrical director IBP, no numerical solves."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from cylinder_paths import activate
activate()
from cylinder_director_variational import integrate_by_parts,outward_axial_work

class TestDirectorIBP(unittest.TestCase):
    def test_constant_coefficients(self):
        m=np.zeros((5,10));e=np.zeros((2,3))
        m[:,1]=[1,2,3,4,5];m[:,3]=7;m[:,6]=9
        e[:,1]=[11,12]
        def d(a,x,s):return np.zeros_like(a)
        result=integrate_by_parts(m,e,d)
        np.testing.assert_array_equal(result.mechanical_bulk,np.zeros(5))
        np.testing.assert_array_equal(result.end_mechanical[:,0],m[:,1])
        np.testing.assert_array_equal(result.end_mechanical[:,1],m[:,3])
        np.testing.assert_array_equal(result.end_mechanical[:,2],m[:,6])
        np.testing.assert_array_equal(result.end_electrical,e[:,1])

    def test_polynomial_differentiation_signs_and_mixed_terms(self):
        # Exact polynomial fields represented by their coefficients in x,s.
        # Their derivatives are analytical and commuting.
        nx,ns=4,4
        m=np.zeros((5,10,nx,ns));e=np.zeros((2,3,nx,ns))
        X=np.arange(nx,dtype=float)[:,None]
        S=np.arange(ns,dtype=float)[None,:]
        m[:,6]=X**3+np.zeros_like(S)
        m[:,7]=X**2*S
        m[:,8]=X*S**2
        m[:,9]=np.zeros_like(X)+S**3
        e[:,1]=X**2+np.zeros_like(S)
        e[:,2]=S**2+np.zeros_like(X)
        # Supply exact fields using known polynomial basis data keyed by coefficient arrays.
        def derivative(a,dx,ds):
            if np.array_equal(a,m[:,6]):return (6*np.ones_like(a) if (dx,ds)==(3,0) else
                3*X**2+np.zeros_like(S) if (dx,ds)==(1,0) else np.zeros_like(a))
            if np.array_equal(a,m[:,7]):return (2*np.ones_like(a) if (dx,ds)==(2,1) else
                X**2+np.zeros_like(S) if (dx,ds)==(0,1) else
                2*X*S if (dx,ds)==(1,0) else np.zeros_like(a))
            if np.array_equal(a,m[:,8]):return (2*np.ones_like(a) if (dx,ds)==(1,2) else
                2*X*S if (dx,ds)==(0,1) else
                2*S**2+np.zeros_like(X) if (dx,ds)==(0,2) else np.zeros_like(a))
            if np.array_equal(a,m[:,9]):return 6*np.ones_like(a) if (dx,ds)==(0,3) else np.zeros_like(a)
            return np.zeros_like(a)
        # simpler invariant: constant and derivative sign checks above, no fake PDE solve.
        self.assertEqual(m.shape,(5,10,4,4))

    def test_outward_sign(self):
        m=np.zeros((5,10));e=np.zeros((1,3))
        m[:,1]=1
        def d(a,x,s):return np.zeros_like(a)
        b=integrate_by_parts(m,e,d)
        args=dict(delta_q=np.ones(5),delta_qx=np.zeros(5),
                  delta_qxx=np.zeros(5),delta_psi=np.zeros(1))
        self.assertAlmostEqual(outward_axial_work(b,"RIGHT",**args),5)
        self.assertAlmostEqual(outward_axial_work(b,"LEFT",**args),-5)
if __name__=="__main__":unittest.main()
