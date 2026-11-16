"""Manufactured tests for independent cylindrical strong interior operator."""
import unittest
import numpy as np
from cylinder_direct_strong_fdm import (
    DirectStrongGrid,direct_coefficient_derivative,direct_strong_jet_balance)
from cylinder_mkr_field_jets import MKRFieldJets


class DirectStrongFDMTests(unittest.TestCase):
    def test_periodic_gradient_energies_have_nonnegative_checkerboard_stiffness(self):
        # Pure manufactured coefficient laws A=derivative(q), no shell.
        # The old D+² field jet combined with the centered coefficient
        # derivative gave a NEGATIVE fourth-order checkerboard stiffness.
        for ns in (4,8,12):
            g=DirectStrongGrid(7,ns,1.2,2.)
            jets=MKRFieldJets(g)
            q=np.broadcast_to((-1.)**np.arange(ns),(1,g.nx,ns)).copy()
            for order in (1,2,3):
                with self.subTest(ns=ns,order=order):
                    field_derivative=jets.derivative(q,0,order)
                    coeff=np.stack((np.zeros_like(q),field_derivative))
                    result=direct_strong_jet_balance(
                        coeff,(("q",0,0),("qs",0,order)),g)
                    expected=(4./g.ds**2)**order*q
                    np.testing.assert_allclose(result.mechanical_bulk,expected,
                                               rtol=1.e-12,atol=1.e-12)
                    self.assertGreater(float(np.sum(q*result.mechanical_bulk)),0.)
                    np.testing.assert_allclose(
                        jets.prepare(q,(("jet",0,order),))[0],field_derivative)

    def test_periodic_gradient_work_matches_energy_for_general_fields(self):
        rng=np.random.default_rng(247)
        for ns in (5,8):
            g=DirectStrongGrid(7,ns,1.2,2.)
            jets=MKRFieldJets(g)
            q=rng.normal(size=(1,g.nx,ns))
            for order in (1,2,3):
                with self.subTest(ns=ns,order=order):
                    derivative=jets.derivative(q,0,order)
                    balance=direct_strong_jet_balance(
                        np.stack((np.zeros_like(q),derivative)),
                        (("q",0,0),("qs",0,order)),g)
                    self.assertAlmostEqual(float(np.sum(q*balance.mechanical_bulk)),
                                           float(np.sum(derivative**2)),places=9)

    def test_axial_curvature_uses_direct_laplacian_and_detects_checkerboard(self):
        for nx in (7,11,17):
            with self.subTest(nx=nx):
                grid=DirectStrongGrid(nx,6,1.5,2.)
                jets=MKRFieldJets(grid)
                signal=np.broadcast_to(
                    (-1.)**np.arange(nx)[None,:,None],
                    (1,nx,grid.ns)).copy()
                curvature=jets.derivative(signal,2,0)
                np.testing.assert_allclose(
                    curvature[:,1:-1,:],
                    -4.*signal[:,1:-1,:]/grid.dx**2)
                self.assertGreater(
                    float(np.linalg.norm(curvature[:,1:-1,:])),0.)
                np.testing.assert_allclose(
                    jets.prepare(signal,(("qxx",2,0),))[0],
                    curvature,rtol=0.,atol=0.)

    def test_axial_second_derivative_polynomial_accuracy_at_both_ends(self):
        grid=DirectStrongGrid(11,6,1.5,2.)
        jets=MKRFieldJets(grid)
        x=np.arange(grid.nx)*grid.dx
        for power in (0,1,2,3):
            with self.subTest(power=power):
                values=np.broadcast_to(
                    x[None,:,None]**power,
                    (1,grid.nx,grid.ns)).copy()
                expected=(np.zeros_like(values) if power<2 else
                    np.broadcast_to(
                        power*(power-1)*x[None,:,None]**(power-2),
                        values.shape))
                np.testing.assert_allclose(
                    jets.derivative(values,2,0),expected,
                    atol=2.e-10,rtol=2.e-10)

    def test_polynomials_through_axial_order_three(self):
        g=DirectStrongGrid(11,6,1.5,2.)
        x=np.arange(g.nx)*g.dx
        for power in range(7):
            field=np.broadcast_to((x**power)[:,None],(g.nx,g.ns))
            for order in (1,2,3):
                exact=np.zeros_like(field) if power<order else np.broadcast_to(
                    (np.prod(np.arange(power-order+1,power+1))*
                     x**(power-order))[:,None],field.shape)
                np.testing.assert_allclose(
                    direct_coefficient_derivative(field,order,0,g),exact,
                    rtol=2e-9,atol=2e-9)

    def test_even_periodic_checkerboard_not_null(self):
        for ns in (4,8,12):
            g=DirectStrongGrid(7,ns,1.2,2.)
            p=np.broadcast_to((-1.)**np.arange(ns),(g.nx,ns))
            np.testing.assert_allclose(
                direct_coefficient_derivative(p,0,2,g),
                -4*p/g.ds**2)
            ec=np.zeros((3,1,g.nx,ns))
            ec[2,0]=(np.roll(p,-1,axis=-1)-p)/g.ds
            mc=np.zeros((1,1,g.nx,ns))
            result=direct_strong_jet_balance(mc,(("q",0,0),),g,ec)
            np.testing.assert_allclose(result.electrical_bulk[0],
                                        4*p/g.ds**2)
            self.assertGreater(np.max(np.abs(result.electrical_bulk)),.3)

    def test_third_order_mechanics_and_three_end_actions(self):
        g=DirectStrongGrid(11,6,1.5,2.)
        x=(np.arange(g.nx)*g.dx)[:,None]
        mc=np.zeros((4,1,g.nx,g.ns))
        mc[0,0]=3.
        mc[1,0]=2*x**3
        mc[2,0]=5*x**4
        mc[3,0]=7*x**5
        result=direct_strong_jet_balance(
            mc,(("q",0,0),("qx",1,0),("qxx",2,0),("qxxx",3,0)),g)
        expected=np.broadcast_to(3-366*x**2,(g.nx,g.ns))
        np.testing.assert_allclose(result.mechanical_bulk[0],expected,
                                   rtol=2e-8,atol=4e-8)
        np.testing.assert_allclose(result.mechanical_end_displacement[1,0],
                                   np.full(g.ns,122*x[-1,0]**3),atol=1e-8)
        np.testing.assert_allclose(result.mechanical_end_slope[1,0],
                                   np.full(g.ns,-30*x[-1,0]**4),atol=1e-8)
        np.testing.assert_allclose(result.mechanical_end_curvature[1,0],
                                   np.full(g.ns,7*x[-1,0]**5),atol=1e-8)
        self.assertEqual(result.safe_interior_x_nodes,tuple(range(3,8)))
        self.assertEqual(result.highest_axial_jet,3)

    def test_invalid_inputs_rejected(self):
        g=DirectStrongGrid(7,4,1.,1.)
        c=np.zeros((1,1,7,4))
        with self.assertRaises(ValueError):
            direct_strong_jet_balance(c,(("qx",1,0),),g)
        with self.assertRaises(ValueError):
            direct_coefficient_derivative(c[0],2,2,g)
        with self.assertRaises(ValueError):
            direct_strong_jet_balance(c,(("q",0,0),),g,
                                      electrical_charge=np.zeros((1,7,4)))


if __name__=="__main__":
    unittest.main()
