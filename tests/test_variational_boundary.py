#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests of static/dynamic conservative boundary work, no shell solves."""
import unittest
import numpy as np

from cylinder_functional_assembly import CylinderFunctional, surface_trapezoid
from flexo_gradient_g3d import G3DMaterial
from cylinder_variational_boundary import (
    boundary_passport, validate_boundary_passport,
    compile_boundary_program, combine_functional_and_boundary
)


class VariationalBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.ops=surface_trapezoid(5,6,1.2,2.)
        self.kw=dict(radius_m=2.,thickness_m=.2,surface_operators=self.ops,n_gauss=2)
        self.mechanical=G3DMaterial(100.,.25,.08,0.,0.,0.)
        self.electric=G3DMaterial(100.,.25,.08,.1,-.02,1.5)
        self.rng=np.random.default_rng(12)

    def make(self,theory="CUF_TE2",electric=False):
        f=CylinderFunctional(theory,self.electric if electric else self.mechanical,**self.kw)
        q=self.rng.normal(0,.002,(f.n_mechanical,*self.ops.shape))
        p=self.rng.normal(0,.001,(2,*self.ops.shape)) if electric else None
        return f,q,p

    def test_all_theories_edge_energy_gradient(self):
        for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE4"):
            with self.subTest(theory=theory):
                f,q,p=self.make(theory)
                n=f.n_mechanical
                spec={"mechanical":{"left":{"traction":self.rng.normal(0,.1,n).tolist(),
                    "double_traction":self.rng.normal(0,.001,n).tolist(),
                    "spring":(np.ones(n)*.7).tolist()}}}
                result=compile_boundary_program(f,q,p,spec=spec)
                d=self.rng.normal(0,.002,q.shape)
                step=1e-5
                hi=compile_boundary_program(f,q+step*d,p,spec=spec).potential_correction_j
                lo=compile_boundary_program(f,q-step*d,p,spec=spec).potential_correction_j
                np.testing.assert_allclose(
                    (hi-lo)/(2*step),np.sum(result.mechanical_residual_correction*d),
                    atol=1e-9,rtol=1e-6)

    def test_cuf_component_order(self):
        f,q,p=self.make("CUF_TE4")
        spec={"mechanical":{"left":{"support":"SIMPLY_SUPPORTED"}}}
        result=compile_boundary_program(f,q,p,spec=spec)
        m=self.ops.shape[0]*self.ops.shape[1]
        self.assertEqual({row.indices[0]//m for row in result.constraints},{0,5,10})

    def test_outward_normal_double_traction_left_right_signs(self):
        f=self.functional("CUF_TE1",False)
        q=np.zeros((f.n_mechanical,*self.ops.shape))
        n=f.n_mechanical
        magnitude=[0.]*n;magnitude[0]=2.
        for side,sign in (("left",-1.),("right",1.)):
            with self.subTest(side=side):
                normal={"mechanical":{side:{"normal_double_traction":magnitude}}}
                axial={"mechanical":{side:{"double_traction":[sign*x for x in magnitude]}}}
                a=compile_boundary_program(f,q,spec=normal)
                b=compile_boundary_program(f,q,spec=axial)
                np.testing.assert_allclose(a.mechanical_residual_correction,
                                           b.mechanical_residual_correction)
                self.assertAlmostEqual(a.potential_correction_j,b.potential_correction_j)

    def test_outward_normal_triple_traction_same_on_both_ends(self):
        f=self.functional("FSDT",False)
        q=np.zeros((f.n_mechanical,*self.ops.shape))
        force=[1.,0.,0.,0.,0.]
        for side in ("left","right"):
            a=compile_boundary_program(f,q,spec={"mechanical":{
                side:{"normal_triple_traction":force}}})
            b=compile_boundary_program(f,q,spec={"mechanical":{
                side:{"triple_traction":force}}})
            np.testing.assert_array_equal(a.mechanical_residual_correction,
                                          b.mechanical_residual_correction)

    def test_mixed_axial_and_normal_traces_fail(self):
        f=self.functional("CUF_TE1",False)
        q=np.zeros((f.n_mechanical,*self.ops.shape))
        spec={"mechanical":{"left":{
            "double_traction":[0.]*f.n_mechanical,
            "normal_double_traction":[0.]*f.n_mechanical}}}
        with self.assertRaisesRegex(ValueError,"OR outward-normal"):
            compile_boundary_program(f,q,spec=spec)

    def test_reference_higher_trace_switching(self):
        for theory in ("KL","FSDT","REDDY","CUF_TE3"):
            with self.subTest(theory=theory):
                f,q,p=self.make(theory)
                spec={"mechanical":{"right":{"support":"CLAMPED","higher":"FIX_FIRST_NORMAL"}}}
                result=compile_boundary_program(f,q,p,spec=spec)
                self.assertGreater(len(result.constraints),f.n_mechanical*self.ops.shape[1])
                self.assertTrue(any("micro d_x" in row.label for row in result.constraints))
        f,q,p=self.make("CUF_TE2")
        with self.assertRaisesRegex(ValueError,"second normal derivative"):
            compile_boundary_program(f,q,p,spec={
                "mechanical":{"left":{"higher":"FIX_FIRST_AND_SECOND_NORMAL"}}})

    def test_electrodes_and_charge_balance(self):
        f,q,p=self.make("FSDT",electric=True)
        e=compile_boundary_program(f,q,p,spec={"electrical":{
            "outer":{"mode":"POTENTIAL","value":10.},
            "inner":{"mode":"POTENTIAL","value":-10.}}})
        self.assertFalse(e.gauge_required)
        self.assertTrue(any(c.field=="electrical" for c in e.constraints))
        paired={"electrical":{
            "outer":{"mode":"CHARGE","value":2.},
            "inner":{"mode":"CHARGE","value":-2.*(1+.1/f.R)/(1-.1/f.R)}}}
        balanced=compile_boundary_program(f,q,p,spec=paired)
        self.assertTrue(balanced.gauge_required)
        self.assertAlmostEqual(balanced.free_electric_charge_c,0.,places=8)
        with self.assertRaisesRegex(ValueError,"zero total"):
            compile_boundary_program(f,q,p,spec={"electrical":{"outer":{
                "mode":"CHARGE","value":2.}}})
        with self.assertRaisesRegex(ValueError,"conflicting"):
            compile_boundary_program(f,q,p,spec={"electrical":{
                "left":{"mode":"POTENTIAL","value":-2.},
                "outer":{"mode":"POTENTIAL","value":3.}}})

    def test_electric_charge_energy_gradient(self):
        f,q,p=self.make("CUF_TE1",electric=True)
        spec={"electrical":{"outer":{"mode":"CHARGE","value":1.},
                            "inner":{"mode":"POTENTIAL","value":0.}}}
        result=compile_boundary_program(f,q,p,spec=spec)
        d=self.rng.normal(0,.002,p.shape)
        step=1e-5
        plus=compile_boundary_program(f,q,p+step*d,spec=spec).potential_correction_j
        minus=compile_boundary_program(f,q,p-step*d,spec=spec).potential_correction_j
        np.testing.assert_allclose(
            (plus-minus)/(2*step),np.sum(result.electrical_residual_correction*d),
            atol=1e-10,rtol=1e-6)

    def test_dynamic_bc_timing_and_static_rejection(self):
        f,q,p=self.make("FSDT",electric=True)
        law={"kind":"HARMONIC","frequency_hz":5.,"phase_rad":0.}
        spec={"mechanical":{"left":{"traction":[1.]*f.n_mechanical,"time_law":law}},
              "electrical":{"outer":{"mode":"CHARGE","value":1.,"time_law":law},
                            "inner":{"mode":"POTENTIAL","value":0.}}}
        with self.assertRaisesRegex(ValueError,"STATIC"):
            compile_boundary_program(f,q,p,spec=spec)
        a=compile_boundary_program(f,q,p,spec=spec,calculation_kind="DYNAMIC",time_s=0.)
        b=compile_boundary_program(f,q,p,spec=spec,calculation_kind="DYNAMIC",time_s=.05)
        self.assertAlmostEqual(a.potential_correction_j,0.,places=9)
        self.assertNotAlmostEqual(b.potential_correction_j,0.,places=9)
        with self.assertRaisesRegex(ValueError,"time_s"):
            compile_boundary_program(f,q,p,spec=spec,calculation_kind="DYNAMIC")

    def test_electric_off_guard(self):
        f,q,p=self.make("KL")
        with self.assertRaisesRegex(ValueError,"electric BC requires"):
            compile_boundary_program(f,q,p,spec={"electrical":{
                "outer":{"mode":"POTENTIAL","value":3.}}})

    def test_boundary_passport_and_merged_functional(self):
        f,q,p=self.make("FSDT",electric=True)
        spec=boundary_passport(mechanical_left="CLAMPED",mechanical_right="FREE",
            electric_outer="POTENTIAL",electric_values={"outer":0.})
        self.assertTrue(validate_boundary_passport(
            spec,electrical_enabled=True,theory="FSDT"))
        total,rm,re,bc=combine_functional_and_boundary(f,q,p,bc_spec=spec)
        self.assertTrue(np.isfinite(total))
        self.assertEqual(rm.shape,q.shape)
        self.assertEqual(re.shape,p.shape)
        self.assertGreater(len(bc.constraints),0)
        self.assertFalse(bc.gauge_required)


if __name__=="__main__":
    unittest.main()
