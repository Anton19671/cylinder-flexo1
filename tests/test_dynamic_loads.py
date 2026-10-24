#!/usr/bin/env python3
"""Static unit checks of time-dependent load laws; no shell solve."""
import unittest
import numpy as np

from cylinder_dynamic_loads import (
    normalize_time_law, LoadChannel, DynamicLoadProgram
)


class TestDynamicLoads(unittest.TestCase):
    def law(self,kind,**extra):
        return normalize_time_law(dict(kind=kind,**extra))

    def test_step_and_ramp(self):
        s=self.law("STEP",amplitude=3,start_time_s=2)
        self.assertEqual(s.value(1.999),0)
        self.assertEqual(s.value(2),3)
        r=self.law("RAMP",amplitude=3,start_time_s=2,duration_s=2)
        np.testing.assert_allclose(r.value(np.array([0,2,3,4,7])),[0,0,1.5,3,3])

    def test_harmonic_multifrequency_damped(self):
        h=self.law("HARMONIC",frequency_hz=2,amplitude=4)
        self.assertAlmostEqual(h.value(.125),4)
        self.assertEqual(h.value(-.1),0)
        self.assertTrue(h.validate_time_grid(0,.01,100))
        with self.assertRaisesRegex(ValueError,"undersamples"):
            h.validate_time_grid(0,.04,100)
        m=self.law("MULTIHARMONIC",harmonics=[
            {"amplitude":2,"frequency_hz":1},
            {"amplitude":1,"frequency_hz":3}],amplitude=1)
        self.assertAlmostEqual(m.value(.25),1,places=10)
        d=self.law("DAMPED_HARMONIC",frequency_hz=1,amplitude=2,decay_rate_per_s=1)
        self.assertAlmostEqual(d.value(.25),2*np.exp(-.25))

    def test_pulses_and_chirp(self):
        pulse=self.law("HALF_SINE_PULSE",amplitude=5,start_time_s=2,duration_s=.4)
        self.assertAlmostEqual(pulse.value(2.2),5)
        self.assertAlmostEqual(pulse.value(2.5),0)
        triangle=self.law("TRIANGULAR_PULSE",amplitude=7,start_time_s=1,duration_s=2)
        np.testing.assert_allclose(triangle.value(np.array([0,1,2,3,4])),[0,0,7,0,0])
        c=self.law("CHIRP_LINEAR",amplitude=2,frequency_start_hz=3,
                   frequency_end_hz=8,start_time_s=1,duration_s=2)
        self.assertEqual(c.value(.5),0)
        self.assertEqual(c.value(3.1),0)

    def test_table_extrapolation_and_window(self):
        tab=self.law("TABULATED",times_s=[0,1,2],factors=[0,5,1])
        self.assertEqual(tab.value(.5),2.5)
        with self.assertRaisesRegex(ValueError,"outside tabulated"):
            tab.value(-1)
        self.assertTrue(tab.validate_time_grid(0,.1,20))
        with self.assertRaisesRegex(ValueError,"full time window"):
            tab.validate_time_grid(0,.1,21)
        hold=self.law("TABULATED",times_s=[0,1],factors=[2,3],extrapolation="HOLD")
        self.assertEqual(hold.value(-1),2)
        zero=self.law("TABULATED",times_s=[0,1],factors=[2,3],extrapolation="ZERO")
        self.assertEqual(zero.value(1.1),0)

    def test_superposition_and_charge(self):
        q=np.ones((3,4,5));e=np.full((2,4,5),2.)
        program=DynamicLoadProgram([
            LoadChannel("radial","MECHANICAL",q,self.law("STEP",amplitude=2)),
            LoadChannel("axial","MECHANICAL",q*3,self.law("RAMP",amplitude=1,duration_s=2)),
            LoadChannel("charge","FREE_CHARGE",e,self.law("CONSTANT",amplitude=-4)),
        ],mechanical_shape=q.shape,electrical_shape=e.shape)
        m,c=program.at(1)
        np.testing.assert_allclose(m,q*3.5)
        np.testing.assert_allclose(c,e*(-4))
        np.testing.assert_allclose(program.force_callback()(1),m.flatten())

    def test_validation(self):
        for spec in (
            {"kind":"RAMP","amplitude":1,"duration_s":0},
            {"kind":"HARMONIC","frequency_hz":0},
            {"kind":"TABULATED","times_s":[1,0],"factors":[0,1]},
            {"kind":"PYTHON","expression":"__import__('os')"},
        ):
            with self.assertRaises(ValueError):normalize_time_law(spec)


if __name__=="__main__":
    unittest.main()
