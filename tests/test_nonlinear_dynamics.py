#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast, synthetic-signal tests for the NON-solver dynamics postprocessor."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from cylinder_paths import activate
activate()
from nonlinear_dynamics import (read_trace,sample_interval,spectral_estimators,
    stroboscopic_poincare,autocorrelation,read_bifurcation_samples,save_analysis)


class TestTraceDiagnostics(unittest.TestCase):
    def test_hann_fft_peak_and_psd_variance(self):
        fs=1024.
        t=np.arange(4096,dtype=float)/fs
        amp=0.35
        sig=amp*np.sin(2*np.pi*64*t)
        freq,a,psd=spectral_estimators(t,sig)
        imax=int(np.argmax(a[1:])+1)
        self.assertAlmostEqual(freq[imax],64,places=12)
        self.assertAlmostEqual(a[imax],amp,delta=5e-4)
        self.assertAlmostEqual(float(np.sum(psd)*(fs/len(t))),amp**2/2,delta=2e-4)

    def test_stroboscopic_phase_and_transient_alignment(self):
        freq=10.
        t=np.arange(3000,dtype=float)/1000
        # Observable x=sin(2*pi*f*t); derivative=2*pi*f*cos(...)
        x=np.sin(2*np.pi*freq*t)
        v=2*np.pi*freq*np.cos(2*np.pi*freq*t)
        cut=t>0.725
        tp,xp,vp=stroboscopic_poincare(t[cut],x[cut],v[cut],
            excitation_hz=freq,phase_rad=0)
        self.assertTrue(np.allclose(xp,0,atol=1e-10))
        self.assertTrue(np.allclose(vp,2*np.pi*freq,atol=1e-10))
        self.assertAlmostEqual((tp[0]*freq)%1.,0.,delta=1e-10)

    def test_invalid_sampling_rejected(self):
        t=np.arange(100,dtype=float)*0.01
        t[30:]+=0.02
        with self.assertRaisesRegex(ValueError,"Nonuniform"):
            sample_interval(t)
        t=np.arange(100,dtype=float)*0.01
        x=np.zeros(100)
        with self.assertRaisesRegex(ValueError,"forcing period"):
            stroboscopic_poincare(t,x,x,excitation_hz=50)

    def test_finite_trace_validation(self):
        with TemporaryDirectory() as d:
            f=Path(d)/"trace.csv"
            with f.open("w",newline="",encoding="utf-8") as fd:
                w=csv.writer(fd);w.writerow(["time_s","w","velocity"])
                for n in range(64):w.writerow([n*.01,n*.002,.2])
            t,x,v=read_trace(f,velocity_column="velocity")
            self.assertEqual(len(t),64)
            self.assertEqual(len(v),64)
            self.assertTrue(np.all(np.diff(t)>0))

    def test_bifurcation_requires_repeated_poincare_samples(self):
        with TemporaryDirectory() as d:
            csvpath=Path(d)/"bif.csv"
            with csvpath.open("w",newline="",encoding="utf-8") as fd:
                w=csv.writer(fd);w.writerow(["parameter","poincare_w"])
                for p in [1,2,3]:
                    for i in range(5):w.writerow([p,p*.1+i*.001])
            a=read_bifurcation_samples(csvpath)
            self.assertEqual(a.shape,(15,2))
            with csvpath.open("w",newline="",encoding="utf-8") as fd:
                w=csv.writer(fd);w.writerow(["parameter","poincare_w"])
                for i in range(9):w.writerow([i,i*.1])
            with self.assertRaisesRegex(ValueError,">=3 parameter"):
                read_bifurcation_samples(csvpath)

    def test_save_creates_plot_data_and_metadata(self):
        with TemporaryDirectory() as d:
            f=Path(d)/"trace.csv"
            with f.open("w",newline="",encoding="utf-8") as fd:
                w=csv.writer(fd);w.writerow(["time_s","w"])
                for i in range(800):
                    t=i*.01
                    w.writerow([t,np.sin(2*np.pi*2*t)])
            out=Path(d)/"DYNAMICS"
            report=save_analysis(f,out,discard_s=.3,excitation_hz=2)
            self.assertGreater(report["poincare_sample_count"],2)
            self.assertEqual(report["lyapunov_spectrum"].split("_")[0],"NOT")
            self.assertEqual(report["status"],"POSTPROCESSING_ONLY_NO_DYNAMIC_SOLVER_EXECUTED")
            for path in ("TIME_HISTORY.csv","SPECTRUM.csv",
                         "DYNAMICS_ANALYSIS_MANIFEST.json","02_PHASE_PORTRAIT.png",
                         "04_POWER_SPECTRAL_DENSITY.png","06_POINCARE.png"):
                self.assertTrue((out/path).is_file(),path)
                self.assertGreater((out/path).stat().st_size,0)
            meta=json.loads((out/"DYNAMICS_ANALYSIS_MANIFEST.json").read_text("utf-8"))
            self.assertEqual(meta["velocity_source"],"numerical_gradient")

    def test_autocorrelation_zero_lag(self):
        t=np.arange(1000,dtype=float)/100
        sig=np.cos(2*np.pi*4*t)
        lags,r=autocorrelation(t,sig)
        self.assertEqual(lags[0],0)
        self.assertAlmostEqual(r[0],1,places=12)
        self.assertTrue(np.all(np.abs(r)<=1+1.e-10))


if __name__=="__main__":
    unittest.main()
