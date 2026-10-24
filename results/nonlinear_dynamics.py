#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace-backed nonlinear dynamics diagnostics for a cylindrical shell.

All graphs are computed from PROVIDED time samples, never from load increments,
iteration logs, static solution steps, or generated/fabricated trajectories.

CSV time-series contract: header time_s,w; optional velocity column.
Time is in seconds and strictly increasing, with a near-uniform sampling grid.
Displacements and derivatives retain user-supplied units. Analysis is independent
of MACRO/MICRO solver readiness and cannot claim a dynamic simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


_MIN_SAMPLES = 32
_MAX_SPECTROGRAM_WINDOWS = 512


def _rows(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("CSV file not found: " + str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no column header")
        header = [str(name or "").strip() for name in reader.fieldnames]
        if len(set(header)) != len(header):
            raise ValueError("CSV contains duplicate column names")
        return tuple(header), list(reader)


def read_trace(path, *, time_column="time_s", signal_column="w", velocity_column=""):
    header, rows = _rows(path)
    cols = (time_column, signal_column) + ((velocity_column,) if velocity_column else ())
    for name in cols:
        if name not in header:
            raise ValueError(f"CSV column {name!r} is missing. Available: {header!r}")
    if len(rows) < _MIN_SAMPLES:
        raise ValueError(f"At least {_MIN_SAMPLES} temporal samples are required")
    try:
        a = np.asarray([[float(row[key]) for key in cols] for row in rows], dtype=float)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("CSV contains missing or nonnumeric samples") from exc
    if not np.isfinite(a).all():
        raise ValueError("CSV contains NaN or infinite time/field values")
    t = a[:, 0]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time must be strictly increasing: do not use static load-step data")
    return t, a[:, 1], (a[:, 2] if velocity_column else None)


def sample_interval(t, *, rtol=1e-3):
    t = np.asarray(t, dtype=float)
    if len(t) < _MIN_SAMPLES:
        raise ValueError(f"At least {_MIN_SAMPLES} samples required")
    d = np.diff(t)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        raise ValueError("Time samples must be finite and increasing")
    dt = float(np.mean(d))
    relative = float(np.max(np.abs(d-dt))/dt)
    if relative > rtol:
        raise ValueError(
            f"Nonuniform time grid (max relative step error {relative:.3g}). "
            "Resample explicitly before FFT, PSD or stroboscopic Poincare."
        )
    return dt


def transient_cut(t, x, v, discard_s):
    discard_s = float(discard_s)
    if not np.isfinite(discard_s) or discard_s < 0:
        raise ValueError("Transient duration must be finite and >= 0")
    select = t >= t[0] + discard_s
    if int(np.count_nonzero(select)) < _MIN_SAMPLES:
        raise ValueError("Transient cut leaves too few time samples")
    return t[select], x[select], (v[select] if v is not None else None)


def spectral_estimators(t, x):
    """One-sided Hann coherent-gain amplitude, and density-normalized PSD.

    Peak-amplitude spectrum is an estimate of a monochromatic signal amplitude
    at bin-centered frequencies. PSD units are (signal unit)^2/Hz.
    """
    dt = sample_interval(t)
    x = np.asarray(x, dtype=float)
    n = x.size
    centered = x - float(np.mean(x))
    window = np.hanning(n)
    rfft = np.fft.rfft(centered*window)
    freq = np.fft.rfftfreq(n, dt)
    factor = np.full(len(freq), 2.0)
    factor[0] = 1.0
    if n % 2 == 0:
        factor[-1] = 1.0
    magnitude = factor*np.abs(rfft)/float(np.sum(window))
    psd = factor*np.abs(rfft)**2/(float(1.0/dt)*float(np.sum(window**2)))
    return freq, magnitude, psd


def stroboscopic_poincare(t, signal, velocity, *, excitation_hz, phase_rad=0.):
    """Poincare phase is measured against the absolute experiment time t=0.

    The external excitation must use f*t+phase_rad. No dominant FFT frequency
    is silently substituted for a known periodic forcing frequency.
    """
    frequency = float(excitation_hz)
    phase = float(phase_rad)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Excitation frequency must be positive and finite")
    if not np.isfinite(phase):
        raise ValueError("Excitation phase must be finite")
    dt = sample_interval(t)
    if frequency*dt > 0.25:
        raise ValueError("At least 4 temporal samples per forcing period required")
    period = 1.0/frequency
    phase_time = (-phase/(2*np.pi*frequency)) % period
    start = int(np.ceil((t[0]-phase_time)/period - 1.e-10))
    end = int(np.floor((t[-1]-phase_time)/period + 1.e-10))
    if end-start+1 < 2:
        raise ValueError("Selected trace covers fewer than 2 forcing cycles")
    if end-start+1 > 1000000:
        raise ValueError("Too many Poincare sections requested")
    tp = phase_time+np.arange(start,end+1)*period
    tp = np.clip(tp, t[0], t[-1])
    return tp, np.interp(tp,t,signal), np.interp(tp,t,velocity)


def autocorrelation(t, signal, *, max_lags=400):
    dt = sample_interval(t)
    x = np.asarray(signal, dtype=float)-float(np.mean(signal))
    denom=float(np.dot(x,x))
    if denom <= np.finfo(float).tiny:
        return np.array([0.]), np.array([1.])
    max_lags=min(int(max_lags),len(x)//2)
    # FFT convolution, normalized by available pairs (biased correlation).
    nfft=1 << (2*len(x)-1).bit_length()
    fft=np.fft.rfft(x,nfft)
    raw=np.fft.irfft(fft*np.conjugate(fft),nfft)[:max_lags+1]
    return np.arange(max_lags+1)*dt, raw/denom


def read_bifurcation_samples(path, *, parameter_column="parameter",
                             signal_column="poincare_w"):
    """Read already established steady-state Poincare samples from a sweep.

    Does not run sweeps, detect attractors, or infer bifurcation thresholds.
    Reject a simple load-deflection curve masquerading as a bifurcation series.
    """
    header, rows = _rows(path)
    if parameter_column not in header or signal_column not in header:
        raise ValueError("Sweep CSV requires parameter and Poincare response columns")
    if len(rows) < 9:
        raise ValueError("Need >=9 steady-state Poincare samples across parameters")
    try:
        a = np.asarray([[float(row[parameter_column]), float(row[signal_column])]
                        for row in rows],dtype=float)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Non-numeric bifurcation sweep samples") from exc
    if not np.isfinite(a).all():
        raise ValueError("Bifurcation sweep contains nonfinite samples")
    pars,counts=np.unique(a[:,0],return_counts=True)
    if len(pars) < 3 or np.any(counts < 3):
        raise ValueError(
            "Bifurcation sweep requires >=3 parameter values with >=3 "
            "post-transient Poincare samples EACH. Static load curves are not valid."
        )
    return a


def _figure(path, x, y, xlabel, ylabel, heading, *, scatter=False, log_y=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    if scatter:
        ax.scatter(x,y,s=9,alpha=.8)
    else:
        ax.plot(x,y,linewidth=1)
    if log_y:ax.set_yscale("log")
    ax.set(xlabel=xlabel,ylabel=ylabel,title=heading)
    ax.grid(True,alpha=.25)
    fig.tight_layout()
    fig.savefig(path,dpi=165)
    plt.close(fig)


def save_analysis(csv_path, output_dir, *, time_column="time_s", signal_column="w",
                  velocity_column="", discard_s=0., excitation_hz=None,
                  phase_rad=0., bifurcation_csv=None,
                  bifurcation_parameter_column="parameter",
                  bifurcation_signal_column="poincare_w"):
    """Generate numerical CSV and lossless configuration manifest plus PNG.

    Results require a genuine dynamic trace. The solver is not launched.
    """
    t,signal,velocity = read_trace(
        csv_path,time_column=time_column,signal_column=signal_column,
        velocity_column=velocity_column
    )
    t,signal,velocity = transient_cut(t,signal,velocity,discard_s)
    dt = sample_interval(t)
    derivative_source = "provided_velocity_column" if velocity is not None else "numerical_gradient"
    if velocity is None:
        velocity=np.gradient(signal,t,edge_order=2)
    freq,amplitude,psd = spectral_estimators(t,signal)
    points = None
    if excitation_hz is not None:
        points=stroboscopic_poincare(
            t,signal,velocity,excitation_hz=excitation_hz,phase_rad=phase_rad
        )
    sweep = None
    if bifurcation_csv:
        sweep=read_bifurcation_samples(
            bifurcation_csv,parameter_column=bifurcation_parameter_column,
            signal_column=bifurcation_signal_column
        )
    out=Path(output_dir).expanduser()
    out.mkdir(parents=True,exist_ok=True)
    images=[]
    def add(name,x,y,xlabel,ylabel,heading,**opts):
        _figure(out/name,x,y,xlabel,ylabel,heading,**opts)
        images.append(name)
    add("01_TIME_HISTORY.png",t,signal,"t (s)",signal_column,
        "Displacement time history")
    add("02_PHASE_PORTRAIT.png",signal,velocity,signal_column,
        "velocity ("+derivative_source+")","Phase portrait")
    add("03_FFT_AMPLITUDE.png",freq,amplitude,"f (Hz)",
        "Peak amplitude","One-sided FFT amplitude (Hann)")
    add("04_POWER_SPECTRAL_DENSITY.png",freq,psd,"f (Hz)",
        signal_column+"^2/Hz","Power spectral density (Hann periodogram)")
    lags,corr=autocorrelation(t,signal)
    add("05_AUTOCORRELATION.png",lags,corr,"lag (s)","correlation",
        "Autocorrelation")
    np.savetxt(out/"TIME_HISTORY.csv",np.column_stack((t,signal,velocity)),
               delimiter=",",header="time_s,signal,velocity",comments="")
    np.savetxt(out/"SPECTRUM.csv",np.column_stack((freq,amplitude,psd)),
               delimiter=",",header="frequency_hz,peak_amplitude,psd",comments="")
    np.savetxt(out/"AUTOCORRELATION.csv",np.column_stack((lags,corr)),
               delimiter=",",header="lag_s,autocorrelation",comments="")
    if points is not None:
        tp,xp,vp=points
        add("06_POINCARE.png",xp,vp,signal_column,"velocity",
            "Stroboscopic Poincare section",scatter=True)
        np.savetxt(out/"POINCARE.csv",np.column_stack((tp,xp,vp)),
                   delimiter=",",header="time_s,signal,velocity",comments="")
    if sweep is not None:
        add("07_BIFURCATION_SAMPLES.png",sweep[:,0],sweep[:,1],
            bifurcation_parameter_column,bifurcation_signal_column,
            "Steady-state Poincare samples versus parameter",scatter=True)
        np.savetxt(out/"BIFURCATION_SAMPLES.csv",sweep,
                   delimiter=",",header="parameter,poincare_signal",comments="")
    result={
        "input_csv":str(Path(csv_path).resolve()),
        "sampling":"near_uniform","sample_count":int(len(t)),
        "dt_s":float(dt),"sampling_frequency_hz":1./float(dt),
        "discard_transient_s":float(discard_s),
        "signal_column":signal_column,
        "velocity_source":derivative_source,
        "excitation_hz":float(excitation_hz) if excitation_hz is not None else None,
        "excitation_phase_rad":float(phase_rad) if points is not None else None,
        "poincare_sample_count":int(len(points[0])) if points is not None else 0,
        "bifurcation_input_csv":str(bifurcation_csv) if sweep is not None else None,
        "images":images,
        "lyapunov_spectrum":"NOT_COMPUTED_REQUIRES_VARIATIONAL_FLOW_OR_VALIDATED_ESTIMATOR",
        "status":"POSTPROCESSING_ONLY_NO_DYNAMIC_SOLVER_EXECUTED",
    }
    (out/"DYNAMICS_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description="Trace-backed nonlinear dynamics graphs.")
    p.add_argument("csv_file",help="CSV with time_s,w (and optional velocity)")
    p.add_argument("output_dir",help="Analysis output directory")
    p.add_argument("--time-column",default="time_s")
    p.add_argument("--signal-column",default="w")
    p.add_argument("--velocity-column",default="")
    p.add_argument("--discard-s",type=float,default=0.)
    p.add_argument("--excitation-hz",type=float)
    p.add_argument("--phase-rad",type=float,default=0.)
    p.add_argument("--bifurcation-csv")
    p.add_argument("--bifurcation-parameter-column",default="parameter")
    p.add_argument("--bifurcation-signal-column",default="poincare_w")
    a=p.parse_args(argv)
    r=save_analysis(a.csv_file,a.output_dir,
        time_column=a.time_column,signal_column=a.signal_column,
        velocity_column=a.velocity_column,discard_s=a.discard_s,
        excitation_hz=a.excitation_hz,phase_rad=a.phase_rad,
        bifurcation_csv=a.bifurcation_csv,
        bifurcation_parameter_column=a.bifurcation_parameter_column,
        bifurcation_signal_column=a.bifurcation_signal_column)
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
