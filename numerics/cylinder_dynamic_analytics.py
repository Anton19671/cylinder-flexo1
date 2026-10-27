#!/usr/bin/env python3
"""Dynamic response analytics for the seven cylindrical shell theories.

Postprocesses accepted full-state NPZ files; does not alter the PDE solver.
Single-run outputs: 00 index, 01 signals, 02 Fourier/PSD, 03 phase
portraits, 04 Poincare, 05 scalar-signal Lyapunov estimates (XLSX + PNG).
Parameter-study output: 06 bifurcation (XLSX + PNG).

An embedding-based Rosenstein slope is an ESTIMATE of the maximal Lyapunov
exponent and is not a validated Lyapunov spectrum for the shell PDE. The
QR spectrum requires the *actual* reduced time-step tangent maps, including
all mechanical/electrical coupling and applied boundary constraints.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import detrend, find_peaks
from scipy.spatial import cKDTree
import xlsxwriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROBE_LOCATIONS = (
    ("P01_CENTER_0", 0.50, 0.00),
    ("P02_CENTER_90", 0.50, 0.25),
    ("P03_CENTER_180", 0.50, 0.50),
    ("P04_CENTER_270", 0.50, 0.75),
    ("P05_QUARTER_0", 0.25, 0.00),
    ("P06_THREE_QUARTER_0", 0.75, 0.00),
    ("P07_QUARTER_180", 0.25, 0.50),
)


@dataclass(frozen=True)
class Probe:
    name: str
    ix: int
    js: int
    x_m: float
    theta_rad: float


@dataclass(frozen=True)
class DynamicRun:
    root: Path
    case: dict
    summary: dict
    time_s: np.ndarray
    load_factor: np.ndarray
    probes: tuple[Probe, ...]
    signals: dict[str, dict[str, np.ndarray]]
    complete: bool


def _finite_array(arr, name, shape=None):
    a = np.asarray(arr, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError(f"{name}: invalid shape or nonfinite data")
    return a


def _radial_component(theory, count):
    t = str(theory).upper()
    if t.startswith("CUF_TE") and t[-1] in "1234":
        n = int(t[-1]) + 1
        if count != 3*n:
            raise ValueError("CUF mechanical component ordering mismatch")
        return 2*n
    if t == "KL" and count == 3 or t in ("FSDT", "REDDY") and count == 5:
        return 2
    raise ValueError(f"unsupported shell theory or components: {theory}, {count}")


def _probes(shape, length_m, radius_m, definitions=None):
    nx, ns = shape
    if nx < 4 or ns < 4:
        raise ValueError("a cylindrical dynamic grid needs at least 4x4 nodes")
    if definitions is None:
        definitions = PROBE_LOCATIONS
    if not definitions or len(definitions) > 64:
        raise ValueError("provide 1..64 probe locations")
    result = []
    used = set()
    for item in definitions:
        if isinstance(item, dict):
            name = str(item["name"])
            ax = float(item["axial_fraction"])
            circ = float(item["circum_fraction"])
        else:
            name, ax, circ = item
        if not name or len(name) > 28 or not all(ch.isalnum() or ch=="_" for ch in name):
            raise ValueError("probe name requires <=28 alphanumeric/underscore characters")
        if not math.isfinite(ax) or not 0 <= ax <= 1 or not math.isfinite(circ) or not 0 <= circ < 1:
            raise ValueError("probe fractions must lie in [0,1] axial / [0,1) circumferential")
        ix = min(nx-1, int(math.floor(ax*(nx-1) + 0.5)))
        js = int(math.floor(circ*ns + 0.5)) % ns
        if name in used:
            raise ValueError("duplicate probe name")
        used.add(name)
        result.append(Probe(name, ix, js, float(ix*length_m/(nx-1)),
                            float(js*2*math.pi/ns)))
    return tuple(result)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_dynamic_run(run_dir, *, probes=None):
    """Read ONLY accepted solver steps, and reject missing/inconsistent data."""
    root = Path(run_dir).expanduser().resolve()
    case = _load_json(root/"REFERENCE_CASE.json")
    summary = _load_json(root/"FINAL_RUN_SUMMARY.json")
    if summary.get("calculation_kind") != "DYNAMIC":
        raise ValueError("dynamic history required")
    steps = summary.get("steps", [])
    accepted = int(summary.get("accepted_steps", -1))
    if accepted < 0 or accepted > len(steps):
        raise ValueError("accepted_steps inconsistent with dynamic summary")
    if accepted == 0:
        raise ValueError("no accepted time steps to analyze")
    # Failed states must never be interpreted as accepted trajectory points.
    records = steps[:accepted]
    for idx, row in enumerate(records, start=1):
        if row.get("index") != idx or row.get("converged") is not True:
            raise ValueError("accepted dynamic state sequence is not contiguous")
        if row.get("state_file") != f"REFERENCE_DYNAMIC_STEP_{idx:04d}.npz":
            raise ValueError("unexpected dynamic step file name")
    with np.load(root/"REFERENCE_DYNAMIC_INITIAL.npz", allow_pickle=False) as raw:
        q0 = _finite_array(raw["mechanical"], "initial mechanical")
        v0 = _finite_array(raw["velocity"], "initial velocity", q0.shape)
        a0 = _finite_array(raw["acceleration"], "initial acceleration", q0.shape)
        p0 = _finite_array(raw["electrical"], "initial electrical") if "electrical" in raw else None
        t0 = float(raw["time_s"])
    if q0.ndim != 3 or not all(z > 0 for z in q0.shape):
        raise ValueError("mechanical field must have (components,nx,ns) shape")
    geometry = case.get("geometry", {})
    length = float(geometry["L"])
    radius = float(geometry["R"])
    if not math.isfinite(length) or length <= 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("invalid cylinder geometry")
    rad = _radial_component(summary["theory"], q0.shape[0])
    points = _probes(q0.shape[1:], length, radius, probes)
    time = [t0]
    factors = [float("nan")]
    keys = ["w_m", "dw_dt_m_s", "d2w_dt2_m_s2", "u_m", "v_tangential_m", "psi0_v",
            "inner_w_m", "outer_w_m"]
    signals = {pt.name: {key: [] for key in keys} for pt in points}

    def append(q, v, a, p, inner, outer):
        for pt in points:
            i, j = pt.ix, pt.js
            row = signals[pt.name]
            row["w_m"].append(float(q[rad,i,j]))
            row["dw_dt_m_s"].append(float(v[rad,i,j]))
            row["d2w_dt2_m_s2"].append(float(a[rad,i,j]))
            row["u_m"].append(float(q[0,i,j]))
            row["v_tangential_m"].append(float(q[(rad//2) if rad>2 else 1,i,j]))
            row["psi0_v"].append(float(p[0,i,j]) if p is not None else float("nan"))
            row["inner_w_m"].append(float(inner[i,j]) if inner is not None else float("nan"))
            row["outer_w_m"].append(float(outer[i,j]) if outer is not None else float("nan"))

    append(q0,v0,a0,p0,None,None)
    for idx, row in enumerate(records, start=1):
        with np.load(root/row["state_file"], allow_pickle=False) as raw:
            q = _finite_array(raw["mechanical"], f"mechanical step {idx}", q0.shape)
            v = _finite_array(raw["velocity"], f"velocity step {idx}", q0.shape)
            a = _finite_array(raw["acceleration"], f"acceleration step {idx}", q0.shape)
            p = _finite_array(raw["electrical"], f"electrical step {idx}", p0.shape) if p0 is not None else None
            if p0 is None and "electrical" in raw:
                raise ValueError("electric state appears after initialization")
            inner = (_finite_array(raw["face_inner_radial_displacement_m"],"inner face",q0.shape[1:])
                     if "face_inner_radial_displacement_m" in raw else None)
            outer = (_finite_array(raw["face_outer_radial_displacement_m"],"outer face",q0.shape[1:])
                     if "face_outer_radial_displacement_m" in raw else None)
            t = float(raw["time_s"])
            factor = float(raw["load_factor"])
        if not math.isfinite(t) or not math.isfinite(factor):
            raise ValueError("nonfinite time/load in accepted state")
        if not np.isclose(t, float(row["time_s"]), atol=1e-11, rtol=1e-8):
            raise ValueError("NPZ time and summary disagree")
        append(q,v,a,p,inner,outer)
        time.append(t)
        factors.append(factor)
    time = np.asarray(time)
    if not np.isfinite(t0) or np.any(np.diff(time) <= 0):
        raise ValueError("accepted time stamps must be strictly increasing")
    dt = float(summary["newmark"]["dt_s"])
    if not dt > 0 or not np.allclose(np.diff(time), dt, rtol=1e-7, atol=1e-11):
        raise ValueError("nonuniform steps: uniform Newmark sampling required")
    series = {pt: {field: np.asarray(values) for field,values in rows.items()}
              for pt,rows in signals.items()}
    complete = bool(summary.get("converged")) and accepted == int(summary.get("requested_steps", accepted))
    return DynamicRun(root,case,summary,time,np.asarray(factors),points,series,complete)


def forcing_frequency_hz(case, explicit=None):
    if explicit is not None:
        freq = float(explicit)
        if not math.isfinite(freq) or freq <= 0:
            raise ValueError("forcing_hz must be finite and positive")
        return freq
    time_law = case.get("dynamics",{}).get("time_law",{})
    if time_law.get("kind","").upper() in ("HARMONIC","DAMPED_HARMONIC"):
        return forcing_frequency_hz(case,time_law.get("frequency_hz"))
    return None


def fourier_spectrum(signal, dt, *, transient_fraction=0.2):
    a = _finite_array(signal,"FFT signal")
    if a.ndim != 1:
        raise ValueError("FFT requires 1D signal")
    if not 0 <= transient_fraction < 0.9:
        raise ValueError("transient_fraction must be within [0,0.9)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("FFT requires positive sample interval")
    start = int(math.floor(len(a)*transient_fraction))
    y = a[start:]
    if len(y) < 16:
        return {"status":"INSUFFICIENT_SAMPLES", "frequency_hz":np.array([]),
                "amplitude":np.array([]),"psd":np.array([]),"dominant_hz":None}
    y = detrend(y, type="linear")
    n = len(y)
    window = np.hanning(n)
    fourier = np.fft.rfft(y*window)
    frequency = np.fft.rfftfreq(n,dt)
    factor = np.full(len(frequency), 2.)
    factor[0] = 1.
    if n%2==0:
        factor[-1] = 1.
    amplitude = factor*np.abs(fourier)/np.sum(window)
    psd = factor*np.abs(fourier)**2/((1/dt)*np.sum(window**2))
    best = int(1+np.argmax(amplitude[1:])) if len(amplitude)>1 else 0
    dominant = float(frequency[best]) if amplitude[best]>1e-12*max(np.max(np.abs(a)),1e-30) else None
    return {"status":"COMPUTED", "frequency_hz":frequency,"amplitude":amplitude,
            "psd":psd,"dominant_hz":dominant}


def poincare_section(time_s, displacement, velocity, frequency_hz, *, transient_fraction=0.2,
                     reference_time_s=0.):
    t = _finite_array(time_s,"Poincare time")
    q = _finite_array(displacement,"Poincare displacement",t.shape)
    v = _finite_array(velocity,"Poincare velocity",t.shape)
    if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0):
        raise ValueError("Poincare samples require increasing time")
    if frequency_hz is None:
        return np.empty((0,3),float)
    freq=float(frequency_hz)
    dt = float(np.median(np.diff(t)))
    if not math.isfinite(freq) or freq<=0:
        raise ValueError("forcing frequency must be positive")
    if freq >= 0.5/dt:
        raise ValueError("forcing frequency exceeds Nyquist; Poincare section unreliable")
    if not 0<=transient_fraction<0.9:
        raise ValueError("invalid transient_fraction")
    tmin=t[int(len(t)*transient_fraction)]
    ref=float(reference_time_s)
    first=int(math.ceil((tmin-ref)*freq-1e-10))
    last=int(math.floor((t[-1]-ref)*freq+1e-10))
    if last<first:
        return np.empty((0,3),float)
    strobes=ref+np.arange(first,last+1,dtype=float)/freq
    strobes=strobes[(strobes>=t[0])&(strobes<=t[-1])]
    return np.column_stack((strobes,np.interp(strobes,t,q),np.interp(strobes,t,v)))


def rosenstein_lyapunov(signal, dt, *, transient_fraction=0.2,
                       embedding_dimension=3,delay=None,theiler=None,fit_steps=8,
                       max_pairs=1200):
    """Embedding-based scalar maximal-LE slope; NEVER a PDE tangent spectrum.

    Average log-distance growth of temporally separated nearest neighbors.
    Report fit quality rather than classifying deterministic chaos from its sign.
    """
    y=_finite_array(signal,"Lyapunov signal")
    if y.ndim!=1 or not math.isfinite(dt) or dt<=0:
        raise ValueError("Lyapunov signal must be 1D with positive dt")
    if not 0<=transient_fraction<0.9:
        raise ValueError("invalid transient_fraction")
    y=y[int(len(y)*transient_fraction):]
    unavailable={"status":"INSUFFICIENT_DATA", "lambda_per_s":None,"r_squared":None,
                 "pairs":0,"time_s":np.array([]),"log_separation":np.array([]),
                 "fit":np.array([]),"delay_samples":None,"theiler_samples":None}
    if len(y)<256 or np.std(y)<1e-12*max(np.max(np.abs(y)),1e-30):
        return unavailable
    y=(y-np.mean(y))/np.std(y)
    if embedding_dimension<2 or embedding_dimension>12:
        raise ValueError("embedding_dimension must be in 2..12")
    if delay is None:
        max_lag=min(100,len(y)//10)
        centered=y-y.mean()
        fft=np.fft.rfft(centered,n=2*len(y))
        ac=np.fft.irfft(fft*np.conjugate(fft),n=2*len(y))[:max_lag+1]
        ac /= ac[0]
        crossings=np.nonzero(ac[1:]<math.exp(-1))[0]
        delay=int(crossings[0]+1) if len(crossings) else max(1,max_lag//5)
    delay=int(delay)
    if delay<1:
        raise ValueError("delay must be >=1")
    if theiler is None:
        theiler=max(10,2*embedding_dimension*delay)
    theiler=int(theiler)
    if theiler<1:
        raise ValueError("theiler must be positive")
    horizon=min(30,max(8,len(y)//60))
    usable=len(y)-(embedding_dimension-1)*delay-horizon
    if usable<128:
        return unavailable
    embedding=np.stack([y[i*delay:i*delay+usable+horizon]
                        for i in range(embedding_dimension)],axis=1)
    anchors=np.unique(np.linspace(0,usable-1,min(usable,max_pairs),dtype=int))
    tree=cKDTree(embedding[:usable])
    k=min(64,usable)
    dist,index=tree.query(embedding[anchors],k=k)
    if k==1:
        index=index[:,None]
    pairs=[]
    for anchor,candidates in zip(anchors,index):
        for candidate in np.atleast_1d(candidates):
            candidate=int(candidate)
            if abs(candidate-int(anchor))>theiler:
                pairs.append((int(anchor),candidate))
                break
    if len(pairs)<50:
        return unavailable
    pair=np.asarray(pairs,dtype=int)
    log_growth=[]
    for step in range(horizon+1):
        distance=np.linalg.norm(embedding[pair[:,0]+step]-embedding[pair[:,1]+step],axis=1)
        log_growth.append(float(np.mean(np.log(np.maximum(distance,np.finfo(float).tiny)))))
    time=np.arange(horizon+1,dtype=float)*dt
    fit_last=min(horizon,int(fit_steps))
    if fit_last<3:
        return unavailable
    slope,intercept=np.polyfit(time[1:fit_last+1],np.asarray(log_growth)[1:fit_last+1],1)
    fitted=intercept+slope*time
    residual=np.asarray(log_growth)[1:fit_last+1]-fitted[1:fit_last+1]
    observed=np.asarray(log_growth)[1:fit_last+1]
    denominator=np.sum((observed-np.mean(observed))**2)
    r2=float(1-np.sum(residual**2)/denominator) if denominator>0 else 0.
    return {"status":"ESTIMATE_ONLY" if r2>=0.8 else "LOW_FIT_QUALITY",
            "lambda_per_s":float(slope),"r_squared":r2,"pairs":len(pair),
            "time_s":time,"log_separation":np.asarray(log_growth),"fit":fitted,
            "delay_samples":delay,"theiler_samples":theiler}


def lyapunov_qr_spectrum(transition_maps, dt, *, discard_steps=0):
    """Benettin/QR for supplied TRUE state-transition tangent maps only.

    Each map must advance a perturbation of the same reduced dynamical state
    from t_k to t_{k+1}. Ordinary static shell tangents are NOT these maps.
    """
    maps=_finite_array(transition_maps,"tangent transition maps")
    if maps.ndim!=3 or maps.shape[1]!=maps.shape[2] or maps.shape[1]<1:
        raise ValueError("expected (n_steps,n_state,n_state) maps")
    if not math.isfinite(dt) or dt<=0:
        raise ValueError("dt must be positive")
    if isinstance(discard_steps,bool) or not isinstance(discard_steps,(int,np.integer)) or not 0<=discard_steps<len(maps):
        raise ValueError("invalid QR transient")
    n=maps.shape[1]
    q=np.eye(n)
    totals=np.zeros(n)
    count=0
    for k,mat in enumerate(maps):
        q,r=np.linalg.qr(mat@q)
        diag=np.abs(np.diag(r))
        if k>=discard_steps:
            if np.any(diag<=np.finfo(float).tiny):
                raise ValueError("singular tangent map; finite QR spectrum unavailable")
            totals+=np.log(diag)
            count+=1
    return np.sort(totals/(count*dt))[::-1]


def _worksheet(wb, name, headers, rows, *, chart=None, chart_title=None):
    sh=wb.add_worksheet(name[:31])
    heading=wb.add_format({"bold":True,"bg_color":"#E7E7E7","border":1,"text_wrap":True})
    scientific=wb.add_format({"num_format":"0.000000E+00"})
    for col,title in enumerate(headers):
        sh.write(0,col,title,heading)
        sh.set_column(col,col,22,scientific)
    count=0
    for count,row in enumerate(rows,start=1):
        for col,val in enumerate(row):
            if val is None:
                continue
            if isinstance(val,(int,float,np.floating,np.integer)) and not isinstance(val,bool):
                if np.isfinite(val):
                    sh.write_number(count,col,float(val))
            else:
                sh.write_string(count,col,str(val))
    sh.freeze_panes(1,0)
    if count:
        sh.autofilter(0,0,count,len(headers)-1)
    if chart and count>=2:
        chart_type,xx,yy=chart
        plot=wb.add_chart({"type":"scatter" if chart_type=="scatter" else "line",
                           "subtype":"straight" if chart_type=="scatter" else None})
        plot.add_series({"name":str(headers[yy]),"categories":[sh.name,1,xx,count,xx],
                         "values":[sh.name,1,yy,count,yy],"marker":{"type":"none"}})
        plot.set_title({"name":chart_title or name})
        plot.set_x_axis({"name":headers[xx]})
        plot.set_y_axis({"name":headers[yy]})
        sh.insert_chart(2,len(headers)+1,plot)
    return sh


def _safe_plot(path, x, y, xlabel, ylabel, title, *, scatter=False):
    fig,ax=plt.subplots(figsize=(9,5))
    if scatter:
        ax.scatter(x,y,s=7,alpha=.75)
    else:
        ax.plot(x,y,lw=1.)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.grid(True,alpha=.25)
    fig.tight_layout()
    fig.savefig(path,dpi=145)
    plt.close(fig)


def _write_manifest(path, obj):
    with Path(path).open("w",encoding="utf-8") as fp:
        json.dump(obj,fp,ensure_ascii=False,indent=2,allow_nan=False)


def export_dynamic_analytics(run_dir, *, output_dir=None,transient_fraction=.2,
                             forcing_hz=None,probes=None,tangent_npz=None,overwrite=False):
    """Generate real spreadsheets and images for one completed/partial run."""
    if not 0<=transient_fraction<0.9:
        raise ValueError("transient_fraction must lie in [0,.9)")
    run=load_dynamic_run(run_dir,probes=probes)
    target=Path(output_dir).expanduser().resolve() if output_dir else run.root/"DYNAMIC_ANALYTICS"
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise ValueError("analytics output directory already contains files; use overwrite=True")
    target.mkdir(parents=True,exist_ok=True)
    plots=target/"PLOTS";plots.mkdir(exist_ok=True)
    freq=forcing_frequency_hz(run.case,forcing_hz)
    law=run.case.get("dynamics",{}).get("time_law",{})
    phase_start=float(law.get("start_time_s",run.time_s[0]))
    dt=float(np.median(np.diff(run.time_s)))
    first=int(len(run.time_s)*transient_fraction)
    statuses={}
    books=[]

    # 01: physical fixed-point time series and per-probe Excel time charts.
    path=target/"01_SIGNALS.xlsx"
    with xlsxwriter.Workbook(str(path)) as wb:
        for pt in run.probes:
            series=run.signals[pt.name]
            headers=["time_s","load_factor",*series.keys()]
            rows=zip(run.time_s,run.load_factor,*(series[k] for k in series))
            _worksheet(wb,pt.name,headers,rows,chart=("line",0,2),chart_title="Radial displacement vs time")
            _safe_plot(plots/f"01_SIGNAL_{pt.name}.png",run.time_s,series["w_m"],
                       "time, s","radial displacement, m",pt.name)
    books.append(path.name)

    # 02: windowed one-sided FFT with physically normalized amplitude and PSD.
    path=target/"02_FOURIER_FFT_PSD.xlsx"
    with xlsxwriter.Workbook(str(path)) as wb:
        for pt in run.probes:
            transform=fourier_spectrum(run.signals[pt.name]["w_m"],dt,
                                        transient_fraction=transient_fraction)
            statuses.setdefault(pt.name,{})["fourier"]=transform["status"]
            values=zip(transform["frequency_hz"],2*np.pi*transform["frequency_hz"],
                       transform["amplitude"],transform["psd"])
            _worksheet(wb,pt.name,["frequency_Hz","omega_rad_s","amplitude_m","PSD_m2_per_Hz"],
                       values,chart=("line",0,2),chart_title="Windowed FFT amplitude")
            if transform["status"]=="COMPUTED":
                _safe_plot(plots/f"02_FFT_{pt.name}.png",transform["frequency_hz"],
                           transform["amplitude"],"frequency, Hz","amplitude, m",pt.name)
            statuses[pt.name]["dominant_frequency_hz"]=transform["dominant_hz"]
    books.append(path.name)

    # 03: fixed-point phase trajectories; no claims of chaotic attractors.
    path=target/"03_PHASE_ATTRACTORS.xlsx"
    with xlsxwriter.Workbook(str(path)) as wb:
        for pt in run.probes:
            s=run.signals[pt.name]
            y=s["w_m"][first:]
            v=s["dw_dt_m_s"][first:]
            a=s["d2w_dt2_m_s2"][first:]
            rows=zip(run.time_s[first:],y,v,a)
            _worksheet(wb,pt.name,["time_s","w_m","dw_dt_m_s","d2w_dt2_m_s2"],rows,
                       chart=("scatter",1,2),chart_title="Phase portrait: w vs dw/dt")
            _safe_plot(plots/f"03_PHASE_{pt.name}.png",y,v,"w, m","dw/dt, m/s",pt.name)
            fig=plt.figure(figsize=(8,6))
            ax=fig.add_subplot(111,projection="3d")
            ax.plot(y,v,a,lw=.65)
            ax.set(xlabel="w, m",ylabel="dw/dt, m/s",zlabel="d2w/dt2, m/s2",
                   title="3D phase trajectory: "+pt.name)
            fig.tight_layout();fig.savefig(plots/f"03_PHASE_3D_{pt.name}.png",dpi=145);plt.close(fig)
    books.append(path.name)

    # 04: forced-period section; never use an inferred natural FFT peak as the forcing.
    path=target/"04_POINCARE.xlsx"
    with xlsxwriter.Workbook(str(path)) as wb:
        for pt in run.probes:
            s=run.signals[pt.name]
            section=poincare_section(run.time_s,s["w_m"],s["dw_dt_m_s"],freq,
                                      transient_fraction=transient_fraction,
                                      reference_time_s=phase_start)
            status=("FORCING_FREQUENCY_UNKNOWN" if freq is None else
                    "INSUFFICIENT_CYCLES" if len(section)==0 else "COMPUTED")
            statuses[pt.name]["poincare"]=status
            _worksheet(wb,pt.name,["strobe_time_s","w_m","dw_dt_m_s"],section,
                       chart=("scatter",1,2),chart_title="Poincare section")
            if len(section):
                _safe_plot(plots/f"04_POINCARE_{pt.name}.png",section[:,1],section[:,2],
                           "w, m","dw/dt, m/s",pt.name,scatter=True)
    books.append(path.name)

    # 05: one embedding-based maximal LE ESTIMATE for each distinct probe.
    path=target/"05_LYAPUNOV.xlsx"
    with xlsxwriter.Workbook(str(path)) as wb:
        sh=wb.add_worksheet("ESTIMATES")
        head=["probe","method","status","lambda_est_1_per_s","fit_R2","neighbor_pairs",
              "delay_samples","theiler_samples"]
        for j,h in enumerate(head):sh.write(0,j,h)
        for i,pt in enumerate(run.probes,start=1):
            res=rosenstein_lyapunov(run.signals[pt.name]["w_m"],dt,
                                    transient_fraction=transient_fraction)
            statuses[pt.name]["lyapunov"]=res["status"]
            vals=[pt.name,"ROSENSTEIN_SCALAR_MAX_ESTIMATE",res["status"],res["lambda_per_s"],
                  res["r_squared"],res["pairs"],res["delay_samples"],res["theiler_samples"]]
            for col,val in enumerate(vals):
                if val is not None:sh.write(i,col,val)
            _worksheet(wb,pt.name,["evolution_s","mean_log_distance","linear_fit"],
                       zip(res["time_s"],res["log_separation"],res["fit"]),
                       chart=("line",0,1),chart_title="Rosenstein log-distance evolution")
            if len(res["time_s"]):
                _safe_plot(plots/f"05_LYAPUNOV_{pt.name}.png",res["time_s"],
                           res["log_separation"],"evolution time, s","mean log separation",
                           pt.name+" (Rosenstein estimate)")
        if tangent_npz is not None:
            with np.load(tangent_npz,allow_pickle=False) as raw:
                maps=_finite_array(raw["state_transition_maps"],"tangent maps")
                tdt=float(raw["dt_s"])
            full=lyapunov_qr_spectrum(maps,tdt,discard_steps=int(len(maps)*transient_fraction))
            _worksheet(wb,"QR_FULL_SPECTRUM",["index","lyapunov_1_per_s"],
                       ((i+1,float(value)) for i,value in enumerate(full)))
            statuses["full_spectrum"]="COMPUTED_FROM_PROVIDED_REDUCED_TANGENT_MAPS"
        else:
            statuses["full_spectrum"]="NOT_AVAILABLE_WITHOUT_FULL_DYNAMIC_TANGENT_MAPS"
    books.append(path.name)

    # 00: complete provenance and explicit interpretation for the artifacts.
    path=target/"00_DYNAMIC_INDEX.xlsx"
    with xlsxwriter.Workbook(str(path)) as wb:
        _worksheet(wb,"RUN",["property","value"],[
            ("theory",run.summary["theory"]),("method",run.summary.get("method")),
            ("rank",run.summary.get("rank")),("complete",run.complete),
            ("accepted_time_steps",len(run.time_s)-1),("dt_s",dt),
            ("transient_fraction",transient_fraction),("forcing_hz",freq),
            ("full_lyapunov_spectrum",statuses["full_spectrum"]),
            ("lyapunov_interpretation","Rosenstein estimate from a scalar displacement, NOT exact PDE spectrum"),
            ("bifurcation_interpretation","requires independent parameter sweep; no diagram from one run"),
            ("phase_interpretation","phase-space trajectories, not a proof of chaotic attractor"),
            ("result_quality","COMPLETE" if run.complete else "PARTIAL_ACCEPTED_STEPS_ONLY")])
        _worksheet(wb,"PROBES",["probe","node_x","node_s","x_m","theta_rad"],
                   ((p.name,p.ix,p.js,p.x_m,p.theta_rad) for p in run.probes))
        _worksheet(wb,"FILES",["workbook"],((item,) for item in books))
        _worksheet(wb,"DIAGNOSTICS",["probe","fft","dominant_frequency_Hz","poincare","lyapunov"],
                   ((pt.name,statuses[pt.name].get("fourier"),statuses[pt.name].get("dominant_frequency_hz"),
                     statuses[pt.name].get("poincare"),statuses[pt.name].get("lyapunov")) for pt in run.probes))
    books.insert(0,path.name)
    template=target/"BIFURCATION_SWEEP_TEMPLATE.csv"
    with template.open("w",newline="",encoding="utf-8") as fp:
        csv.writer(fp).writerow(["parameter_value","run_dir"])
    _write_manifest(target/"00_ANALYTICS_MANIFEST.json",{
        "source_run":str(run.root),"theory":run.summary["theory"],"method":run.summary.get("method"),
        "rank":run.summary.get("rank"),"complete":run.complete,
        "time_steps":len(run.time_s)-1,"transient_fraction":transient_fraction,
        "forcing_hz":freq,"probes":[vars(pt) for pt in run.probes],
        "workbooks":books,"plots_subdir":"PLOTS","statuses":statuses,
        "bifurcation":"REQUIRES_PARAMETER_SWEEP_MANIFEST"})
    return {"output_dir":str(target),"workbooks":books,"probes":len(run.probes),
            "complete":run.complete,"statuses":statuses}


def export_bifurcation_study(manifest_csv, *, output_dir=None, parameter_name="parameter",
                            probe_name="P01_CENTER_0",transient_fraction=.5,
                            forcing_hz=None,overwrite=False):
    """True multi-run parameter diagram: post-transient peaks and strobe points."""
    path=Path(manifest_csv).expanduser().resolve()
    with path.open("r",encoding="utf-8-sig",newline="") as fp:
        rows=list(csv.DictReader(fp))
    if len(rows)<2:
        raise ValueError("bifurcation study needs at least two independent parameter runs")
    runs=[]
    for row in rows:
        value=float(row["parameter_value"])
        if not math.isfinite(value):
            raise ValueError("nonfinite continuation parameter")
        candidate=Path(row["run_dir"])
        if not candidate.is_absolute():candidate=path.parent/candidate
        record=load_dynamic_run(candidate)
        if not record.complete:
            raise ValueError("bifurcation diagram requires completed runs")
        if probe_name not in record.signals:
            raise ValueError(f"unknown monitor probe {probe_name}")
        runs.append((value,record))
    runs.sort(key=lambda r:r[0])
    if len(set(x[0] for x in runs))!=len(runs):
        raise ValueError("parameter sweep requires distinct parameter values")
    first=runs[0][1]
    for value,record in runs[1:]:
        if record.summary.get("theory")!=first.summary.get("theory") or record.summary.get("method")!=first.summary.get("method") or record.summary.get("rank")!=first.summary.get("rank"):
            raise ValueError("one bifurcation study requires matching theory/method/MVI rank")
        if not np.isclose(np.diff(record.time_s)[0],np.diff(first.time_s)[0],rtol=1e-8):
            raise ValueError("one parameter diagram requires the same time step")
    output=Path(output_dir).expanduser().resolve() if output_dir else path.parent/"BIFURCATION_ANALYTICS"
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError("nonempty bifurcation output directory")
    output.mkdir(parents=True,exist_ok=True)
    peak_rows=[];poincare_rows=[];summary_rows=[]
    for value,record in runs:
        t=record.time_s
        trace=record.signals[probe_name]["w_m"]
        vel=record.signals[probe_name]["dw_dt_m_s"]
        dt=float(np.diff(t)[0])
        f=forcing_frequency_hz(record.case,forcing_hz)
        start=int(len(t)*transient_fraction)
        q=trace[start:]
        min_distance=max(1,int(.2/(f*dt))) if f else 1
        peaks,_=find_peaks(q,distance=min_distance)
        for i in peaks:
            peak_rows.append((value,float(t[start+i]),float(q[i])))
        law=record.case.get("dynamics",{}).get("time_law",{})
        ref=float(law.get("start_time_s",t[0]))
        points=poincare_section(t,trace,vel,f,transient_fraction=transient_fraction,
                                reference_time_s=ref)
        for ts,w,v in points:
            poincare_rows.append((value,float(ts),float(w),float(v)))
        lyap=rosenstein_lyapunov(trace,dt,transient_fraction=transient_fraction)
        summary_rows.append((value,len(peaks),len(points),float(np.mean(q)),float(np.std(q)),
                             lyap["lambda_per_s"],lyap["status"]))
    dest=output/"06_BIFURCATION.xlsx"
    with xlsxwriter.Workbook(str(dest)) as wb:
        _worksheet(wb,"PEAK_BRANCHES",[parameter_name,"peak_time_s","radial_peak_m"],
                   peak_rows,chart=("scatter",0,2),chart_title="Bifurcation from post-transient maxima")
        _worksheet(wb,"POINCARE_BRANCHES",[parameter_name,"strobe_time_s","w_m","dw_dt_m_s"],
                   poincare_rows,chart=("scatter",0,2),chart_title="Stroboscopic branches")
        _worksheet(wb,"PARAMETER_SUMMARY",[parameter_name,"n_peaks","n_strobes","w_mean_m",
                 "w_std_m","rosenstein_lambda_est_1_s","estimate_status"],summary_rows,
                 chart=("scatter",0,5),chart_title="Scalar Lyapunov estimate vs parameter")
        _worksheet(wb,"METHODOLOGY",["setting","value"],[
            ("theory",first.summary["theory"]),("method",first.summary.get("method")),
            ("rank",first.summary.get("rank")),("probe",probe_name),
            ("transient_fraction",transient_fraction),
            ("peak_method","local maxima after transient"),
            ("Poincare_method","forcing-period stroboscopic samples if frequency known"),
            ("Lyapunov_method","Rosenstein scalar estimate, NOT full PDE spectrum")])
    if peak_rows:
        _safe_plot(output/"06_BIFURCATION_PEAKS.png",
                   [p[0] for p in peak_rows],[p[2] for p in peak_rows],
                   parameter_name,"radial displacement maxima, m","Bifurcation: maxima",
                   scatter=True)
    if poincare_rows:
        _safe_plot(output/"06_BIFURCATION_POINCARE.png",
                   [p[0] for p in poincare_rows],[p[2] for p in poincare_rows],
                   parameter_name,"stroboscopic w, m","Bifurcation: Poincare",
                   scatter=True)
    finite=[s for s in summary_rows if s[5] is not None]
    if finite:
        _safe_plot(output/"06_LYAPUNOV_PARAMETER.png",[s[0] for s in finite],
                   [s[5] for s in finite],parameter_name,"lambda estimate, 1/s",
                   "Rosenstein estimates (not certified)")
    report={"parameter_name":parameter_name,"probe":probe_name,"theory":first.summary["theory"],
            "method":first.summary.get("method"),"rank":first.summary.get("rank"),
            "runs":len(runs),"post_transient_maxima":len(peak_rows),
            "poincare_samples":len(poincare_rows),"output_dir":str(output),
            "lyapunov":"ROSENSTEIN_SCALAR_ESTIMATE_ONLY"}
    _write_manifest(output/"06_BIFURCATION_MANIFEST.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description="Cylinder dynamics Excel/FFT/phase/Lyapunov/bifurcation export")
    sub=parser.add_subparsers(dest="command",required=True)
    one=sub.add_parser("run",help="export one Newmark history")
    one.add_argument("run_dir")
    one.add_argument("--output-dir")
    one.add_argument("--transient-fraction",type=float,default=.2)
    one.add_argument("--forcing-hz",type=float)
    one.add_argument("--tangent-npz")
    one.add_argument("--overwrite",action="store_true")
    study=sub.add_parser("bifurcation",help="combine multiple completed runs")
    study.add_argument("manifest_csv")
    study.add_argument("--output-dir")
    study.add_argument("--parameter-name",default="parameter")
    study.add_argument("--probe-name",default="P01_CENTER_0")
    study.add_argument("--transient-fraction",type=float,default=.5)
    study.add_argument("--forcing-hz",type=float)
    study.add_argument("--overwrite",action="store_true")
    opts=parser.parse_args(argv)
    if opts.command=="run":
        result=export_dynamic_analytics(opts.run_dir,output_dir=opts.output_dir,
            transient_fraction=opts.transient_fraction,forcing_hz=opts.forcing_hz,
            tangent_npz=opts.tangent_npz,overwrite=opts.overwrite)
    else:
        result=export_bifurcation_study(opts.manifest_csv,output_dir=opts.output_dir,
            parameter_name=opts.parameter_name,probe_name=opts.probe_name,
            transient_fraction=opts.transient_fraction,forcing_hz=opts.forcing_hz,
            overwrite=opts.overwrite)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    main()
