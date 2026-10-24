#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physically explicit time laws for dynamic shell load channels.

Time in seconds; every law returns a DIMENSIONLESS multiplier of an actual
reference spatial mechanical load (N per generalized area) or surface free
charge (C per generalized area). No substitute for the spatial load projection,
no assumption that load increments are time steps, and NO shell PDE solve.

The spatial load's type (pressure/patch/line/point/end force/moment) remains
independent of the time law. Time-dependent essential mechanical BCs require
consistent velocity/acceleration constraints and are not simulated here.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
import numpy as np

TIME_LAWS = (
    "CONSTANT", "STEP", "RAMP", "HARMONIC", "DAMPED_HARMONIC",
    "MULTIHARMONIC", "CHIRP_LINEAR", "RECTANGULAR_PULSE",
    "HALF_SINE_PULSE", "TRIANGULAR_PULSE", "GAUSSIAN_PULSE",
    "TABULATED",
)
TARGETS = ("MECHANICAL", "FREE_CHARGE")


def _float(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{label}: numeric value required")
    try:
        a = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: numeric value required") from exc
    if not math.isfinite(a):
        raise ValueError(f"{label}: numeric value must be finite")
    if positive and not a > 0 or nonnegative and not a >= 0:
        raise ValueError(f"{label}: invalid sign")
    return a


def read_time_table(path, *, time_column="time_s", value_column="factor"):
    """CSV with explicit time seconds and multiplier (no extrapolation guess)."""
    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("tabulated load requires at least two rows")
    try:
        t = np.array([_float(r[time_column],time_column) for r in rows])
        a = np.array([_float(r[value_column],value_column) for r in rows])
    except KeyError as exc:
        raise ValueError(f"table is missing column {exc.args[0]}") from exc
    if np.any(np.diff(t) <= 0):
        raise ValueError("tabulated load times must be strictly increasing")
    return t,a


def normalize_time_law(raw, *, base_dir=None):
    """Validate and freeze one time law. No eval() or executable expressions."""
    if not isinstance(raw, dict):
        raise ValueError("load time_law must be an object")
    name = str(raw.get("kind", "")).strip().upper()
    if name not in TIME_LAWS:
        raise ValueError(f"unknown load time law {name!r}; allowed: {TIME_LAWS}")
    params = {"kind":name}
    def get(key, default=None, *, positive=False, nonnegative=False):
        value = raw.get(key, default)
        if value is None:
            raise ValueError(f"{name}: missing {key}")
        params[key] = _float(value,key,positive=positive,nonnegative=nonnegative)
        return params[key]
    if name != "TABULATED":
        get("amplitude",1.)
    if name in ("STEP","RAMP","HARMONIC","DAMPED_HARMONIC","MULTIHARMONIC",
                "CHIRP_LINEAR","RECTANGULAR_PULSE","HALF_SINE_PULSE",
                "TRIANGULAR_PULSE"):
        get("start_time_s",0.)
    if name in ("RAMP","CHIRP_LINEAR","RECTANGULAR_PULSE","HALF_SINE_PULSE",
                "TRIANGULAR_PULSE"):
        get("duration_s",None,positive=True)
    if name in ("HARMONIC","DAMPED_HARMONIC"):
        get("frequency_hz",None,positive=True)
        get("phase_rad",0.)
        if name=="DAMPED_HARMONIC": get("decay_rate_per_s",None,nonnegative=True)
    if name == "MULTIHARMONIC":
        parts = raw.get("harmonics")
        if not isinstance(parts,list) or not parts:
            raise ValueError("MULTIHARMONIC needs a nonempty harmonics list")
        waves=[]
        for p in parts:
            if not isinstance(p,dict):raise ValueError("harmonic must be an object")
            waves.append((
                _float(p.get("amplitude"),"harmonic amplitude"),
                _float(p.get("frequency_hz"),"harmonic frequency",positive=True),
                _float(p.get("phase_rad",0),"harmonic phase"),
            ))
        params["harmonics"]=tuple(waves)
    if name == "CHIRP_LINEAR":
        get("frequency_start_hz",None,positive=True)
        get("frequency_end_hz",None,positive=True)
        get("phase_rad",0.)
    if name == "GAUSSIAN_PULSE":
        get("center_time_s",None)
        get("sigma_s",None,positive=True)
    if name == "TABULATED":
        has_file="csv_path" in raw
        has_array="times_s" in raw or "factors" in raw
        if has_file == has_array:
            raise ValueError("TABULATED requires either csv_path or times_s+factors")
        if has_file:
            path=Path(str(raw["csv_path"])).expanduser()
            if base_dir is not None and not path.is_absolute():path=Path(base_dir)/path
            t,v=read_time_table(path,
                time_column=str(raw.get("time_column","time_s")),
                value_column=str(raw.get("value_column","factor")))
        else:
            try:
                t=np.asarray(raw["times_s"],float)
                v=np.asarray(raw["factors"],float)
            except (ValueError,TypeError,KeyError) as exc:
                raise ValueError("invalid tabulated time/factor arrays") from exc
        if t.ndim!=1 or v.shape!=t.shape or t.size<2:
            raise ValueError("tabulated times/factors require equal 1D arrays >=2")
        if not (np.isfinite(t).all() and np.isfinite(v).all() and np.all(np.diff(t)>0)):
            raise ValueError("tabulated times must be strictly increasing and finite")
        ext=str(raw.get("extrapolation","ERROR")).strip().upper()
        if ext not in ("ERROR","ZERO","HOLD"):
            raise ValueError("tabulated extrapolation must be ERROR/ZERO/HOLD")
        params.update(times_s=tuple(t.tolist()),factors=tuple(v.tolist()),extrapolation=ext)
    return TimeLaw(params)


@dataclass(frozen=True)
class TimeLaw:
    settings: dict

    @property
    def kind(self):return self.settings["kind"]

    def value(self,t_s):
        arr=np.asarray(t_s,dtype=float)
        if not np.isfinite(arr).all():raise ValueError("time must be finite")
        x=arr
        p=self.settings
        k=self.kind
        A=p.get("amplitude",1.)
        start=p.get("start_time_s",0.)
        tau=x-start
        if k=="CONSTANT":y=np.full_like(x,A)
        elif k=="STEP":y=A*(tau>=0)
        elif k=="RAMP":y=A*np.clip(tau/p["duration_s"],0,1)
        elif k=="HARMONIC":
            y=np.where(tau>=0,A*np.sin(2*np.pi*p["frequency_hz"]*tau+p["phase_rad"]),0.)
        elif k=="DAMPED_HARMONIC":
            y=np.where(tau>=0,A*np.exp(-p["decay_rate_per_s"]*np.maximum(tau,0))*
                     np.sin(2*np.pi*p["frequency_hz"]*tau+p["phase_rad"]),0.)
        elif k=="MULTIHARMONIC":
            y=np.zeros_like(x)
            for amp,freq,phase in p["harmonics"]:
                y+=amp*np.sin(2*np.pi*freq*tau+phase)
            y=np.where(tau>=0,A*y,0.)
        elif k=="CHIRP_LINEAR":
            T=p["duration_s"]
            phase=p["phase_rad"]+2*np.pi*(p["frequency_start_hz"]*tau+
                .5*(p["frequency_end_hz"]-p["frequency_start_hz"])*tau**2/T)
            y=np.where((tau>=0)&(tau<=T),A*np.sin(phase),0.)
        elif k=="RECTANGULAR_PULSE":
            y=np.where((tau>=0)&(tau<=p["duration_s"]),A,0.)
        elif k=="HALF_SINE_PULSE":
            T=p["duration_s"]
            y=np.where((tau>=0)&(tau<=T),A*np.sin(np.pi*tau/T),0.)
        elif k=="TRIANGULAR_PULSE":
            T=p["duration_s"]
            y=A*np.clip(1-np.abs(2*tau/T-1),0,1)
        elif k=="GAUSSIAN_PULSE":
            y=A*np.exp(-.5*((x-p["center_time_s"])/p["sigma_s"])**2)
        elif k=="TABULATED":
            t=np.asarray(p["times_s"]);v=np.asarray(p["factors"])
            if p["extrapolation"]=="ERROR" and (np.any(x<t[0]) or np.any(x>t[-1])):
                raise ValueError("time is outside tabulated load range")
            y=np.interp(x,t,v)
            if p["extrapolation"]=="ZERO":y=np.where((x>=t[0])&(x<=t[-1]),y,0.)
        else:raise RuntimeError("unreachable time-law kind")
        return float(y) if arr.ndim==0 else y

    def bandwidth_hz(self):
        p=self.settings
        if self.kind in ("HARMONIC","DAMPED_HARMONIC"):
            return p["frequency_hz"]
        if self.kind=="MULTIHARMONIC":return max(h[1] for h in p["harmonics"])
        if self.kind=="CHIRP_LINEAR":
            return max(p["frequency_start_hz"],p["frequency_end_hz"])
        return None

    def validate_time_grid(self, start_time_s, time_step_s, n_steps, *, points_per_period=16):
        t0=_float(start_time_s,"start_time_s")
        dt=_float(time_step_s,"time_step_s",positive=True)
        if isinstance(n_steps,bool) or not isinstance(n_steps,int) or n_steps<1:
            raise ValueError("n_steps must be a positive integer")
        freq=self.bandwidth_hz()
        if freq is not None and dt*freq>1/points_per_period:
            raise ValueError(f"time step undersamples excitation: {dt*freq:.3g} cycles/step; "
                             f"require >= {points_per_period} samples/period")
        if self.kind=="TABULATED" and self.settings["extrapolation"]=="ERROR":
            end=t0+dt*n_steps
            if t0<self.settings["times_s"][0] or end>self.settings["times_s"][-1]:
                raise ValueError("tabulated load does not cover full time window")
        if self.kind in ("RECTANGULAR_PULSE","HALF_SINE_PULSE","TRIANGULAR_PULSE"):
            if dt>self.settings["duration_s"]/6:
                raise ValueError("pulse is insufficiently time-resolved; need >=6 steps in pulse")
        return True


@dataclass(frozen=True)
class LoadChannel:
    name: str
    target: str
    reference: np.ndarray
    time_law: TimeLaw

    def __post_init__(self):
        if not isinstance(self.name,str) or not self.name.strip():
            raise ValueError("load channel must have a name")
        if self.target not in TARGETS:raise ValueError("target must be MECHANICAL/FREE_CHARGE")
        field=np.asarray(self.reference,float)
        if field.ndim!=3 or not np.isfinite(field).all():
            raise ValueError("reference spatial load must be (components,nx,ns), finite")
        if not isinstance(self.time_law,TimeLaw):raise TypeError("time_law must be TimeLaw")
        object.__setattr__(self,"reference",field.copy())


class DynamicLoadProgram:
    """Superposition of several independent time laws on reference fields.

    All channels contribute generalized DEAD forcing consistent with -F(t)·q.
    A follower pressure, moving force or base excitation needs a different
    residual/tangent/mass pullback and is intentionally not emulated here.
    """
    def __init__(self,channels,*,mechanical_shape,electrical_shape=None):
        self.channels=tuple(channels)
        if not self.channels:raise ValueError("at least one dynamic load channel is required")
        self.mechanical_shape=tuple(mechanical_shape)
        self.electrical_shape=None if electrical_shape is None else tuple(electrical_shape)
        if len(self.mechanical_shape)!=3 or any(k<=0 for k in self.mechanical_shape):
            raise ValueError("mechanical_shape must be (components,nx,ns)")
        if self.electrical_shape is not None and (len(self.electrical_shape)!=3 or
                                                  self.electrical_shape[1:]!=self.mechanical_shape[1:]):
            raise ValueError("electrical_shape must have the same surface grid")
        names=set()
        for channel in self.channels:
            if not isinstance(channel,LoadChannel):raise TypeError("expected LoadChannel")
            if channel.name in names:raise ValueError("duplicate dynamic load channel name")
            names.add(channel.name)
            expected=self.mechanical_shape if channel.target=="MECHANICAL" else self.electrical_shape
            if expected is None or channel.reference.shape!=expected:
                raise ValueError("dynamic channel target and field shape do not match")

    def at(self,time_s):
        mechanical=np.zeros(self.mechanical_shape)
        charge=None if self.electrical_shape is None else np.zeros(self.electrical_shape)
        for channel in self.channels:
            scaled=channel.reference*channel.time_law.value(time_s)
            if channel.target=="MECHANICAL":mechanical+=scaled
            else:charge+=scaled
        return mechanical,charge

    def force_callback(self):
        """Factory compatible with Newmark external_force(t) signature."""
        return lambda time_s:self.at(time_s)[0].reshape(-1)


__all__=("TIME_LAWS","TARGETS","TimeLaw","LoadChannel","DynamicLoadProgram",
         "normalize_time_law","read_time_table")
