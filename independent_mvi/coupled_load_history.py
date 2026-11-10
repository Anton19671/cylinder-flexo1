"""Method-owned, time-dependent generalized loads for coupled MVI.

A CASE may supply a tabulated dimensionless scale for its existing nodal
mechanical/electric loads.  The physical nodal load SHAPES remain unchanged:
no spatial interpolation, reference solver, MKR, or hidden rescaling.
All integration stages receive loads evaluated at their actual times.

Physical inner/outer face charge is a distinct, time-INDEPENDENT boundary
source, and is never multiplied by the tabulated volume-charge scale.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _numbers(sequence, name):
    if (not isinstance(sequence, list) or len(sequence) < 2 or
            any(type(value) not in (int, float) for value in sequence)):
        raise ValueError(name + "_REQUIRES_NUMERIC_LIST")
    value=np.asarray(sequence,dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(name + "_REQUIRES_FINITE_NUMBERS")
    return value.copy()


@dataclass(frozen=True)
class CoupledTabulatedLoadLaw:
    """Piecewise linear scalar law, no extrapolation or interpolation in x/s."""

    times_s: np.ndarray
    mechanical_scale: np.ndarray
    electric_scale: np.ndarray

    @classmethod
    def from_case(cls, definition, *, horizon_s, electric_active):
        if not isinstance(definition, dict):
            raise ValueError("MVI_LOAD_TIME_LAW_MUST_BE_OBJECT")
        allowed={"kind","time_s","mechanical_scale","electric_scale"}
        if (set(definition)-allowed or
                not {"kind","time_s","mechanical_scale"}<=set(definition) or
                definition["kind"]!="TABULATED_LINEAR"):
            raise ValueError("MVI_LOAD_TIME_LAW_KEYS_INVALID")
        if not electric_active and "electric_scale" in definition:
            raise ValueError("MVI_LOAD_ELECTRIC_SCALE_WITH_ELECTRICITY_OFF")
        times=_numbers(definition["time_s"],"MVI_LOAD_TIMES")
        mechanical=_numbers(definition["mechanical_scale"],"MVI_LOAD_MECHANICAL_SCALE")
        electrical=(_numbers(definition["electric_scale"],"MVI_LOAD_ELECTRIC_SCALE")
                    if "electric_scale" in definition else mechanical.copy())
        if times.size!=mechanical.size or times.size!=electrical.size:
            raise ValueError("MVI_LOAD_TIME_LAW_LENGTH_MISMATCH")
        if times[0]!=0. or np.any(np.diff(times)<=0.):
            raise ValueError("MVI_LOAD_TIMES_MUST_BEGIN_AT_ZERO_AND_INCREASE")
        if (not math.isfinite(float(horizon_s)) or horizon_s<=0. or
                times[-1]<horizon_s):
            raise ValueError("MVI_LOAD_TIMES_DO_NOT_COVER_CALCULATION")
        for arr in (times,mechanical,electrical):
            arr.setflags(write=False)
        return cls(times,mechanical,electrical)

    def factors_at(self,time_s):
        if (isinstance(time_s,(bool,np.bool_)) or
                not isinstance(time_s,(int,float,np.integer,np.floating)) or
                not math.isfinite(float(time_s))):
            raise ValueError("MVI_LOAD_TIME_OUTSIDE_TABULATION")
        t=float(time_s)
        horizon=float(self.times_s[-1])
        # Cumulative dt and predeclared n*dt may differ by a few ULPs.
        # Admit only that roundoff, never extrapolate physical loads.
        tolerance=32.*np.finfo(float).eps*max(
            abs(t),abs(horizon),np.finfo(float).tiny)
        if t < -tolerance or t > horizon+tolerance:
            raise ValueError("MVI_LOAD_TIME_OUTSIDE_TABULATION")
        t=min(max(t,0.),horizon)
        return (float(np.interp(t,self.times_s,self.mechanical_scale)),
                float(np.interp(t,self.times_s,self.electric_scale)))


class CoupledLoadProgram:
    """Prepared loads at time t, including constant prescribed face charge."""

    def __init__(self,mechanical,charge=None,face_charge=None,law=None):
        if law is not None and not isinstance(law,CoupledTabulatedLoadLaw):
            raise TypeError("MVI_LOAD_PROGRAM_REQUIRES_TABULATED_LAW")
        self.mechanical=np.array(mechanical,dtype=float,copy=True)
        self.charge=(None if charge is None else
                     np.array(charge,dtype=float,copy=True))
        self.face_charge=(None if face_charge is None else
                          np.array(face_charge,dtype=float,copy=True))
        if self.mechanical.ndim!=3 or not np.isfinite(self.mechanical).all():
            raise ValueError("MVI_LOAD_INVALID_MECHANICAL_FIELD")
        if self.charge is not None:
            if (self.charge.ndim!=3 or not np.isfinite(self.charge).all() or
                    self.charge.shape[1:]!=self.mechanical.shape[1:]):
                raise ValueError("MVI_LOAD_INVALID_ELECTRIC_FIELD")
        if self.face_charge is not None and (
                self.charge is None or self.face_charge.shape!=self.charge.shape
                or not np.isfinite(self.face_charge).all()):
            raise ValueError("MVI_LOAD_INVALID_FACE_FIELD")
        self.law=law
        self.mechanical.setflags(write=False)
        if self.charge is not None:self.charge.setflags(write=False)
        if self.face_charge is not None:self.face_charge.setflags(write=False)

    def factors_at(self,time_s):
        if self.law is None:
            if not isinstance(time_s,(int,float,np.integer,np.floating)) or (
                    isinstance(time_s,(bool,np.bool_)) or
                    not math.isfinite(float(time_s)) or time_s<0):
                raise ValueError("MVI_LOAD_TIME_INVALID")
            return 1.,1.
        return self.law.factors_at(time_s)

    def at(self,time_s):
        mechanical_factor,electric_factor=self.factors_at(time_s)
        mechanical=self.mechanical*mechanical_factor
        electric=(None if self.charge is None else
                  self.charge*electric_factor)
        if self.face_charge is not None:
            electric=electric+self.face_charge
        if not np.isfinite(mechanical).all() or (
                electric is not None and not np.isfinite(electric).all()):
            raise ValueError("MVI_LOAD_NONFINITE_SCALED_FIELD")
        return mechanical,electric

    def characteristic_times(self):
        return (self.law.times_s if self.law is not None
                else np.array([0.]))
