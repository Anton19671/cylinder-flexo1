"""Compare Newmark, explicit RK4 and existing Gauss IRK4 on ONE shell eigenmode.

This isolates temporal discretization for q'' + omega**2 q = 0; it is
NOT a full 3D shell dynamic solve, not fixed-rank MVI+RK, and not a
validation of nonlinear or loaded dynamics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cylinder_gauss2_rk import GAUSS2_A, GAUSS2_B


def frequency_from_npz(path, mode=1):
    if type(mode) is not int or mode < 1:
        raise ValueError("MODE_INDEX_INVALID")
    with np.load(path, allow_pickle=False) as z:
        name = "frequencies_hz" if "frequencies_hz" in z else "frequencies"
        if name not in z:
            raise ValueError("MODAL_FREQUENCIES_MISSING")
        frequencies = np.asarray(z[name], dtype=float)
    if frequencies.ndim != 1 or mode > len(frequencies):
        raise ValueError("MODE_INDEX_INVALID")
    frequency = float(frequencies[mode - 1])
    if not np.isfinite(frequency) or frequency <= 0.:
        raise ValueError("MODE_FREQUENCY_INVALID")
    return frequency


def compare_modal_integrators(frequency_hz, *, steps_per_period=100,
                              periods=10, amplitude_m=1.e-6,
                              include_histories=False):
    """Compare the same linear, undamped modal oscillator over exact periods.

    State: u/amplitude, velocity/(omega*amplitude); energy/E(0)=|state|^2.
    """
    f, amplitude = float(frequency_hz), float(amplitude_m)
    if not np.isfinite(f) or f <= 0 or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("MODAL_COMPARE_POSITIVE_FREQUENCY_AND_AMPLITUDE_REQUIRED")
    if (type(steps_per_period) is not int or steps_per_period < 16
            or type(periods) is not int or periods < 1
            or steps_per_period * periods > 500000):
        raise ValueError("MODAL_COMPARE_INVALID_TIME_GRID")
    omega = 2.*np.pi*f
    dt = 1./(steps_per_period*f)
    h = omega*dt
    j = np.array([[0., 1.], [-1., 0.]])
    a = h*j
    rk4 = (np.eye(2)+a+a@a/2.+a@a@a/6.+a@a@a@a/24.)
    stages = np.linalg.solve(np.eye(4)-h*np.kron(GAUSS2_A,j),
                             np.tile(np.eye(2), (2, 1)))
    gauss = np.eye(2)+h*(np.kron(GAUSS2_B[None,:],j)@stages)
    times = np.arange(steps_per_period*periods+1)*dt
    exact = np.column_stack((np.cos(omega*times), -np.sin(omega*times)))
    report, histories = {}, {}
    for name, transition in (("NEWMARK", None),("RK4",rk4),("GAUSS_IRK4",gauss)):
        fields = np.empty((len(times),2))
        fields[0] = [1., 0.]
        for k in range(1,len(times)):
            if transition is not None:
                fields[k] = transition @ fields[k-1]
            else:
                u,v = fields[k-1]
                predicted = u+h*v-.25*h*h*u
                u_next = predicted/(1.+.25*h*h)
                fields[k] = (u_next, v-.5*h*(u+u_next))
        phase = np.unwrap(np.arctan2(-fields[:,1], fields[:,0]))
        estimated = float((phase[-1]-phase[0])/(2*np.pi*times[-1]))
        energy = np.einsum("ij,ij->i",fields,fields)
        report[name] = {
            "frequency_hz": estimated,
            "frequency_error_percent": float(100*(estimated/f-1)),
            "final_phase_error_deg": float((phase[-1]-omega*times[-1])*180/np.pi),
            "energy_change_percent": float(100*(energy[-1]-1)),
            "maximum_energy_variation_percent": float(100*np.max(np.abs(energy-1))),
            "maximum_normalized_state_error": float(np.max(np.linalg.norm(fields-exact,axis=1)))
        }
        if include_histories:
            histories[name] = fields
    result = {
        "scope":"ONE_LINEAR_UNFORCED_EIGENMODE_NOT_FULL_SHELL",
        "reference_frequency_hz":f,
        "initial_displacement_amplitude_m":amplitude,
        "initial_velocity_m_s":0.,
        "steps_per_period":steps_per_period,
        "periods":periods,
        "time_step_s":float(dt),
        "integrators":report
    }
    if include_histories:
        result["histories"] = {"time_s":times,"EXACT":exact,**histories}
    return result


def save_comparison(result, prefix):
    prefix = Path(prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True,exist_ok=True)
    header = {k:v for k,v in result.items() if k != "histories"}
    prefix.with_suffix(".json").write_text(json.dumps(header,indent=2)+"\n",
                                           encoding="utf-8")
    with prefix.with_suffix(".csv").open("w",newline="",encoding="utf-8") as out:
        names=("method",)+tuple(next(iter(result["integrators"].values())).keys())
        writer=csv.DictWriter(out,fieldnames=names)
        writer.writeheader()
        for method, data in result["integrators"].items():
            writer.writerow({"method":method,**data})
    if "histories" in result:
        histories=result["histories"]
        with prefix.with_name(prefix.name+"_histories").with_suffix(".csv").open(
                "w",newline="",encoding="utf-8") as out:
            writer=csv.writer(out)
            writer.writerow(("time_s","exact_u_m","newmark_u_m","rk4_u_m","gauss_irk4_u_m"))
            for i,t in enumerate(histories["time_s"]):
                writer.writerow((float(t),*(
                    result["initial_displacement_amplitude_m"]*float(histories[key][i,0])
                    for key in ("EXACT","NEWMARK","RK4","GAUSS_IRK4"))))
    return prefix


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes-npz",required=True)
    parser.add_argument("--mode-index",type=int,default=1)
    parser.add_argument("--steps-per-period",type=int,default=100)
    parser.add_argument("--periods",type=int,default=10)
    parser.add_argument("--amplitude-m",type=float,default=1.e-6)
    parser.add_argument("--output-prefix",required=True)
    args=parser.parse_args(argv)
    f=frequency_from_npz(args.modes_npz,args.mode_index)
    result=compare_modal_integrators(
        f,steps_per_period=args.steps_per_period,periods=args.periods,
        amplitude_m=args.amplitude_m,include_histories=True)
    save_comparison(result,args.output_prefix)


if __name__=="__main__":
    main()
