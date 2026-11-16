"""Explicit opt-in nonlinear cylindrical ENERGY-FDM research CASE runner.

This is the native FDM first-variation branch, NOT the former direct-strong
route and NOT an MVI or reference SBP import. Reuses only the validated
physical CASE metadata and shell constitutive laws. Requires the explicit
INDEPENDENT_ENERGY_FDM_COUPLED_RESEARCH_V1 schema; existing strong CASEs
continue on their existing dispatcher without changing scientific method.
No numerical time step or directory mutation occurs in --validate-only.

Research-only: a converged nonlinear algebraic solve does not certify
mesh convergence, physical gradient boundary closure or benchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cylinder_paths import activate
activate()

import numpy as np

from cylinder_native_run_control import cooperate
from cylinder_native_energy_fdm import NativeEnergyFDM,NativeEnergyEndSpace
from cylinder_native_energy_dynamics import NativeEnergyDynamics
from cylinder_direct_strong_newton import StrongNewtonSettings
from independent_mkr import run_coupled_research as physical_case

SCHEMA="INDEPENDENT_ENERGY_FDM_COUPLED_RESEARCH_V1"
METHOD="INDEPENDENT_ENERGY_FDM_COUPLED_RESEARCH"
STATUS="ENERGY_CONSISTENT_DISCRETE_FIRST_VARIATION_NOT_PHYSICALLY_CERTIFIED"


def _norm_projection(space,physical_mechanical,physical_electric,label):
    """Map positive physical scaling to actual constrained FDM coordinates.

    Physical scales can vary with component, axial position and angle.
    Each reduced coordinate is a column of the actual nullspace. The
    Euclidean magnitude of its weighted physical column gives a positive
    dimensional scale; the gauge pivot contributes its affine correction.
    This is a Newton normalization, not an altered physical equation.
    """
    shape=space.engine.model.shape
    a=np.asarray(physical_mechanical,float)
    if a.shape!=shape or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError(label+"_MECHANICAL_SCALE_INVALID")
    fields=list(a)
    if space.electrical_spaces:
        b=np.asarray(physical_electric,float)
        expected=(len(space.electrical_spaces),*shape[1:])
        if b.shape!=expected or not np.isfinite(b).all() or np.any(b<=0):
            raise ValueError(label+"_ELECTRICAL_SCALE_INVALID")
        fields.extend(list(b))
    elif physical_electric is not None:
        raise ValueError(label+"_ELECTRICITY_OFF")
    values=[]
    for c,subspace in enumerate(space.spaces):
        # Each reduced coefficient acts on one circumferential column.
        norms=np.sqrt(np.einsum(
            "ik,ij,ik->kj",
            subspace.null,fields[c]**2,subspace.null,optimize=True))
        values.append(norms.ravel())
    output=np.concatenate(values)
    if space.gauge_pivot is not None:
        pivot=space.gauge_pivot
        other=np.arange(space.free_size)!=pivot
        ratio=space.gauge_row[other]/space.gauge_row[pivot]
        output=np.hypot(output[other],ratio*output[pivot])
    if (output.shape!=(space.reduced_size,) or
            not np.isfinite(output).all() or np.any(output<=0)):
        raise ValueError(label+"_REDUCED_SCALE_INVALID")
    return output


def load_case(filename):
    """Check the whole original physical case, then instantiate native FDM."""
    data=physical_case.load_case(filename,expected_schema=SCHEMA)
    model=data["model"]
    program=data["program"]
    space=NativeEnergyEndSpace(
        NativeEnergyFDM(model),axial_orders=program.axial_orders,
        mechanical_ends=program.mechanical_ends,
        electrical_ends=program.electrical_ends,
        electric_gauge=program.electric_gauge)
    opts=data["opts"]
    required=space.reduced_size*(2 if opts["integrator"]=="GAUSS_IRK4" else 1)
    if required>opts["max_dofs"]:
        raise ValueError(
            "ENERGY_FDM_REDUCED_DOF_BUDGET_TOO_SMALL: "+str(required))
    data["settings"]=StrongNewtonSettings(
        max_dofs=opts["max_dofs"],
        max_iterations=opts["max_iterations"],
        linear_solver=opts.get("linear_solver","AUTO"),
        dense_lu_dof_limit=opts.get("dense_lu_dof_limit",384))
    data["space"]=space
    data["energy"]=space.engine
    data["dynamics"]=NativeEnergyDynamics(
        space,density_kg_m3=data["engine"].density_kg_m3,
        eta_s=data["engine"].eta_s,
        mass_damping_rate_per_s=data["engine"].mass_damping_rate_per_s)
    data["reduced_state_scale"]=_norm_projection(
        space,data["qscale"],data["escale"],"ENERGY_FDM_STATE")
    data["reduced_equation_scale"]=_norm_projection(
        space,data["rscale"],data["erescale"],"ENERGY_FDM_EQUATION")
    return data


def _atomic_json(path,value):
    text=json.dumps(value,allow_nan=False,indent=2)+"\n"
    temp=path.with_name(path.name+".pending")
    try:
        with temp.open("x",encoding="utf-8") as stream:
            stream.write(text)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_npz(path,**arrays):
    temp=path.with_name(path.name+".pending.npz")
    try:
        with temp.open("xb") as stream:
            np.savez_compressed(stream,**arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _save_state(path,state):
    _atomic_npz(
        path,time_s=np.array(state.time_s),
        mechanical=state.mechanical,
        velocity=state.velocity,
        acceleration=state.acceleration,
        **({"electrical":state.electrical}
           if state.electrical is not None else {}))


def _diagnostics(nonlin):
    def finite(value):
        if value is None:
            return None
        f=float(value)
        return f if math.isfinite(f) else None
    fields=dict(
        status=nonlin.status,
        iterations=int(nonlin.iterations),
        converged=bool(nonlin.converged),
        scaled_equation_norm=finite(nonlin.scaled_equation_norm))
    if nonlin.status=="NEWTON_LINE_SEARCH_FAILED":
        fields.update(
            best_trial_scaled_norm=finite(nonlin.line_search_best_scaled_norm),
            best_trial_alpha=finite(nonlin.line_search_best_alpha),
            admissible_trials=int(nonlin.line_search_admissible_trials),
            invalid_trials=int(nonlin.line_search_invalid_trials),
            linearized_relative_defect=finite(
                nonlin.line_search_linearized_relative_defect))
    return fields


def run(filename,*,validate_only):
    data=load_case(filename)
    opts=data["opts"]
    model=data["model"]
    space=data["space"]
    summary=dict(
        schema=SCHEMA,method=METHOD,solver_backend="INDEPENDENT_MKR",
        spatial_method="NATIVE_ENERGY_FDM_NO_MVI_SPATIAL_REUSE",
        research_only=True,validated_only=bool(validate_only),
        scientific_accuracy_certified=False,
        physical_boundary_closure_certified=False,
        theory=model.theory,gradient_model=model.material.gradient_model,
        geometry_model=model.geometry_model,kind=opts["kind"],
        integrator=opts["integrator"],steps=opts["steps"],
        requested_steps=opts["steps"],
        physical_steps_computed=0,accepted_steps=0,time_s=0.,
        physical_unknowns=(
            model.n_mechanical+(data["case"]["physics"]["potential_order"]+1
                                if model.electrical_on else 0)
        )*model.grid.nx*model.grid.ns,
        reduced_unknowns=space.reduced_size,
        output_dir=str(data["output"]))
    if validate_only:
        return summary
    out=data["output"]
    out.mkdir(parents=True,exist_ok=False)
    _atomic_json(out/"CASE.json",data["case"])
    summary["status"]="IN_PROGRESS"
    _atomic_json(out/"RUN_SUMMARY.json",summary)
    kwargs=dict(reduced_state_scale=data["reduced_state_scale"],
                reduced_equation_scale=data["reduced_equation_scale"],
                settings=data["settings"])
    seed=data["seed"]
    sources=data["sources"]
    loads=(sources["mechanical_load"],sources.get("electrical_charge"))
    def at_time(_time):
        return loads
    try:
        if opts["kind"]=="STATIC":
            result=space.solve_static(
                seed["mechanical"],seed.get("electrical"),
                mechanical_load=loads[0],electrical_charge=loads[1],
                **kwargs)
            summary["static_nonlinear_diagnostics"]=_diagnostics(result.nonlinear)
            if result.mechanical is None:
                raise RuntimeError("ENERGY_FDM_STATIC_"+result.nonlinear.status)
            _atomic_npz(out/"STATIC_ACCEPTED.npz",
                        mechanical=result.mechanical,
                        **({"electrical":result.electrical}
                           if result.electrical is not None else {}))
            summary.update(status="EXPERIMENTAL_STATIONARY_UNCERTIFIED",
                           accepted_steps=1,physical_steps_computed=1)
        else:
            driver=data["dynamics"]
            initial=driver.initialize(
                seed["mechanical"],seed["velocity"],
                seed.get("electrical"),
                acceleration_guess=seed.get("acceleration"),
                loads_at=at_time,**kwargs)
            summary["initial_nonlinear_diagnostics"]=_diagnostics(
                initial.nonlinear)
            if initial.state is None:
                raise RuntimeError("ENERGY_FDM_INITIAL_"+initial.status)
            state=initial.state
            _save_state(out/"STEP_0000.npz",state)
            method={"NEWMARK":driver.newmark,
                    "BATHE":driver.bathe,
                    "GAUSS_IRK4":driver.gauss}[opts["integrator"]]
            for k in range(1,opts["steps"]+1):
                cooperate(out)
                step=method(state,opts["dt_s"],loads_at=at_time,**kwargs)
                if step.state is None:
                    summary["last_step_nonlinear_diagnostics"]=dict(
                        step=k,integrator=opts["integrator"],
                        driver_status=step.status,
                        **_diagnostics(step.nonlinear))
                    raise RuntimeError("ENERGY_FDM_STEP_"+step.status)
                state=step.state
                _save_state(out/f"STEP_{k:04d}.npz",state)
                summary.update(accepted_steps=k,physical_steps_computed=k,
                               time_s=float(state.time_s))
                _atomic_json(out/"RUN_SUMMARY.json",summary)
            summary["status"]="EXPERIMENTAL_COMPLETED_UNCERTIFIED"
        summary["validated_only"]=False
        _atomic_json(out/"RUN_SUMMARY.json",summary)
        return summary
    except Exception as exc:
        summary.update(status="FAILED",error_type=type(exc).__name__,
                       detail=str(exc),scientific_accuracy_certified=False)
        _atomic_json(out/"RUN_SUMMARY.json",summary)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only",action="store_true")
    group.add_argument("--run",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.case,validate_only=args.validate_only),
                     allow_nan=False))


if __name__=="__main__":
    main()
