"""Explicit opt-in, fail-closed CASE runner for independent coupled MVI.

Seven theories, LINEAR/VK/GL, G3D/Gz, stationary parallel ends,
explicit grounded electric modes, static or native Newmark/Bathe/Gauss2.
All surface work, inertia, viscosity and X/Y Newton are OWNED by MVI;
only local continuum physics is shared. Nothing starts on import or
--validate-only. The runner never silently substitutes linear/reference
MVI, MKR/FDM, SVD or a different boundary program.

IMPORTANT: high-order natural/microfree and electric face closure have
NOT been scientifically certified. An explicit research_opt_in=True
is mandatory even for validation. No production GUI defaults.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cylinder_paths import activate
activate()
import numpy as np

from flexo_gradient_g3d import G3DMaterial
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid, CoupledMVIFactors, NativeCoupledMVISurface,
)
from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram, _check_end_factors,
)
from independent_mvi.coupled_static import (
    CoupledMVISweepSettings, solve_coupled_mvi_static,
)
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_newmark import NativeCoupledMVINewmark
from independent_mvi.coupled_bathe import NativeCoupledMVIBathe
from independent_mvi.coupled_gauss2 import NativeCoupledMVIGauss2
from independent_mvi.coupled_initial import close_mvi_instantaneous_state
from independent_mvi.coupled_discrete_green import coupled_mvi_discrete_green_identity

SCHEMA="INDEPENDENT_MVI_COUPLED_RESEARCH_V1"
METHOD="MVI_COUPLED_RESEARCH_FIXED_RANK"
THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
MATERIAL_KEYS=frozenset((
    "young_pa","poisson","ell_m","mu1_c_per_m","mu2_c_per_m",
    "kappa_f_per_m","gradient_model",
))
PHYSICS_KEYS=frozenset((
    "theory","geometry_model","material","thickness_m",
    "density_kg_m3","eta_s","mass_damping_rate_per_s",
    "n_gauss","potential_order",
))
GRID_KEYS=frozenset(("nx","ns","length_m","radius_m"))
SOLVER_KEYS=frozenset((
    "kind","integrator","dt_s","steps","field_scales",
    "energy_scale_j","max_sweeps","max_acceleration_coordinates",
    "max_halfstep_coordinates",
))
OPTIONAL_LINEAR_KEYS=frozenset((
    "linear_solver","dense_coordinate_limit","krylov_restart",
    "krylov_maxiter","krylov_relative_tolerance",
))
CASE_KEYS=frozenset((
    "schema","physics","grid","boundary","solver","rank",
    "initial_state_npz","loads_npz","output_dir","research_opt_in",
))


def _exact(value,keys,name):
    if not isinstance(value,dict) or set(value)!=set(keys):
        raise ValueError(name+"_KEYS_INVALID")
    return value


def _positive(value,name,*,allow_zero=False):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(name+"_INVALID")
    try:
        v=float(value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(name+"_INVALID") from exc
    if not math.isfinite(v) or (v<0 if allow_zero else v<=0):
        raise ValueError(name+"_INVALID")
    return v


def _resource(path,parent,label):
    if not isinstance(path,str) or not path.strip():
        raise ValueError(label+"_PATH_REQUIRED")
    p=Path(path).expanduser()
    if not p.is_absolute():
        p=parent/p
    p=p.resolve()
    if not p.is_file() or p.suffix.lower()!=".npz":
        raise ValueError(label+"_REQUIRES_EXISTING_NPZ")
    return p


def _read_npz(path,required,optional=()):
    with np.load(path,allow_pickle=False) as content:
        names=set(content.files)
        if not set(required)<=names or names-set(required)-set(optional):
            raise ValueError("COUPLED_MVI_NPZ_KEYS_INVALID")
        result={name:np.array(content[name],dtype=float,copy=True)
                for name in names}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("COUPLED_MVI_NPZ_NONFINITE")
    return result


def load_case(path):
    """Preflight is deliberately non-solving and has no output side effects."""
    path=Path(path).expanduser().resolve()
    if path.suffix.lower()!=".json" or not path.is_file():
        raise ValueError("COUPLED_MVI_CASE_REQUIRES_JSON_FILE")
    try:
        raw=json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError,OSError) as exc:
        raise ValueError("COUPLED_MVI_CASE_CANNOT_BE_READ") from exc
    case=_exact(raw,CASE_KEYS,"COUPLED_MVI_CASE")
    if case["schema"]!=SCHEMA:
        raise ValueError("COUPLED_MVI_CASE_SCHEMA_MISMATCH")
    if case["research_opt_in"] is not True:
        raise ValueError("COUPLED_MVI_RESEARCH_OPT_IN_REQUIRED")
    p=_exact(case["physics"],PHYSICS_KEYS,"COUPLED_MVI_PHYSICS")
    grid=_exact(case["grid"],GRID_KEYS,"COUPLED_MVI_GRID")
    mat=_exact(p["material"],MATERIAL_KEYS,"COUPLED_MVI_MATERIAL")
    if p["theory"] not in THEORIES:
        raise ValueError("COUPLED_MVI_UNKNOWN_THEORY")
    surface=NativeCoupledMVISurface(
        p["theory"],G3DMaterial(**mat),
        grid=CoupledMVISurfaceGrid(**grid),
        thickness_m=p["thickness_m"],n_gauss=p["n_gauss"],
        potential_order=p["potential_order"],
        geometry_model=p["geometry_model"])
    if type(case["rank"]) is not int or not 1<=case["rank"]<=8:
        raise ValueError("COUPLED_MVI_FIXED_RANK_MUST_BE_1_TO_8")
    b=_exact(case["boundary"],{"mechanical","electric"},
             "COUPLED_MVI_BOUNDARY")
    program=CoupledMVIEndProgram(**b)
    program.validate(surface)
    opts=case["solver"]
    if (not isinstance(opts,dict) or
            not SOLVER_KEYS.issubset(opts) or
            set(opts)-SOLVER_KEYS-OPTIONAL_LINEAR_KEYS):
        raise ValueError("COUPLED_MVI_SOLVER_KEYS_INVALID")
    if opts["kind"] not in ("STATIC","DYNAMIC"):
        raise ValueError("COUPLED_MVI_UNKNOWN_CALCULATION_KIND")
    if opts["kind"]=="STATIC":
        if (opts["integrator"] is not None or opts["dt_s"] is not None or
                opts["steps"]!=0):
            raise ValueError("COUPLED_MVI_STATIC_FORBIDS_TIME_PARAMETERS")
    else:
        if opts["integrator"] not in ("NEWMARK","BATHE","GAUSS_IRK4"):
            raise ValueError("COUPLED_MVI_UNKNOWN_TIME_INTEGRATOR")
        _positive(opts["dt_s"],"COUPLED_MVI_TIME_STEP")
        if type(opts["steps"]) is not int or not 1<=opts["steps"]<=120:
            raise ValueError("COUPLED_MVI_TIME_STEP_COUNT_INVALID")
    if type(opts["max_sweeps"]) is not int or not 1<=opts["max_sweeps"]<=500:
        raise ValueError("COUPLED_MVI_MAX_SWEEPS_INVALID")
    if (type(opts["max_acceleration_coordinates"]) is not int or
            not 1<=opts["max_acceleration_coordinates"]<=4096):
        raise ValueError("COUPLED_MVI_ACCELERATION_CAP_INVALID")
    field_scales=np.asarray(opts["field_scales"],dtype=float)
    if (field_scales.shape!=(surface.n_fields,) or
            not np.isfinite(field_scales).all() or
            np.any(field_scales<=0)):
        raise ValueError("COUPLED_MVI_EXPLICIT_FIELD_SCALES_REQUIRED")
    energy_scale=_positive(opts["energy_scale_j"],"COUPLED_MVI_ENERGY_SCALE")
    density=_positive(p["density_kg_m3"],"COUPLED_MVI_DENSITY")
    eta=_positive(p["eta_s"],"COUPLED_MVI_VISCOSITY",allow_zero=True)
    damping=_positive(p["mass_damping_rate_per_s"],
                      "COUPLED_MVI_MASS_DAMPING",allow_zero=True)
    src=_resource(case["initial_state_npz"],path.parent,
                  "COUPLED_MVI_INITIAL_STATE")
    load_path=_resource(case["loads_npz"],path.parent,"COUPLED_MVI_LOADS")
    expected_initial={"x_profiles","y_profiles"}
    if opts["kind"]=="DYNAMIC":
        expected_initial.add("velocity")
    init=_read_npz(src,expected_initial)
    factors=CoupledMVIFactors(init["x_profiles"],init["y_profiles"])
    if factors.rank!=case["rank"]:
        raise ValueError("COUPLED_MVI_SEED_RANK_MISMATCH")
    free=_check_end_factors(surface,factors,program)
    expected_load={"mechanical_load"}
    if surface.electrical_on:
        expected_load.add("free_charge")
    loads=_read_npz(load_path,expected_load)
    for name,shape in (
        ("mechanical_load",(surface.n_mechanical,surface.grid.nx,surface.grid.ns)),
        ("free_charge",(surface.n_electric,surface.grid.nx,surface.grid.ns))):
        if name in loads and loads[name].shape!=shape:
            raise ValueError("COUPLED_MVI_"+name.upper()+"_SHAPE_INVALID")
    if opts["kind"]=="DYNAMIC":
        shape=(surface.n_mechanical,surface.grid.nx,surface.grid.ns)
        if init["velocity"].shape!=shape:
            raise ValueError("COUPLED_MVI_INITIAL_VELOCITY_SHAPE_INVALID")
        for c in range(surface.n_mechanical):
            blocked=np.ones(surface.grid.nx,bool)
            blocked[free[c]]=False
            if np.max(np.abs(init["velocity"][c,blocked]),initial=0.)>1e-10:
                raise ValueError("COUPLED_MVI_INITIAL_MOVING_SUPPORT_UNSUPPORTED")
    if not isinstance(case["output_dir"],str) or not case["output_dir"].strip():
        raise ValueError("COUPLED_MVI_OUTPUT_DIR_REQUIRED")
    output=Path(case["output_dir"]).expanduser()
    if not output.is_absolute():
        output=path.parent/output
    output=output.resolve()
    if (output in (path.parent,path.parent.parent,src.parent,load_path.parent)
            or output in (path,src,load_path)):
        raise ValueError("COUPLED_MVI_UNSAFE_OUTPUT_DIRECTORY")
    if output.exists():
        raise ValueError("COUPLED_MVI_OUTPUT_ALREADY_EXISTS")
    from independent_mvi.coupled_halfstep import CoupledMVIHalfstepSettings
    cap=opts["max_halfstep_coordinates"]
    if type(cap) is not int or not 1<=cap<=4096:
        raise ValueError("COUPLED_MVI_HALFSTEP_CAP_INVALID")
    # Every field carries rank*ns active angular factors in Y and
    # rank*available-axial factors in X. Report the needed cap before
    # any physical run, rather than silently failing after output creation.
    free=program.free_axial_indices(surface)
    if opts["kind"]=="DYNAMIC":
        mass_dofs=sum(len(free[c])*surface.grid.ns
                      for c in range(surface.n_mechanical))
        if mass_dofs>opts["max_acceleration_coordinates"]:
            raise ValueError(
                "COUPLED_MVI_ACCELERATION_CAP_TOO_SMALL_FOR_GRID: "
                +str(mass_dofs))
    needed_x=sum(factors.rank*len(ids) for ids in free)
    needed_y=factors.rank*surface.n_fields*surface.grid.ns
    if max(needed_x,needed_y)>cap:
        raise ValueError(
            "COUPLED_MVI_HALFSTEP_CAP_TOO_SMALL_FOR_RANK_AND_GRID: "
            +str(max(needed_x,needed_y)))
    settings=CoupledMVISweepSettings(
        max_sweeps=opts["max_sweeps"],
        halfstep=CoupledMVIHalfstepSettings(
            max_coordinates=cap,
            linear_solver=opts.get("linear_solver","AUTO"),
            dense_coordinate_limit=opts.get("dense_coordinate_limit",96),
            krylov_restart=opts.get("krylov_restart",24),
            krylov_maxiter=opts.get("krylov_maxiter",60),
            krylov_relative_tolerance=opts.get(
                "krylov_relative_tolerance",1.e-5)))
    return dict(
        path=path,case=case,surface=surface,factors=factors,initial=init,
        program=program,load=loads,opts=opts,settings=settings,
        field_scales=field_scales,energy_scale=energy_scale,
        density=density,eta=eta,damping=damping,output=output)


def _write_json(path,data):
    temporary=path.with_name(path.name+".pending")
    try:
        temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_npz(path,**arrays):
    temporary=path.with_name(path.stem+".pending.npz")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream,**arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _green_audit(surface,factors,mechanical_load,free_charge):
    """Audit only the conservative G3D/Gz weak-work decomposition.

    Dynamic inertia/Kelvin-Voigt actions are deliberately excluded:
    the returned diagnostic is NEVER a total dynamic natural end force.
    The caller aborts if the identical method-owned nodal virtual work
    cannot be decomposed into volume and boundary work.
    """
    work=surface.weak_work(
        factors,mechanical_load=mechanical_load,free_charge=free_charge)
    green=coupled_mvi_discrete_green_identity(surface,work)
    return dict(
        status=green.status,
        conservative_only=True,
        physical_boundary_closure_certified=False,
        identity_max_abs_error=float(green.identity_max_abs_error),
        identity_relative_error=float(green.identity_relative_error),
        boundary_work_coefficient_max_abs=float(np.max(
            np.abs(green.boundary_gradient),initial=0.)),
        field_trace_orders=[len(items) for items in green.normal_left],
    )


def _store_state(output,step,state,surface):
    field=surface.reconstruct(state.factors)
    if (not np.isfinite(field).all() or not
            np.isfinite(state.velocity).all() or not
            np.isfinite(state.acceleration).all()):
        raise ValueError("COUPLED_MVI_NONFINITE_ACCEPTED_FIELD")
    _write_npz(output/f"STEP_{step:04d}.npz",
               x_profiles=state.factors.x,y_profiles=state.factors.y,
               mechanical=field[:surface.n_mechanical],
               electrical=field[surface.n_mechanical:],
               velocity=state.velocity,acceleration=state.acceleration,
               time_s=np.array(state.time_s))


def run(path,*,validate_only):
    items=load_case(path)
    surface=items["surface"]
    opts=items["opts"]
    result=dict(method=METHOD,schema=SCHEMA,
        research_only=True,scientific_accuracy_certified=False,
        validated_only=bool(validate_only),
        calculation_kind=opts["kind"],integrator=opts["integrator"],
        theory=surface.theory,gradient_model=surface.material.gradient_model,
        geometry_model=surface.geometry_model,rank=items["factors"].rank,
        steps=opts["steps"],output_dir=str(items["output"]),
        physical_steps_computed=0,
        physical_boundary_closure_certified=False)
    if validate_only:
        return result

    # Only an explicit --run or explicit final dispatch reaches this point.
    output=items["output"]
    output.mkdir(parents=True,exist_ok=False)
    _write_json(output/"CASE.json",items["case"])
    result.update(status="IN_PROGRESS",accepted_steps=0,time_s=0.)
    _write_json(output/"RUN_SUMMARY.json",result)
    mechanical=items["load"]["mechanical_load"]
    charge=items["load"].get("free_charge")
    scales=items["field_scales"]
    energy=items["energy_scale"]
    settings=items["settings"]
    try:
        if opts["kind"]=="STATIC":
            answer=solve_coupled_mvi_static(
                surface,items["factors"],program=items["program"],
                field_scales=scales,energy_scale_j=energy,
                mechanical_load=mechanical,free_charge=charge,
                settings=settings)
            if not answer.converged:
                raise RuntimeError("COUPLED_MVI_STATIC_"+answer.status)
            q=surface.reconstruct(answer.factors)
            _write_npz(output/"STATIC_ACCEPTED.npz",
                x_profiles=answer.factors.x,y_profiles=answer.factors.y,
                mechanical=q[:surface.n_mechanical],
                electrical=q[surface.n_mechanical:])
            result["discrete_green"]= _green_audit(
                surface,answer.factors,mechanical,charge)
            result.update(status="EXPERIMENTAL_STATIONARY_UNCERTIFIED",
                          accepted_steps=1,physical_steps_computed=1)
        else:
            configuration=NativeCoupledMVINewmark(
                surface,NativeCoupledMVIInertia(
                    surface,density_kg_m3=items["density"],
                    mass_damping_rate_per_s=items["damping"]),
                program=items["program"],eta_s=items["eta"])
            initial=close_mvi_instantaneous_state(
                configuration,items["factors"],items["initial"]["velocity"],
                time_s=0.,field_scales=scales,energy_scale_j=energy,
                mechanical_load=mechanical,free_charge=charge,
                electric_settings=settings,
                max_acceleration_coordinates=opts["max_acceleration_coordinates"])
            if initial.state is None:
                raise RuntimeError("COUPLED_MVI_INITIAL_"+initial.status)
            state=initial.state
            result["initial_discrete_green"]=_green_audit(
                surface,state.factors,mechanical,charge)
            _store_state(output,0,state,surface)
            driver=(configuration if opts["integrator"]=="NEWMARK"
                    else NativeCoupledMVIBathe(configuration)
                    if opts["integrator"]=="BATHE"
                    else NativeCoupledMVIGauss2(configuration))
            for step in range(1,opts["steps"]+1):
                common=dict(field_scales=scales,energy_scale_j=energy,
                            settings=settings)
                if opts["integrator"]=="NEWMARK":
                    advance=driver.advance(state,opts["dt_s"],
                        mechanical_load=mechanical,free_charge=charge,**common)
                else:
                    advance=driver.advance(
                        state,opts["dt_s"],
                        loads_at=lambda _t:(mechanical,charge),**common,
                        **({"max_acceleration_coordinates":
                            opts["max_acceleration_coordinates"]}
                           if opts["integrator"]=="GAUSS_IRK4" else {}))
                if advance.state is None:
                    raise RuntimeError("COUPLED_MVI_TIME_STEP_"+advance.status)
                if advance.state.factors.rank!=items["factors"].rank:
                    raise RuntimeError("COUPLED_MVI_RANK_CHANGED")
                state=advance.state
                result["last_discrete_green"]=_green_audit(
                    surface,state.factors,mechanical,charge)
                _store_state(output,step,state,surface)
                result.update(accepted_steps=step,
                              physical_steps_computed=step,
                              time_s=float(state.time_s))
                _write_json(output/"RUN_SUMMARY.json",result)
            result["status"]="EXPERIMENTAL_COMPLETED_UNCERTIFIED"
        result["validated_only"]=False
        _write_json(output/"RUN_SUMMARY.json",result)
        return result
    except Exception as exc:
        result.update(status="FAILED",error_type=type(exc).__name__,
                      detail=str(exc),scientific_accuracy_certified=False)
        _write_json(output/"RUN_SUMMARY.json",result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only",action="store_true")
    group.add_argument("--run",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.case,validate_only=args.validate_only),indent=2))


if __name__=="__main__":
    main()
