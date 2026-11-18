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
from cylinder_native_run_control import cooperate, NativeUserStop
import numpy as np

from flexo_gradient_g3d import G3DMaterial
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid, CoupledMVIFactors, NativeCoupledMVISurface,
    mvi_separated_term_norms,
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
from independent_mvi.coupled_load_history import (
    CoupledTabulatedLoadLaw, CoupledLoadProgram,
)
from independent_mvi.coupled_physical_recovery import (
    recover_cuf_displacements, radial_history_row,
)
from independent_mvi.coupled_checkpoint import (
    input_manifest, accepted_state_sha256, exclusive_run_lock,
    solver_implementation_fingerprint,
    prepare_resume, append_history_row,
)
from independent_mvi.coupled_discrete_green import coupled_mvi_discrete_green_identity
from independent_mvi.coupled_dynamic_green import audit_mvi_dynamic_discrete_green
from independent_mvi.coupled_mechanical_end_work import compile_mechanical_end_work
from independent_mvi.coupled_electric_faces import (
    face_generalized_free_charge,recover_mvi_electric_faces,
    summarize_mvi_electric_faces,
)

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
    "krylov_dense_fallback_limit","allow_projected_static",
    "allow_projected_dynamic",
    "acceleration_linear_solver","acceleration_dense_limit",
    "acceleration_krylov_restart","acceleration_krylov_maxiter",
    "acceleration_krylov_relative_tolerance",
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


def load_case(path, *, allow_existing_output=False):
    """Non-solving preflight; existing outputs require explicit resume."""
    path=Path(path).expanduser().resolve()
    if path.suffix.lower()!=".json" or not path.is_file():
        raise ValueError("COUPLED_MVI_CASE_REQUIRES_JSON_FILE")
    try:
        raw=json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError,OSError) as exc:
        raise ValueError("COUPLED_MVI_CASE_CANNOT_BE_READ") from exc
    if (not isinstance(raw,dict) or not CASE_KEYS.issubset(raw) or
            set(raw)-CASE_KEYS-{"electric_faces","mechanical_end_work","load_time_law","kl_radial_face_traction"}):
        raise ValueError("COUPLED_MVI_CASE_KEYS_INVALID")
    case=raw
    if case["schema"]!=SCHEMA:
        raise ValueError("COUPLED_MVI_CASE_SCHEMA_MISMATCH")
    if case["research_opt_in"] is not True:
        raise ValueError("COUPLED_MVI_RESEARCH_OPT_IN_REQUIRED")
    p=_exact(case["physics"],PHYSICS_KEYS,"COUPLED_MVI_PHYSICS")
    grid=_exact(case["grid"],GRID_KEYS,"COUPLED_MVI_GRID")
    mat=_exact(p["material"],MATERIAL_KEYS,"COUPLED_MVI_MATERIAL")
    if p["theory"] not in THEORIES:
        raise ValueError("COUPLED_MVI_UNKNOWN_THEORY")
    b=_exact(case["boundary"],{"mechanical","electric"},
             "COUPLED_MVI_BOUNDARY")
    floating=(b["electric"]=="INSULATED_FLOATING")
    surface=NativeCoupledMVISurface(
        p["theory"],G3DMaterial(**mat),
        grid=CoupledMVISurfaceGrid(**grid),
        thickness_m=p["thickness_m"],n_gauss=p["n_gauss"],
        potential_order=p["potential_order"],
        geometry_model=p["geometry_model"],
        floating_gauge=floating)
    if type(case["rank"]) is not int or not 1<=case["rank"]<=8:
        raise ValueError("COUPLED_MVI_FIXED_RANK_MUST_BE_1_TO_8")
    program=CoupledMVIEndProgram(**b)
    program.validate(surface)
    end_work=compile_mechanical_end_work(
        surface,case.get("mechanical_end_work"))
    surface.mechanical_end_work=end_work
    from cylinder_kl_face_load import compile_kl_face_load
    if case.get("kl_radial_face_traction") is not None and case.get("load_time_law") is not None:
        raise ValueError("KL_FACE_LOAD_TIME_LAW_NOT_SUPPORTED")
    surface.kl_face_load=compile_kl_face_load(
        case.get("kl_radial_face_traction"),theory=surface.theory,
        shape=(surface.grid.nx,surface.grid.ns),
        radius_m=surface.grid.radius_m,thickness_m=surface.thickness_m,
        geometry_model=surface.geometry_model)
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
        if type(opts["steps"]) is not int or not 1<=opts["steps"]<=100000:
            raise ValueError("COUPLED_MVI_TIME_STEP_COUNT_INVALID")
    if type(opts["max_sweeps"]) is not int or not 1<=opts["max_sweeps"]<=500:
        raise ValueError("COUPLED_MVI_MAX_SWEEPS_INVALID")
    if (type(opts["max_acceleration_coordinates"]) is not int or
            not 1<=opts["max_acceleration_coordinates"]<=300000):
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
    # Validate archive shapes before indexing a charge mode, adding face
    # charge, or checking floating-gauge compatibility.  Otherwise malformed
    # free_charge inputs leak NumPy IndexError/broadcast errors instead of
    # the stable public preflight diagnostic below.
    for name,shape in (
        ("mechanical_load",(surface.n_mechanical,surface.grid.nx,surface.grid.ns)),
        ("free_charge",(surface.n_electric,surface.grid.nx,surface.grid.ns))):
        if name in loads and loads[name].shape!=shape:
            raise ValueError("COUPLED_MVI_"+name.upper()+"_SHAPE_INVALID")
    # Preserve the original volume-charge source: face charge is a
    # distinct physical boundary input and never scales with the
    # time-dependent volume-charge multiplier.
    original_charge=(None if "free_charge" not in loads else
                     loads["free_charge"].copy())
    face_charge=None
    faces=case.get("electric_faces")
    if faces is not None:
        if not isinstance(faces,dict) or set(faces)!={"inner","outer"}:
            raise ValueError("MVI_ELECTRIC_FACES_REQUIRE_INNER_OUTER")
        # Electric face CHARGE is converted into method-owned generalized
        # electric work on the reference midsurface, never into a
        # pointwise penalty or an unconstrained full-grid correction.
        generalized=face_generalized_free_charge(
            surface,inner=faces["inner"],outer=faces["outer"])
        if generalized is not None:
            face_charge=np.asarray(generalized,dtype=float).copy()
            loads["free_charge"]+=face_charge
    elif surface.electrical_on:
        # Omitted specification means zero free charge on both faces.
        faces={"inner":None,"outer":None}
    if surface.floating_gauge and "load_time_law" not in case:
        # With a dynamic load law, test EVERY knot below instead of
        # rejecting the unscaled baseline at time zero.
        # A floating electrostatic problem is solvable only if net
        # generalized physical free charge is zero. Check the full
        # sum of volume and curvature-weighted face loads BEFORE any
        # physical material evaluation or output directory creation.
        source=np.asarray(loads["free_charge"][0],dtype=float)
        weights=surface.grid.weights
        net=float(np.sum(source*weights))
        absolute=float(np.sum(np.abs(source)*weights))
        if absolute>0 and abs(net)>5.e-11*absolute:
            raise ValueError(
                "MVI_FLOATING_GAUGE_INCOMPATIBLE_TOTAL_FREE_CHARGE")
    load_law=None
    if "load_time_law" in case:
        if opts["kind"]!="DYNAMIC":
            raise ValueError("MVI_STATIC_FORBIDS_TIME_DEPENDENT_LOAD")
        load_law=CoupledTabulatedLoadLaw.from_case(
            case["load_time_law"],
            horizon_s=float(opts["dt_s"])*opts["steps"],
            electric_active=surface.electrical_on)
    load_program=CoupledLoadProgram(
        loads["mechanical_load"],original_charge,
        face_charge=face_charge,law=load_law)
    # A floating electric gauge requires charge compatibility at EVERY
    # knot, not just in the initially supplied source fields.
    if surface.floating_gauge and load_law is not None:
        for knot in load_law.times_s:
            _,charge_at=load_program.at(float(knot))
            weights=surface.grid.weights
            total=float(np.sum(charge_at[0]*weights))
            absolute=float(np.sum(np.abs(charge_at[0])*weights))
            if absolute>0. and abs(total)>5.e-11*absolute:
                raise ValueError(
                    "MVI_FLOATING_GAUGE_INCOMPATIBLE_TIME_VARYING_CHARGE")
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
    if output.exists() and not allow_existing_output:
        raise ValueError("COUPLED_MVI_OUTPUT_ALREADY_EXISTS")
    if allow_existing_output and (not output.is_dir()):
        raise ValueError("COUPLED_MVI_CHECKPOINT_OUTPUT_DIRECTORY_MISSING")
    from independent_mvi.coupled_halfstep import CoupledMVIHalfstepSettings
    cap=opts["max_halfstep_coordinates"]
    if type(cap) is not int or not 1<=cap<=300000:
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
    # The acceleration cap is only a physical nodal-coordinate safety
    # budget; the matrix-free mass operator never allocates cap**2.
    acceleration_options=dict(
        acceleration_linear_solver=opts.get(
            "acceleration_linear_solver","AUTO"),
        acceleration_dense_limit=opts.get("acceleration_dense_limit",192),
        acceleration_krylov_restart=opts.get(
            "acceleration_krylov_restart",32),
        acceleration_krylov_maxiter=opts.get(
            "acceleration_krylov_maxiter",200),
        acceleration_krylov_relative_tolerance=opts.get(
            "acceleration_krylov_relative_tolerance",1.e-8),
    )
    # Reject invalid choices BEFORE starting or creating output.
    if acceleration_options["acceleration_linear_solver"] not in (
            "AUTO","DENSE_LU","MATRIX_FREE_GMRES"):
        raise ValueError("COUPLED_MVI_ACCELERATION_BACKEND_INVALID")
    for name in ("acceleration_dense_limit","acceleration_krylov_restart",
                 "acceleration_krylov_maxiter"):
        value=acceleration_options[name]
        if type(value) is not int or value<1:
            raise ValueError("COUPLED_MVI_"+name.upper()+"_INVALID")
    tol=acceleration_options["acceleration_krylov_relative_tolerance"]
    if (isinstance(tol,(bool,np.bool_)) or
            not isinstance(tol,(int,float)) or
            not math.isfinite(float(tol)) or not 0<float(tol)<.1):
        raise ValueError("COUPLED_MVI_ACCELERATION_KRYLOV_TOLERANCE_INVALID")
    if opts["kind"]=="DYNAMIC":
        nacc=sum(len(free[c])*surface.grid.ns
                 for c in range(surface.n_mechanical))
        if (acceleration_options["acceleration_linear_solver"]=="DENSE_LU"
                and nacc>acceleration_options["acceleration_dense_limit"]):
            raise ValueError("COUPLED_MVI_DENSE_ACCELERATION_LIMIT_EXCEEDED")
    projected_static=opts.get("allow_projected_static",False)
    projected_dynamic=opts.get("allow_projected_dynamic",False)
    if type(projected_static) is not bool or (projected_static and opts["kind"]!="STATIC"):
        raise ValueError("COUPLED_MVI_PROJECTED_STATIC_ONLY_BOOLEAN_REQUIRED")
    if type(projected_dynamic) is not bool or (projected_dynamic and opts["kind"]!="DYNAMIC"):
        raise ValueError("COUPLED_MVI_PROJECTED_DYNAMIC_ONLY_BOOLEAN_REQUIRED")
    settings=CoupledMVISweepSettings(
        allow_projected_convergence=(projected_static or projected_dynamic),
        max_sweeps=opts["max_sweeps"],
        halfstep=CoupledMVIHalfstepSettings(
            max_coordinates=cap,
            linear_solver=opts.get("linear_solver","AUTO"),
            dense_coordinate_limit=opts.get("dense_coordinate_limit",96),
            krylov_restart=opts.get("krylov_restart",24),
            krylov_maxiter=opts.get("krylov_maxiter",60),
            krylov_relative_tolerance=opts.get(
                "krylov_relative_tolerance",1.e-5),
            krylov_dense_fallback_limit=opts.get(
                "krylov_dense_fallback_limit",0)))
    return dict(
        path=path,case=case,surface=surface,factors=factors,initial=init,
        program=program,load=loads,opts=opts,settings=settings,
        field_scales=field_scales,energy_scale=energy_scale,
        density=density,eta=eta,damping=damping,output=output,
        electric_faces=faces,mechanical_end_work=end_work,
        acceleration_options=acceleration_options,load_program=load_program,
        initial_path=src,loads_path=load_path)


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


def _finite_diagnostic(value):
    """Unmeasured/nonfinite diagnostics are null, never false convergence.

    Failed halfsteps legitimately carry inf norms. Preserve their status
    and history without masking the original failure with a JSON error.
    """
    if isinstance(value,dict):
        return {key:_finite_diagnostic(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [_finite_diagnostic(item) for item in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value,np.generic):
        return value.item()
    return value


def _sweep_diagnostics(answer):
    return _finite_diagnostic(dict(
        status=answer.status,converged=bool(answer.converged),
        sweeps=answer.sweeps,field_change=answer.field_change,
        projected_norm_x=answer.projected_norm_x,
        projected_norm_y=answer.projected_norm_y,
        full_nodal_norm=answer.full_nodal_norm,history=list(answer.history)))


def _dynamic_step_diagnostics(advance,step,rank):
    solves=[]
    for name in ("solver","first_solve","final_solve"):
        answer=getattr(advance,name,None)
        if answer is not None:
            solves.append(dict(stage=name,**_sweep_diagnostics(answer)))
    for index,answer in enumerate(getattr(advance,"stage_solvers",())):
        solves.append(dict(stage=f"gauss_{index+1}",**_sweep_diagnostics(answer)))
    return _finite_diagnostic(dict(
        step=step,requested_rank=rank,status=advance.status,
        attempted_time_s=getattr(advance,"attempted_time_s",None),
        state_accepted=advance.state is not None,solves=solves,
        joint_nodal_norm=getattr(advance,"joint_nodal_norm",None),
        endpoint_rank_error=getattr(advance,"endpoint_rank_error",None),
        scientific_accuracy_certified=False))


def _dynamic_solution_quality(diagnostic,tolerance):
    solves=diagnostic["solves"]
    if not solves:
        return {}  # Missing evidence must not manufacture equilibrium.
    projected=(str(diagnostic["status"]).endswith("CONVERGED_PROJECTED") or
               any(item["status"]=="CONVERGED_PROJECTED" for item in solves))
    norms=[item["full_nodal_norm"] for item in solves]
    joint=diagnostic["joint_nodal_norm"]
    if joint is not None:
        norms.append(joint)
    measured=all(value is not None for value in norms)
    full=(not projected and all(item["converged"] for item in solves)
          and measured and max(norms)<=tolerance)
    return dict(
        solution_status=("CONVERGED_PROJECTED" if projected else diagnostic["status"]),
        solution_quality=("PROJECTED_EQUILIBRIUM" if projected else
                          "FULL_NODAL_EQUILIBRIUM" if full else "UNREPORTED_EQUILIBRIUM"),
        full_nodal_equilibrium_satisfied=(full if measured or projected else None),
        full_nodal_norm=max(norms) if all(value is not None for value in norms) else None,
        projected_norm_x=max(item["projected_norm_x"] for item in solves)
            if all(item["projected_norm_x"] is not None for item in solves) else None,
        projected_norm_y=max(item["projected_norm_y"] for item in solves)
            if all(item["projected_norm_y"] is not None for item in solves) else None)


_DYNAMIC_QUALITY_CLASSES=("FULL_NODAL_EQUILIBRIUM","PROJECTED_EQUILIBRIUM",
                          "UNREPORTED_EQUILIBRIUM")


def _accumulate_dynamic_quality(summary,step_quality):
    """Keep the weakest physical equilibrium certificate over ALL time steps.

    A later full-nodal step must not erase an earlier rank-projected one.
    Existing accepted checkpoint steps without quality evidence are marked
    unreported, never retroactively certified. The physical equations and
    fixed MVI rank are untouched.
    """
    counts=summary.get("dynamic_quality_step_counts")
    if counts is None:
        counts={key:0 for key in _DYNAMIC_QUALITY_CLASSES}
        counts["UNREPORTED_EQUILIBRIUM"]=summary.get("accepted_steps",0)
    else:
        if (not isinstance(counts,dict) or
                set(counts)!=set(_DYNAMIC_QUALITY_CLASSES) or
                any(type(x) is not int or x<0 for x in counts.values()) or
                sum(counts.values())!=summary.get("accepted_steps",0)):
            raise ValueError("MVI_DYNAMIC_QUALITY_CHECKPOINT_INCONSISTENT")
        counts=dict(counts)
    quality=step_quality.get("solution_quality","UNREPORTED_EQUILIBRIUM")
    if quality not in _DYNAMIC_QUALITY_CLASSES:
        raise ValueError("MVI_DYNAMIC_UNKNOWN_QUALITY_CLASS")
    counts[quality]+=1
    summary["dynamic_quality_step_counts"]=counts
    summary["last_step_solution_quality"]=quality
    summary["last_step_solution_status"]=step_quality.get(
        "solution_status","MVI_DYNAMIC_EQUILIBRIUM_UNREPORTED")
    projected=counts["PROJECTED_EQUILIBRIUM"]>0
    unknown=counts["UNREPORTED_EQUILIBRIUM"]>0
    if projected:
        summary["solution_status"]="CONVERGED_PROJECTED"
        summary["solution_quality"]=(
            "PROJECTED_WITH_UNREPORTED_STEPS" if unknown
            else "PROJECTED_EQUILIBRIUM")
        summary["full_nodal_equilibrium_satisfied"]=False
    elif unknown:
        summary["solution_status"]="MVI_DYNAMIC_EQUILIBRIUM_UNREPORTED"
        summary["solution_quality"]="UNREPORTED_EQUILIBRIUM"
        summary["full_nodal_equilibrium_satisfied"]=None
    else:
        summary["solution_status"]=step_quality["solution_status"]
        summary["solution_quality"]="FULL_NODAL_EQUILIBRIUM"
        summary["full_nodal_equilibrium_satisfied"]=True
    for name in ("full_nodal_norm","projected_norm_x","projected_norm_y"):
        current=step_quality.get(name)
        old=summary.get(name)
        if current is not None:
            if (not isinstance(current,(float,int)) or
                    not math.isfinite(current) or current<0):
                raise ValueError("MVI_DYNAMIC_INVALID_QUALITY_NORM")
            summary[name]=max(float(old),float(current)) if old is not None else float(current)
    return summary


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
        floating_gauge_reaction_included=(work.gauge_reaction_density is not None),
        physical_boundary_closure_certified=False,
        identity_max_abs_error=float(green.identity_max_abs_error),
        identity_relative_error=float(green.identity_relative_error),
        boundary_work_coefficient_max_abs=float(np.max(
            np.abs(green.boundary_gradient),initial=0.)),
        field_trace_orders=[len(items) for items in green.normal_left],
    )


def _dynamic_green_audit(surface,configuration,state,
                         mechanical_load,free_charge):
    """Whole instantaneous dynamic virtual work from the SAME physical jets.

    No experimental force is automatically applied at an end. The
    printed boundary actions are diagnostic, not source-certified
    microfree or electric face conditions.
    """
    audit=audit_mvi_dynamic_discrete_green(
        surface,state.factors,state.velocity,state.acceleration,
        inertia=configuration.inertia,viscosity=configuration.viscosity,
        mechanical_load=mechanical_load,free_charge=free_charge)
    return dict(
        status=audit.status,
        conservative_only=False,
        includes_inertia=True,
        includes_kelvin_voigt=True,
        electric_inertia=False,
        physical_boundary_closure_certified=False,
        conservative_identity_error=float(
            audit.conservative.identity_relative_error),
        inertia_identity_error=float(audit.inertia.identity_relative_error),
        viscosity_identity_error=float(audit.kelvin_voigt.identity_relative_error),
        total_identity_error=float(audit.total.identity_relative_error),
        mechanical_force_consistency_error=float(
            audit.mechanical_nodal_work_error),
        electric_mass_viscosity_error=float(
            audit.electric_nodal_work_error),
        total_boundary_work_coefficient_max_abs=float(
            np.max(np.abs(audit.total.boundary_gradient),initial=0.)),
        field_trace_orders=[len(items) for items in audit.total.normal_left],
    )


def _store_state(output,step,state,surface):
    field=surface.reconstruct(state.factors)
    if (not np.isfinite(field).all() or not
            np.isfinite(state.velocity).all() or not
            np.isfinite(state.acceleration).all()):
        raise ValueError("COUPLED_MVI_NONFINITE_ACCEPTED_FIELD")
    mechanical=field[:surface.n_mechanical]
    recovered=recover_cuf_displacements(surface,mechanical)
    payload=dict(x_profiles=state.factors.x,y_profiles=state.factors.y,
                 mechanical=mechanical,
                 electrical=field[surface.n_mechanical:],
                 velocity=state.velocity,acceleration=state.acceleration,
                 time_s=np.array(state.time_s))
    if recovered is not None:
        payload["physical_z_stations_m"]=recovered.z_stations_m
        payload["physical_uvw_stations_m"]=recovered.displacement_stations_m
    _write_npz(output/f"STEP_{step:04d}.npz",**payload)


def _run_loaded_case(items,*,validate_only,resume=False):
    surface=items["surface"]
    opts=items["opts"]
    result=dict(method=METHOD,schema=SCHEMA,
        research_only=True,scientific_accuracy_certified=False,
        validated_only=bool(validate_only),
        calculation_kind=opts["kind"],integrator=opts["integrator"],
        theory=surface.theory,gradient_model=surface.material.gradient_model,
        geometry_model=surface.geometry_model,rank=items["factors"].rank,
        electric_floating_gauge=bool(surface.floating_gauge),
        generalized_mechanical_end_work=(
            items["mechanical_end_work"] is not None),
        steps=opts["steps"],output_dir=str(items["output"]),
        physical_steps_computed=0,
        physical_boundary_closure_certified=False,
        time_dependent_loads=bool(items["load_program"].law is not None),
        projected_dynamic_opt_in=bool(opts.get("allow_projected_dynamic",False)))
    if validate_only:
        return result

    # Never interpret a dormant rank slot as permission to reduce R.
    # These per-slot terms are coordinate-dependent diagnostics; every
    # accepted state is still represented at the user's requested R.
    result["initial_separated_term_norms"]=(
        mvi_separated_term_norms(items["factors"]).tolist())
    result["rank_reduction_performed"]=False

    # Directory creation and exclusive process lock are owned by run().
    output=items["output"]
    manifest=input_manifest(items["path"],items["initial_path"],
                            items["loads_path"])
    result["input_manifest"]=manifest
    implementation_fingerprint=solver_implementation_fingerprint()
    result["solver_implementation_sha256"]=implementation_fingerprint
    if not resume:
        _write_json(output/"CASE.json",items["case"])
        result.update(status="IN_PROGRESS",accepted_steps=0,time_s=0.)
        _write_json(output/"RUN_SUMMARY.json",result)
    program=items["load_program"]
    mechanical,charge=program.at(0.)
    face_specs=items["electric_faces"]
    # Append a committed physical observation after its NPZ has been
    # written. The summary is the LAST per-step commit marker.
    history_path=output/"PHYSICAL_RADIAL_HISTORY.csv"

    def record_physical_step(step,state):
        q=surface.reconstruct(state.factors)[:surface.n_mechanical]
        row=radial_history_row(surface,q,step=step,time_s=state.time_s)
        append_history_row(history_path,row)

    def electric_face_diagnostic(factors):
        if not surface.electrical_on:
            return {"electrical_enabled":False}
        physical=recover_mvi_electric_faces(
            surface,factors,inner=face_specs["inner"],
            outer=face_specs["outer"])
        return summarize_mvi_electric_faces(physical)
    scales=items["field_scales"]
    energy=items["energy_scale"]
    settings=items["settings"]
    user_stop_requested=False
    def run_cancelled():
        # cooperate raises on Stop and waits on Pause; polling also runs
        # inside reduced Newton columns and Krylov actions.
        nonlocal user_stop_requested
        try:
            cooperate(output)
        except NativeUserStop:
            # Reduced drivers may translate InterruptedError into a
            # CANCELLED result. Preserve the cause at this runner boundary.
            user_stop_requested=True
            raise
        return False
    # A rejected restart must NEVER overwrite a previously committed
    # summary with failure metadata from this new invocation.
    checkpoint_validated=not resume
    try:
        if opts["kind"]=="STATIC":
            answer=solve_coupled_mvi_static(
                surface,items["factors"],program=items["program"],
                field_scales=scales,energy_scale_j=energy,
                mechanical_load=mechanical,free_charge=charge,
                settings=settings,cancelled=run_cancelled)
            if not answer.converged:
                # Preserve full reduced X/Y failure evidence before raising.
                # Never write an accepted physical state on failed convergence.
                _write_json(output/"MVI_FAILED_SWEEP_DIAGNOSTICS.json",dict(
                    **_sweep_diagnostics(answer),
                    requested_rank=items["factors"].rank,
                    scientific_accuracy_certified=False))
                raise RuntimeError("COUPLED_MVI_STATIC_"+answer.status)
            result.update(
                solution_status=answer.status,
                solution_quality=("PROJECTED_EQUILIBRIUM" if answer.status=="CONVERGED_PROJECTED"
                                  else "FULL_NODAL_EQUILIBRIUM"),
                full_nodal_equilibrium_satisfied=(answer.status!="CONVERGED_PROJECTED"),
                projected_norm_x=answer.projected_norm_x,
                projected_norm_y=answer.projected_norm_y,
                full_nodal_norm=answer.full_nodal_norm,
                field_change=answer.field_change,sweeps=answer.sweeps)
            q=surface.reconstruct(answer.factors)
            result["discrete_green"]=_green_audit(
                surface,answer.factors,mechanical,charge)
            result["electric_faces"]=electric_face_diagnostic(answer.factors)
            if answer.factors.rank!=items["factors"].rank:
                raise RuntimeError("COUPLED_MVI_RANK_CHANGED")
            result["last_separated_term_norms"]=(
                mvi_separated_term_norms(answer.factors).tolist())
            static_payload=dict(
                x_profiles=answer.factors.x,y_profiles=answer.factors.y,
                mechanical=q[:surface.n_mechanical],
                electrical=q[surface.n_mechanical:])
            recovered=recover_cuf_displacements(
                surface,q[:surface.n_mechanical])
            if recovered is not None:
                static_payload["physical_z_stations_m"]=recovered.z_stations_m
                static_payload["physical_uvw_stations_m"]=recovered.displacement_stations_m
            _write_npz(output/"STATIC_ACCEPTED.npz",**static_payload)
            result.update(status=("EXPERIMENTAL_PROJECTED_STATIONARY_UNCERTIFIED"
                                  if answer.status=="CONVERGED_PROJECTED" else
                                  "EXPERIMENTAL_STATIONARY_UNCERTIFIED"),
                          accepted_steps=1,physical_steps_computed=1)
        else:
            configuration=NativeCoupledMVINewmark(
                surface,NativeCoupledMVIInertia(
                    surface,density_kg_m3=items["density"],
                    mass_damping_rate_per_s=items["damping"]),
                program=items["program"],eta_s=items["eta"])
            if resume:
                state,prior=prepare_resume(
                    output,items["case"],manifest,surface=surface,
                    configuration=configuration,steps=opts["steps"],
                    dt_s=opts["dt_s"],input_rank=items["factors"].rank,
                    implementation_fingerprint=implementation_fingerprint)
                first_step=prior["accepted_steps"]+1
                checkpoint_validated=True
                result=prior
                # --resume explicitly starts a new execution from the checked
                # checkpoint. Only now, under the output lock, retire controls
                # belonging to the terminated process. A rejected resume must
                # preserve them as well as the accepted result files.
                for name in ("CONTROL_STOP","CONTROL_PAUSE"):
                    (output/name).unlink(missing_ok=True)
                result.pop("stopped_by_user",None)
                result.pop("error_type",None)
                result.pop("detail",None)
                result.pop("failed_time_step",None)
                result["status"]="IN_PROGRESS"
                _write_json(output/"RUN_SUMMARY.json",result)
            else:
                initial=close_mvi_instantaneous_state(
                    configuration,items["factors"],items["initial"]["velocity"],
                    time_s=0.,field_scales=scales,energy_scale_j=energy,
                    mechanical_load=mechanical,free_charge=charge,
                    electric_settings=settings,cancelled=run_cancelled,
                    max_acceleration_coordinates=opts["max_acceleration_coordinates"],
                    **items["acceleration_options"])
                if initial.state is None:
                    raise RuntimeError("COUPLED_MVI_INITIAL_"+initial.status)
                state=initial.state
                if state.factors.rank!=items["factors"].rank:
                    raise RuntimeError("COUPLED_MVI_RANK_CHANGED")
                result["initial_separated_term_norms"]=(
                    mvi_separated_term_norms(state.factors).tolist())
                result["initial_electric_status"]=initial.electric_status
                result["initial_discrete_green"]=_dynamic_green_audit(
                    surface,configuration,state,mechanical,charge)
                result["initial_electric_faces"]=electric_face_diagnostic(
                    state.factors)
                _store_state(output,0,state,surface)
                record_physical_step(0,state)
                result["last_state_sha256"]=accepted_state_sha256(output,0)
                _write_json(output/"RUN_SUMMARY.json",result)
                first_step=1
            driver=(configuration if opts["integrator"]=="NEWMARK"
                    else NativeCoupledMVIBathe(configuration)
                    if opts["integrator"]=="BATHE"
                    else NativeCoupledMVIGauss2(configuration))
            for step in range(first_step,opts["steps"]+1):
                cooperate(output)
                common=dict(field_scales=scales,energy_scale_j=energy,
                            settings=settings,cancelled=run_cancelled)
                t_end=float(state.time_s+opts["dt_s"])
                if opts["integrator"]=="NEWMARK":
                    load_end,charge_end=program.at(t_end)
                    advance=driver.advance(state,opts["dt_s"],
                        mechanical_load=load_end,free_charge=charge_end,**common)
                else:
                    advance=driver.advance(
                        state,opts["dt_s"],
                        loads_at=program.at,**common,
                        **({"max_acceleration_coordinates":
                            opts["max_acceleration_coordinates"]}
                           if opts["integrator"]=="GAUSS_IRK4" else {}),
                        **({"acceleration_options":items["acceleration_options"]}
                           if opts["integrator"]=="GAUSS_IRK4" else {}))
                diagnostic=_dynamic_step_diagnostics(
                    advance,step,items["factors"].rank)
                _write_json(output/f"MVI_STEP_{step:04d}_DIAGNOSTICS.json",diagnostic)
                if advance.state is None:
                    result["failed_time_step"]=diagnostic
                    raise RuntimeError("COUPLED_MVI_TIME_STEP_"+advance.status)
                if advance.state.factors.rank!=items["factors"].rank:
                    raise RuntimeError("COUPLED_MVI_RANK_CHANGED")
                state=advance.state
                mechanical,charge=program.at(state.time_s)
                result["last_separated_term_norms"]=(
                    mvi_separated_term_norms(state.factors).tolist())
                result["last_discrete_green"]=_dynamic_green_audit(
                    surface,configuration,state,mechanical,charge)
                result["last_electric_faces"]=electric_face_diagnostic(
                    state.factors)
                _store_state(output,step,state,surface)
                record_physical_step(step,state)
                result["last_state_sha256"]=accepted_state_sha256(output,step)
                result["last_time_step_diagnostics"]=diagnostic
                _accumulate_dynamic_quality(result,_dynamic_solution_quality(
                    diagnostic,settings.halfstep.tolerance_scaled))
                result.update(accepted_steps=step,
                              physical_steps_computed=step,
                              time_s=float(state.time_s))
                _write_json(output/"RUN_SUMMARY.json",result)
            result["status"]="EXPERIMENTAL_COMPLETED_UNCERTIFIED"
        result["validated_only"]=False
        _write_json(output/"RUN_SUMMARY.json",result)
        return result
    except Exception as exc:
        if not checkpoint_validated:
            # Preserve all previously accepted results unchanged if a
            # checkpoint, input hash, output history or rank fails audit.
            raise
        stopped=isinstance(exc,NativeUserStop) or user_stop_requested
        result.update(status="STOPPED_BY_USER" if stopped else "FAILED",
                      error_type=type(exc).__name__,
                      detail=str(exc),scientific_accuracy_certified=False)
        if stopped:
            result["stopped_by_user"]=True
        _write_json(output/"RUN_SUMMARY.json",result)
        raise


def run(path,*,validate_only,resume=False):
    """Run under an exclusive output lock; resume only on explicit request.

    Validation never creates or changes a result directory. The original
    CASE and source NPZ fingerprints are rechecked on every restart.
    """
    if type(resume) is not bool:
        raise TypeError("COUPLED_MVI_RESUME_REQUIRES_BOOLEAN")
    if resume and validate_only:
        raise ValueError("COUPLED_MVI_RESUME_REQUIRES_RUN")
    items=load_case(path,allow_existing_output=resume)
    if resume and items["opts"]["kind"]!="DYNAMIC":
        raise ValueError("COUPLED_MVI_RESUME_REQUIRES_DYNAMIC_CASE")
    if validate_only:
        return _run_loaded_case(items,validate_only=True)
    output=items["output"]
    if resume:
        if not output.is_dir():
            raise ValueError("COUPLED_MVI_CHECKPOINT_OUTPUT_DIRECTORY_MISSING")
    else:
        output.mkdir(parents=True,exist_ok=False)
    with exclusive_run_lock(output):
        return _run_loaded_case(items,validate_only=False,resume=resume)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only",action="store_true")
    group.add_argument("--run",action="store_true")
    group.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.case,validate_only=args.validate_only,
                         resume=args.resume),indent=2))


if __name__=="__main__":
    main()
