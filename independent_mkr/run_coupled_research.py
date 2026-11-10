"""Native nonlinear G3D/Gz STRONG MKR research CASE executor.

Independent of the MVI model and all variational-reference solvers.
Its own strong jet differentiation, physical end rows, strong Newton,
initial acceleration, native Newmark/Bathe/Gauss2, and saved results
are selected through one explicit schema. No work is performed on
import or on --validate-only.

Uncertified source microfree radial face conditions, high-order strong
boundary convergence and physical benchmarks are NOT silently claimed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from cylinder_paths import activate
activate()
from cylinder_native_run_control import cooperate

import numpy as np
from flexo_gradient_g3d import G3DMaterial
from cylinder_direct_strong_fdm import DirectStrongGrid
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_direct_strong_newton import StrongNewtonSettings
from cylinder_native_mkr_end_program import compile_native_mkr_end_program
from cylinder_native_mkr_newmark import NativeNewmarkState,NativeStrongNewmark
from cylinder_native_mkr_initial import initialize_consistent_native_mkr
from cylinder_native_mkr_bathe import NativeStrongBathe
from cylinder_native_mkr_gauss2 import NativeStrongGauss2
from cylinder_native_mkr_face_recovery import recover_native_mkr_face_induction

SCHEMA="INDEPENDENT_MKR_COUPLED_RESEARCH_V1"
METHOD="MKR_DIRECT_STRONG_COUPLED_RESEARCH"
THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
MAT_KEYS=frozenset(("young_pa","poisson","ell_m","mu1_c_per_m",
                     "mu2_c_per_m","kappa_f_per_m","gradient_model"))
PHYS_KEYS=frozenset(("theory","geometry_model","material",
                      "thickness_m","density_kg_m3","eta_s",
                      "mass_damping_rate_per_s","n_gauss","potential_order"))
GRID_KEYS=frozenset(("nx","ns","length_m","radius_m"))
SOLVER_KEYS=frozenset(("kind","integrator","dt_s","steps","max_dofs",
                        "max_iterations","mechanical_state_scale",
                        "mechanical_equation_scale","electrical_state_scale",
                        "electrical_equation_scale","potential_scale_v"))
SOLVER_OPTIONAL=frozenset(("linear_solver","dense_lu_dof_limit"))
CASE_KEYS=frozenset(("schema","research_opt_in","physics","grid","boundary",
                      "solver","initial_state_npz","loads_npz","output_dir"))
OPTIONAL_KEYS=frozenset(("mechanical_end_work","electric_faces"))


def _strict(value,keys,label):
    if not isinstance(value,dict) or set(value)!=set(keys):
        raise ValueError(label+"_KEYS_INVALID")
    return value


def _finite(value,label,positive=True):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(label+"_INVALID")
    try:
        val=float(value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(label+"_INVALID") from exc
    if not math.isfinite(val) or (val<=0 if positive else val<0):
        raise ValueError(label+"_INVALID")
    return val


def _resource(raw,parent,label):
    if not isinstance(raw,str) or not raw.strip():
        raise ValueError(label+"_PATH_INVALID")
    path=Path(raw).expanduser()
    if not path.is_absolute():
        path=parent/path
    path=path.resolve()
    if not path.is_file() or path.suffix.lower()!=".npz":
        raise ValueError(label+"_REQUIRES_NPZ_FILE")
    return path


def _npz(path,required,optional=()):
    with np.load(path,allow_pickle=False) as f:
        keys=set(f.files)
        if not set(required)<=keys or keys-set(required)-set(optional):
            raise ValueError("MKR_COUPLED_NPZ_KEYS_INVALID")
        data={name:np.array(f[name],dtype=float,copy=True) for name in keys}
    if not all(np.isfinite(x).all() for x in data.values()):
        raise ValueError("MKR_COUPLED_NPZ_NONFINITE")
    return data


def _face_charge(surface,spec,potential_order):
    """MKR-owned physical face charge -> normalized TE electric coefficients.

    sigma is physical C/m^2 on real cylindrical faces with area
    multiplier (1+z/R). It is the same physical electric enthalpy sign,
    not an MVI spatial integration routine.
    """
    if spec is None:
        return None
    if not isinstance(spec,dict) or set(spec)!={"inner","outer"}:
        raise ValueError("MKR_FACE_REQUIRES_INNER_OUTER")
    if not surface.electrical_on:
        if any(value is not None for value in spec.values()):
            raise ValueError("MKR_FACE_CHARGE_WITH_ELECTRICITY_OFF")
        return None
    shape=(surface.grid.nx,surface.grid.ns)
    order=potential_order
    result=np.zeros((order+1,*shape))
    for side,n in (("inner",-1),("outer",1)):
        item=spec[side]
        if item is None:
            continue
        if not isinstance(item,dict) or set(item)!={"mode","charge_c_per_m2"}:
            raise ValueError("MKR_FACE_ENTRY_KEYS_INVALID")
        if item["mode"]=="INSULATED":
            if item["charge_c_per_m2"] is not None:
                raise ValueError("MKR_FACE_INSULATED_REQUIRES_NULL")
            continue
        if item["mode"]!="CHARGE":
            raise ValueError("MKR_FACE_POTENTIAL_NOT_IMPLEMENTED")
        raw=item["charge_c_per_m2"]
        if raw is None or isinstance(raw,(bool,np.bool_)):
            raise ValueError("MKR_FACE_CHARGE_INVALID")
        try:
            values=np.asarray(raw,float)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError("MKR_FACE_CHARGE_INVALID") from exc
        if values.ndim==0:
            values=np.full(shape,float(values))
        if values.shape!=shape or not np.isfinite(values).all():
            raise ValueError("MKR_FACE_CHARGE_SHAPE_INVALID")
        z=n*surface.thickness_m/2.
        factor=1.+z/surface.grid.radius_m
        result+=factor*np.power(float(n),np.arange(order+1))[:,None,None]*values
    return result


def load_case(filename):
    path=Path(filename).expanduser().resolve()
    if not path.is_file() or path.suffix.lower()!=".json":
        raise ValueError("MKR_COUPLED_REQUIRES_EXISTING_JSON_CASE")
    case=json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(case,dict) or not CASE_KEYS<=set(case) or
            set(case)-CASE_KEYS-OPTIONAL_KEYS):
        raise ValueError("MKR_COUPLED_CASE_KEYS_INVALID")
    if case["schema"]!=SCHEMA or case["research_opt_in"] is not True:
        raise ValueError("MKR_COUPLED_EXPLICIT_RESEARCH_SCHEMA_REQUIRED")
    p=_strict(case["physics"],PHYS_KEYS,"MKR_PHYSICS")
    g=_strict(case["grid"],GRID_KEYS,"MKR_GRID")
    m=_strict(p["material"],MAT_KEYS,"MKR_MATERIAL")
    if p["theory"] not in THEORIES:
        raise ValueError("MKR_COUPLED_UNKNOWN_THEORY")
    grid=DirectStrongGrid(**g)
    model=DirectStrongShell(
        p["theory"],G3DMaterial(**m),grid=grid,
        thickness_m=p["thickness_m"],n_gauss=p["n_gauss"],
        geometry_model=p["geometry_model"])
    if model.electrical_on:
        potential_order=p["potential_order"]
        if type(potential_order) is not int or not 0<=potential_order<=4:
            raise ValueError("MKR_COUPLED_POTENTIAL_ORDER_REQUIRED")
    elif p["potential_order"] is not None:
        raise ValueError("MKR_COUPLED_ELECTRIC_OFF_REQUIRES_NULL_ORDER")
    potential_order=p["potential_order"]
    b=_strict(case["boundary"],("mechanical","electric","axial_orders"),
              "MKR_BOUNDARY")
    opts=case["solver"]
    if (not isinstance(opts,dict) or not SOLVER_KEYS.issubset(opts)
            or set(opts)-SOLVER_KEYS-SOLVER_OPTIONAL):
        raise ValueError("MKR_SOLVER_KEYS_INVALID")
    if opts["kind"] not in ("STATIC","DYNAMIC"):
        raise ValueError("MKR_COUPLED_UNKNOWN_CALCULATION_KIND")
    if opts["kind"]=="STATIC":
        if (opts["integrator"] is not None or opts["dt_s"] is not None or
                opts["steps"]!=0):
            raise ValueError("MKR_COUPLED_STATIC_FORBIDS_TIME_FIELDS")
    else:
        if opts["integrator"] not in ("NEWMARK","BATHE","GAUSS_IRK4"):
            raise ValueError("MKR_COUPLED_UNKNOWN_INTEGRATOR")
        _finite(opts["dt_s"],"MKR_TIME_STEP")
        if type(opts["steps"]) is not int or not 1<=opts["steps"]<=120:
            raise ValueError("MKR_COUPLED_TIME_STEPS_INVALID")
    if (type(opts["max_dofs"]) is not int or not 1<=opts["max_dofs"]<=20000
            or type(opts["max_iterations"]) is not int or
            not 1<=opts["max_iterations"]<=150):
        raise ValueError("MKR_COUPLED_NEWTON_LIMIT_INVALID")
    max_unknowns=(model.n_mechanical+
                  (potential_order+1 if model.electrical_on else 0))*grid.nx*grid.ns
    # Gauss has two independent implicit stage fields in ONE Newton system.
    if opts["integrator"]=="GAUSS_IRK4":
        max_unknowns*=2
    if (b["electric"]=="INSULATED_FLOATING") and model.electrical_on:
        max_unknowns+=1
    if max_unknowns>opts["max_dofs"]:
        raise ValueError("MKR_COUPLED_STRONG_DOF_BUDGET_TOO_SMALL: "+str(max_unknowns))
    program=compile_native_mkr_end_program(
        model,axial_orders=b["axial_orders"],mechanical=b["mechanical"],
        electric=b["electric"],potential_order=potential_order,
        mechanical_end_work=case.get("mechanical_end_work"),
        potential_scale_v=opts["potential_scale_v"])
    qm_shape=model.shape
    e_shape=(potential_order+1,grid.nx,grid.ns) if model.electrical_on else None
    from cylinder_direct_strong_newton import _positive_scale
    qscale=_positive_scale(opts["mechanical_state_scale"],qm_shape,
                           "MKR mechanical state")
    rscale=_positive_scale(opts["mechanical_equation_scale"],qm_shape,
                           "MKR mechanical equation")
    if model.electrical_on:
        escale=_positive_scale(opts["electrical_state_scale"],e_shape,
                               "MKR electric state")
        erescale=_positive_scale(opts["electrical_equation_scale"],e_shape,
                                 "MKR electric equation")
    else:
        if (opts["electrical_state_scale"] is not None or
                opts["electrical_equation_scale"] is not None):
            raise ValueError("MKR_ELECTRIC_SCALES_WHEN_ELECTRICITY_OFF")
        escale=erescale=None
    density=_finite(p["density_kg_m3"],"MKR_DENSITY")
    eta=_finite(p["eta_s"],"MKR_VISCOSITY",positive=False)
    damping=_finite(p["mass_damping_rate_per_s"],"MKR_MASS_DAMPING",
                    positive=False)
    source=_resource(case["initial_state_npz"],path.parent,"MKR_INITIAL")
    loads_file=_resource(case["loads_npz"],path.parent,"MKR_LOADS")
    seed_keys={"mechanical"}
    if opts["kind"]=="DYNAMIC":
        seed_keys.add("velocity")
    if model.electrical_on:
        seed_keys.add("electrical")
    seed=_npz(source,seed_keys,("acceleration",))
    if seed["mechanical"].shape!=qm_shape:
        raise ValueError("MKR_INITIAL_MECHANICAL_SHAPE_INVALID")
    if "velocity" in seed and seed["velocity"].shape!=qm_shape:
        raise ValueError("MKR_INITIAL_VELOCITY_SHAPE_INVALID")
    if "acceleration" in seed and seed["acceleration"].shape!=qm_shape:
        raise ValueError("MKR_INITIAL_ACCELERATION_SHAPE_INVALID")
    if model.electrical_on and seed["electrical"].shape!=e_shape:
        raise ValueError("MKR_INITIAL_ELECTRICAL_SHAPE_INVALID")
    sources=_npz(loads_file,("mechanical_load",),
                 ("electrical_charge",))
    if sources["mechanical_load"].shape!=qm_shape:
        raise ValueError("MKR_MECHANICAL_LOAD_SHAPE_INVALID")
    if model.electrical_on:
        raw=sources.get("electrical_charge",np.zeros(e_shape))
        if raw.shape!=e_shape:
            raise ValueError("MKR_ELECTRIC_CHARGE_SHAPE_INVALID")
        face=_face_charge(model,case.get("electric_faces"),potential_order)
        sources["electrical_charge"]=raw+(face if face is not None else 0.)
    elif "electrical_charge" in sources:
        raise ValueError("MKR_ELECTRIC_CHARGE_WHEN_OFF")
    elif case.get("electric_faces") is not None:
        _face_charge(model,case["electric_faces"],potential_order)
    output=case["output_dir"]
    if not isinstance(output,str) or not output.strip():
        raise ValueError("MKR_OUTPUT_PATH_INVALID")
    output=Path(output).expanduser()
    if not output.is_absolute():
        output=path.parent/output
    output=output.resolve()
    if (output.exists() or output in (path.parent,path.parent.parent,
        source.parent,loads_file.parent,source,loads_file,path)):
        raise ValueError("MKR_OUTPUT_ALREADY_EXISTS_OR_UNSAFE")
    from cylinder_native_mkr_initial import _verify_fixed_initial_supports
    engine=NativeStrongNewmark(
        model,axial_orders=program.axial_orders,
        mechanical_ends=program.mechanical_ends,
        electrical_ends=program.electrical_ends,
        electric_gauge=program.electric_gauge,
        density_kg_m3=density,eta_s=eta,
        mass_damping_rate_per_s=damping,
        mechanical_load=sources["mechanical_load"],
        electrical_charge=sources.get("electrical_charge"))
    if opts["kind"]=="DYNAMIC":
        _verify_fixed_initial_supports(
            engine,seed["mechanical"],seed["velocity"],
            absolute_tolerance=1.e-9)
    return dict(path=path,case=case,model=model,program=program,engine=engine,
                settings=StrongNewtonSettings(
                    max_dofs=opts["max_dofs"],
                    max_iterations=opts["max_iterations"],
                    linear_solver=opts.get("linear_solver","AUTO"),
                    dense_lu_dof_limit=opts.get("dense_lu_dof_limit",384)),
                seed=seed,sources=sources,opts=opts,output=output,
                qscale=qscale,rscale=rscale,escale=escale,
                erescale=erescale)


def _json(path,data):
    tmp=path.with_suffix(path.suffix+".pending")
    try:
        tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",
                       encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _npz_save(path,**data):
    tmp=path.with_suffix(".pending.npz")
    try:
        with tmp.open("xb") as stream:
            np.savez_compressed(stream,**data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _save_state(path,state):
    _npz_save(path,time_s=np.array(state.time_s),
              mechanical=state.mechanical,velocity=state.velocity,
              acceleration=state.acceleration,
              **({"electrical":state.electrical}
                 if state.electrical is not None else {}))


def run(filename,*,validate_only):
    args=load_case(filename)
    model=args["model"]
    opts=args["opts"]
    summary=dict(schema=SCHEMA,method=METHOD,
        validated_only=bool(validate_only),research_only=True,
        scientific_accuracy_certified=False,
        physical_boundary_closure_certified=False,
        solver_backend="INDEPENDENT_MKR",
        theory=model.theory,geometry_model=model.geometry_model,
        gradient_model=model.material.gradient_model,
        kind=opts["kind"],calculation_kind=opts["kind"],
        integrator=opts["integrator"],steps=opts["steps"],
        requested_steps=opts["steps"],
        time_s=0.,physical_steps_computed=0,
        output_dir=str(args["output"]),
        spatial_method="DIRECT_STRONG_MKR_NO_MVI_SPATIAL_REUSE")
    if validate_only:
        return summary
    out=args["output"]
    out.mkdir(parents=True,exist_ok=False)
    _json(out/"CASE.json",args["case"])
    summary.update(status="IN_PROGRESS",accepted_steps=0)
    _json(out/"RUN_SUMMARY.json",summary)
    kw=dict(mechanical_state_scale=args["qscale"],
            mechanical_equation_scale=args["rscale"],
            electrical_state_scale=args["escale"],
            electrical_equation_scale=args["erescale"],
            settings=args["settings"])
    seed=args["seed"]
    engine=args["engine"]
    def face_diagnostic(q,p):
        if not model.electrical_on:
            return {"electrical_enabled":False}
        return recover_native_mkr_face_induction(
            model,q,p,face_specification=args["case"].get("electric_faces"))
    try:
        if opts["kind"]=="STATIC":
            answer=model.solve(
                seed["mechanical"],seed.get("electrical"),
                axial_orders=args["program"].axial_orders,
                mechanical_ends=args["program"].mechanical_ends,
                electrical_ends=args["program"].electrical_ends,
                electric_gauge=args["program"].electric_gauge,
                mechanical_load=args["sources"]["mechanical_load"],
                electrical_charge=args["sources"].get("electrical_charge"),
                **kw)
            if not answer.converged:
                raise RuntimeError("MKR_STATIC_"+answer.status)
            summary["electric_faces"]=face_diagnostic(
                answer.mechanical,answer.electrical)
            _npz_save(out/"STATIC_ACCEPTED.npz",
                      mechanical=answer.mechanical,
                      **({"electrical":answer.electrical}
                         if answer.electrical is not None else {}))
            summary.update(status="EXPERIMENTAL_STATIONARY_UNCERTIFIED",
                           accepted_steps=1,physical_steps_computed=1)
        else:
            closed=initialize_consistent_native_mkr(
                engine,seed["mechanical"],seed["velocity"],
                seed.get("electrical"),time_s=0.,
                acceleration_guess=seed.get("acceleration"),
                **kw)
            if closed.state is None:
                raise RuntimeError("MKR_INITIAL_"+closed.nonlinear.status)
            state=closed.state
            summary["initial_electric_faces"]=face_diagnostic(
                state.mechanical,state.electrical)
            _save_state(out/"STEP_0000.npz",state)
            driver=(engine if opts["integrator"]=="NEWMARK"
                    else NativeStrongBathe(engine)
                    if opts["integrator"]=="BATHE"
                    else NativeStrongGauss2(engine))
            for k in range(1,opts["steps"]+1):
                cooperate(out)
                if opts["integrator"]=="NEWMARK":
                    next_step=driver.advance(state,opts["dt_s"],**kw)
                else:
                    next_step=driver.advance(state,opts["dt_s"],
                                             loads_at=None,**kw)
                if next_step.state is None:
                    status=(next_step.status if hasattr(next_step,"status")
                            else next_step.nonlinear.status)
                    raise RuntimeError("MKR_STEP_"+status)
                state=next_step.state
                summary["last_electric_faces"]=face_diagnostic(
                    state.mechanical,state.electrical)
                _save_state(out/f"STEP_{k:04d}.npz",state)
                summary.update(accepted_steps=k,physical_steps_computed=k,
                               time_s=float(state.time_s))
                _json(out/"RUN_SUMMARY.json",summary)
            summary["status"]="EXPERIMENTAL_COMPLETED_UNCERTIFIED"
        summary["validated_only"]=False
        _json(out/"RUN_SUMMARY.json",summary)
        return summary
    except Exception as exc:
        summary.update(status="FAILED",error_type=type(exc).__name__,
                       detail=str(exc),scientific_accuracy_certified=False)
        _json(out/"RUN_SUMMARY.json",summary)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only",action="store_true")
    group.add_argument("--run",action="store_true")
    settings=parser.parse_args()
    print(json.dumps(run(settings.case,validate_only=settings.validate_only),
                     indent=2))


if __name__=="__main__":
    main()
