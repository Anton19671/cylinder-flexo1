#!/usr/bin/env python3
"""Explicit small-grid CASE route for the variational static FDM backend."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import numpy as np

from cylinder_functional_assembly import CylinderFunctional, surface_trapezoid
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_variational_fdm import StaticFDMSettings, solve_static_fdm
from flexo_gradient_g3d import G3DMaterial
from flexo_scale_policy import micro_material_from_case, flexoelectric_enabled
import final_progress as progress

BACKEND="VARIATIONAL_REFERENCE"


def selected_backend(case):
    name=str(case.get("solver_backend","CLASSICAL")).strip().upper()
    if name not in ("CLASSICAL",BACKEND):
        raise ValueError("solver_backend must be CLASSICAL or VARIATIONAL_REFERENCE")
    return name


def _real(value,label,positive=False):
    if isinstance(value,bool):raise ValueError(label+": boolean is not numeric")
    try:v=float(value)
    except (TypeError,ValueError) as exc:raise ValueError(label+": finite number required") from exc
    if not np.isfinite(v) or (positive and v<=0):raise ValueError(label+": invalid number/sign")
    return v


def _integer(value,label,minimum):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError(f"{label}: integer >= {minimum} required")
    return value


def _load_field(value,shape,label):
    raw=np.asarray(value,dtype=object)
    if any(isinstance(x,bool) for x in raw.ravel()):raise ValueError(label+": boolean is not numeric")
    a=np.asarray(value,float)
    if a.shape==(shape[0],):a=np.broadcast_to(a[:,None,None],shape).copy()
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(label+": wrong finite field shape")
    return a


@dataclass(frozen=True)
class ReferenceCase:
    problem: BoundarySpatialProblem
    mechanical: np.ndarray
    electrical: np.ndarray | None
    dead_load: np.ndarray
    free_charge: np.ndarray | None
    factors: tuple
    settings: StaticFDMSettings


def prepare_reference_case(case):
    """Validate without creating directories or substituting another backend."""
    if selected_backend(case)!=BACKEND:raise ValueError("explicit reference backend required")
    if str(case.get("calculation_kind","")).upper()!="STATIC":
        raise ValueError("reference backend currently requires STATIC")
    if str(case.get("method","")).upper()!="FDM":
        raise ValueError("reference backend currently requires FDM; MVI cycles are not connected")
    if str(case.get("geometry_model","")).upper()!="GREEN_LAGRANGE":
        raise ValueError("reference backend requires full GREEN_LAGRANGE")
    if str(case.get("material_model","")).upper()!="HOOKE":
        raise ValueError("reference backend requires HOOKE")
    if str(case.get("mode","")).upper()!="NONLINEAR_ELASTIC":
        raise ValueError("reference backend requires mode=NONLINEAR_ELASTIC")
    if (case.get("units") or {}).get("system")!="SI_N_M_PA":
        raise ValueError("reference backend requires SI_N_M_PA")
    theory=str(case.get("theory","")).upper()
    if theory=="CUF":theory="CUF_"+str(case.get("cuf_model","")).upper()
    scale=str(case.get("scale_model","")).upper()
    if scale=="MICRO":
        material=micro_material_from_case(case)
        order=_integer(int(case["micro"]["potential_order"]),"potential_order",0) if flexoelectric_enabled(case) else 0
        if order>4:raise ValueError("potential_order must be 0..4")
    elif scale=="MACRO":
        m=case["material"]
        material=G3DMaterial(_real(m["E0"],"E0",True),_real(m["nu0"],"nu0"),0.,0.,0.,0.)
        order=0
    else:raise ValueError("explicit scale_model=MACRO/MICRO required")
    geo=case["geometry"]
    length,radius,thickness=(_real(geo[key],key,True) for key in ("L","R","h"))
    grid=case.get("fdm_grid") or case.get("grid") or {}
    nx=_integer(grid.get("nx"),"nx",4);ns=_integer(grid.get("ns"),"ns",4)
    if grid.get("stencil") is not None:
        raise ValueError("reference operators are fixed; omit legacy grid.stencil")
    if nx*ns>256:raise ValueError("reference backend limited to 256 surface nodes")
    quad=case.get("quadrature") or {}
    if str(quad.get("type","")).upper()!="GAUSS":raise ValueError("reference quadrature requires GAUSS")
    n_gauss=_integer(quad.get("n_gauss"),"n_gauss",2)
    if n_gauss>32:raise ValueError("reference n_gauss must be <=32")
    # Full GL enthalpy contains fourth powers of the displacement ansatz.
    # Avoid obvious thickness underintegration of high-order CUF/Reddy modes;
    # this floor does not replace a quadrature convergence study for H(z).
    minimum_gauss=(2*int(theory[-1])+1 if theory.startswith("CUF_TE")
                   and theory[-1] in "1234" else 7 if theory=="REDDY" else 3)
    if n_gauss<minimum_gauss:
        raise ValueError(f"{theory}: n_gauss >= {minimum_gauss} required to avoid thickness underintegration")
    functional=CylinderFunctional(theory,material,radius_m=radius,thickness_m=thickness,
        surface_operators=surface_trapezoid(nx,ns,length,radius),n_gauss=n_gauss)
    q=np.zeros((functional.n_mechanical,nx,ns))
    p=np.zeros((order+1,nx,ns)) if functional.electrical_on else None
    if q.size+(0 if p is None else p.size)>4096:raise ValueError("reference nodal DOF limit exceeded")
    bc=case.get("boundary_conditions") or {}
    spec=bc.get("variational")
    if not isinstance(spec,dict):raise ValueError("explicit boundary_conditions.variational required")
    if bc.get("advanced") is not None or bc.get("engineering") is not None:
        raise ValueError("reference backend uses only explicit variational BC")
    for side in ("left","right"):
        if bc.get(side) is not None and str(bc[side]).upper()!=str(
                spec.get("mechanical",{}).get(side,{}).get("support","FREE")).upper():
            raise ValueError(side+": conflicting classical/variational support")
    if scale=="MACRO" and any(str(site.get("higher","NATURAL")).upper()!="NATURAL"
                             for site in spec.get("mechanical",{}).values()):
        raise ValueError("higher-order reference traces require MICRO G3D")
    for key in ("loads_npz","initial_profiles_npz","initial_state","initial_GB",
                "newton","linear","inner_mvi","mvi_grid","fdm_operator_scheme"):
        if case.get(key) is not None:
            raise ValueError(key+": legacy fields are not accepted by reference backend")
    problem=BoundarySpatialProblem(functional,bc_spec=spec)
    loads=case.get("reference_loads")
    definition=case.get("load_definition")
    if loads is not None and definition is not None:raise ValueError("choose reference_loads or load_definition")
    if loads is not None:
        if not isinstance(loads,dict) or set(loads)-{"mechanical","free_charge","load_factors"}:
            raise ValueError("unknown reference_loads fields")
        if "mechanical" not in loads:raise ValueError("reference_loads.mechanical required")
        dead=_load_field(loads["mechanical"],q.shape,"mechanical load")
        charge=None
        if "free_charge" in loads:
            if p is None:raise ValueError("free_charge requires electrical model ON")
            charge=_load_field(loads["free_charge"],p.shape,"free charge")
        factors=loads.get("load_factors")
    else:
        if not isinstance(definition,dict):raise ValueError("explicit spatial load required")
        allowed={"type","q","face","surface","kinematics","load_factors"}
        if set(definition)-allowed:raise ValueError("unsupported reference load_definition fields")
        if (str(definition.get("type","")).upper()!="UNIFORM_RADIAL_FACE" or
                str(definition.get("face",definition.get("surface",""))).upper()!="MIDDLE" or
                str(definition.get("kinematics","")).upper()!="DEAD_REFERENCE"):
            raise ValueError("reference pressure requires UNIFORM_RADIAL_FACE + MIDDLE + DEAD_REFERENCE")
        if "face" in definition and "surface" in definition and definition["face"]!=definition["surface"]:
            raise ValueError("conflicting pressure face/surface")
        dead=np.zeros_like(q)
        radial=2*(int(theory[-1])+1) if theory.startswith("CUF_") else 2
        dead[radial]=_real(definition.get("q"),"pressure q")
        charge=None;factors=definition.get("load_factors")
    if not isinstance(factors,list) or not factors:raise ValueError("explicit nonempty load_factors required")
    factors=tuple(_real(x,"load factor") for x in factors)
    if len(factors)>100:raise ValueError("reference static series limited to 100 load factors")
    options=case.get("reference_newton",{})
    if not isinstance(options,dict):raise ValueError("reference_newton must be an object")
    try:settings=StaticFDMSettings(**options)
    except TypeError as exc:raise ValueError("invalid reference_newton fields") from exc
    # Compile charge compatibility and all trace equations before declaring a
    # CASE valid; dry-run validation must not generate any output files.
    a=problem.assemble(q,p,dead_load=dead,free_charge=charge)
    if a.space.basis.shape[1]>settings.max_free_dofs:
        raise ValueError("reference free-DOF limit exceeded; use a smaller verification grid")
    if not isinstance(case.get("output_dir"),str) or not case["output_dir"].strip():
        raise ValueError("explicit output_dir required")
    return ReferenceCase(problem,q,p,dead,charge,factors,settings)


def _write_json(path,data):
    temp=path.with_name(path.name+".tmp")
    with temp.open("w",encoding="utf-8") as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def run_reference_case(case,case_path):
    prepared=prepare_reference_case(case)
    out=Path(case["output_dir"])
    if not out.is_absolute():out=Path(case_path).parent/out
    out=out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("OUTPUT_DIR_NOT_EMPTY: reference results must not be mixed")
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"REFERENCE_CASE.json",case)
    summary={"solver_backend":BACKEND,"experimental":True,"calculation_kind":"STATIC",
             "method":"FDM","theory":prepared.problem.functional.theory,
             "geometry_model":"GREEN_LAGRANGE","material_model":"HOOKE",
             "field_order":"component,x,s","thickness_basis":"(2*z/h)**k",
             "units":{"system":"SI_N_M_PA"},"output_dir":str(out),
             "status":"NOT_STARTED","converged":False,"accepted_steps":0,
             "steps":[],"mesh_convergence_verified":False}
    functional=prepared.problem.functional
    summary.update(scale_model=case["scale_model"],
        gradient_enabled=functional.material.ell_m>0,
        electrical_enabled=functional.electrical_on,
        quadrature=case["quadrature"],
        residual_scaling={"energy_j":float(functional.material.young_pa*functional.h*np.sum(functional.W)),
                          "translation_m":functional.h,"rotation_rad":1.,
                          "potential_v":prepared.settings.potential_scale_v})
    q,p=prepared.mechanical,prepared.electrical
    try:
        progress.configure(out,solver_backend=BACKEND)
        for index,factor in enumerate(prepared.factors,1):
            def report(record):
                progress.emit("reference_newton",step=index,load_factor=factor,**record)
                print(f"REFERENCE_FDM step={index} it={record['iteration']} Rscaled={record['residual_scaled']:.6e}",flush=True)
            result=solve_static_fdm(prepared.problem,q,p,dead_load=factor*prepared.dead_load,
                free_charge=None if prepared.free_charge is None else factor*prepared.free_charge,
                settings=prepared.settings,on_iteration=report)
            state_name=f"REFERENCE_STEP_{index:04d}.npz" if result.converged else "REFERENCE_FAILED_ITERATE.npz"
            temp=out/(state_name+".tmp")
            arrays={"mechanical":result.mechanical,"full_residual":result.full_residual,
                    "constraint_reaction":result.constraint_reaction,
                    "x_m":np.linspace(0,float(case["geometry"]["L"]),q.shape[1]),
                    "s_m":np.arange(q.shape[2])*2*np.pi*prepared.problem.functional.R/q.shape[2]}
            if result.electrical is not None:arrays["electrical"]=result.electrical
            with temp.open("wb") as stream:np.savez_compressed(stream,**arrays)
            os.replace(temp,out/state_name)
            history_name=f"REFERENCE_NEWTON_{index:04d}.json"
            # Singular tangents can have infinite condition number: store
            # that diagnostic as null rather than nonstandard JSON Infinity.
            history=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v)
                      for k,v in row.items()} for row in result.history]
            _write_json(out/history_name,history)
            step={"index":index,"load_factor":factor,"status":result.status,
                  "converged":result.converged,"iterations":result.iterations,
                  "potential_j":result.total_potential_j,
                  "admissible_residual_scaled":result.admissible_residual_scaled,
                  "residual_relative_to_initial":result.residual_relative_to_initial,
                  "constraint_defect_scaled":result.constraint_defect_scaled,
                  "max_mechanical_coefficient_abs":float(np.max(abs(result.mechanical))),
                  "max_potential_coefficient_abs":None if result.electrical is None else float(np.max(abs(result.electrical))),
                  "state_file":state_name,"history_file":history_name}
            radial=2*(int(functional.theory[-1])+1) if functional.theory.startswith("CUF_") else 2
            step["w_middle_abs_max_m"]=float(np.max(abs(result.mechanical[radial])))
            summary["steps"].append(step)
            summary["status"]=result.status
            if result.converged:
                summary["accepted_steps"]+=1
                q,p=result.mechanical,result.electrical
            else:
                summary["failed_step_index"]=index
                summary["failed_step_code"]=result.status
            summary["converged"]=summary["accepted_steps"]==len(prepared.factors)
            if result.converged and not summary["converged"]:summary["status"]="IN_PROGRESS"
            _write_json(out/"FINAL_RUN_SUMMARY.json",summary)
            if not result.converged:break
    finally:progress.reset()
    return summary


__all__=("BACKEND","selected_backend","prepare_reference_case","run_reference_case")
