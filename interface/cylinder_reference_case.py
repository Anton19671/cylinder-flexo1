#!/usr/bin/env python3
"""Explicit small-grid CASE route for coupled variational FDM/MVI solves."""
from dataclasses import dataclass
from pathlib import Path
import copy
import csv
import json
import os
import numpy as np

from cylinder_functional_assembly import (CylinderFunctional,
    surface_trapezoid, surface_sbp21, surface_sbp21_fourier, surface_sbp21_csr,
    surface_sbp42_fourier, surface_sbp42_fourier_csr)
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_variational_fdm import StaticFDMSettings, solve_static_fdm, state_scales
from cylinder_variational_mvi import StaticMVISettings, solve_static_mvi
from cylinder_physical_fields import recover_thickness_fields
from cylinder_face_conditions import face_conditions_from_recovery
from cylinder_variational_dynamic import solve_dynamic_fdm, solve_dynamic_mvi
from cylinder_newmark_mvi import NewmarkSettings, validate_dynamic_case
from cylinder_dynamic_loads import normalize_time_law
from cylinder_variational_loads import project_radial_load
from cylinder_dynamic_trace import (
    MechanicalTraceNewmark, mechanical_trace_matrix,
    SparseMechanicalTraceNewmark, sparse_mechanical_trace_matrix,
    effective_step_acceptance, require_zero_initial_support_motion)
from cylinder_newmark_audit import audit_newmark_step
from cylinder_initial_acceleration import consistent_initial_acceleration
from cylinder_spatial_spectrum_audit import spatial_operator_diagnostics
from cylinder_director_rotation_audit import director_rotation_diagnostic
from cylinder_mvi_essential_trace_guard import require_homogeneous_mvi_traces
from flexo_gradient_g3d import G3DMaterial
from cylinder_thickness_locking_diagnostic import thickness_locking_diagnostic
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
    settings: object
    method: str
    mvi_settings: StaticMVISettings | None
    calculation_kind: str
    dynamics: object | None
    fdm_backend: str = "DENSE_REFERENCE"
    initial_acceleration: np.ndarray | None = None


@dataclass(frozen=True)
class ReferenceDynamics:
    settings: NewmarkSettings
    density_kg_m3: float
    time_law: object
    mass_damping_rate_per_s: float = 0.0
    eta_s: float = 0.0


def prepare_reference_case(case, *, base_dir=None):
    """Validate without creating directories or substituting another backend."""
    if selected_backend(case)!=BACKEND:raise ValueError("explicit reference backend required")
    _postprocess_options(case)
    kind=str(case.get("calculation_kind","")).upper()
    if kind not in ("STATIC","DYNAMIC"):
        raise ValueError("reference backend requires STATIC or DYNAMIC")
    method=str(case.get("method","")).upper()
    if method not in ("FDM","MVI"):
        raise ValueError("reference backend requires FDM or MVI")
    fdm_backend=str(case.get("fdm_backend","DENSE_REFERENCE")).upper()
    if fdm_backend not in ("DENSE_REFERENCE","MATRIX_FREE_GMRES"):
        raise ValueError("fdm_backend must be DENSE_REFERENCE or MATRIX_FREE_GMRES")
    if method!="FDM" and fdm_backend!="DENSE_REFERENCE":
        raise ValueError("MATRIX_FREE_GMRES currently available for FDM only; MVI unchanged")
    matrix_free=(method=="FDM" and fdm_backend=="MATRIX_FREE_GMRES")
    from cylinder_kinematic_models import normalized_model
    geometry_model=normalized_model(case.get("geometry_model",""))
    if str(case.get("material_model","")).upper()!="HOOKE":
        raise ValueError("reference backend requires HOOKE")
    mode=str(case.get("mode","")).upper()
    expected_mode="LINEAR_ELASTIC" if geometry_model=="LINEAR" else "NONLINEAR_ELASTIC"
    if mode!=expected_mode:
        raise ValueError(f"{geometry_model} requires mode={expected_mode}")
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
    other_grid="mvi_grid" if method=="FDM" else "fdm_grid"
    if case.get(other_grid) is not None:
        raise ValueError(other_grid+": grid belongs to the other spatial method")
    grid=case.get("fdm_grid" if method=="FDM" else "mvi_grid") or case.get("grid") or {}
    nx=_integer(grid.get("nx"),"nx",4);ns=_integer(grid.get("ns"),"ns",4)
    if grid.get("stencil") is not None:
        raise ValueError("reference operators are selected with grid.operator_scheme, not stencil")
    operator_scheme=str(grid.get("operator_scheme","LEGACY_REFERENCE")).upper()
    if operator_scheme not in (
            "LEGACY_REFERENCE","SBP21_PERIODIC_ODD","SBP21_PERIODIC_ODD_CSR",
            "SBP21_FOURIER_ODD","SBP42_FOURIER_ODD","SBP42_FOURIER_ODD_CSR"):
        raise ValueError("grid.operator_scheme must be LEGACY_REFERENCE, "
                         "SBP21_PERIODIC_ODD, SBP21_PERIODIC_ODD_CSR or SBP21_FOURIER_ODD")
    if operator_scheme in ("SBP21_FOURIER_ODD","SBP42_FOURIER_ODD") and matrix_free:
        raise ValueError("SBP21_FOURIER_ODD is currently dense only; use DENSE_REFERENCE")
    if operator_scheme in ("SBP21_PERIODIC_ODD_CSR","SBP42_FOURIER_ODD_CSR") and not matrix_free:
        raise ValueError("SBP21_PERIODIC_ODD_CSR requires FDM MATRIX_FREE_GMRES")
    if matrix_free:
        from cylinder_matrixfree_fdm import MatrixFreeNewtonSettings
        early=case.get("reference_newton",{})
        if not isinstance(early,dict):
            raise ValueError("reference_newton must be an object")
        max_nodes=_integer(early.get("max_surface_nodes",
            MatrixFreeNewtonSettings.max_surface_nodes),
            "reference_newton.max_surface_nodes",1)
        max_dofs=_integer(early.get("max_total_dofs",
            MatrixFreeNewtonSettings.max_total_dofs),
            "reference_newton.max_total_dofs",1)
        count=(3 if theory=="KL" else 5 if theory in ("FSDT","REDDY")
               else 3*(int(theory[-1])+1) if theory.startswith("CUF_TE")
                  and theory[-1] in "1234" else None)
        if count is None:
            raise ValueError("unsupported variational shell theory")
        potential_fields=order+1 if material.kappa_f_per_m>0 else 0
        if nx*ns*(count+potential_fields)>max_dofs:
            raise ValueError("MATRIX_FREE_NODAL_DOF_SAFETY_LIMIT")
    else:
        max_nodes=256
    if nx*ns>max_nodes:
        raise ValueError(f"{fdm_backend}: surface node count exceeds {max_nodes}")
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
    surface=(surface_sbp42_fourier_csr(nx,ns,length,radius)
             if operator_scheme=="SBP42_FOURIER_ODD_CSR"
             else surface_sbp42_fourier(nx,ns,length,radius)
             if operator_scheme=="SBP42_FOURIER_ODD"
             else surface_sbp21_csr(nx,ns,length,radius)
             if operator_scheme=="SBP21_PERIODIC_ODD_CSR"
             else surface_sbp21(nx,ns,length,radius)
             if operator_scheme=="SBP21_PERIODIC_ODD"
             else surface_sbp21_fourier(nx,ns,length,radius)
             if operator_scheme=="SBP21_FOURIER_ODD"
             else surface_trapezoid(nx,ns,length,radius))
    functional=CylinderFunctional(theory,material,radius_m=radius,thickness_m=thickness,
        surface_operators=surface,n_gauss=n_gauss,geometry_model=geometry_model)
    q=np.zeros((functional.n_mechanical,nx,ns))
    p=np.zeros((order+1,nx,ns)) if functional.electrical_on else None
    total_dofs=q.size+(0 if p is None else p.size)
    if not matrix_free and total_dofs>4096:
        raise ValueError("dense reference nodal DOF limit exceeded")
    bc=case.get("boundary_conditions") or {}
    spec=bc.get("variational")
    if not isinstance(spec,dict):raise ValueError("explicit boundary_conditions.variational required")
    spec=copy.deepcopy(spec)  # resolve CASE-local CSV paths without editing caller input
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
                "newton","linear","inner_mvi","fdm_operator_scheme"):
        if case.get(key) is not None:
            raise ValueError(key+": legacy fields are not accepted by reference backend")
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
        projection=project_radial_load(functional,definition)
        dead=projection.generalized
        charge=None;factors=definition.get("load_factors")
    if kind=="STATIC":
        if not isinstance(factors,list) or not factors:raise ValueError("explicit nonempty load_factors required")
        factors=tuple(_real(x,"load factor") for x in factors)
        if len(factors)>100:raise ValueError("reference static series limited to 100 load factors")
    else:
        if factors not in (None,[]):raise ValueError("DYNAMIC uses dynamics.time_law, not static load_factors")
        factors=()
    if matrix_free:
        from cylinder_matrixfree_fdm import MatrixFreeNewtonSettings

    options=case.get("reference_newton",{})
    if not isinstance(options,dict):raise ValueError("reference_newton must be an object")
    try:
        settings=(MatrixFreeNewtonSettings(**options) if fdm_backend=="MATRIX_FREE_GMRES"
                  else StaticFDMSettings(**options))
    except TypeError as exc:raise ValueError("invalid reference_newton fields") from exc
    if matrix_free and total_dofs>settings.max_total_dofs:
        raise ValueError("MATRIX_FREE_NODAL_DOF_SAFETY_LIMIT")
    mvi_settings=None
    if method=="MVI":
        mvi_options=case.get("reference_mvi",{})
        if not isinstance(mvi_options,dict) or set(mvi_options)-{"max_sweeps"}:
            raise ValueError("reference_mvi accepts only max_sweeps; rank belongs at CASE root")
        rank=_integer(case.get("rank"),"MVI rank",1)
        mvi_settings=StaticMVISettings(rank=rank,**mvi_options)
        if rank>min(nx,ns):raise ValueError("MVI rank exceeds grid dimensions; increase nx and ns")
    elif case.get("reference_mvi") is not None or case.get("rank") is not None:
        raise ValueError("MVI settings require method=MVI")
    dynamic=None
    if kind=="DYNAMIC":
        contract=validate_dynamic_case(case)
        if contract["damping"]=="EXTERNAL_OPERATOR":
            raise ValueError("reference backend does not accept an external damping operator")
        damping_rate=contract["mass_damping_rate_per_s"]
        material_eta_s=0.0
        if scale=="MICRO":
            material_density=_real(case["micro"]["rho_kg_m3"],"micro.rho_kg_m3",True)
            if not np.isclose(contract["density_kg_m3"],material_density,rtol=1.e-12,atol=0.):
                raise ValueError("dynamics.density_kg_m3 must equal micro.rho_kg_m3")
            material_eta_s=_real(case["micro"]["eta_s"],"micro.eta_s")
            if material_eta_s<0:raise ValueError("micro.eta_s must be nonnegative")
        d=case["dynamics"]
        initial=d.get("initial_conditions",{"type":"ZERO"})
        if not isinstance(initial,dict) or set(initial)!={"type"} or str(initial["type"]).upper()!="ZERO":
            raise ValueError("reference dynamics currently requires initial_conditions.type=ZERO")
        law=normalize_time_law(d.get("time_law"),base_dir=base_dir)
        law.validate_time_grid(contract["start_time_s"],contract["time_step_s"],contract["n_steps"])
        settings_dynamic=NewmarkSettings(contract["time_step_s"],contract["n_steps"],
            beta=contract["newmark_beta"],gamma=contract["newmark_gamma"],
            t_start_s=contract["start_time_s"],
            residual_tolerance_relative=contract["residual_tolerance_relative"])
        dynamic=ReferenceDynamics(settings_dynamic,contract["density_kg_m3"],law,damping_rate,material_eta_s)
        # All end loads, electrode potentials and free charges share a physical
        # time grid with Newmark. Validate their sampling and table coverage
        # before producing any output, not after a step has been accepted.
        for family in ("mechanical","electrical"):
            for site,data in spec.get(family,{}).items():
                if not isinstance(data,dict) or "time_law" not in data:
                    continue
                raw_law=data["time_law"]
                boundary_law=normalize_time_law(raw_law,base_dir=base_dir)
                boundary_law.validate_time_grid(
                    contract["start_time_s"],contract["time_step_s"],
                    contract["n_steps"])
                # The boundary assembler parses time laws each step. It has no
                # CASE directory argument, so bind relative CSV paths explicitly.
                if "csv_path" in raw_law:
                    csv_path=Path(str(raw_law["csv_path"])).expanduser()
                    if not csv_path.is_absolute():
                        csv_path=(Path.cwd() if base_dir is None else
                                  Path(base_dir))/csv_path
                    data["time_law"]["csv_path"]=str(csv_path.resolve())
                if family=="mechanical":
                    support=str(data.get("support","FREE")).upper()
                    # CONSTANT displacement may be nonzero, but its prescribed
                    # velocity and acceleration vanish. Moving constraints
                    # require a constrained Newmark DAE, not a mere time factor.
                    if support!="FREE" and boundary_law.kind not in ("CONSTANT","SMOOTHSTEP"):
                        raise ValueError(
                            f"{site}: time-dependent essential mechanical BC "
                            "needs C2 SMOOTHSTEP displacement law")
                    if support!="FREE":
                        from cylinder_variational_boundary import _fixed_components, _vector
                        selected=_fixed_components(support,theory,functional.n_mechanical,data)
                        prescribed=_vector(data.get("prescribed"),functional.n_mechanical,
                                           site+" prescribed")
                        if selected:
                            require_zero_initial_support_motion(
                                boundary_law,prescribed[list(selected)],contract["start_time_s"])
                # Electrical POTENTIAL is quasi-static at every time step:
                # its time-dependent Dirichlet value has no electric inertia
                # and is imposed by the same affine admissible trace space.
    # Compile charge compatibility and all trace equations before declaring a
    # CASE valid; dry-run validation must not generate any output files.
    # Create the problem only after CASE-local time tables were resolved above.
    problem=BoundarySpatialProblem(functional,bc_spec=spec,calculation_kind=kind)
    if method=="MVI":
        require_homogeneous_mvi_traces(spec,theory=theory,
                                       n_mechanical=functional.n_mechanical)
    if matrix_free:
        from cylinder_matrixfree_fdm import SparseTraceProjector
        scales=state_scales(functional,q,p,settings)
        time0=None if kind=="STATIC" else dynamic.settings.t_start_s
        projector=SparseTraceProjector.from_problem(
            problem,q,p,scales,time_s=time0,
            max_rows=settings.max_constraint_rows)
        if kind=="STATIC":
            # Full variational residual also validates charge conservation,
            # spring work, loads and electrodes, but requires no SVD.
            problem.evaluate_effective_residual(
                q,p,dead_load=dead,free_charge=charge)
        else:
            factor0=dynamic.time_law.value(time0)
            loaded=factor0*dead
            loaded_charge=None if charge is None else factor0*charge
            # Project prescribed mechanical/electric traces without building
            # a full nodal identity, dense constraint matrix or SVD basis.
            initial=q.ravel() if p is None else np.concatenate(
                (q.ravel(),p.ravel()))
            u=scales*projector.project_affine(initial/scales)
            q=u[:q.size].reshape(q.shape)
            p=None if p is None else u[q.size:].reshape(p.shape)
            _,initial_force,initial_edge=problem.evaluate_effective_residual(
                q,p,dead_load=loaded,free_charge=loaded_charge,time_s=time0,
                acceleration=np.zeros_like(q),
                density_kg_m3=dynamic.density_kg_m3)
            initial_assembly=None
    else:
        # Original dense reference route remains bit-for-bit unchanged.
        if kind=="STATIC":
            a=problem.assemble(q,p,dead_load=dead,free_charge=charge)
        else:
            t0=dynamic.settings.t_start_s;factor0=dynamic.time_law.value(t0)
            a=problem.assemble(q,p,dead_load=factor0*dead,
                free_charge=None if charge is None else factor0*charge,time_s=t0,
                acceleration=np.zeros_like(q),density_kg_m3=dynamic.density_kg_m3)
            state=a.space.origin
            q=state[:q.size].reshape(q.shape)
            p=None if p is None else state[q.size:].reshape(p.shape)
            a=problem.assemble(q,p,dead_load=factor0*dead,
                free_charge=None if charge is None else factor0*charge,time_s=t0,
                acceleration=np.zeros_like(q),density_kg_m3=dynamic.density_kg_m3)
            initial_force=a.full_residual
            initial_edge=a.edge
            initial_assembly=a
        # The full admissible grid space certifies an MVI result, but its
        # dimension is NOT the number of unknowns solved in an X/Y half-step.
        # The dense FDM cap applies only to the full-nodal FDM Newton solve.
        # Every frozen-partner MVI subspace independently enforces this cap.
        if method == "FDM" and a.space.basis.shape[1]>settings.max_free_dofs:
            raise ValueError("reference free-DOF limit exceeded; use a smaller verification grid")
    initial_acceleration=None
    if kind=="DYNAMIC":
        initial_acceleration=consistent_initial_acceleration(
            problem,q,p,initial_residual=initial_force,
            initial_edge=initial_edge,settings=settings,
            density_kg_m3=dynamic.density_kg_m3,
            backend=fdm_backend,full_assembly=initial_assembly,
            sparse_projector=projector if matrix_free else None)
    if kind=="DYNAMIC" and method=="MVI":
        # The genuine separated Newmark solution has NO additive nodal lift.
        # Projecting a prescribed initial electrode/support field into the
        # affine boundary space can raise its surface rank. Detect this before
        # creating an output directory or accepting a temporal history.
        from cylinder_variational_mvi import _initial_profiles, reconstruct_profiles
        scales0=state_scales(functional,q,p,settings)
        field0=q if p is None else np.concatenate((q,p))
        normalized=(field0.ravel()/scales0).reshape(field0.shape)
        try:
            xx,yy=_initial_profiles(
                normalized,mvi_settings.rank,
                np.linalg.norm(normalized,axis=(1,2)))
        except ValueError as exc:
            raise ValueError(
                "DYNAMIC_MVI_INITIAL_STATE_EXCEEDS_FIXED_RANK: "
                "prescribed initial mechanical/electrical traces cannot be "
                "represented without an additive field or a rank change"
            ) from exc
        reconstructed=reconstruct_profiles(xx,yy)
        if not np.allclose(reconstructed,normalized,rtol=1.e-10,atol=1.e-12):
            raise ValueError(
                "DYNAMIC_MVI_INITIAL_STATE_EXCEEDS_FIXED_RANK: "
                "initial field must have the specified separation rank")
    if not isinstance(case.get("output_dir"),str) or not case["output_dir"].strip():
        raise ValueError("explicit output_dir required")
    return ReferenceCase(problem,q,p,dead,charge,factors,settings,method,mvi_settings,kind,dynamic,fdm_backend,initial_acceleration)


def _write_json(path,data):
    temp=path.with_name(path.name+".tmp")
    with temp.open("w",encoding="utf-8") as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def _write_dynamic_history(path,steps):
    temp=path.with_name(path.name+".tmp")
    columns=("index","time_s","load_factor","status","converged","w_middle_abs_max_m",
             "velocity_abs_max_m_per_s","acceleration_abs_max_m_per_s2",
             "max_potential_coefficient_abs","admissible_residual_scaled")
    with temp.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for step in steps:writer.writerow({key:step.get(key) for key in columns})
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def _postprocess_options(case):
    data=case.get("postprocess",{})
    if not isinstance(data,dict) or set(data)-{
            "thickness_stations","electric_induction_scale_c_per_m2",
            "dynamic_analytics"}:
        raise ValueError(
            "postprocess permits thickness_stations, "
            "electric_induction_scale_c_per_m2 and dynamic_analytics")
    n=_integer(data.get("thickness_stations",11),
               "postprocess.thickness_stations",3)
    if n>201:
        raise ValueError("postprocess.thickness_stations must be <=201")
    scale=data.get("electric_induction_scale_c_per_m2")
    if scale is not None:
        scale=_real(scale,"postprocess.electric_induction_scale_c_per_m2",True)
    raw=data.get("dynamic_analytics",{})
    if isinstance(raw,bool):
        raw={"enabled":raw}
    if not isinstance(raw,dict) or set(raw)-{
            "enabled","transient_fraction","forcing_hz"}:
        raise ValueError(
            "postprocess.dynamic_analytics must be a boolean or an object "
            "with enabled, transient_fraction and forcing_hz")
    default_enabled=str(case.get("calculation_kind","")).upper()=="DYNAMIC"
    enabled=raw.get("enabled",default_enabled)
    if not isinstance(enabled,bool):
        raise ValueError("postprocess.dynamic_analytics.enabled must be boolean")
    fraction=_real(raw.get("transient_fraction",.2),
                   "postprocess.dynamic_analytics.transient_fraction")
    if not 0<=fraction<.9:
        raise ValueError(
            "postprocess.dynamic_analytics.transient_fraction must be in [0,0.9)")
    forcing=raw.get("forcing_hz")
    if forcing is not None:
        forcing=_real(forcing,"postprocess.dynamic_analytics.forcing_hz",True)
    return {"thickness_stations":n,
            "electric_induction_scale_c_per_m2":scale,
            "dynamic_analytics":{"enabled":enabled,
                                  "transient_fraction":fraction,
                                  "forcing_hz":forcing}}

def run_reference_case(case,case_path):
    prepared=prepare_reference_case(case,base_dir=Path(case_path).parent)
    if prepared.calculation_kind=="DYNAMIC":
        return run_reference_dynamic_case(case,case_path,prepared=prepared)
    out=Path(case["output_dir"])
    if not out.is_absolute():out=Path(case_path).parent/out
    out=out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("OUTPUT_DIR_NOT_EMPTY: reference results must not be mixed")
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"REFERENCE_CASE.json",case)
    summary={"solver_backend":BACKEND,"experimental":True,"calculation_kind":"STATIC",
             "method":prepared.method,"theory":prepared.problem.functional.theory,
             "geometry_model":case["geometry_model"],"material_model":"HOOKE",
             "field_order":"component,x,s","thickness_basis":"(2*z/h)**k",
             "electric_charge_convention":"PLUS_FREE_CHARGE_IN_ENTHALPY_D_N_EQUALS_MINUS_SIGMA",
             "units":{"system":"SI_N_M_PA"},"output_dir":str(out),
             "status":"NOT_STARTED","converged":False,"accepted_steps":0,
             "steps":[],"mesh_convergence_verified":False}
    functional=prepared.problem.functional
    postprocess=_postprocess_options(case)
    summary['thickness_locking_diagnostic']=thickness_locking_diagnostic(functional.theory,functional.material.poisson)
    summary['director_kinematics_diagnostic']=director_rotation_diagnostic(functional.theory)
    grid=case.get('mvi_grid' if prepared.method=='MVI' else 'fdm_grid') or case.get('grid') or {}
    summary['spatial_operator_diagnostics']=spatial_operator_diagnostics(functional.ops,grid.get('operator_scheme','LEGACY_REFERENCE'))
    summary.update(scale_model=case["scale_model"],
        gradient_enabled=functional.material.ell_m>0,
        gradient_model=functional.material.gradient_model_label,
        active_strain_gradient_components=functional.material.active_gradient_components,
        electrical_enabled=functional.electrical_on,
        quadrature=case["quadrature"],
        residual_scaling={"energy_j":float(functional.material.young_pa*functional.h*np.sum(functional.W)),
                          "translation_m":functional.h,"rotation_rad":1.,
                          "potential_v":prepared.settings.potential_scale_v})
    if prepared.mvi_settings is not None:
        summary.update(rank=prepared.mvi_settings.rank,
            convergence_meaning="full admissible nodal residual; projected residual reported separately")
    q,p=prepared.mechanical,prepared.electrical
    previous_mvi_result=None
    try:
        progress.configure(out,solver_backend=BACKEND)
        for index,factor in enumerate(prepared.factors,1):
            def report(record):
                progress.emit("reference_"+record.get("event","newton"),step=index,load_factor=factor,**{k:v for k,v in record.items() if k!="event"})
                print(f"REFERENCE_{prepared.method} step={index} sweep={record.get('sweep',0)} "
                      f"{record.get('direction','FULL')} it={record['iteration']} "
                      f"Rscaled={record['residual_scaled']:.6e}",flush=True)
            if prepared.fdm_backend=="MATRIX_FREE_GMRES":
                from cylinder_matrixfree_fdm import solve_matrixfree_static_fdm
            solve=(solve_matrixfree_static_fdm if prepared.fdm_backend=="MATRIX_FREE_GMRES"
                   else solve_static_fdm if prepared.method=="FDM" else solve_static_mvi)
            extra=({} if prepared.method=="FDM" else
                   {"mvi_settings":prepared.mvi_settings,
                    "previous_mvi_result":previous_mvi_result})
            result=solve(prepared.problem,q,p,dead_load=factor*prepared.dead_load,
                free_charge=None if prepared.free_charge is None else factor*prepared.free_charge,
                settings=prepared.settings,on_iteration=report,**extra)
            state_name=f"REFERENCE_STEP_{index:04d}.npz" if result.converged else "REFERENCE_FAILED_ITERATE.npz"
            temp=out/(state_name+".tmp")
            arrays={"mechanical":result.mechanical,"full_residual":result.full_residual,
                    "constraint_reaction":result.constraint_reaction,
                    "x_m":np.linspace(0,float(case["geometry"]["L"]),q.shape[1]),
                    "s_m":np.arange(q.shape[2])*2*np.pi*prepared.problem.functional.R/q.shape[2]}
            if result.electrical is not None:arrays["electrical"]=result.electrical
            if prepared.method=="MVI":
                arrays.update(x_profiles=result.x_profiles,y_profiles=result.y_profiles,
                              state_scales=result.scales)
            # Recover physical 3D fields for accepted or failed iterate,
            # preserving the same actual nonlinear state of MVI/FDM.
            arrays.update(recover_thickness_fields(
                functional,result.mechanical,result.electrical,
                n_thickness=postprocess["thickness_stations"]))
            face_arrays,face_summary=face_conditions_from_recovery(
                functional,arrays,prepared.problem.spec,
                calculation_kind="STATIC",
                electric_induction_scale_c_per_m2=postprocess[
                    "electric_induction_scale_c_per_m2"])
            arrays.update(face_arrays)
            with temp.open("wb") as stream:np.savez_compressed(stream,**arrays)
            os.replace(temp,out/state_name)
            history_name=f"REFERENCE_{'MVI' if prepared.method=='MVI' else 'NEWTON'}_{index:04d}.json"
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
                  "state_file":state_name,"history_file":history_name,
                  "face_conditions":face_summary,
                  "director_kinematics_diagnostic":director_rotation_diagnostic(
                      functional.theory,result.mechanical),
                  "radial_face_displacement_inner_abs_max_m":float(
                      np.max(np.abs(arrays["face_inner_radial_displacement_m"]))),
                  "radial_face_displacement_outer_abs_max_m":float(
                      np.max(np.abs(arrays["face_outer_radial_displacement_m"])))}
            if prepared.method=="MVI":
                step.update(sweeps=result.sweeps,projected_residual_scaled=result.projected_residual_scaled,
                            projected_equilibrium_satisfied=result.projected_equilibrium_satisfied,
                            full_equilibrium_satisfied=bool(result.mvi_certificate["full_equilibrium_satisfied"]),
                            mvi_certificate=result.mvi_certificate,
                            mvi_initialization=("CONTINUATION" if previous_mvi_result is not None
                                                else "COLD_START"))
            radial=2*(int(functional.theory[-1])+1) if functional.theory.startswith("CUF_") else 2
            step["w_middle_abs_max_m"]=float(np.max(abs(result.mechanical[radial])))
            summary["steps"].append(step)
            summary["status"]=result.status
            if result.converged:
                summary["accepted_steps"]+=1
                q,p=result.mechanical,result.electrical
                if prepared.method=="MVI":
                    previous_mvi_result=result
            else:
                summary["failed_step_index"]=index
                summary["failed_step_code"]=result.status
            summary["converged"]=summary["accepted_steps"]==len(prepared.factors)
            if result.converged and not summary["converged"]:summary["status"]="IN_PROGRESS"
            _write_json(out/"FINAL_RUN_SUMMARY.json",summary)
            if not result.converged:break
    finally:progress.reset()
    return summary


def run_reference_dynamic_case(case,case_path,*,prepared=None):
    """Advance the coupled mechanical/electric fields by Newmark in time."""
    prepared=prepare_reference_case(case,base_dir=Path(case_path).parent) if prepared is None else prepared
    if prepared.calculation_kind!="DYNAMIC":raise ValueError("dynamic reference CASE required")
    out=Path(case["output_dir"])
    if not out.is_absolute():out=Path(case_path).parent/out
    out=out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("OUTPUT_DIR_NOT_EMPTY: reference results must not be mixed")
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"REFERENCE_CASE.json",case)
    dyn=prepared.dynamics;newmark=dyn.settings;f=prepared.problem.functional
    postprocess=_postprocess_options(case)
    summary={"solver_backend":BACKEND,"experimental":True,"calculation_kind":"DYNAMIC",
        "method":prepared.method,"theory":f.theory,"geometry_model":case["geometry_model"],
        "material_model":"HOOKE","field_order":"component,x,s",
        "thickness_basis":"(2*z/h)**k",
        "electric_charge_convention":"PLUS_FREE_CHARGE_IN_ENTHALPY_D_N_EQUALS_MINUS_SIGMA",
        "units":{"system":"SI_N_M_PA"},
        "output_dir":str(out),"status":"NOT_STARTED","converged":False,
        "accepted_steps":0,"requested_steps":newmark.n_steps,"steps":[],
        "mesh_convergence_verified":False,"time_step_convergence_verified":False,
        "scale_model":case["scale_model"],"gradient_enabled":f.material.ell_m>0,
        "gradient_model":f.material.gradient_model_label,
        "active_strain_gradient_components":f.material.active_gradient_components,
        "electrical_enabled":f.electrical_on,"density_kg_m3":dyn.density_kg_m3,
        "newmark":{"beta":newmark.beta,"gamma":newmark.gamma,
                   "dt_s":newmark.dt_s,"start_time_s":newmark.t_start_s},
        "electric_inertia":False,
        "material_viscosity":{"kind":"KELVIN_VOIGT_GREEN_LAGRANGE","eta_s":dyn.eta_s},
        "damping":{"kind":"MASS_PROPORTIONAL" if dyn.mass_damping_rate_per_s else "NONE",
                   "mass_damping_rate_per_s":dyn.mass_damping_rate_per_s},
        "convergence_meaning":"full admissible effective Newmark residual"}
    summary['thickness_locking_diagnostic']=thickness_locking_diagnostic(f.theory,f.material.poisson)
    summary['director_kinematics_diagnostic']=director_rotation_diagnostic(f.theory)
    grid=case.get('mvi_grid' if prepared.method=='MVI' else 'fdm_grid') or case.get('grid') or {}
    summary['spatial_operator_diagnostics']=spatial_operator_diagnostics(f.ops,grid.get('operator_scheme','LEGACY_REFERENCE'))
    summary["fdm_backend"]=prepared.fdm_backend
    if prepared.mvi_settings is not None:
        from cylinder_dynamic_case_passport import dynamic_physics_sha256
        summary["rank"]=prepared.mvi_settings.rank
        summary["physical_case_sha256"]=dynamic_physics_sha256(case)
    q,p=prepared.mechanical,prepared.electrical
    previous_mvi_result=None
    velocity=np.zeros_like(q)
    if prepared.initial_acceleration is None:
        raise ValueError("DYNAMIC_CONSISTENT_INITIAL_ACCELERATION_MISSING")
    acceleration=np.asarray(prepared.initial_acceleration,float).copy()
    summary["initial_acceleration"]={
        "method":"CONSISTENT_CONSTRAINED_MASS",
        "max_abs_m_per_s2":float(np.max(np.abs(acceleration),initial=0)),
        "initial_velocity_type":"ZERO"}
    if prepared.fdm_backend=="MATRIX_FREE_GMRES":
        from cylinder_variational_boundary import compile_boundary_program
        initial_edge=compile_boundary_program(
            f,q,p,spec=prepared.problem.spec,calculation_kind="DYNAMIC",
            time_s=newmark.t_start_s)
    else:
        initial_edge=prepared.problem.assemble(
            q,p,time_s=newmark.t_start_s,
            acceleration=np.zeros_like(q),
            density_kg_m3=dyn.density_kg_m3).edge
    sparse_trace=prepared.fdm_backend=="MATRIX_FREE_GMRES"
    trace_matrix=(sparse_mechanical_trace_matrix if sparse_trace
                  else mechanical_trace_matrix)
    monitor_class=(SparseMechanicalTraceNewmark if sparse_trace
                   else MechanicalTraceNewmark)
    trace_c,trace_d=trace_matrix(initial_edge.constraints,q.size)
    trace_monitor=monitor_class(trace_c,trace_d,q.ravel(),velocity.ravel(),acceleration.ravel(),
        t=newmark.t_start_s,dt=newmark.dt_s,beta=newmark.beta,gamma=newmark.gamma)
    initial=out/"REFERENCE_DYNAMIC_INITIAL.npz"
    arrays={"time_s":np.array(newmark.t_start_s),"mechanical":q,
            "velocity":velocity,"acceleration":acceleration}
    if p is not None:arrays["electrical"]=p
    with initial.open("wb") as stream:np.savez_compressed(stream,**arrays)
    try:
        progress.configure(out,solver_backend=BACKEND)
        for index in range(1,newmark.n_steps+1):
            time_s=newmark.t_start_s+index*newmark.dt_s
            factor=dyn.time_law.value(time_s)
            if prepared.fdm_backend=="MATRIX_FREE_GMRES":
                trace_edge=compile_boundary_program(
                    f,q,p,spec=prepared.problem.spec,
                    calculation_kind="DYNAMIC",time_s=time_s)
            else:
                trace_edge=prepared.problem.assemble(
                    q,p,time_s=time_s,acceleration=np.zeros_like(q),
                    density_kg_m3=dyn.density_kg_m3).edge
            trace_new_c,trace_new_d=trace_matrix(
                trace_edge.constraints,q.size)
            expected_trace=trace_monitor.preview(trace_new_c,trace_new_d,time_s)
            q_predictor=q+newmark.dt_s*velocity+newmark.dt_s**2*(.5-newmark.beta)*acceleration
            velocity_predictor=velocity+newmark.dt_s*(1-newmark.gamma)*acceleration
            def report(record):
                clean={k:v for k,v in record.items() if k!="event"}
                progress.emit("reference_dynamic_"+record.get("event","newton"),
                    step=index,time_s=time_s,load_factor=factor,**clean)
                print(f"REFERENCE_DYNAMIC_{prepared.method} step={index} t={time_s:.9g} "
                      f"sweep={record.get('sweep',0)} {record.get('direction','FULL')} "
                      f"it={record['iteration']} Rscaled={record['residual_scaled']:.6e}",flush=True)
            if prepared.fdm_backend=="MATRIX_FREE_GMRES":
                from cylinder_matrixfree_fdm import solve_matrixfree_dynamic_fdm
            solve=(solve_matrixfree_dynamic_fdm if prepared.fdm_backend=="MATRIX_FREE_GMRES"
                   else solve_dynamic_fdm if prepared.method=="FDM" else solve_dynamic_mvi)
            extra=({} if prepared.method=="FDM" else
                   {"mvi_settings":prepared.mvi_settings,
                    "previous_mvi_result":previous_mvi_result})
            result=solve(prepared.problem,q,p,q_predictor=q_predictor,
                velocity_predictor=velocity_predictor,time_s=time_s,dt_s=newmark.dt_s,
                beta=newmark.beta,gamma=newmark.gamma,density_kg_m3=dyn.density_kg_m3,
                mass_damping_rate_per_s=dyn.mass_damping_rate_per_s,
                eta_s=dyn.eta_s,
                dead_load=factor*prepared.dead_load,
                free_charge=None if prepared.free_charge is None else factor*prepared.free_charge,
                settings=prepared.settings,on_iteration=report,**extra)
            # Reject a spatial or full-Newmark-residual failure BEFORE writing
            # the accepted-state name, updating history, or changing q/v/a.
            accepted,gate_status=effective_step_acceptance(
                bool(result.converged),result.residual_relative_to_initial,
                newmark.residual_tolerance_relative,
                absolute_residual=result.admissible_residual_scaled,
                absolute_tolerance=prepared.settings.absolute_tolerance_scaled)
            trace_defects=None
            newmark_kinematic_defects=None
            if accepted:
                try:
                    newmark_kinematic_defects=audit_newmark_step(
                        q,velocity,acceleration,
                        result.mechanical,result.velocity,result.acceleration,
                        dt_s=newmark.dt_s,beta=newmark.beta,gamma=newmark.gamma).as_dict()
                    trace_defects=trace_monitor.accept(expected_trace,
                        result.mechanical.ravel(),result.velocity.ravel(),result.acceleration.ravel())
                except ValueError as exc:
                    accepted=False
                    gate_status=("NEWMARK_KINEMATICS_DRIFT"
                                 if "NEWMARK_KINEMATICS_DRIFT" in str(exc)
                                 else "MOVING_MECHANICAL_BC_INCONSISTENT")
                    trace_failure=str(exc)
            state_name=(f"REFERENCE_DYNAMIC_STEP_{index:04d}.npz" if accepted
                        else "REFERENCE_DYNAMIC_FAILED_ITERATE.npz")
            temp=out/(state_name+".tmp")
            arrays={"time_s":np.array(time_s),"load_factor":np.array(factor),
                "mechanical":result.mechanical,"velocity":result.velocity,
                "acceleration":result.acceleration,
                "damping_force":(
                    dyn.mass_damping_rate_per_s*
                    f.mass_action(result.velocity,density_kg_m3=dyn.density_kg_m3,
                                  mechanical=result.mechanical)),
                "full_residual":result.full_residual,
                "constraint_reaction":result.constraint_reaction,
                "x_m":np.linspace(0,float(case["geometry"]["L"]),q.shape[1]),
                "s_m":np.arange(q.shape[2])*2*np.pi*f.R/q.shape[2]}
            if dyn.eta_s:
                visc=f.kelvin_voigt_action(result.mechanical,result.velocity,eta_s=dyn.eta_s)
                arrays["material_viscous_force"]=visc.mechanical_force
                arrays["material_dissipation_power_w"]=np.array(visc.power_w)
                arrays["damping_force"]+=visc.mechanical_force
            if result.electrical is not None:arrays["electrical"]=result.electrical
            if prepared.method=="MVI":arrays.update(x_profiles=result.x_profiles,
                y_profiles=result.y_profiles,
                state_scales=result.scales)
            # Recover physical 3D fields for accepted or failed iterate,
            # preserving the same actual nonlinear state of MVI/FDM.
            arrays.update(recover_thickness_fields(
                f,result.mechanical,result.electrical,
                n_thickness=postprocess["thickness_stations"]))
            face_arrays,face_summary=face_conditions_from_recovery(
                f,arrays,prepared.problem.spec,
                calculation_kind="DYNAMIC",time_s=time_s,
                electric_induction_scale_c_per_m2=postprocess[
                    "electric_induction_scale_c_per_m2"])
            arrays.update(face_arrays)
            with temp.open("wb") as stream:np.savez_compressed(stream,**arrays)
            os.replace(temp,out/state_name)
            history_name=f"REFERENCE_DYNAMIC_{prepared.method}_{index:04d}.json"
            history=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v)
                      for k,v in row.items()} for row in result.history]
            _write_json(out/history_name,history)
            radial=2*(int(f.theory[-1])+1) if f.theory.startswith("CUF_") else 2
            step={"index":index,"time_s":time_s,"load_factor":factor,
                "status":result.status if accepted else gate_status,
                "converged":accepted,
                "iterations":result.iterations,"sweeps":result.sweeps,
                "admissible_residual_scaled":result.admissible_residual_scaled,
                "residual_relative_to_initial":result.residual_relative_to_initial,
                "constraint_defect_scaled":result.constraint_defect_scaled,
                "w_middle_abs_max_m":float(np.max(abs(result.mechanical[radial]))),
                "velocity_abs_max_m_per_s":float(np.max(abs(result.velocity))),
                "acceleration_abs_max_m_per_s2":float(np.max(abs(result.acceleration))),
                "max_potential_coefficient_abs":None if result.electrical is None else float(np.max(abs(result.electrical))),
                "state_file":state_name,"history_file":history_name,
                "full_equilibrium_satisfied":(bool(result.mvi_certificate["full_equilibrium_satisfied"])
                                               if prepared.method=="MVI" and result.mvi_certificate is not None
                                               else bool(accepted)),
                "full_newmark_residual_tolerance":newmark.residual_tolerance_relative,
                "face_conditions":face_summary,
                "director_kinematics_diagnostic":director_rotation_diagnostic(
                    f.theory,result.mechanical),
                "radial_face_displacement_inner_abs_max_m":float(
                    np.max(np.abs(arrays["face_inner_radial_displacement_m"]))),
                "radial_face_displacement_outer_abs_max_m":float(
                    np.max(np.abs(arrays["face_outer_radial_displacement_m"])))}
            if prepared.method=="MVI":step.update(
                projected_residual_scaled=result.projected_residual_scaled,
                projected_equilibrium_satisfied=result.projected_equilibrium_satisfied,
                mvi_certificate=result.mvi_certificate,
                mvi_initialization=("CONTINUATION" if previous_mvi_result is not None
                                    else "COLD_START"))
            if trace_defects is not None:
                step["mechanical_trace_defects"]=trace_defects
            if newmark_kinematic_defects is not None:
                step["newmark_kinematic_defects"]=newmark_kinematic_defects
            if not accepted and gate_status in ("MOVING_MECHANICAL_BC_INCONSISTENT",
                                                 "NEWMARK_KINEMATICS_DRIFT"):
                step["kinematic_or_boundary_error"]=trace_failure
            summary["steps"].append(step)
            summary["status"]=step["status"]
            _write_dynamic_history(out/"REFERENCE_DYNAMIC_HISTORY.csv",summary["steps"])
            if not accepted:
                summary["failed_step_index"]=index
                summary["failed_step_code"]=gate_status
                _write_json(out/"FINAL_RUN_SUMMARY.json",summary)
                break
            q,p=result.mechanical,result.electrical
            velocity,acceleration=result.velocity,result.acceleration
            if prepared.method=="MVI":
                previous_mvi_result=result
            summary["accepted_steps"]+=1
            summary["converged"]=summary["accepted_steps"]==newmark.n_steps
            summary["status"]="CONVERGED" if summary["converged"] else "IN_PROGRESS"
            _write_json(out/"FINAL_RUN_SUMMARY.json",summary)
    finally:progress.reset()
    # Export after the accepted Newmark states and final summary have been
    # persisted. This is strictly postprocessing: never change solver success.
    analytics=postprocess["dynamic_analytics"]
    if not analytics["enabled"]:
        summary["dynamic_analytics"]={"status":"DISABLED"}
    elif summary["accepted_steps"]<1:
        summary["dynamic_analytics"]={"status":"NO_ACCEPTED_STEPS"}
    else:
        try:
            from cylinder_dynamic_analytics import export_dynamic_analytics
            exported=export_dynamic_analytics(
                out,transient_fraction=analytics["transient_fraction"],
                forcing_hz=analytics["forcing_hz"])
            summary["dynamic_analytics"]={
                "status":"EXPORTED","output_dir":exported["output_dir"],
                "workbooks":exported["workbooks"],
                "probe_count":exported["probes"],
                "complete":exported["complete"]}
        except Exception as exc:
            # Leave complete/partial accepted physical states intact. A failed
            # optional visualization must never invalidate a Newmark step.
            summary["dynamic_analytics"]={
                "status":"EXPORT_FAILED","error_type":type(exc).__name__,
                "message":str(exc)}
    _write_json(out/"FINAL_RUN_SUMMARY.json",summary)
    return summary


def run_reference_mvi_series(case,case_path,ranks=range(1,9)):
    """Independent fixed-rank runs; preflight every rank before any output."""
    import copy
    if str(case.get("method","")).upper()!="MVI":
        raise ValueError("MVI rank series requires method=MVI")
    ranks=[_integer(rank,"MVI rank",1) for rank in ranks]
    if not ranks or len(set(ranks))!=len(ranks):raise ValueError("nonempty distinct MVI ranks required")
    prepared=[]
    for rank in ranks:
        candidate=copy.deepcopy(case);candidate["rank"]=rank
        prepare_reference_case(candidate,base_dir=Path(case_path).parent)
        # Rank-specific CASE files are written into a different directory.
        # Preserve the original meaning of any CASE-relative time tables when
        # the saved rank file is reopened and validated by the runner.
        laws=[]
        dynamics=candidate.get("dynamics") or {}
        if isinstance(dynamics,dict):laws.append(dynamics.get("time_law"))
        bc=(candidate.get("boundary_conditions") or {}).get("variational") or {}
        if isinstance(bc,dict):
            for family in ("mechanical","electrical"):
                sites=bc.get(family) or {}
                if isinstance(sites,dict):
                    laws.extend(data.get("time_law") for data in sites.values() if isinstance(data,dict))
        for law in laws:
            if isinstance(law,dict) and "csv_path" in law:
                table=Path(str(law["csv_path"])).expanduser()
                if not table.is_absolute():table=Path(case_path).resolve().parent/table
                law["csv_path"]=str(table.resolve())
        prepared.append(candidate)
    base=Path(case["output_dir"])
    if not base.is_absolute():base=Path(case_path).parent/base
    out=base.resolve()/"MVI_1_8_SERIES"
    if out.exists() and any(out.iterdir()):raise ValueError("OUTPUT_DIR_NOT_EMPTY: MVI rank series")
    out.mkdir(parents=True,exist_ok=True)
    summary={"solver_backend":BACKEND,"method":"MVI_RANK_SERIES","ranks":ranks,
             "results":[],"all_converged":False,"output_dir":str(out),
             "mesh_convergence_verified":False}
    for candidate in prepared:
        rank=candidate["rank"]
        candidate["output_dir"]=str(out/f"MVI_{rank}")
        path=out/f"CASE_MVI_{rank}.json"
        _write_json(path,candidate)
        try:result=run_reference_case(candidate,path)
        except Exception as exc:
            result={"rank":rank,"converged":False,"status":"RANK_EXCEPTION",
                    "error_type":type(exc).__name__,"error":str(exc)}
        summary["results"].append(result)
        summary["all_converged"]=(len(summary["results"])==len(ranks) and
                                  all(row["converged"] for row in summary["results"]))
        _write_json(out/"MVI_1_8_SUMMARY.json",summary)
    return summary


__all__=("BACKEND","selected_backend","prepare_reference_case","run_reference_case",
         "run_reference_dynamic_case","run_reference_mvi_series")

