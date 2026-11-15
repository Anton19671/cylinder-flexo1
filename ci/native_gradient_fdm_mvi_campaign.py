"""Native gradient-flexoelectric FDM versus fixed-rank MVI physical campaign.

7 shell kinematics x G3D/Gz x STATIC/DYNAMIC x (FDM,MVI-1..8).
Shared: physical data, SI loading, material, initial continuum fields.
NEVER shared: spatial derivative assembly, solver, rank update, fallback.

This is a research comparison, NOT a certified free-face or external benchmark.
All physical calculations are executed by GitHub Actions.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np
from cylinder_paths import activate
activate()
from independent_mkr import run_coupled_research as fdm
from independent_mvi import run_coupled_research as mvi

THEORIES = ("KL", "FSDT", "REDDY", "CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")
GRADIENTS = ("G3D", "Gz")
KINDS = ("STATIC", "DYNAMIC")
GRID = (11, 9)
L, R, H = 1.2, 2.0, 0.08
PRESSURE = 100.0
DT, NSTEPS = 1.e-4, 2
GRADIENT_LENGTH_M = 0.002
MATERIAL_BASE = dict(young_pa=1.e9, poisson=.3, ell_m=GRADIENT_LENGTH_M,
                     mu1_c_per_m=1.e-3, mu2_c_per_m=-5.e-4, kappa_f_per_m=1.e-8)


def nmechanical(theory):
    return 3 if theory == "KL" else 5 if theory in ("FSDT", "REDDY") else 3*(int(theory[-1])+1)


def electric_order(theory):
    return int(theory[-1]) if theory.startswith("CUF_TE") else 1


def all_cases():
    for theory in THEORIES:
        for gradient in GRADIENTS:
            for kind in KINDS:
                yield (theory, gradient, kind, "FDM", None)
                for rank in range(1,9):
                    yield (theory, gradient, kind, "MVI", rank)


def case_id(item):
    theory,gradient,kind,method,rank=item
    return "_".join((theory,gradient,kind,method)) + (f"_R{rank}" if rank else "")


def included(scope, item):
    if scope == "full":
        return True
    if scope == "pilot":
        theory,gradient,kind,method,rank=item
        return theory == "KL" and (method == "FDM" or rank in (1,3,8))
    raise ValueError("INVALID_CAMPAIGN_SCOPE")


def pressure_profile(ns):
    theta=2*np.pi*np.arange(ns)/ns
    p=(1. + .18*np.cos(theta) + .12*np.sin(theta)
         + .09*np.cos(2*theta) + .06*np.sin(2*theta)
         + .04*np.cos(3*theta) + .025*np.sin(3*theta)
         + .015*np.cos(4*theta))
    if not np.isfinite(p).all() or np.min(p)<=0:
        raise AssertionError("INVALID_NONAXISYMMETRIC_PRESSURE")
    return PRESSURE*p


def physical_data(theory,gradient,kind):
    if theory not in THEORIES or gradient not in GRADIENTS or kind not in KINDS:
        raise ValueError("INVALID_PHYSICAL_PASSPORT")
    nx,ns=GRID
    nm=nmechanical(theory)
    pe=electric_order(theory)
    forcing=np.zeros((nm,nx,ns),dtype=float)
    q=-pressure_profile(ns)*(1.+H/(2.*R))
    if theory.startswith("CUF_TE"):
        k=pe+1
        forcing[2*k:3*k,:,:]=q[np.newaxis,np.newaxis,:]
    else:
        forcing[2,:,:]=q[np.newaxis,:]
    if not np.isfinite(forcing).all() or not np.any(forcing):
        raise AssertionError("INVALID_PHYSICAL_LOAD")
    physics={
        "theory":theory,"geometry_model":"GREEN_LAGRANGE",
        "material":dict(MATERIAL_BASE,gradient_model=gradient),
        "thickness_m":H,"density_kg_m3":2500.,
        "eta_s":0.,"mass_damping_rate_per_s":0.,
        "n_gauss":12,"potential_order":pe,
    }
    grid=dict(nx=nx,ns=ns,length_m=L,radius_m=R)
    passport=dict(physics=physics,grid=grid,kind=kind,
        integrator="NEWMARK" if kind=="DYNAMIC" else None,
        dt_s=DT if kind=="DYNAMIC" else None,
        steps=NSTEPS if kind=="DYNAMIC" else 0,
        mechanical_essential="IMMOBILE_PARALLEL_ENDS",
        electrical_essential="GROUNDED_BOTH",
        nominal_electric_faces="ZERO_FREE_CHARGE",
        nominal_outer_face_pressure_pa=PRESSURE,
        mechanical_load_sha256=hashlib.sha256(forcing.tobytes()).hexdigest(),
        full_higher_boundary_conditions_certified=False)
    digest=hashlib.sha256(json.dumps(passport,sort_keys=True).encode()).hexdigest()
    return physics,grid,forcing,passport,digest


def fourier_seed(rank,ns):
    angle=2*np.pi*np.arange(ns)/ns
    modes=[np.ones(ns)]
    for k in range(1,5):
        modes.extend((np.cos(k*angle),np.sin(k*angle)))
    return np.stack(modes[:rank])


def write_input(folder,item):
    theory,gradient,kind,method,rank=item
    nx,ns=GRID
    physics,grid,forcing,passport,digest=physical_data(theory,gradient,kind)
    nm=nmechanical(theory)
    ne=electric_order(theory)+1
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"PHYSICAL_PASSPORT.json").write_text(
        json.dumps(dict(passport,hash_sha256=digest),indent=2)+"\n")
    if method=="FDM":
        initial={"mechanical":np.zeros((nm,nx,ns)),
                 "electrical":np.zeros((ne,nx,ns))}
        if kind=="DYNAMIC":
            initial["velocity"]=np.zeros((nm,nx,ns))
        np.savez_compressed(folder/"initial.npz",**initial)
        np.savez_compressed(folder/"loads.npz",mechanical_load=forcing,
                            electrical_charge=np.zeros((ne,nx,ns)))
        case=dict(schema=fdm.SCHEMA,research_opt_in=True,
            physics=physics,grid=grid,
            boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                          electric="GROUNDED_BOTH",
                          axial_orders=[2]*nm),
            solver=dict(kind=kind,
                integrator="NEWMARK" if kind=="DYNAMIC" else None,
                dt_s=DT if kind=="DYNAMIC" else None,
                steps=NSTEPS if kind=="DYNAMIC" else 0,
                max_dofs=5000,max_iterations=35,
                linear_solver="AUTO",dense_lu_dof_limit=650,
                mechanical_state_scale=1.e-4,
                mechanical_equation_scale=1.e4,
                electrical_state_scale=1.,
                electrical_equation_scale=1.,
                potential_scale_v=1.),
            initial_state_npz="initial.npz",loads_npz="loads.npz",
            output_dir="RESULTS")
    elif method=="MVI":
        if not isinstance(rank,int) or not 1<=rank<=8 or rank>ns or rank>nx-2:
            raise ValueError("MVI_RANK_GRID_CAPACITY_EXCEEDED")
        x=np.zeros((nm+ne,rank,nx))
        y=np.broadcast_to(fourier_seed(rank,ns),(nm+ne,rank,ns)).copy()
        initial=dict(x_profiles=x,y_profiles=y)
        if kind=="DYNAMIC":
            initial["velocity"]=np.zeros((nm,nx,ns))
        np.savez_compressed(folder/"initial.npz",**initial)
        np.savez_compressed(folder/"loads.npz",
                            mechanical_load=forcing,free_charge=np.zeros((ne,nx,ns)))
        case=dict(schema=mvi.SCHEMA,research_opt_in=True,
            physics=physics,grid=grid,
            boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                          electric="GROUNDED_BOTH"),
            rank=rank,
            solver=dict(kind=kind,
                integrator="NEWMARK" if kind=="DYNAMIC" else None,
                dt_s=DT if kind=="DYNAMIC" else None,
                steps=NSTEPS if kind=="DYNAMIC" else 0,
                field_scales=[1.e-4]*nm+[1.]*ne,
                energy_scale_j=1.,max_sweeps=40,
                max_acceleration_coordinates=5000,
                max_halfstep_coordinates=5000,
                allow_projected_static=(kind=="STATIC"),
                linear_solver="AUTO",
                dense_coordinate_limit=200,
                krylov_restart=48,
                krylov_maxiter=90,
                krylov_relative_tolerance=1.e-7),
            initial_state_npz="initial.npz",loads_npz="loads.npz",
            output_dir="RESULTS")
    else:
        raise ValueError("UNKNOWN_NUMERICAL_METHOD")
    path=folder/"CASE.json"
    path.write_text(json.dumps(case,indent=2)+"\n")
    return path,forcing,digest


def radial_recovery(q,theory):
    if theory.startswith("CUF_TE"):
        k=electric_order(theory)+1
        coefficients=q[2*k:3*k]
        outer=np.sum(coefficients,axis=0)
        inner=np.einsum("k,kij->ij",(-1.)**np.arange(k),coefficients)
        mid=coefficients[0]
    else:
        outer=inner=mid=q[2]
    return mid,outer,inner


def run_case(identifier,output):
    cases={case_id(c):c for c in all_cases()}
    if identifier not in cases:
        raise ValueError("UNKNOWN_CASE_ID")
    item=cases[identifier]
    theory,gradient,kind,method,rank=item
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    evidence=dict(id=identifier,theory=theory,gradient_model=gradient,
        kind=kind,method=method,rank=rank,grid=list(GRID),
        git_sha=os.environ.get("GITHUB_SHA","UNKNOWN"),
        status="IN_PROGRESS",independent_spatial_methods=True,
        physical_higher_order_boundary_certified=False,
        pointwise_electric_faces_certified=False,
        external_benchmark_certified=False)
    marker=root/"EVIDENCE.json"
    def save():
        marker.write_text(json.dumps(evidence,indent=2,allow_nan=False)+"\n")
    save()
    try:
        folder=root/"input"
        casefile,forcing,passport_hash=write_input(folder,item)
        evidence["physical_passport_sha256"]=passport_hash
        save()
        solver=fdm if method=="FDM" else mvi
        preview=solver.run(casefile,validate_only=True)
        if preview.get("validated_only") is not True:
            raise AssertionError("NATIVE_CASE_PREFLIGHT_REJECTED")
        answer=solver.run(casefile,validate_only=False)
        output_state=(folder/"RESULTS"/
                      ("STATIC_ACCEPTED.npz" if kind=="STATIC" else
                       f"STEP_{NSTEPS:04d}.npz"))
        expected_steps=1 if kind=="STATIC" else NSTEPS
        if answer.get("accepted_steps")!=expected_steps or not output_state.is_file():
            raise AssertionError("NATIVE_CASE_NOT_ACCEPTED")
        with np.load(output_state,allow_pickle=False) as data:
            q=np.asarray(data["mechanical"],float)
            phi=np.asarray(data["electrical"],float)
            if method=="MVI":
                x=np.asarray(data["x_profiles"],float)
                y=np.asarray(data["y_profiles"],float)
                if x.shape!=(len(q)+len(phi),rank,GRID[0]) or \
                   y.shape!=(len(q)+len(phi),rank,GRID[1]):
                    raise AssertionError("MVI_RANK_NOT_RETAINED")
            if kind=="DYNAMIC":
                v=np.asarray(data["velocity"],float)
                a=np.asarray(data["acceleration"],float)
                if v.shape!=q.shape or a.shape!=q.shape or \
                   not np.isfinite(v).all() or not np.isfinite(a).all():
                    raise AssertionError("BAD_DYNAMIC_FIELDS")
        if q.shape!=forcing.shape or phi.shape!=(electric_order(theory)+1,*GRID) \
           or not np.isfinite(q).all() or not np.isfinite(phi).all():
            raise AssertionError("NONFINITE_OR_MISMATCHED_FIELDS")
        middle,outer,inner=radial_recovery(q,theory)
        if np.max(np.abs(outer))<=1.e-14 or np.max(np.abs(phi))<=1.e-14:
            raise AssertionError("NO_LOADED_ELECTROMECHANICAL_RESPONSE")
        fields=dict(mechanical=q,electrical=phi,
                    radial_middle=middle,radial_outer=outer,radial_inner=inner)
        if kind=="DYNAMIC":
            fields.update(velocity=v,acceleration=a)
        np.savez_compressed(root/"FULL_FIELDS.npz",**fields)
        evidence.update(status="COMPLETED_RESEARCH_ONLY",
            solver_status=answer.get("status"),
            solution_status=answer.get("solution_status"),
            solution_quality=answer.get("solution_quality"),
            full_nodal_norm=answer.get("full_nodal_norm"),
            projected_norm_x=answer.get("projected_norm_x"),
            projected_norm_y=answer.get("projected_norm_y"),
            full_nodal_equilibrium_satisfied=answer.get("full_nodal_equilibrium_satisfied"),
            accepted_steps=expected_steps,
            physical_passport_sha256=passport_hash,
            radial_outer_at_xcenter_theta0_m=float(outer[GRID[0]//2,0]),
            radial_outer_max_abs_m=float(np.max(np.abs(outer))),
            potential_max_abs_v=float(np.max(np.abs(phi))),
            mechanical_l2=float(np.linalg.norm(q)),
            electric_l2=float(np.linalg.norm(phi)),
            electric_face_diagnostics=answer.get("electric_faces",
                answer.get("last_electric_faces",{})))
        save()
        print("NATIVE_GRADIENT_ACCEPTED",json.dumps(
            {k:evidence[k] for k in ("id","method","rank",
                "radial_outer_at_xcenter_theta0_m","potential_max_abs_v")}),flush=True)
    except BaseException as exc:
        evidence.update(status="FAILED",
            error_type=type(exc).__name__,error=str(exc))
        save()
        (root/"TRACEBACK.txt").write_text(traceback.format_exc())
        print("NATIVE_GRADIENT_FAILED",identifier,type(exc).__name__,str(exc),flush=True)
        raise


def compare(root,output,scope):
    destination=Path(output)
    destination.mkdir(parents=True,exist_ok=True)
    records={}
    for file in Path(root).rglob("EVIDENCE.json"):
        value=json.loads(file.read_text())
        records.setdefault(value["id"],[]).append((value,file.parent))
    rows=[]
    for item in all_cases():
        if not included(scope,item) or item[3]!="MVI":
            continue
        theory,gradient,kind,_,rank=item
        reference=case_id((theory,gradient,kind,"FDM",None))
        target=case_id(item)
        def pick(identifier):
            found=records.get(identifier,[])
            return found[0] if len(found)==1 else None
        f=pick(reference)
        m=pick(target)
        row=dict(theory=theory,gradient=gradient,kind=kind,
            rank=rank,fdm_id=reference,mvi_id=target,
            fdm_status=f[0]["status"] if f else "MISSING_OR_DUPLICATE",
            mvi_status=m[0]["status"] if m else "MISSING_OR_DUPLICATE",
            mvi_solution_status=m[0].get("solution_status") if m else None,
            mvi_full_nodal_norm=m[0].get("full_nodal_norm") if m else None,
            mvi_full_nodal_equilibrium_satisfied=(m[0].get("full_nodal_equilibrium_satisfied") if m else None),
            relative_mechanical_l2=None,relative_electric_l2=None,
            relative_outer_radial_l2=None,midpoint_radial_difference_m=None,
            status="MISSING_OR_FAILED")
        if f and m and f[0]["status"]=="COMPLETED_RESEARCH_ONLY" and \
           m[0]["status"]=="COMPLETED_RESEARCH_ONLY":
            if f[0]["physical_passport_sha256"]!=m[0]["physical_passport_sha256"]:
                row["status"]="PHYSICAL_PASSPORT_MISMATCH"
            elif not (f[1]/"FULL_FIELDS.npz").is_file() or \
                 not (m[1]/"FULL_FIELDS.npz").is_file():
                row["status"]="FIELDS_MISSING"
            else:
                with np.load(f[1]/"FULL_FIELDS.npz",allow_pickle=False) as fa, \
                     np.load(m[1]/"FULL_FIELDS.npz",allow_pickle=False) as ma:
                    for key,label in (("mechanical","mechanical"),
                                      ("electrical","electric"),
                                      ("radial_outer","outer_radial")):
                        x=np.asarray(fa[key],float)
                        y=np.asarray(ma[key],float)
                        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                            raise AssertionError("INVALID_COMPARISON_"+key)
                        row["relative_"+label+"_l2"]=float(np.linalg.norm(y-x)/max(np.linalg.norm(x),1.e-30))
                    row["midpoint_radial_difference_m"]=float(
                        ma["radial_outer"][GRID[0]//2,0]-fa["radial_outer"][GRID[0]//2,0])
                row["status"]="COMPARED_RESEARCH_ONLY"
        rows.append(row)
    if not rows:
        raise RuntimeError("NO_EXPECTED_COMPARISONS")
    with (destination/"COMPARISON.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed=sum(r["status"]=="COMPARED_RESEARCH_ONLY" for r in rows)
    report=dict(scope=scope,expected_cases=sum(included(scope,c) for c in all_cases()),
        expected_pairs=len(rows),compared_pairs=completed,
        status="COMPARED_RESEARCH_ONLY" if completed==len(rows) else "INCOMPLETE",
        physical_boundary_certified=False,external_accuracy_certified=False,
        results=rows)
    (destination/"COMPARISON.json").write_text(json.dumps(report,indent=2)+"\n")
    print("NATIVE_GRADIENT_COMPARISON",json.dumps(
        {k:report[k] for k in ("scope","expected_cases","expected_pairs",
                               "compared_pairs","status")}),flush=True)
    if completed!=len(rows):
        raise RuntimeError("INCOMPLETE_NATIVE_FDM_MVI_GRADIENT_CAMPAIGN")


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="cmd",required=True)
    matrix=sub.add_parser("matrix")
    matrix.add_argument("--scope",choices=("pilot","full"),required=True)
    run=sub.add_parser("run")
    run.add_argument("--id",required=True)
    run.add_argument("--output",required=True)
    summary=sub.add_parser("compare")
    summary.add_argument("--scope",choices=("pilot","full"),required=True)
    summary.add_argument("--root",required=True)
    summary.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.cmd=="matrix":
        print(json.dumps({"include":[{"id":case_id(c)} for c in all_cases()
            if included(args.scope,c)]},separators=(",",":")))
    elif args.cmd=="run":
        run_case(args.id,args.output)
    else:
        compare(args.root,args.output,args.scope)


if __name__=="__main__":
    main()
