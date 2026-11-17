"""Published-parameter large KL flexoelectric cylinder: energy FDM vs MVI-1..8.

Reference: Deng & Shang, Aerospace 2024, 11(8), 661,
https://doi.org/10.3390/aerospace11080661, Table 1 and Sec. 4.1.3.

Not exact validation: the paper has an OPEN BaTiO3/PVDF FG cylindrical
PANEL, consistent couple stress theory and simply supported edges.
Our solver uses a CLOSED homogeneous BaTiO3 cylinder, fixed axial ends,
G3D/Gz strain-gradient flexoelectricity. Original scalar f is mapped
to mu1=f, mu2=0 as an EXPLICIT MODELING ASSUMPTION (not a derivation).
Published geometry, material, l/h and full q0 amplitude are preserved.
The nonaxisymmetric load is adapted to be periodic on the closed shell.
All runs are genuine loaded nonlinear Green-Lagrange, never fake tests.
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
from independent_mkr import run_energy_fdm_research as fdm
from independent_mvi import run_coupled_research as mvi
from cylinder_kl_face_load import kl_radial_displacement

DOI="10.3390/aerospace11080661"
NX, NS = 25, 33
L, R = math.pi*1.e-3, 12.e-3
H = .1*L
ELL = .2*H
Q0 = 100.e6
E, NU = 113.7e9, .325
F, KAPPA, RHO = 1.e-5, 1.24e-8, 6017.
GRADIENTS=("G3D","Gz")
METHODS=("ENERGY_FDM","MVI")
RANKS=tuple(range(1,9))


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")


def pressure():
    """Periodic adapted q=-q0 sin(pi x/L)(1+cos(theta))/2, max=q0."""
    x=np.linspace(0.,L,NX)
    theta=2*math.pi*np.arange(NS)/NS
    q=-Q0*np.sin(math.pi*x/L)[:,None]*(.5+.5*np.cos(theta))[None,:]
    assert q.shape==(NX,NS) and np.isclose(q[NX//2,0],-Q0)
    assert not np.any(q>1.e-7) and np.max(np.abs(q[0]))<1.e-7
    return q


def physics(gradient):
    if gradient not in GRADIENTS:
        raise ValueError("INVALID_GRADIENT")
    return dict(theory="KL",geometry_model="GREEN_LAGRANGE",
        material=dict(young_pa=E,poisson=NU,ell_m=ELL,
                      mu1_c_per_m=F,mu2_c_per_m=0.,
                      kappa_f_per_m=KAPPA,gradient_model=gradient),
        thickness_m=H,density_kg_m3=RHO,
        eta_s=0.,mass_damping_rate_per_s=0.,
        n_gauss=12,potential_order=1)


def physical_passport(gradient):
    data=dict(
        paper_doi=DOI,paper_section="4.1.3",paper_table=1,
        material="HOMOGENEOUS_BaTiO3",
        geometry=dict(nx=NX,ns=NS,length_m=L,
                      radius_m=R,thickness_m=H),
        physics=physics(gradient),
        boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                      electric="GROUNDED_BOTH"),
        physical_outer_face_pressure_pa_sha256=hashlib.sha256(
            pressure().astype("<f8").tobytes()).hexdigest(),
        load="q=-q0*sin(pi*x/L)*(1+cos(theta))/2",
        q0_pa=Q0,material_length_ratio_l_over_h=.2,
        adapted_from_open_FG_panel_to_closed_homogeneous_cylinder=True,
        adapted_CCST_to_G3D_or_Gz_not_equivalent=True,
        adapted_pressure_to_closed_periodic_nonaxisymmetric=True,
        assumed_mu1_equals_published_f_and_mu2_zero=True,
        fixed_axial_ends_instead_of_simply_supported=True,
        physical_boundary_closure_certified=False,
        scientific_accuracy_certified=False,
        exact_article_replication=False)
    serial=json.dumps(data,sort_keys=True,allow_nan=False,
                      separators=(",",":"))
    return data,hashlib.sha256(serial.encode("utf-8")).hexdigest()


def fourier_seed(rank):
    theta=2*math.pi*np.arange(NS)/NS
    modes=[np.ones(NS)]
    for k in range(1,5):
        modes.extend((np.cos(k*theta),np.sin(k*theta)))
    return np.broadcast_to(np.stack(modes[:rank]),(5,rank,NS)).copy()


def prepare(folder, gradient, method, rank):
    if method not in METHODS or gradient not in GRADIENTS:
        raise ValueError("UNKNOWN_LITERATURE_CASE")
    if method=="MVI" and rank not in RANKS:
        raise ValueError("FIXED_RANK_1_TO_8_REQUIRED")
    if method=="ENERGY_FDM" and rank is not None:
        raise ValueError("FDM_HAS_NO_MVI_RANK")
    folder.mkdir(parents=True,exist_ok=False)
    passport,phash=physical_passport(gradient)
    write_json(folder/"PHYSICAL_PASSPORT.json",
               dict(passport,sha256=phash))
    q=np.zeros((3,NX,NS),float)
    v=np.zeros((2,NX,NS),float)
    face=dict(face="OUTER",traction_pa=pressure().tolist())
    grid=dict(nx=NX,ns=NS,length_m=L,radius_m=R)
    boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                  electric="GROUNDED_BOTH")
    if method=="ENERGY_FDM":
        np.savez_compressed(folder/"initial.npz",mechanical=q,electrical=v)
        np.savez_compressed(folder/"load.npz",
                            mechanical_load=q,electrical_charge=v)
        case=dict(schema=fdm.SCHEMA,research_opt_in=True,
             physics=physics(gradient),grid=grid,
             boundary=dict(boundary,axial_orders=[2,2,2]),
             solver=dict(kind="STATIC",integrator=None,dt_s=None,steps=0,
                 max_dofs=8500,max_iterations=45,
                 linear_solver="AUTO",dense_lu_dof_limit=450,
                 mechanical_state_scale=1.e-4,
                 mechanical_equation_scale=1.e8,
                 electrical_state_scale=1.,
                 electrical_equation_scale=1.e-4,
                 potential_scale_v=1.),
             initial_state_npz="initial.npz",loads_npz="load.npz",
             output_dir="RESULTS",kl_radial_face_traction=face)
    else:
        np.savez_compressed(folder/"initial.npz",
                            x_profiles=np.zeros((5,rank,NX)),
                            y_profiles=fourier_seed(rank))
        np.savez_compressed(folder/"load.npz",
                            mechanical_load=q,free_charge=v)
        case=dict(schema=mvi.SCHEMA,research_opt_in=True,
             physics=physics(gradient),grid=grid,boundary=boundary,
             rank=rank,
             solver=dict(kind="STATIC",integrator=None,dt_s=None,steps=0,
                 max_sweeps=65,max_acceleration_coordinates=8500,
                 max_halfstep_coordinates=8500,
                 field_scales=[1.e-4,1.e-4,1.e-4,1.,1.],
                 energy_scale_j=.1,
                 allow_projected_static=True,
                 linear_solver="AUTO",dense_coordinate_limit=180,
                 krylov_restart=48,krylov_maxiter=150,
                 krylov_relative_tolerance=1.e-7,
                 krylov_dense_fallback_limit=0),
             initial_state_npz="initial.npz",loads_npz="load.npz",
             output_dir="RESULTS",kl_radial_face_traction=face)
    write_json(folder/"CASE.json",case)
    return folder/"CASE.json",phash


def outer_radial(field,owner):
    if hasattr(owner,"field_jets"):
        jets=owner.field_jets.prepare(field,owner.derivative_orders)[:3]
    else:
        grid=owner.grid
        jets=(field,np.einsum("ij,cjk->cik",grid.dx,field),
                    np.einsum("jk,cik->cij",grid.ds,field))
    return np.asarray(kl_radial_displacement(
        *jets,radius_m=R,z_m=H/2,geometry_model="GREEN_LAGRANGE"))


def run_case(gradient,method,rank,output):
    if gradient not in GRADIENTS or method not in METHODS:
        raise ValueError("LARGE_STUDY_INVALID_SELECTION")
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    ident=f"KL_{gradient}_{method}"+(f"_R{rank}" if rank is not None else "")
    report=dict(id=ident,gradient=gradient,method=method,rank=rank,
        grid=[NX,NS],reference_doi=DOI,pressure_pa=Q0,
        sha=os.getenv("GITHUB_SHA","UNKNOWN"),
        scientific_accuracy_certified=False,
        exact_paper_replication=False,status="IN_PROGRESS")
    marker=root/"EVIDENCE.json"
    write_json(marker,report)
    try:
        case_path,phash=prepare(root/"input",gradient,method,rank)
        report["physical_passport_sha256"]=phash
        write_json(marker,report)
        solver=(fdm if method=="ENERGY_FDM" else mvi)
        check=solver.run(case_path,validate_only=True)
        if not check.get("validated_only"):
            raise AssertionError("LARGE_CASE_PREFLIGHT_NOT_VALIDATED")
        prepared=solver.load_case(case_path)
        print("LITERATURE_LARGE_REAL_BEGIN",json.dumps(
            dict(id=ident,grid=[NX,NS],pressure_pa=Q0,
                 passport_sha256=phash)),flush=True)
        answer=solver.run(case_path,validate_only=False)
        stored=root/"input"/"RESULTS"/"STATIC_ACCEPTED.npz"
        if answer.get("accepted_steps")!=1 or not stored.is_file():
            raise AssertionError("LARGE_CASE_NO_ACCEPTED_STATE")
        with np.load(stored,allow_pickle=False) as values:
            u=np.asarray(values["mechanical"],float)
            phi=np.asarray(values["electrical"],float)
            if u.shape!=(3,NX,NS) or phi.shape!=(2,NX,NS) or not (
                    np.isfinite(u).all() and np.isfinite(phi).all()):
                raise AssertionError("LARGE_CASE_INVALID_PHYSICAL_FIELDS")
            extra={}
            if method=="MVI":
                x=np.asarray(values["x_profiles"],float)
                y=np.asarray(values["y_profiles"],float)
                if x.shape!=(5,rank,NX) or y.shape!=(5,rank,NS):
                    raise AssertionError("LARGE_CASE_MVI_RANK_LOST")
                extra=dict(x_profiles=x,y_profiles=y)
        owner=(prepared["space"].engine.model if method=="ENERGY_FDM"
               else prepared["surface"])
        outer=outer_radial(u,owner)
        peak=float(np.max(np.abs(outer)))
        volts=float(np.max(np.abs(phi)))
        if (outer.shape!=(NX,NS) or not np.isfinite(outer).all() or
                peak<=1.e-15 or volts<=1.e-15):
            raise AssertionError("LARGE_CASE_MISSING_PHYSICAL_RESPONSE")
        if method=="MVI":
            quality=answer.get("solution_quality")
            full=answer.get("full_nodal_equilibrium_satisfied")
            if quality not in ("PROJECTED_EQUILIBRIUM",
                               "FULL_NODAL_EQUILIBRIUM"):
                raise AssertionError("LARGE_CASE_EQUILIBRIUM_UNREPORTED")
            if quality=="FULL_NODAL_EQUILIBRIUM" and full is not True:
                raise AssertionError("LARGE_CASE_FALSE_FULL_NODAL")
            if quality=="PROJECTED_EQUILIBRIUM" and full is not False:
                raise AssertionError("LARGE_CASE_FALSE_PROJECTED")
        np.savez_compressed(root/"FIELDS.npz",
             mechanical=u,electrical=phi,radial_outer=outer,**extra)
        report.update(status="ACCEPTED_RESEARCH_ONLY",
            solver_status=answer.get("status"),
            solution_status=answer.get("solution_status"),
            solution_quality=answer.get("solution_quality"),
            full_nodal_norm=answer.get("full_nodal_norm"),
            full_nodal_equilibrium_satisfied=answer.get(
                "full_nodal_equilibrium_satisfied"),
            projected_norm_x=answer.get("projected_norm_x"),
            projected_norm_y=answer.get("projected_norm_y"),
            radial_mid_theta0_m=float(outer[NX//2,0]),
            radial_max_abs_m=peak,potential_max_abs_v=volts)
        write_json(marker,report)
        print("LITERATURE_LARGE_REAL_ACCEPTED",
              json.dumps(report,allow_nan=False),flush=True)
    except BaseException as exc:
        report.update(status="FAILED",error_type=type(exc).__name__,
                      detail=str(exc))
        write_json(marker,report)
        (root/"TRACEBACK.txt").write_text(
            traceback.format_exc(),encoding="utf-8")
        print("LITERATURE_LARGE_REAL_FAILED",ident,
              type(exc).__name__,str(exc),flush=True)
        raise


def compare(source,destination):
    src=Path(source)
    target=Path(destination)
    target.mkdir(parents=True,exist_ok=True)
    records={}
    for file in src.rglob("EVIDENCE.json"):
        value=json.loads(file.read_text(encoding="utf-8"))
        records.setdefault(value["id"],[]).append((value,file.parent))
    rows=[]
    for gradient in GRADIENTS:
        reference=records.get(f"KL_{gradient}_ENERGY_FDM",[])
        for rank in RANKS:
            matches=records.get(f"KL_{gradient}_MVI_R{rank}",[])
            base=reference[0] if len(reference)==1 else None
            mvirow=matches[0] if len(matches)==1 else None
            row=dict(gradient=gradient,rank=rank,grid=f"{NX}x{NS}",
                fdm_status=(base[0]["status"] if base else "MISSING"),
                mvi_status=(mvirow[0]["status"] if mvirow else "MISSING"),
                mvi_quality=(mvirow[0].get("solution_quality") if mvirow else None),
                mvi_full_nodal_norm=(mvirow[0].get("full_nodal_norm") if mvirow else None),
                relative_radial_l2=None,relative_electric_l2=None,
                relative_mechanical_l2=None,center_radial_difference_m=None,
                status="NOT_COMPARABLE",detail=(
                    mvirow[0].get("detail") if mvirow else "Missing rank result"))
            if (base and mvirow and
                    base[0]["status"]=="ACCEPTED_RESEARCH_ONLY" and
                    mvirow[0]["status"]=="ACCEPTED_RESEARCH_ONLY"):
                if (base[0].get("physical_passport_sha256")!=
                        mvirow[0].get("physical_passport_sha256")):
                    row["status"]="PHYSICAL_PASSPORT_MISMATCH"
                elif base[0].get("sha")!=mvirow[0].get("sha"):
                    row["status"]="CODE_REVISION_MISMATCH"
                else:
                    with np.load(base[1]/"FIELDS.npz",allow_pickle=False) as a, \
                         np.load(mvirow[1]/"FIELDS.npz",allow_pickle=False) as b:
                        for label,key in (("radial","radial_outer"),
                                          ("electric","electrical"),
                                          ("mechanical","mechanical")):
                            x=np.asarray(a[key],float)
                            y=np.asarray(b[key],float)
                            if x.shape!=y.shape or not (
                                    np.isfinite(x).all() and np.isfinite(y).all()):
                                raise ValueError("LARGE_COMPARISON_BAD_FIELDS")
                            row["relative_"+label+"_l2"]=float(
                                np.linalg.norm(y-x)/max(np.linalg.norm(x),1.e-30))
                        row["center_radial_difference_m"]=float(
                            b["radial_outer"][NX//2,0]-a["radial_outer"][NX//2,0])
                    row.update(status="COMPARED_RESEARCH_ONLY",detail=None)
            rows.append(row)
    with (target/"COMPARISON.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report=dict(reference_doi=DOI,exact_paper_replication=False,
        grid=[NX,NS],pressure_pa=Q0,expected_runs=18,expected_pairs=16,
        recorded_runs=sum(len(x)==1 for x in records.values()),
        compared_pairs=sum(x["status"]=="COMPARED_RESEARCH_ONLY" for x in rows),
        results=rows,external_accuracy_certified=False)
    report["status"]=(
        "COMPARED_RESEARCH_ONLY" if report["compared_pairs"]==16
        else "INCOMPLETE_WITH_EXPLICIT_FAILURES")
    write_json(target/"COMPARISON.json",report)
    print("LITERATURE_LARGE_COMPARISON",json.dumps({
        k:report[k] for k in ("expected_runs","expected_pairs",
                              "recorded_runs","compared_pairs","status")}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="action",required=True)
    item=sub.add_parser("run")
    item.add_argument("--gradient",choices=GRADIENTS,required=True)
    item.add_argument("--method",choices=METHODS,required=True)
    item.add_argument("--rank",type=int)
    item.add_argument("--output",required=True)
    group=sub.add_parser("compare")
    group.add_argument("--source",required=True)
    group.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.action=="run":
        run_case(args.gradient,args.method,args.rank,args.output)
    else:
        compare(args.source,args.output)


if __name__=="__main__":
    main()
