"""Independent FDM/MVI axial refinement for the same closed KL Gz cylinder.

The 21x4 pressure pilot is not used as a same-grid reference because MVI
requires an odd Fourier grid with at least 5 nodes. Both independent
methods use 21/31/41 by 5, the same 12 Gauss points, zero initial fields,
100 Pa outer-face dead traction, and identical physical CASE data.
Rank 1 is sufficient for the imposed axisymmetric solution family.
No external accuracy or nonaxisymmetric stability is claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from cylinder_paths import activate
activate()
from ci.matched_sat_physical_pilot import make_case
from independent_mkr import run_coupled_research as fdm
from independent_mvi import run_coupled_research as mvi
from cylinder_kl_face_load import kl_radial_displacement

MESHES=(21,31,41)
NS=5

def save(path,content):
    path.write_text(json.dumps(content,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def prepare(folder,nx,method):
    if nx not in MESHES or method not in ("FDM","MVI1"):
        raise ValueError("UNKNOWN_MESH_COMPARISON_CASE")
    folder.mkdir(parents=True,exist_ok=False)
    path,nm,pe=make_case(folder,"KL","Gz","STATIC",nx=nx,n_gauss=12)
    case=json.loads(path.read_text())
    case["grid"]["ns"]=NS
    q=np.zeros((nm,nx,NS));p=np.zeros((pe+1,nx,NS))
    # Mechanical generalized forcing is zero because the exact nominal
    # outer-face work is already contained in the shared physical CASE.
    # Explicitly retain that load, never replace it by a midsurface pressure.
    passport=dict(physics=case["physics"],
        grid={key:case["grid"][key] for key in ("nx","ns","length_m","radius_m")},
        boundary={key:case["boundary"][key] for key in ("mechanical","electric")},
        kl_radial_face_traction=case["kl_radial_face_traction"],
        generalized_mechanical_load_sha256=hashlib.sha256(q.astype("<f8").tobytes()).hexdigest(),
        generalized_electric_charge_sha256=hashlib.sha256(p.astype("<f8").tobytes()).hexdigest())
    digest=hashlib.sha256(json.dumps(passport,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if method=="FDM":
        np.savez_compressed(folder/"seed.npz",mechanical=q,electrical=p)
        np.savez_compressed(folder/"loads.npz",mechanical_load=q,electrical_charge=p)
    else:
        np.savez_compressed(folder/"seed.npz",
            x_profiles=np.zeros((nm+pe+1,1,nx)),y_profiles=np.ones((nm+pe+1,1,NS)))
        np.savez_compressed(folder/"loads.npz",mechanical_load=q,free_charge=p)
        case.update(schema=mvi.SCHEMA,grid=passport["grid"],boundary=passport["boundary"],rank=1,
            solver=dict(kind="STATIC",integrator=None,dt_s=None,steps=0,
                field_scales=[1.e-4]*nm+[1.]*(pe+1),energy_scale_j=1.,
                max_sweeps=40,max_acceleration_coordinates=5000,max_halfstep_coordinates=5000,
                allow_projected_static=True,linear_solver="AUTO",dense_coordinate_limit=200,
                krylov_restart=48,krylov_maxiter=90,krylov_relative_tolerance=1.e-7))
    save(path,case)
    save(folder/"PHYSICAL_PASSPORT.json",dict(passport,sha256=digest))
    return path,digest

def run(root,nx,method):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(method=method,rank=1 if method=="MVI1" else None,grid=[nx,NS],
        theory="KL",gradient="Gz",n_gauss=12,pressure_pa=100.,
        git_sha=os.getenv("GITHUB_SHA"),status="STARTED",
        external_accuracy_certified=False,physical_boundary_closure_certified=False)
    marker=root/"EVIDENCE.json";save(marker,report)
    try:
        path,digest=prepare(root/"input",nx,method)
        solver=fdm if method=="FDM" else mvi
        prepared=solver.load_case(path)
        report["physical_passport_sha256"]=digest
        started=time.perf_counter()
        answer=solver.run(path,validate_only=False)
        report["elapsed_s"]=time.perf_counter()-started
        assert answer["accepted_steps"]==1
        with np.load(root/"input"/"RESULTS"/"STATIC_ACCEPTED.npz",allow_pickle=False) as state:
            q=np.asarray(state["mechanical"],float)
            p=np.asarray(state["electrical"],float)
        assert q.shape==(3,nx,NS) and p.shape==(2,nx,NS)
        assert np.isfinite(q).all() and np.isfinite(p).all()
        assert np.max(np.abs(q[:,[0,-1],:]))<=1.e-9
        assert np.max(np.abs(p[:,[0,-1],:]))<=1.e-8
        assert q[2,nx//2,0]<-1.e-14 and np.max(abs(p))>1.e-14
        # Reject a nonaxisymmetric branch instead of comparing it to
        # the axisymmetrically restricted FDM solve.
        np.testing.assert_allclose(q,np.broadcast_to(q[..., :1],q.shape),rtol=1.e-5,atol=1.e-11)
        np.testing.assert_allclose(p,np.broadcast_to(p[..., :1],p.shape),rtol=1.e-5,atol=1.e-8)
        if method=="FDM":
            owner=prepared["model"]
            jets=owner.field_jets.prepare(q,owner.derivative_orders)[:3]
        else:
            owner=prepared["surface"]
            jets=(q,np.einsum("ij,cjk->cik",owner.grid.dx,q),
                     np.einsum("jk,cik->cij",owner.grid.ds,q))
        outer=kl_radial_displacement(*jets,radius_m=owner.grid.radius_m,
              z_m=owner.thickness_m/2,geometry_model=owner.geometry_model)
        z=np.linspace(-1.,1.,81)
        phi=np.einsum("zk,kij->zij",z[:,None]**np.arange(2)[None,:],p)
        np.savez_compressed(root/"FIELDS.npz",x_m=np.linspace(0.,owner.grid.length_m,nx),
            mechanical=q,electrical=p,radial_outer=outer)
        report.update(status="ACCEPTED_RESEARCH_ONLY",solver_status=answer["status"],
            solution_status=answer.get("solution_status"),
            solution_quality=answer.get("solution_quality"),
            w_mid_m=float(q[2,nx//2,0]),w_outer_m=float(outer[nx//2,0]),
            phi_sampled_max_v=float(np.max(np.abs(phi))))
    except Exception as exc:
        report.update(status="FAILED",error_type=type(exc).__name__,error=str(exc))
        save(marker,report);print("MESH_FAILURE",json.dumps(report),flush=True);raise
    save(marker,report);print("MESH_RESULT",json.dumps(report),flush=True)

def summarize(root,output):
    records=[json.loads(p.read_text()) for p in Path(root).rglob("EVIDENCE.json")]
    rows=[]
    for nx in MESHES:
        pair={method:[r for r in records if r["method"]==method and r["grid"]==[nx,NS]]
              for method in ("FDM","MVI1")}
        row=dict(grid=[nx,NS],status="INCOMPLETE")
        if all(len(v)==1 and v[0]["status"]=="ACCEPTED_RESEARCH_ONLY" for v in pair.values()):
            f,m=pair["FDM"][0],pair["MVI1"][0]
            if f["physical_passport_sha256"]!=m["physical_passport_sha256"]:
                raise ValueError("PHYSICAL_PASSPORT_MISMATCH")
            row.update(status="COMPARED_RESEARCH_ONLY",fdm=f,mvi=m,
                w_mid_relative_difference_percent=100*(m["w_mid_m"]-f["w_mid_m"])/abs(f["w_mid_m"]),
                phi_relative_difference_percent=100*(m["phi_sampled_max_v"]-f["phi_sampled_max_v"])/abs(f["phi_sampled_max_v"]))
        rows.append(row)
    result=dict(rows=rows,records=records,external_accuracy_certified=False)
    save(Path(output),result)
    print("MESH_COMPARISON",json.dumps(result),flush=True)
    if not all(row["status"]=="COMPARED_RESEARCH_ONLY" for row in rows):
        raise RuntimeError("MESH_COMPARISON_INCOMPLETE")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--nx",type=int,choices=MESHES)
    p.add_argument("--method",choices=("FDM","MVI1"))
    p.add_argument("--output",required=True)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--summarize")
    a=p.parse_args()
    if a.summarize:
        summarize(a.summarize,a.output)
    elif a.preflight:
        root=Path(a.output)
        for nx in MESHES:
            digests=[]
            for method in ("FDM","MVI1"):
                path,digest=prepare(root/f"{method}-{nx}",nx,method)
                (fdm if method=="FDM" else mvi).load_case(path)
                digests.append(digest)
            assert digests[0]==digests[1]
        print("MESH_PREFLIGHT: six independent CASEs validated; three matching physical passports",flush=True)
    else:
        run(a.output,a.nx,a.method)
