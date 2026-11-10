"""Tiny physical zero-equilibrium native shell smoke, never a benchmark.

Run an unloaded LINEAR KL physical shell through each OWN research
static backend and one native strong MKR Newmark step. This checks
material/geometry/case/boundary/solver/NPZ execution, not convergence
or validity for loaded G3D/Gz research. Uses a temporary folder.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import numpy as np

from cylinder_paths import activate
activate()

from independent_mkr import run_coupled_research as mkr
from independent_mvi import run_coupled_research as mvi


def common_physics(electric=False):
    return dict(theory="KL",geometry_model="LINEAR",
                material=dict(young_pa=1.e9,poisson=.3,ell_m=.002,
                              mu1_c_per_m=0.,mu2_c_per_m=0.,
                              kappa_f_per_m=1.e-9 if electric else 0.,
                              gradient_model="G3D"),
                thickness_m=.08,density_kg_m3=2500.,
                eta_s=0.,mass_damping_rate_per_s=0.,
                n_gauss=2,potential_order=1 if electric else None)


def run_mkr(work,kind,electric=False):
    folder=work/("mkr_"+kind.lower()+
                 ("_electric" if electric else ""))
    folder.mkdir()
    nx,ns=7,5
    q=np.zeros((3,nx,ns))
    initial={"mechanical":q}
    if kind=="DYNAMIC":
        initial["velocity"]=q.copy()
    if electric:
        initial["electrical"]=np.zeros((2,nx,ns))
    np.savez_compressed(folder/"initial.npz",**initial)
    sources={"mechanical_load":q.copy()}
    if electric:
        sources["electrical_charge"]=np.zeros((2,nx,ns))
    np.savez_compressed(folder/"load.npz",**sources)
    case=dict(schema=mkr.SCHEMA,research_opt_in=True,
              physics=common_physics(electric),
              grid=dict(nx=nx,ns=ns,length_m=1.2,radius_m=2.),
              boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                            electric="GROUNDED_BOTH" if electric else "OFF",
                            axial_orders=[1,1,1]),
              solver=dict(kind=kind,
                          integrator="NEWMARK" if kind=="DYNAMIC" else None,
                          dt_s=.001 if kind=="DYNAMIC" else None,
                          steps=1 if kind=="DYNAMIC" else 0,
                          max_dofs=300,max_iterations=3,
                          mechanical_state_scale=1.,
                          mechanical_equation_scale=1.e7,
                          electrical_state_scale=1. if electric else None,
                          electrical_equation_scale=1. if electric else None,
                          potential_scale_v=1.),
              initial_state_npz="initial.npz",loads_npz="load.npz",
              output_dir="out")
    path=folder/"case.json"
    path.write_text(json.dumps(case))
    answer=mkr.run(path,validate_only=False)
    assert answer["accepted_steps"]==1
    assert answer["scientific_accuracy_certified"] is False
    if electric:
        assert answer["electric_faces"]["physical_face_flux_certified"] is False
        for face in answer["electric_faces"]["faces"].values():
            assert face["natural_flux_difference_abs_max_c_per_m2"]<1.e-10
    filename="STATIC_ACCEPTED.npz" if kind=="STATIC" else "STEP_0001.npz"
    with np.load(folder/"out"/filename,allow_pickle=False) as data:
        np.testing.assert_allclose(data["mechanical"],0.,rtol=0.,atol=1e-9)
    return answer["status"]


def run_mvi_static(work,electric=False):
    folder=work/("mvi_static_electric" if electric else "mvi_static")
    folder.mkdir()
    nx,ns=9,5
    n=5 if electric else 3
    x=np.zeros((n,1,nx))
    y=np.ones((n,1,ns))
    np.savez_compressed(folder/"initial.npz",x_profiles=x,y_profiles=y)
    sources={"mechanical_load":np.zeros((3,nx,ns))}
    if electric:
        sources["free_charge"]=np.zeros((2,nx,ns))
    np.savez_compressed(folder/"load.npz",**sources)
    case=dict(schema=mvi.SCHEMA,research_opt_in=True,
              physics=common_physics(electric),
              grid=dict(nx=nx,ns=ns,length_m=1.2,radius_m=2.),
              boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                            electric="GROUNDED_BOTH" if electric else "OFF"),
              solver=dict(kind="STATIC",integrator=None,dt_s=None,
                          steps=0,field_scales=[1.]*n,
                          energy_scale_j=1.,max_sweeps=3,
                          max_acceleration_coordinates=200,
                          max_halfstep_coordinates=200),
              rank=1,initial_state_npz="initial.npz",
              loads_npz="load.npz",output_dir="out")
    path=folder/"case.json"
    path.write_text(json.dumps(case))
    result=mvi.run(path,validate_only=False)
    assert result["accepted_steps"]==1
    assert result["scientific_accuracy_certified"] is False
    if electric:
        assert result["electric_faces"]["physical_face_flux_certified"] is False
        for face in result["electric_faces"]["faces"].values():
            assert face["natural_flux_difference_abs_max_c_per_m2"]<1.e-10
    with np.load(folder/"out"/"STATIC_ACCEPTED.npz",
                 allow_pickle=False) as data:
        np.testing.assert_allclose(
            data["mechanical"],0.,rtol=0.,atol=1.e-9)
    return result["status"]


def main():
    with tempfile.TemporaryDirectory(prefix="cylinder-native-smoke-") as temp:
        folder=Path(temp)
        outcomes={
            "MKR_STATIC_KL":run_mkr(folder,"STATIC"),
            "MKR_NEWMARK_KL":run_mkr(folder,"DYNAMIC"),
            "MVI_STATIC_RANK1_KL":run_mvi_static(folder),
            "MKR_STATIC_KL_ELECTRIC":run_mkr(folder,"STATIC",electric=True),
            "MVI_STATIC_RANK1_KL_ELECTRIC":run_mvi_static(folder,electric=True),
        }
        print(json.dumps(dict(
            status="ZERO_PHYSICAL_SHELL_SMOKE_ONLY_NOT_BENCHMARK",
            scientific_accuracy_certified=False,
            outcomes=outcomes),indent=2))


if __name__=="__main__":
    main()
