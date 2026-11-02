#!/usr/bin/env python3
"""Cammalleri (2023): independent plane-stress constitutive *diagnostic*.

The production solver and material law are never modified on disk.
For LINEAR KL, uncoupled MACRO material only, construct the exact local
B.T @ Q @ B tangent after eliminating sigma_zz by a Schur complement.
Compare with the existing 3D-Lame/epsilon_zz=0 tangent on identical SBP42
grids and with identical inertia/boundary conditions. Do NOT call either
case an independently validated m=1 mode until axial profiles are checked.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np

from cylinder_paths import activate
activate()

import cylinder_modal_mvi
from cylinder_modal_reference import solve_modal_case
from cylinder_kl_g3d_weak import (
    _kl_small_virtual_F_library, _variation, gauss_thickness)
from modal_literature_diagnostics import axial_harmonic_signature

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/"config/verification/rawat2016/modal_linear/fdm_13x13/RAWAT2016_KL_13x13_MODAL_FDM.json"
OUTPUT=ROOT/"modal_cammalleri_3d_vs_plane_stress.json"
PUBLISHED={5:{"model_hz":605.,"ansys_hz":581.},
           6:{"model_hz":555.,"ansys_hz":538.},
           7:{"model_hz":611.,"ansys_hz":600.}}
GRIDS=((13,21),(17,31))


def reduced_plane_stress_matrix(material):
    """Exact 3D isotropic elasticity Schur complement sigma_zz=0.

    Engineering strains: [ex, es, ez, gamma_xs, gamma_xz, gamma_sz].
    We keep transverse shear entries; KL transverse shear should vanish
    by its kinematics and is not forcibly deleted from this material test.
    """
    lam=material.lame_lambda
    shear=material.shear_modulus
    c=np.zeros((6,6),dtype=float)
    c[:3,:3]=lam
    c[np.arange(3),np.arange(3)]+=2.*shear
    c[3,3]=c[4,4]=c[5,5]=shear
    ids=[0,1,3,4,5]
    reduced=c[np.ix_(ids,ids)]-np.outer(c[ids,2],c[2,ids])/c[2,2]
    q=np.zeros((6,6))
    q[np.ix_(ids,ids)]=reduced
    e,nu=material.young_pa,material.poisson
    np.testing.assert_allclose(q[0,0],e/(1-nu**2),rtol=1.e-13)
    np.testing.assert_allclose(q[0,1],nu*e/(1-nu**2),rtol=1.e-13)
    np.testing.assert_allclose(q[3,3],shear,rtol=1.e-13)
    if not np.all(np.linalg.eigvalsh(reduced)>0):
        raise ValueError("CAMMALLERI_PLANE_STRESS_MATRIX_NOT_POSITIVE")
    return q


def plane_stress_local_tangent(functional):
    """Build a separate diagnostic linear KL B^T Q B Hessian, never a patch."""
    f=functional
    if (f.theory!="KL" or f.geometry_model!="LINEAR" or
            f.material.ell_m!=0 or f.electrical_on):
        raise ValueError("CAMMALLERI_PLANE_STRESS_DIAGNOSTIC_REQUIRES_LINEAR_MACRO_KL")
    q=reduced_plane_stress_matrix(f.material)
    local=np.zeros((30,30))
    identity=np.eye(3)
    zero=np.zeros((3,3))
    max_normal_strain=0.
    for z,weight in zip(*gauss_thickness(f.h,f.n_gauss)):
        z=float(z)
        library=_kl_small_virtual_F_library(
            f.R,f.h,z,"LINEAR").reshape(30,4,3,3)
        strains=np.empty((30,6))
        for i,direction in enumerate(library):
            strain,_=_variation(identity,zero,zero,zero,*direction,"LINEAR")
            strains[i]=strain
        max_normal_strain=max(max_normal_strain,float(np.max(np.abs(strains[:,2]))))
        local+=float(weight)*(1.+z/f.R)*(strains@q@strains.T)
    if max_normal_strain>1.e-12:
        raise ValueError("CAMMALLERI_KL_NONZERO_TRANSVERSE_NORMAL_STRAIN")
    if not np.isfinite(local).all():
        raise ValueError("CAMMALLERI_PLANE_STRESS_NONFINITE_STIFFNESS")
    asym=np.linalg.norm(local-local.T)/max(np.linalg.norm(local),1.e-30)
    if asym>1.e-12:
        raise ValueError("CAMMALLERI_PLANE_STRESS_NON_SYMMETRIC")
    return .5*(local+local.T)


def run_case(base,nx,ns,variant):
    case=json.loads(json.dumps(base))
    case["geometry"]={"L":.305,"R":.076,"h":.000254}
    case["material"]={"E0":207e9,"nu0":.3}
    case["modal"].update({"density_kg_m3":7833,"n_modes":12})
    case["fdm_grid"]={"nx":nx,"ns":ns,
                      "operator_scheme":"SBP42_FOURIER_ODD_CSR"}
    case["output_dir"]=f"OUTPUT_CAMMALLERI_{variant}_{nx}x{ns}"
    old=cylinder_modal_mvi._kl_linear_local_tangent
    started=time.monotonic()
    try:
        if variant=="plane_stress_diagnostic":
            cylinder_modal_mvi._kl_linear_local_tangent=plane_stress_local_tangent
        elif variant!="production_3d":
            raise ValueError("Unknown constitutive diagnostic")
        result=solve_modal_case(case)
    finally:
        cylinder_modal_mvi._kl_linear_local_tangent=old
    freq=np.asarray(result["frequencies_hz"],float)
    modes=np.asarray(result["mechanical_modes"],float)
    nums=result.get("circumferential_wavenumbers")
    if nums is None or not result.get("global_spectrum_completeness_verified",False):
        raise ValueError("CAMMALLERI_SPECTRUM_NOT_VERIFIED")
    nums=np.asarray(nums,int)
    if (freq.shape!=(12,) or nums.shape!=(12,) or
            modes.shape!=(12,3,nx,ns) or
            not np.isfinite(modes).all() or
            not np.isfinite(freq).all() or
            np.any(freq<=0)):
        raise ValueError("CAMMALLERI_BAD_EIGENMODES")
    rows=[]
    for n,ref in PUBLISHED.items():
        matches=np.flatnonzero(nums==n)
        if not len(matches):
            raise ValueError("CAMMALLERI_HARMONIC_NOT_IN_FIRST_12:"+str(n))
        i=int(matches[0])
        f=float(freq[i])
        axial=axial_harmonic_signature(modes[i,2],int(n))
        rows.append({"n":n,
                     "mode_index":i+1,
                     "frequency_hz":f,
                     "paper_model_hz":ref["model_hz"],
                     "paper_ansys_hz":ref["ansys_hz"],
                     "model_difference_percent":100.*(f/ref["model_hz"]-1.),
                     "ansys_difference_percent":100.*(f/ref["ansys_hz"]-1.),
                     "axial_profile_diagnostic":axial,
                     "m1_independently_verified":False})
    filename=f"modal_cammalleri_{variant}_{nx}x{ns}_fields.npz"
    np.savez_compressed(ROOT/filename,
                        mechanical_modes=modes,frequencies_hz=freq,
                        circumferential_wavenumbers=nums)
    return {"status":"COMPUTED_NOT_INDEPENDENTLY_VALIDATED",
            "model_variant":variant,"grid":[nx,ns],
            "mechanical_law":("3D_LAME_WITH_KL_EPS_ZZ_ZERO"
                if variant=="production_3d" else
                "PLANE_STRESS_SCHUR_DIAGNOSTIC_ONLY"),
            "elapsed_s":round(time.monotonic()-started,3),
            "frequencies_hz":freq.tolist(),
            "circumferential_n":nums.tolist(),
            "rows":rows,"modes_artifact":filename}


def main():
    base=json.loads(CASE.read_text(encoding="utf-8"))
    result={
        "purpose":"Diagnose Cammalleri 2023 plane-stress versus 3D material mismatch",
        "paper_doi":"10.3390/designs7060122",
        "theory":"KL","operator":"SBP42_FOURIER_ODD_CSR",
        "production_code_changed":False,
        "acceptance_criteria_changed":False,
        "note":"Two DIFFERENT material models: their frequencies must not be conflated.",
        "runs":[],
    }
    failed=False
    for nx,ns in GRIDS:
        for variant in ("production_3d","plane_stress_diagnostic"):
            print(f"[PLANE_STRESS_DIAG] START {variant} {nx}x{ns}",flush=True)
            try:
                row=run_case(base,nx,ns,variant)
            except Exception as exc:
                failed=True
                row={"grid":[nx,ns],"model_variant":variant,
                     "status":"ERROR","error":repr(exc)}
            result["runs"].append(row)
            OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",
                              encoding="utf-8")
            print("[PLANE_STRESS_DIAG] RESULT "+
                  json.dumps(row,ensure_ascii=False),flush=True)
    if not failed:
        comparisons=[]
        for nx,ns in GRIDS:
            current=next(r for r in result["runs"] if r["grid"]==[nx,ns] and
                         r["model_variant"]=="production_3d")
            trial=next(r for r in result["runs"] if r["grid"]==[nx,ns] and
                       r["model_variant"]=="plane_stress_diagnostic")
            for i,n in enumerate(PUBLISHED):
                f0=current["rows"][i]["frequency_hz"]
                f1=trial["rows"][i]["frequency_hz"]
                comparisons.append({"grid":[nx,ns],"n":n,"production_3d_hz":f0,
                    "plane_stress_diagnostic_hz":f1,
                    "difference_percent":100.*(f1/f0-1.)})
        result["comparisons"]=comparisons
    result["diagnostic_complete"]=not failed
    OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",
                      encoding="utf-8")
    print("[PLANE_STRESS_DIAG] SUMMARY "+
          json.dumps(result,ensure_ascii=False),flush=True)
    return 1 if failed else 0

if __name__=="__main__":
    sys.exit(main())
