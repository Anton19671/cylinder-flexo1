#!/usr/bin/env python3
"""Input-only CUF TE4 patch-pressure rank-sensitivity experiment.

The geometry, material, Bathe time grid, displacement kinematics, boundary
conditions, and weak-form source remain unchanged. Only the prescribed
pressure footprint differs from the established axisymmetric benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_linear_bathe_case import make_case

METHODS = {"FDM": None, "MVI1": 1, "MVI8": 8}
SHARED_KEYS = (
    "solver_backend", "theory", "cuf_model", "calculation_kind",
    "scale_model", "mode", "geometry_model", "material_model",
    "units", "geometry", "material", "quadrature",
    "boundary_conditions", "load_definition", "dynamics"
)

PATCH_LOAD = {
    "type": "PATCH_RADIAL_FACE",
    "kinematics": "DEAD_REFERENCE",
    "face": "MIDDLE",
    "direction": "RADIAL_INWARD",
    "q": 100000,
    "x1": 0.3, "x2": 1.6, "edge_width_x": 0.35,
    "full_circumference": False,
    "circumferential_coordinate": "THETA",
    "c1": 0.45, "c2": 3.1, "edge_width_c": 0.6
}


def experiment_cases(steps: int = 1, dt_s: float = 0.002):
    result = {}
    for name, rank in METHODS.items():
        method = "FDM" if rank is None else "MVI"
        case = make_case(
            "CUF_TE4", method, rank, steps, dt_s, "CUF_TE4_PATCH_" + name)
        case["load_definition"] = dict(PATCH_LOAD)
        if rank == 8:
            case["reference_mvi"].update({
                "dynamic_krylov_halfsteps": True,
                "dynamic_krylov_rtol": 1.e-6,
                "dynamic_krylov_restart": 50,
                "dynamic_krylov_max_matvecs": 1200
            })
        result[name] = case
    reference = result["FDM"]
    for name, case in result.items():
        for key in SHARED_KEYS:
            if case.get(key) != reference.get(key):
                raise ValueError("PATCH_PHYSICAL_CASE_MISMATCH:" + name + ":" + key)
        if name != "FDM":
            rank = METHODS[name]
            if case["rank"] != rank or case["method"] != "MVI":
                raise ValueError("PATCH_MVI_RANK_INCORRECT")
            if case["mvi_grid"] != {
                "nx": 9, "ns": 9, "operator_scheme": "SBP42_FOURIER_ODD"}:
                raise ValueError("PATCH_MVI_SPATIAL_GRID_DIFFERENT")
        else:
            if case["method"] != "FDM" or case["fdm_backend"] != "MATRIX_FREE_GMRES":
                raise ValueError("PATCH_REFERENCE_MKR_NOT_FULL_NODAL")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if type(args.steps) is not int or not 1 <= args.steps <= 120:
        raise ValueError("PATCH_STEP_COUNT_OUT_OF_RANGE")
    cases = experiment_cases(args.steps)
    args.output.mkdir(parents=True, exist_ok=True)
    case = cases[args.method]
    path = args.output / ("CUF_TE4_PATCH_" + args.method + ".json")
    if path.exists():
        raise ValueError("PATCH_CASE_REFUSES_OVERWRITE")
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print("CUF_TE4_NONAXISYMMETRIC_REAL_CASE", args.method, path, flush=True)


if __name__ == "__main__":
    main()
