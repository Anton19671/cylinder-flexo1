#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cylinder_final_stage11a_dispatcher as dispatcher


def main():
    app_dir=Path(__file__).resolve().parent
    source=app_dir.parent.parent/"result_mvi2"/"INPUT"/"FINAL_CASE.json"
    if not source.exists():
        # Release ZIP does not contain the user's result folder. Keep a small
        # embedded controller case with the same public schema.
        cfg={
            "theory":"CUF","cuf_model":"TE1","method":"MVI",
            "mode":"NONLINEAR_ELASTIC","rank":1,
            "units":{"system":"N_MM_MPA"},
            "geometry":{"L":2000.0,"R":1000.0,"h":50.0},
            "material":{"E0":210000.0,"nu0":0.3},
            "grid":{"nx":7,"ns":7,"stencil":7},
            "mvi_grid":{"nx":7,"ns":7,"stencil":7},
            "fdm_grid":{"nx":7,"ns":7,"stencil":7},
            "quadrature":{"type":"GAUSS","n_gauss":5},
            "boundary_conditions":{"left":"CLAMPED","right":"CLAMPED","advanced":None},
            "load_definition":{
                "type":"UNIFORM_RADIAL_FACE","kinematics":"DEAD_REFERENCE",
                "load_factors":[0.05,0.10],"face":"OUTER",
                "pressure_direction":"EXTERNAL","q_magnitude":10000.0,
                "q":-10000.0,
            },
            "loads_npz":None,"initial_profiles_npz":None,
            "inner_mvi":{
                "projected_tolerance":1e-7,"increment_tolerance":1e-9,
                "tangent_fd_step":1e-7,"max_halfstep_iterations":2,
                "max_xy_cycles":2,"detf_guarded_correction":True,
                "max_correction_halvings":2,
            },
            "postprocess":{"deformation_scale":1.0,"gb_ratio_thresholds":[],"control_points":[]},
        }
    else:
        cfg=json.loads(source.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="mvi_1_8_controller_") as td:
        root=Path(td)
        cfg["output_dir"]=str(root/"OUT")
        case=root/"FINAL_CASE.json"
        case.write_text(json.dumps(cfg,indent=2),encoding="utf-8")

        seen=[]
        original=dispatcher.run_final_case

        def fake_run(case_path):
            child=json.loads(Path(case_path).read_text(encoding="utf-8"))
            seen.append((int(child["rank"]),child.get("loads_npz")))
            failed=int(child["rank"]) in (3,6)
            return {
                "converged":not failed,
                "status":"SYNTHETIC_FAIL" if failed else "CONVERGED_LOAD_SERIES",
                "accepted_steps":0 if failed else 2,
                "requested_steps":2,
                "failed_step_index":1 if failed else None,
                "failed_step_status":"synthetic" if failed else None,
            }

        dispatcher.run_final_case=fake_run
        try:
            result=dispatcher.run_mvi_rank_series(case,range(1,9))
        finally:
            dispatcher.run_final_case=original

        if [r for r,_ in seen] != list(range(1,9)):
            raise AssertionError(f"series did not execute all ranks: {seen}")
        load_paths={p for _,p in seen}
        if len(load_paths)!=1:
            raise AssertionError(f"ranks did not share one load NPZ: {load_paths}")
        if not result.get("all_ranks_use_identical_load_history"):
            raise AssertionError("identical load history flag is false")
        if len(result.get("rows",[]))!=8:
            raise AssertionError("summary does not contain eight rows")

    print("MVI_1_8_COMMON_LOAD_SELF_CHECK PASS")


if __name__=="__main__":
    main()
