#!/usr/bin/env python3
"""Run user's Cammalleri modal CASE through GUI child runner; keep full log."""
import json,sys,subprocess,time,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"config/verification/cammalleri2023/user_13x21"
NAMES={"FDM":"CAMMALLERI2023_CUF_TE1_13x21_MKR_FDM.json",
       "MVI_1_AUTO":"CAMMALLERI2023_CUF_TE1_13x21_MVI_1__AUTO_20261102_181121_1d45e842.json"}
NAMES.update({f"MVI_{n}":f"CAMMALLERI2023_CUF_TE1_13x21_MVI_{n}.json" for n in range(1,9)})
def main(name):
    casepath=BASE/NAMES[name]
    case=json.loads(casepath.read_text(encoding="utf-8"))
    reportdir=ROOT/"modal_cammalleri_user_results"/name
    reportdir.mkdir(parents=True,exist_ok=True)
    output=(casepath.parent/case["output_dir"]).resolve()
    report={"name":name,"case":str(casepath.relative_to(ROOT)),
            "entrypoint":"interface/restored_gui_runner.py",
            "status":"ERROR","physics_unchanged":True}
    started=time.monotonic()
    try:
        assert case["theory"]=="CUF" and case["cuf_model"]=="TE1" and case["calculation_kind"]=="MODAL"
        assert case["modal"]["n_modes"]==12
        assert case["geometry"]=={"L":.305,"R":.076,"h":.000254}
        assert case["material"]=={"E0":207e9,"nu0":.3}
        assert case["modal"]["density_kg_m3"]==7833
        assert case["quadrature"]["n_gauss"]==9
        assert case["boundary_conditions"]["variational"]["mechanical"]["left"]["support"]=="CLAMPED"
        assert case["boundary_conditions"]["variational"]["mechanical"]["right"]["support"]=="CLAMPED"
        assert (name=="FDM" and case["method"]=="FDM" or
                name!="FDM" and case["method"]=="MVI" and case["rank"]==int(name.split("_")[1]))
        cmd=[sys.executable,str(ROOT/"interface/restored_gui_runner.py"),str(casepath),"MODAL"]
        print("[USER-CAMMALLERI] START "+name,flush=True)
        with (reportdir/"solver_stdout.log").open("w",encoding="utf-8") as log:
            with subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,encoding="utf-8",
                                  errors="replace",text=True,bufsize=1) as proc:
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if any(s in line for s in ("MODAL STEP ","MODAL SOLVE DONE","MODAL PREPARE",
                                              " MODAL FINISHED "," MODAL FAILED ")):
                        print(line,end="",flush=True)
                rc=proc.wait()
        report["exit_code"]=rc
        if rc!=0:raise RuntimeError(f"GUI child runner exit code {rc}; see solver_stdout.log")
        summary=json.loads((output/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
        if (summary.get("status")!="MODAL_COMPLETED" or summary.get("converged") is not True or
                summary.get("n_modes")!=12 or summary.get("method")!=case["method"]):
            raise ValueError("Modal completion status, method, or mode count invalid")
        if name!="FDM" and summary.get("rank")!=case["rank"]:
            raise ValueError("Wrong MVI rank returned")
        scheme=case["mvi_grid" if name!="FDM" else "fdm_grid"]["operator_scheme"]
        if summary.get("effective_operator_scheme")!=scheme:
            raise ValueError("Solver silently changed operator scheme")
        freq=np.asarray(summary["frequencies_hz"],float)
        with np.load(output/"REFERENCE_MODAL_MODES.npz",allow_pickle=False) as fields:
            modes=np.asarray(fields["mechanical_modes"],float)
            stored=np.asarray(fields["frequencies_hz"],float)
        if (freq.shape!=(12,) or modes.shape[0]!=12 or modes.shape[-2:]!=(13,21) or
                not np.isfinite(freq).all() or not np.isfinite(modes).all() or np.any(freq<=0)):
            raise ValueError("Twelve complete positive finite modes not found")
        np.testing.assert_allclose(stored,freq,rtol=1.e-10,atol=1.e-7)
        np.savez_compressed(reportdir/"computed_modes.npz",frequencies_hz=freq,mechanical_modes=modes)
        report.update(status="COMPLETED_12_MODES",frequencies_hz=freq.tolist(),
                      modes_shape=list(modes.shape),mvi_sweeps=summary.get("mvi_sweeps"))
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
    finally:
        report["wall_seconds"]=round(time.monotonic()-started,3)
        (reportdir/"case_check.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print("[USER-CAMMALLERI] RESULT "+json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report["status"]=="COMPLETED_12_MODES" else 1
if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in NAMES:raise SystemExit("Choose exact user CASE identifier")
    sys.exit(main(sys.argv[1]))
