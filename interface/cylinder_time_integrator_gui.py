"""Headless-safe selection and dispatch of dynamic temporal integration.

The GUI selection is deliberately NOT a new physical CASE property.
Newmark remains default; Gauss IRK4 must be explicit, MVI-only,
experimental, and must never write into Newmark's output directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from cylinder_case_io import load_case_file


TIME_INTEGRATOR_CHOICES = (
    "Ньюмарк",
    "Гаусс–Рунге–Кутта IRK4 (эксперимент)",
)
_TIME_CODES = {
    TIME_INTEGRATOR_CHOICES[0]: "NEWMARK",
    TIME_INTEGRATOR_CHOICES[1]: "GAUSS2_IRK4",
    "NEWMARK": "NEWMARK",
    "GAUSS2_IRK4": "GAUSS2_IRK4",
    "MODAL": "NEWMARK",  # modal runner token; modal CASE never time-integrated
}


@dataclass(frozen=True)
class TimeIntegratorPlan:
    case_path: Path
    case_sha256: str
    integrator: str
    output_dir: Path
    calculation_kind: str
    spatial_method: str
    mvi_rank: int | None
    series: bool = False


def time_integrator_code(selection):
    try:
        return _TIME_CODES[selection]
    except (KeyError, TypeError) as exc:
        raise ValueError("Выберите Ньюмарк или Гаусс–Рунге–Кутта IRK4") from exc


def _output(case, path):
    raw=case.get("output_dir")
    if not isinstance(raw,str) or not raw.strip():
        raise ValueError("В CASE требуется отдельная папка результатов")
    result=Path(raw).expanduser()
    if not result.is_absolute():
        result=path.parent/result
    return result.resolve()


def plan_time_integrator_run(case_path, selection="NEWMARK", *, series=False):
    """Validate one saved physical CASE; never create any results directory."""
    if not isinstance(series,bool):
        raise TypeError("series must be Boolean")
    code=time_integrator_code(selection)
    path=Path(case_path).expanduser().resolve()
    raw=path.read_bytes()
    case=load_case_file(path)
    if not isinstance(case,dict):
        raise ValueError("CASE должен быть JSON-объектом")
    # Same complete physics validation as the established Newmark GUI.
    from cylinder_final_stage11a_dispatcher import validate_final_case
    info=validate_final_case(path)
    kind=info["calculation_kind"]
    method=info["method"]
    rank=case.get("rank") if method=="MVI" else None
    output=_output(case,path)
    if kind=="MODAL":
        if series:raise ValueError("MODAL does not support MVI rank series")
        if code!="NEWMARK":raise ValueError("MODAL has no time integrator")
    if code=="GAUSS2_IRK4":
        if series:
            raise ValueError("IRK4 не поддерживает серию МВИ 1–8; выберите Ньюмарк")
        if kind!="DYNAMIC" or method!="MVI":
            raise ValueError("IRK4 доступен только для динамического МВИ")
        # This preflight checks the IRK-specific rank and stage-time
        # limitations BEFORE a directory is created.
        from cylinder_gauss2_case import prepare_gauss2_mvi_case
        prepare_gauss2_mvi_case(case,base_dir=path.parent)
        output=output.with_name(output.name+"_GAUSS2_IRK4")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("GAUSS2_OUTPUT_DIR_NOT_EMPTY: результаты IRK4 не перезаписываются")
    elif series and method!="MVI":
        raise ValueError("Серия МВИ 1–8 требует метод МВИ")
    return TimeIntegratorPlan(
        path,sha256(raw).hexdigest(),
        code if kind=="DYNAMIC" else "MODAL" if kind=="MODAL" else "STATIC",
        output,kind,method,rank,series)


def execute_time_integrator_plan(plan):
    """Execute only a prevalidated, unchanged CASE without fallback."""
    if not isinstance(plan,TimeIntegratorPlan):
        raise TypeError("TimeIntegratorPlan required")
    if sha256(plan.case_path.read_bytes()).hexdigest()!=plan.case_sha256:
        raise ValueError("CASE_CHANGED_AFTER_PREFLIGHT: сохраните и проверьте CASE заново")
    if plan.integrator=="GAUSS2_IRK4":
        from cylinder_gauss2_case import run_gauss2_mvi_case
        result=run_gauss2_mvi_case(plan.case_path,output_dir=plan.output_dir)
        # The physical history writer returns a dataclass. Expose its
        # summary as a plain dict for the GUI log and Results button.
        return {**result.summary,
                "output_dir":str(result.output_dir),
                "time_integrator":"GAUSS2_IRK4"}
    if plan.integrator not in ("NEWMARK","STATIC","MODAL"):
        raise ValueError("INVALID_TIME_INTEGRATOR_PLAN")
    if plan.series:
        from cylinder_final_stage11a_dispatcher import run_mvi_rank_series
        return run_mvi_rank_series(plan.case_path)
    from cylinder_final_stage11a_dispatcher import run_final_case
    return run_final_case(plan.case_path)


__all__=("TIME_INTEGRATOR_CHOICES","TimeIntegratorPlan","time_integrator_code",
         "plan_time_integrator_run","execute_time_integrator_plan")
