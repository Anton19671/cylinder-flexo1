#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained Excel export for one completed cylinder-shell calculation.

Output-only module.  It never evaluates shell equations and never changes a
calculation state.  Every calculation directory receives its own workbook so
the workbook can be shared and plotted independently of other runs.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import final_units as funits

EXCEL_MAX_ROWS = 1_048_576
# Keep margin for header/filter and avoid generating sheets at Excel's hard edge.
DATA_ROWS_PER_SHEET = 900_000


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _as_number(v):
    if v is None or v == "":
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v) if isinstance(v, (float, np.floating)) else int(v)
        return x if not isinstance(x, float) or np.isfinite(x) else str(x)
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        x = float(s)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return s


def _cell(v):
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if isinstance(v, np.ndarray):
        return json.dumps(v.tolist(), ensure_ascii=False, separators=(",", ":"))
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    return _as_number(v)


def _slug(s: Any, max_len=90):
    t = re.sub(r"\s+", "_", str(s or "RUN").strip())
    t = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._+-]+", "_", t).strip("._ ")
    return (t or "RUN")[:max_len]


def _flatten(obj: Any, prefix=""):
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(_flatten(v, p))
    elif isinstance(obj, (list, tuple)):
        # Short scalar lists are easier to use as one value. Structured lists
        # are expanded so no input information is lost.
        if all(not isinstance(x, (dict, list, tuple)) for x in obj) and len(obj) <= 20:
            rows.append((prefix, json.dumps(obj, ensure_ascii=False)))
        else:
            for i, v in enumerate(obj):
                rows.extend(_flatten(v, f"{prefix}[{i}]"))
    else:
        rows.append((prefix, _cell(obj)))
    return rows


def _run_name(root: Path):
    info = _read_json(root / "RUN_INFO.json", {}) or {}
    if info.get("run_name"):
        return str(info["run_name"])
    # MVI rank-series outputs are stored as MVI_N/OUT.  Use the rank folder in
    # the standalone Excel filename instead of the non-descriptive word OUT.
    if root.name.upper()=="OUT" and root.parent.name.upper().startswith("MVI_"):
        return root.parent.name
    return root.name


def _file_prefix(root: Path):
    info=_read_json(root/"RUN_INFO.json",{}) or {}
    try:run=f"RUN_{int(info.get('run_number')):03d}"
    except Exception:
        m=re.match(r"^(RUN_\d+)",str(info.get("run_name") or root.name),re.IGNORECASE)
        run=(m.group(1).upper() if m else _slug(root.name,20))
    variant=_slug(info.get("variant_name") or info.get("run_name") or root.name,54)
    if variant.upper().startswith(run.upper()):
        return variant[:64].rstrip("._ ")
    return f"{run}__{variant}"[:64].rstrip("._ ")


def _summary(root: Path):
    for p in (root / "FINAL_RUN_SUMMARY.json", root / "CASE_RUN_SUMMARY.json", root / "RUN_RESULT.json", root / "SERIES_SUMMARY.json"):
        j = _read_json(p, None)
        if isinstance(j, dict):
            return j
    return {}


def _case(root: Path):
    for p in (root / "INPUT" / "FINAL_CASE.json", root / "FINAL_CASE.json", root / "RUN_PARAMETERS.json"):
        j = _read_json(p, None)
        if isinstance(j, dict): return j
    return {}


def _display_context(root: Path, summary=None, case=None):
    summary=_summary(root) if summary is None else summary
    case=_case(root) if case is None else case
    units=(case.get("units") or summary.get("units") or {})
    system=str(units.get("system","N_MM_MPA"))
    labels=funits.unit_labels(system)
    load_definition=summary.get("load_definition") or case.get("load_definition") or {}
    _,load_label,load_key=funits.load_axis([0.0],load_definition,system)
    return system,labels,load_definition,load_label,load_key


def _header_label(key,labels,load_label):
    """English-first bilingual Excel header with explicit units."""
    k=str(key); low=k.lower()
    load_bi=f"Load / Нагрузка {load_label}"
    exact={
        "step":"Step / Шаг [-]","step_index":"Step / Шаг [-]","load_factor":"Load factor λ / Коэффициент нагрузки λ [-]",
        "load_value":load_bi,"w_abs_max":f"Maximum deflection |w|max / Максимальный прогиб |w|max [{labels['displacement']}]",
        "w_at_abs_max":f"Deflection at |w|max / Прогиб в точке |w|max [{labels['displacement']}]",
        "w_reference":f"Reference deflection w_reference / Контрольный прогиб w_reference [{labels['displacement']}]",
        "w_abs_reference":f"Reference deflection magnitude |w_reference| / Модуль контрольного прогиба [{labels['displacement']}]",
        "w_center_mid":f"Diagnostic center deflection w_center / Диагностический прогиб центра [{labels['displacement']}]",
        "w_abs_center_mid":f"Center deflection magnitude |w_center| / Модуль прогиба центра [{labels['displacement']}]",
        "w_min":f"Minimum deflection w_min / Минимальный прогиб [{labels['displacement']}]",
        "w_max":f"Maximum deflection w_max / Максимальный прогиб [{labels['displacement']}]",
        "x":f"Axial coordinate x / Продольная координата x [{labels['length']}]","s":f"Circumferential coordinate s / Окружная координата s [{labels['length']}]","z":f"Thickness coordinate z / Координата по толщине z [{labels['length']}]",
        "wmax_x":f"x at |w|max / x в точке |w|max [{labels['length']}]","wmax_s":f"s at |w|max / s в точке |w|max [{labels['length']}]",
        "center_x":f"Center x / x центра [{labels['length']}]","center_s":f"Center s / s центра [{labels['length']}]",
        "reference_x":f"Reference x / Контрольная x [{labels['length']}]","reference_s":f"Reference s / Контрольная s [{labels['length']}]",
        "theta_rad":"Angle θ / Угол θ [rad]","wmax_theta":"θ at |w|max / θ в точке |w|max [rad]","center_theta":"Center θ / θ центра [rad]","reference_theta":"Reference θ / Контрольная θ [rad]",
        "u":f"Axial displacement u / Продольное перемещение u [{labels['displacement']}]","v":f"Circumferential displacement v / Окружное перемещение v [{labels['displacement']}]","w":f"Radial displacement w / Радиальный прогиб w [{labels['displacement']}]",
        "u_mid":f"Midsurface u / u средней поверхности [{labels['displacement']}]","v_mid":f"Midsurface v / v средней поверхности [{labels['displacement']}]","w_mid":f"Midsurface w / w средней поверхности [{labels['displacement']}]",
        "radial_surface_load":f"Radial load q / Радиальная нагрузка q [{labels['pressure']}]",
        "load_quantity":"Load physical quantity / Физическая величина нагрузки [-]","load_unit":"Load unit / Единица нагрузки [-]",
        "epsilon_i":"Strain intensity ε_i / Интенсивность деформаций ε_i [-]","sigma_i":f"Stress intensity σ_i / Интенсивность напряжений σ_i [{labels['stress']}]",
        "detf":"det F [-]","detf_min":"min det F [-]","detf_max":"max det F [-]","max_abs_j_minus_1":"max |J−1| [-]","gb":f"Birger shear modulus G_B / Модуль сдвига Биргера G_B [{labels['stress']}]","gb_over_g0":"G_B/G_0 [-]","gb_ratio_min":"min G_B/G_0 [-]","gb_ratio_max":"max G_B/G_0 [-]",
        "birger_nonlinear":"Birger nonlinear zone / Нелинейная зона Биргера [0/1]","plastic":"Plastic zone / Пластическая зона [0/1]",
        "plastic_thickness_fraction":"Plastic thickness fraction h_p/h / Доля пластической толщины [-]",
        "plastic_volume_fraction":"Plastic volume fraction Vp/V / Доля пластического объёма [-]","birger_nonlinear_volume_fraction":"Birger nonlinear volume fraction Vnl/V / Доля нелинейного объёма [-]",
        "elapsed_s":"Elapsed time / Время от начала [s]","elapsed_step_s":"Step time / Время шага [s]","elapsed_cumulative_s":"Cumulative time / Накопленное время [s]",
        "ix":"Index ix / Индекс ix [-]","is":"Index is / Индекс is [-]","g":"Thickness point g / Точка по толщине g [-]","tau":"Index τ / Индекс τ [-]",
        "status":"Status / Статус [-]","converged":"Converged / Сходимость [True/False]","iterations":"Iterations / Итерации [-]",
        "file":"File / Файл [-]","state_file":"State file / Файл состояния [-]","derived_file":"Derived-results file / Файл производных результатов [-]","kind":"Data type / Тип данных [-]","arrays":"Arrays / Массивы [-]",
        "residual_inf":"Residual inf / Невязка inf [solver coherent units]","residual_convergence_inf":"Convergence residual inf / Невязка сходимости inf [solver coherent units]","increment_inf":"Increment inf / Приращение inf [solver coherent units]",
        "max_epsilon_i":"max ε_i [-]","solver_iterations":"Solver iterations / Итерации решателя [-]","outer_iterations":"Outer iterations / Внешние итерации [-]",
    }
    if low in exact:return exact[low]
    if low.startswith(("eps_","strain","epsilon")) or low in ("residual_relative","residual_convergence_relative","field_increment_rel"):
        return f"{k} [-]"
    if low.startswith(("sig_","stress")) or "sigma" in low:return f"{k} [{labels['stress']}]"
    if low.startswith(("eta","gb_ratio","plastic_","birger_nonlinear_")):return f"{k} [-]"
    if low.endswith(("_x","_s","_z")):return f"{k} [{labels['length']}]"
    if low.endswith("_theta"):return f"{k} [rad]"
    if "iteration" in low or "cycle" in low or low in ("ix","is","rank","component"):return f"{k} [-]"
    return k



def _result_measure_names(summary: dict):
    """Human-readable strain/stress measures used by the saved derived arrays."""
    geometry=str((summary or {}).get("geometry_model") or "").upper().strip()
    theory=str((summary or {}).get("theory") or "").upper().strip()
    if theory=="CUF" and geometry=="GREEN_LAGRANGE":
        return (
            "деформация Грина–Лагранжа E",
            "2-й тензор напряжений Пиолы–Кирхгофа S",
        )
    if theory=="CUF" and geometry=="VON_KARMAN":
        return (
            "деформация ветви von Kármán (массив E решателя)",
            "сопряжённое напряжение ветви von Kármán (массив S решателя)",
        )
    if theory=="CUF" and geometry=="LINEAR":
        return (
            "линейная деформация (массив E решателя)",
            "линейное упругое напряжение (массив S решателя)",
        )
    return ("деформация", "напряжение")


def _is_birger_mode(summary: dict) -> bool:
    mode=str((summary or {}).get("mode") or "").upper()
    material=str((summary or {}).get("material_model") or "").upper()
    return mode.endswith("BIRGER") or material=="BIRGER"


def _field_meta(code: Any, summary: dict, labels: dict):
    """Return (readable_name, unit, meaning) for result field codes."""
    c=str(code or "").strip(); low=c.lower()
    strain_measure,stress_measure=_result_measure_names(summary)
    comp={
        "xx":("продольная/осевая", "x–x"),
        "ss":("окружная", "s–s"),
        "rr":("радиальная/по толщине", "r–r"),
        "xs":("продольно-окружная сдвиговая", "x–s"),
        "xr":("продольно-радиальная сдвиговая", "x–r"),
        "sr":("окружно-радиальная сдвиговая", "s–r"),
    }
    if c.startswith("E_") and c[2:] in comp:
        desc,axes=comp[c[2:]]
        extra=" Для недиагональных компонент это тензорная сдвиговая компонента; инженерная γ = 2E_ij."
        return (f"{c} — {desc} компонента тензора: {strain_measure}", "-", f"Компонента {axes}.{extra if c[2:] in ('xs','xr','sr') else ''}")
    if c.startswith("S_") and c[2:] in comp:
        desc,axes=comp[c[2:]]
        return (f"{c} — {desc} компонента тензора: {stress_measure}", labels["stress"], f"Компонента {axes}; для GREEN_LAGRANGE это не Cauchy stress, а S (2PK).")
    if low.startswith("eps_"):
        suffix=low[4:]
        return (f"{c} — компонента деформации", "-", f"Компонента {suffix}; точная мера определяется выбранной кинематикой.")
    if low.startswith("sig_") or low.startswith("tau_"):
        return (f"{c} — компонента напряжения", labels["stress"], "Компонента напряжений выбранной теории.")
    exact={
        "epsilon_i":("ε_i — интенсивность деформаций", "-", "Инвариантная интенсивность деформаций, используемая решателем."),
        "sigma_i":("σ_i — интенсивность напряжений", labels["stress"], "Инвариантная интенсивность напряжений, используемая решателем."),
        "detF":("detF = det(F) — якобиан деформации", "-", "Контроль допустимости отображения; для малых деформаций близок к 1."),
        "GB":("G_B — текущий модуль сдвига Биргера", labels["stress"], "Выводится только при включённой модели Birger."),
        "GB_ratio":("G_B/G0 — относительный модуль сдвига", "-", "Поле модели Birger: 1 — исходный модуль; <1 — снижение модуля."),
        "u":("u — продольное перемещение", labels["displacement"], "Перемещение вдоль оси x."),
        "v":("v — окружное перемещение", labels["displacement"], "Перемещение вдоль окружной координаты s."),
        "w":("w — радиальное перемещение/прогиб", labels["displacement"], "Положительное направление определяется соглашением решателя; внешнее давление обычно даёт отрицательный w."),
        "plastic_mask":("Пластическая зона", "0/1", "1 — точка относится к пластической зоне J2; 0 — нет."),
        "birger_nonlinear_mask":("Нелинейная зона Биргера", "0/1", "1 — активна нелинейная ветвь Биргера; 0 — нет."),
        "unloading_mask":("Зона разгрузки Биргера", "0/1", "1 — обнаружена разгрузка по принятому критерию."),
    }
    if c in exact:return exact[c]
    return (c, "-", "Техническое поле. См. README и параметры конкретной теории.")


def _write_field_legend(book: _Book, summary: dict, labels: dict):
    """A compact Russian legend so every technical code in Excel is interpretable."""
    ws=book.wb.add_worksheet("Field_Legend_Поля")
    headers=["Code / Код","Full name / Полное название","Units / Единицы","Meaning / Что означает"]
    for col,h in enumerate(headers):ws.write(0,col,h,book.header)
    codes=[
        "u","v","w","E_xx","E_ss","E_rr","E_xs","E_xr","E_sr",
        "S_xx","S_ss","S_rr","S_xs","S_xr","S_sr",
        "epsilon_i","sigma_i","detF",
    ]
    if _is_birger_mode(summary):
        codes.extend(["GB","GB_ratio","birger_nonlinear_mask","unloading_mask"])
    codes.append("plastic_mask")
    for r,code in enumerate(codes,1):
        name,unit,meaning=_field_meta(code,summary,labels)
        ws.write(r,0,code);ws.write(r,1,name,book.wrap);ws.write(r,2,unit);ws.write(r,3,meaning,book.wrap)
        ws.set_row(r,34)
    strain_measure,stress_measure=_result_measure_names(summary)
    start=len(codes)+3
    notes=[
        ("Система координат", "x — вдоль оси цилиндра; s=R·θ — по окружности; r — радиально; z — координата по толщине относительно средней поверхности, r=R+z."),
        ("Деформации", f"В этой постановке основной сохранённый тензор: {strain_measure}."),
        ("Напряжения", f"В этой постановке основной сохранённый тензор: {stress_measure}."),
        ("Почему поле может выглядеть плоским по окружности", "При равномерной радиальной нагрузке на замкнутый цилиндр и осесимметричных закреплениях решение должно быть практически независимо от θ/s. Это нормальная осесимметрия; проверяйте изменение по x и z."),
        ("Sections_XZ", "Сечение x–z при окружном индексе is точки max|w|."),
        ("Sections_SZ", "Сечение θ/s–z при продольном индексе ix точки max|w|. Для осесимметричной задачи по θ оно ожидаемо почти постоянное."),
        ("Profiles_Z", "Профиль по толщине z в узле (is,ix), выбранном по max|w|."),
    ]
    ws.write(start,0,"IMPORTANT NOTES / ВАЖНЫЕ ПОЯСНЕНИЯ",book.sub)
    for i,(k,v) in enumerate(notes,start+1):
        ws.write(i,0,k,book.sub);ws.write(i,1,v,book.wrap)
    ws.freeze_panes(1,0);ws.set_column(0,0,28);ws.set_column(1,1,62);ws.set_column(2,2,16);ws.set_column(3,3,82)

def _power_unit(base,exponent):
    exponent=int(exponent)
    if exponent==0:return "-"
    if exponent==1:return base
    if exponent==-1:return f"1/{base}"
    if exponent>1:return f"{base}^{exponent}"
    return f"1/{base}^{abs(exponent)}"


def _cuf_coefficient_unit(tau,labels):
    # u(z)=sum(z^tau*q_tau), hence [q_tau]=L^(1-tau).
    return _power_unit(labels["length"],1-int(tau))


def _resultant_unit(name,tau,labels):
    n=str(name)
    if n in ("Nx","Ns","Nxs"):return labels["line_load"]
    if n in ("Mx","Ms","Mxs"):return labels["force"]
    if tau is None:return "coherent resultant units"
    p=int(tau)
    # CUF F_tau=z^tau: A-type integrals are F*L^(tau-1), while
    # algebraic C terms carry one additional inverse length.
    exponent=p-1 if n.startswith("A") else p-2
    L=labels["length"]
    if exponent==0:return labels["force"]
    if exponent>0:return f"{labels['force']}·{_power_unit(L,exponent)}"
    return f"{labels['force']}/{_power_unit(L,-exponent)}"


def _state_files(root: Path):
    q = sorted(root.glob("STEP_*_STATE.npz"))
    if q: return q
    return sorted(p for p in root.glob("STEP_*.npz") if "STATE" not in p.name)


def _step_number(path: Path):
    m = re.search(r"STEP_(\d+)", path.name)
    return int(m.group(1)) if m else 0


def _state_mid_fields(path: Path):
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if "q" in files:
            q = np.asarray(z["q"], float)
            return q[0, 0], q[1, 0], q[2, 0]
        if "U" in files:
            U = np.asarray(z["U"], float)
            return U[0], U[1], U[2]
        if {"X", "Y"}.issubset(files):
            X = np.asarray(z["X"], float); Y = np.asarray(z["Y"], float)
            if X.ndim == 3:
                U = np.einsum("crx,crs->csx", X, Y)
                return U[0], U[1], U[2]
            if X.ndim == 4:
                q = np.einsum("cmrx,cmrs->cmsx", X, Y)
                return q[0, 0], q[1, 0], q[2, 0]
    return None, None, None


def _load_factor_map(root: Path):
    rows = _read_csv(root / "NUMERIC" / "RESULTS_SUMMARY.csv") or _read_csv(root / "NUMERIC" / "STEP_SUMMARY.csv")
    out = {}
    for i, r in enumerate(rows, 1):
        try: step = int(float(r.get("step", i) or i))
        except Exception: step = i
        try: lf = float(r.get("load_factor", step) or step)
        except Exception: lf = float(step)
        out[step] = lf
    if out: return out
    for p in sorted(root.glob("STEP_*.json")):
        j = _read_json(p, {}) or {}; step = _step_number(p)
        try: out[step] = float(j.get("load_factor", step))
        except Exception: out[step] = float(step)
    return out


class _Book:
    def __init__(self, path: Path):
        try:
            import xlsxwriter
        except Exception as exc:
            raise RuntimeError("Excel export requires package 'XlsxWriter'. Run INSTALL_REQUIREMENTS.cmd.") from exc
        self.xlsxwriter = xlsxwriter
        self.path = path
        self.wb = xlsxwriter.Workbook(str(path), {"constant_memory": True})
        self.wb.set_properties({
            "title": f"Cylinder solver results — {path.stem}",
            "subject": "Independent results workbook for one cylinder-shell calculation",
            "author": "Cylinder Solver",
            "comments": "Generated from saved calculation results; no external workbook links.",
        })
        self.header = self.wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "text_wrap": True})
        self.title = self.wb.add_format({"bold": True, "font_size": 14})
        self.sub = self.wb.add_format({"bold": True, "bg_color": "#EDEDED"})
        self.num = self.wb.add_format({"num_format": "0.0000000000E+00"})
        self.wrap = self.wb.add_format({"text_wrap": True, "valign": "top"})
        self.export_warnings = []

    def warn(self, sheet: str, item: Any, exc: Exception):
        self.export_warnings.append({
            "sheet": str(sheet),
            "item": str(item),
            "error": f"{type(exc).__name__}: {exc}",
        })

    def close(self):
        self.wb.close()

    def simple_sheet(self, name: str, headers: list[str], rows: Iterable[Iterable[Any]], widths=None):
        ws = self.wb.add_worksheet(name[:31])
        for c, h in enumerate(headers): ws.write(0, c, h, self.header)
        rr = 1
        for row in rows:
            for c, v in enumerate(row): ws.write(rr, c, _cell(v))
            rr += 1
        ws.freeze_panes(1, 0)
        if headers: ws.autofilter(0, 0, max(0, rr-1), len(headers)-1)
        if widths:
            for c, w in enumerate(widths): ws.set_column(c, c, w)
        else:
            for c in range(len(headers)): ws.set_column(c, c, 15)
        return ws, rr


class _SplitLongWriter:
    def __init__(self, book: _Book, base_name: str, headers: list[str], widths=None):
        self.book = book; self.base = base_name; self.headers = headers; self.widths = widths
        self.part = 0; self.row = DATA_ROWS_PER_SHEET + 2; self.ws = None

    def _new(self):
        self.part += 1
        suffix = "" if self.part == 1 else f"_{self.part}"
        name = (self.base[:31-len(suffix)] + suffix)[:31]
        self.ws = self.book.wb.add_worksheet(name)
        for c, h in enumerate(self.headers): self.ws.write(0, c, h, self.book.header)
        self.ws.freeze_panes(1, 0); self.ws.autofilter(0, 0, DATA_ROWS_PER_SHEET, len(self.headers)-1)
        widths = self.widths or [14] * len(self.headers)
        for c, w in enumerate(widths): self.ws.set_column(c, c, w)
        self.row = 1

    def write(self, values):
        if self.ws is None or self.row > DATA_ROWS_PER_SHEET:
            self._new()
        for c, v in enumerate(values): self.ws.write(self.row, c, _cell(v))
        self.row += 1


def _write_readme(book: _Book, root: Path, run_name: str, summary: dict, case: dict):
    ws = book.wb.add_worksheet("README")
    ws.write("A1", "CYLINDER SOLVER — ONE CALCULATION / ONE EXCEL FILE / ОДИН РАСЧЁТ — ОДНА КНИГА", book.title)
    lines = [
        ("Run name / Имя расчёта", run_name),
        ("Results folder / Папка результатов", str(root)),
        ("Created / Создано", datetime.now().isoformat(timespec="seconds")),
        ("Theory / Теория", summary.get("theory", case.get("theory"))),
        ("CUF model / Модель CUF", summary.get("cuf_model", case.get("cuf_model"))),
        ("Method / Метод", summary.get("method", case.get("method"))),
        ("Mode / Режим", summary.get("mode", case.get("mode"))),
        ("Unit system / Система единиц", ((case.get("units") or {}).get("system") if isinstance(case.get("units"), dict) else summary.get("units"))),
        ("Purpose / Назначение", "This workbook is self-contained for independent plotting/inspection. / Книга автономна для проверки и построения графиков. Raw NPZ/JSON remain authoritative / исходные NPZ/JSON являются эталонными массивами."),
        ("Excel row policy / Ограничение строк", f"Long 3-D tables are automatically split after {DATA_ROWS_PER_SHEET:,} data rows per sheet. / Длинные 3D-таблицы автоматически делятся по листам."),
        ("Coordinates / Координаты", "x — cylinder axis / ось цилиндра; s=R·θ — circumferential / окружная; r — radial / радиально; z — through thickness from midsurface / по толщине от средней поверхности."),
        ("Main sheets / Главные листы", "Load_Deflection — load–deflection / нагрузка–прогиб; Surface_Fields — midsurface fields / поля средней поверхности; Thickness_Fields — through-thickness fields / поля по толщине; Sections_XZ / Sections_SZ — sections / сечения; Profiles_Z — thickness profile / профиль по толщине; Field_Legend_Поля — field codes / расшифровка кодов."),
        ("Green–Lagrange note / Важно для Green–Lagrange", "E_* — Green–Lagrange strain components / компоненты деформации Green–Lagrange; S_* — second Piola–Kirchhoff stress / 2-й тензор Пиолы–Кирхгофа. S_* is not automatically Cauchy stress / S_* нельзя автоматически трактовать как Cauchy stress."),
    ]
    for r, (k, v) in enumerate(lines, 3):
        ws.write(r-1, 0, k, book.sub); ws.write(r-1, 1, _cell(v), book.wrap)
    ws.set_column("A:A", 25); ws.set_column("B:B", 95); ws.freeze_panes(2, 0)


def _write_parameters(book: _Book, case: dict, summary: dict):
    rows = [["CASE." + k, v] for k, v in _flatten(case)]
    rows += [["SUMMARY." + k, v] for k, v in _flatten(summary)]
    book.simple_sheet("Parameters", ["parameter", "value"], rows, [48, 80])


def _write_step_tables(book: _Book, root: Path):
    summary=_summary(root);system,labels,load_definition,load_label,_=_display_context(root,summary)
    rows = _read_csv(root / "NUMERIC" / "RESULTS_SUMMARY.csv") or _read_csv(root / "NUMERIC" / "STEP_SUMMARY.csv")
    if not rows:
        rows=[]
        for p in sorted(root.glob("STEP_*.json")):
            j=_read_json(p,{}) or {}
            if not isinstance(j,dict): continue
            row=dict(j); row.setdefault("step",_step_number(p)); rows.append(row)
    if not rows: return
    headers = list(rows[0].keys())
    book.simple_sheet("Step_Summary", [_header_label(h,labels,load_label) for h in headers], ([r.get(h) for h in headers] for r in rows), [18]*len(headers))

    ld = book.wb.add_worksheet("Load_Deflection")
    hdr = ["Step / Шаг [-]", "Load factor λ / Коэффициент нагрузки λ [-]", f"Load / Нагрузка {load_label}",
           f"Global maximum |w|max / Глобальный максимум |w|max [{labels['displacement']}]",
           f"w at global maximum / w в точке глобального максимума [{labels['displacement']}]",
           f"Reference deflection w_reference (x=L/2, theta=0) / Контрольный прогиб [{labels['displacement']}]",
           f"|w_reference| / Модуль контрольного прогиба [{labels['displacement']}]",
           f"Diagnostic index-center w_center / Диагностический индексный центр [{labels['displacement']}]",
           f"|w_center| / Модуль прогиба центра [{labels['displacement']}]",
           f"x(|w|max) [{labels['length']}]", "theta(|w|max) [rad]",
           f"x(reference) / x(контрольная) [{labels['length']}]", "theta(reference) / theta(контрольная) [rad]",
           f"x(center) / x(центр) [{labels['length']}]", "theta(center) / theta(центр) [rad]",
           f"max|u| [{labels['displacement']}]", f"max|v| [{labels['displacement']}]",
           "Rproj [-]", "XY cycles / Циклы XY [-]", "Step time / Время шага [s]", "Status / Статус [-]"]
    for c,h in enumerate(hdr): ld.write(0,c,h,book.header)
    # Explicit origin requested by GUI/results convention.
    ld.write_row(1,0,[0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"","","","ORIGIN"])
    for i,r in enumerate(rows,2):
        factor=float(r.get("load_factor",i-1) or 0.0)
        actual=funits.load_axis([factor],load_definition,system)[0][0]
        rproj=r.get("residual_convergence_relative",r.get("residual_relative",r.get("Rproj_rel")))
        xy=r.get("xy_cycles",r.get("cycles",r.get("iterations")))
        ld.write_row(i,0,[_cell(r.get("step",i-1)),factor,actual,_cell(r.get("w_abs_max")),_cell(r.get("w_at_abs_max")),
                          _cell(r.get("w_reference")),_cell(r.get("w_abs_reference")),
                          _cell(r.get("w_center_mid")),_cell(r.get("w_abs_center_mid")),
                          _cell(r.get("wmax_x")),_cell(r.get("wmax_theta")),_cell(r.get("reference_x")),_cell(r.get("reference_theta")),
                          _cell(r.get("center_x")),_cell(r.get("center_theta")),_cell(r.get("u_abs_max")),_cell(r.get("v_abs_max")),
                          _cell(rproj),_cell(xy),_cell(r.get("elapsed_step_s")),_cell(r.get("status",r.get("solver_status")))])
    ld.freeze_panes(1,0); ld.autofilter(0,0,len(rows)+1,len(hdr)-1); ld.set_column(0,len(hdr)-1,20)
    try:
        chart = book.wb.add_chart({"type":"scatter", "subtype":"straight_with_markers"})
        chart.add_series({
            "name":"Global |w|max",
            "categories":["Load_Deflection",1,3,len(rows)+1,3],
            "values":["Load_Deflection",1,2,len(rows)+1,2],
            "marker":{"type":"circle","size":5},
        })
        chart.add_series({
            "name":"Reference |w|: x=L/2, theta=0",
            "categories":["Load_Deflection",1,6,len(rows)+1,6],
            "values":["Load_Deflection",1,2,len(rows)+1,2],
            "marker":{"type":"diamond","size":5},
        })
        chart.add_series({
            "name":"Diagnostic |w_center|",
            "categories":["Load_Deflection",1,8,len(rows)+1,8],
            "values":["Load_Deflection",1,2,len(rows)+1,2],
            "marker":{"type":"square","size":5},
        })
        chart.set_title({"name":"Нагрузка–прогиб"}); chart.set_x_axis({"name":f"deflection [{labels['displacement']}]"}); chart.set_y_axis({"name":load_label})
        ld.insert_chart("G2", chart, {"x_scale":1.25,"y_scale":1.15})
    except Exception as exc:
        book.warn("Load_Deflection", "chart", exc)


def _write_progress(book: _Book, root: Path):
    _,labels,_,load_label,_=_display_context(root)
    p = root / "NUMERIC" / "ONLINE_HISTORY.csv"
    rows = _read_csv(p)
    if not rows:
        # Postprocess may be disabled/still running; fall back to JSONL.
        jp = root / "LIVE_PROGRESS.jsonl"
        if jp.exists():
            for line_no,ln in enumerate(jp.read_text(encoding="utf-8", errors="replace").splitlines(),1):
                try:
                    j=json.loads(ln)
                    if isinstance(j,dict):rows.append(j)
                except Exception as exc:
                    book.warn("Online_History", f"LIVE_PROGRESS.jsonl:{line_no}", exc)
    if not rows: return
    headers = sorted(set().union(*(r.keys() for r in rows)))
    display_headers=[_header_label(h,labels,load_label) for h in headers]
    book.simple_sheet("Online_History", display_headers, ([r.get(h) for h in headers] for r in rows), [18]*len(headers))
    groups = [
        ("Newton_History", {"newton_iteration"}),
        ("MVI_History", {"mvi_halfstep","mvi_cycle"}),
        ("Birger_History", {"birger_iteration","birger_outer","birger_outer_iteration"}),
    ]
    for name, events in groups:
        rr=[r for r in rows if str(r.get("event")) in events]
        if rr: book.simple_sheet(name,display_headers,([r.get(h) for h in headers] for r in rr),[18]*len(headers))


def _write_control_points(book: _Book, root: Path):
    rows = _read_csv(root / "NUMERIC" / "CONTROL_POINTS.csv")
    if rows:
        _,labels,_,load_label,_=_display_context(root)
        headers=list(rows[0].keys()); book.simple_sheet("Control_Points",[_header_label(h,labels,load_label) for h in headers],([r.get(h) for h in headers] for r in rows),[19]*len(headers))


def _grid_coords(summary: dict, shape):
    ns,nx = shape
    g = summary.get("geometry") or {}
    L=float(g.get("L", nx-1)); R=float(g.get("R",1.0))
    xx=np.linspace(0,L,nx); theta=np.linspace(0,2*np.pi,ns,endpoint=False); ss=R*theta
    return xx,ss,theta


def _write_surface_fields(book: _Book, root: Path, summary: dict):
    system,labels,load_definition,load_label,_=_display_context(root,summary)
    lfmap=_load_factor_map(root)
    wr=_SplitLongWriter(book,"Surface_Fields",["Шаг [-]","Коэффициент нагрузки λ [-]",load_label,"Индекс ix [-]","Индекс is [-]",f"x [{labels['length']}]",f"s [{labels['length']}]","θ [rad]",f"u_mid [{labels['displacement']}]",f"v_mid [{labels['displacement']}]",f"w_mid [{labels['displacement']}]",f"Радиальная нагрузка q [{labels['pressure']}]"] ,[10,16,18,10,10,14,14,14,18,18,18,20])
    wrote=False
    for sp in _state_files(root):
        step=_step_number(sp); u,v,w=_state_mid_fields(sp)
        if w is None: continue
        w=np.asarray(w,float);u=np.asarray(u,float);v=np.asarray(v,float);xx,ss,th=_grid_coords(summary,w.shape)
        radial=None; dp=root/"NUMERIC"/f"STEP_{step:04d}_DERIVED.npz"
        if dp.exists():
            try:
                with np.load(dp,allow_pickle=False) as z:
                    if "radial_surface_load" in z.files: radial=np.asarray(z["radial_surface_load"],float)
            except Exception as exc:
                book.warn("Surface_Fields", dp.name, exc)
        for js in range(w.shape[0]):
            for ix in range(w.shape[1]):
                factor=lfmap.get(step,step); actual=funits.load_axis([factor],load_definition,system)[0][0]
                wr.write([step,factor,actual,ix,js,xx[ix],ss[js],th[js],u[js,ix],v[js,ix],w[js,ix],None if radial is None else radial[js,ix]])
                wrote=True
    return wrote


def _write_cuf_coefficients(book: _Book, root: Path, summary: dict):
    system,labels,load_definition,load_label,_=_display_context(root,summary)
    lfmap=_load_factor_map(root); wr=None; wrote=False
    for dp in sorted((root/"NUMERIC").glob("STEP_*_DERIVED.npz")):
        step=_step_number(dp)
        try:
            with np.load(dp,allow_pickle=False) as z:
                if "q_coefficients" not in z.files: continue
                q=np.asarray(z["q_coefficients"],float)
        except Exception as exc:
            book.warn("CUF_Coefficients", dp.name, exc)
            continue
        _,m,ns,nx=q.shape;xx,ss,th=_grid_coords(summary,(ns,nx))
        if wr is None: wr=_SplitLongWriter(book,"CUF_Coefficients",["Шаг [-]","Коэффициент нагрузки λ [-]",load_label,"Компонента [-]","Индекс τ [-]","Индекс ix [-]","Индекс is [-]",f"x [{labels['length']}]",f"s [{labels['length']}]","θ [rad]","Размерность коэффициента","Коэффициент CUF"] ,[10,16,18,12,10,10,10,14,14,14,22,20])
        factor=lfmap.get(step,step);actual=funits.load_axis([factor],load_definition,system)[0][0]
        for c,name in enumerate(("u","v","w")):
            for tau in range(m):
                for js in range(ns):
                    for ix in range(nx):
                        wr.write([step,factor,actual,name,tau,ix,js,xx[ix],ss[js],th[js],_cuf_coefficient_unit(tau,labels),q[c,tau,js,ix]])
                        wrote=True
    return wrote


def _write_thickness_fields(book: _Book, root: Path, summary: dict):
    """Export physical through-thickness arrays in long form.

    For CUF, displacements at Gauss points are exported together with all six
    Green/linear strain and stress components.  For KL the stored strain/stress
    components are exported; no uncomputed through-thickness displacement is
    invented.
    """
    system,labels,load_definition,load_label,_=_display_context(root,summary)
    lfmap=_load_factor_map(root)
    headers=[
        "Шаг [-]","Коэффициент нагрузки λ [-]",load_label,"Точка по толщине g [-]",f"z [{labels['length']}]","Индекс ix [-]","Индекс is [-]",f"x [{labels['length']}]",f"s [{labels['length']}]","θ [rad]",
        f"u(x,θ,z) — axial displacement [{labels['displacement']}]",f"v(x,θ,z) — circumferential displacement [{labels['displacement']}]",f"w(x,θ,z) — radial displacement [{labels['displacement']}]",
        "E_xx — продольная деформация [-]","E_ss — окружная деформация [-]","E_rr — радиальная деформация [-]","E_xs — тензорный сдвиг x-s [-]","E_xr — тензорный сдвиг x-r [-]","E_sr — тензорный сдвиг s-r [-]",
        f"σ_x = S_xx — продольное / axial normal stress [{labels['stress']}]",f"σ_θ = S_θθ — окружное / circumferential normal stress [{labels['stress']}]",f"σ_r = S_rr — радиальное / radial normal stress [{labels['stress']}]",f"τ_xθ = S_xθ — сдвиг x-θ / in-plane shear stress [{labels['stress']}]",f"τ_xr = S_xr — поперечный сдвиг x-r / transverse shear stress [{labels['stress']}]",f"τ_θr = S_θr — поперечный сдвиг θ-r / transverse shear stress [{labels['stress']}]",
        "ε_i — интенсивность деформаций [-]",f"σ_i — интенсивность напряжений [{labels['stress']}]","det(F) [-]",f"G_B [{labels['stress']}] (только Birger)","G_B/G0 [-] (только Birger)",
        "Нелинейная зона Биргера [0/1] (только Birger)","Пластическая зона [0/1] (если J2/Birger)","Доля пластической толщины hp/h [-] (если доступно)"
    ]
    wr=_SplitLongWriter(book,"Thickness_Fields",headers,[10,16,18,10,13,10,10,13,13,13]+[18]*(len(headers)-10))
    wrote=False
    is_birger=_is_birger_mode(summary)
    mat=summary.get("material") or {}; E0=mat.get("E0");nu0=mat.get("nu0")
    try:G0=float(E0)/(2*(1+float(nu0)))
    except Exception as exc:
        G0=None
        book.warn("Thickness_Fields", "material.G0", exc)
    for dp in sorted((root/"NUMERIC").glob("STEP_*_DERIVED.npz")):
        step=_step_number(dp)
        try:
            with np.load(dp,allow_pickle=False) as z:
                keys=set(z.files); zc=np.asarray(z["z"],float) if "z" in keys else None
                ei=np.asarray(z["epsilon_i"],float) if "epsilon_i" in keys else None
                si=np.asarray(z["sigma_i"],float) if "sigma_i" in keys else None
                if zc is None or ei is None: continue
                nz,ns,nx=ei.shape;xx,ss,th=_grid_coords(summary,(ns,nx))
                GB=np.asarray(z["GB"],float) if is_birger and "GB" in keys else None
                det=np.asarray(z["detF"],float) if "detF" in keys else None
                nl=np.asarray(z["birger_nonlinear_mask"]) if "birger_nonlinear_mask" in keys else None
                pm=np.asarray(z["plastic_mask"]) if "plastic_mask" in keys else None
                pf=np.asarray(z["plastic_thickness_fraction"],float) if "plastic_thickness_fraction" in keys else None
                disp=np.asarray(z["disp_gauss"],float) if "disp_gauss" in keys else None
                E=np.asarray(z["E"],float) if "E" in keys else None
                S=np.asarray(z["S"],float) if "S" in keys else None
                strain=np.asarray(z["strain"],float) if "strain" in keys else None
                stress=np.asarray(z["stress"],float) if "stress" in keys else None
                for g in range(nz):
                    for js in range(ns):
                        for ix in range(nx):
                            if E is not None:
                                ep=[E[g,k,js,ix] for k in range(6)]
                            elif strain is not None:
                                # KL stores ex, es, ezz, gamma_xs.  Remaining
                                # transverse shear components are not part of KL.
                                ep=[strain[g,0,js,ix],strain[g,1,js,ix],strain[g,2,js,ix],strain[g,3,js,ix],None,None]
                            else: ep=[None]*6
                            if S is not None:
                                sg=[S[g,k,js,ix] for k in range(6)]
                            elif stress is not None:
                                sg=[stress[g,0,js,ix],stress[g,1,js,ix],0.0,stress[g,2,js,ix],None,None]
                            else: sg=[None]*6
                            uvw=[None,None,None] if disp is None else [disp[g,c,js,ix] for c in range(3)]
                            gb=None if GB is None else GB[g,js,ix]
                            factor=lfmap.get(step,step);actual=funits.load_axis([factor],load_definition,system)[0][0]
                            wr.write([
                                step,factor,actual,g,zc[g],ix,js,xx[ix],ss[js],th[js],*uvw,*ep,*sg,
                                ei[g,js,ix],None if si is None else si[g,js,ix],None if det is None else det[g,js,ix],gb,
                                None if gb is None or G0 is None else gb/G0,
                                None if nl is None else int(nl[g,js,ix]),None if pm is None else int(pm[g,js,ix]),None if pf is None else pf[js,ix]
                            ])
                            wrote=True
        except Exception as exc:
            book.warn("Thickness_Fields", dp.name, exc)
            continue
    return wrote


def _write_resultants(book: _Book, root: Path, summary: dict, family="all"):
    system,labels,load_definition,load_label,_=_display_context(root,summary)
    lfmap=_load_factor_map(root)
    wr=_SplitLongWriter(book,"Resultants",["Шаг [-]","Коэффициент нагрузки λ [-]",load_label,"Результанта","Индекс τ [-]","Размерность","Индекс ix [-]","Индекс is [-]",f"x [{labels['length']}]",f"s [{labels['length']}]","θ [rad]","Значение"] ,[10,16,18,22,10,20,10,10,14,14,14,18])
    wrote=False
    for dp in sorted((root/"NUMERIC").glob("STEP_*_DERIVED.npz")):
        step=_step_number(dp)
        try:
            with np.load(dp,allow_pickle=False) as z:
                for key in z.files:
                    if not key.startswith("resultant_"): continue
                    a=np.asarray(z[key],float); name=key[len("resultant_"):]
                    fam=str(family).lower()
                    if fam=="membrane" and name not in ("Nx","Ns","Nxs"):
                        continue
                    if fam=="moments" and name not in ("Mx","Ms","Mxs"):
                        continue
                    if a.ndim==2:
                        ns,nx=a.shape;xx,ss,th=_grid_coords(summary,(ns,nx))
                        factor=lfmap.get(step,step);actual=funits.load_axis([factor],load_definition,system)[0][0];unit=_resultant_unit(name,None,labels)
                        for js in range(ns):
                            for ix in range(nx):wr.write([step,factor,actual,name,None,unit,ix,js,xx[ix],ss[js],th[js],a[js,ix]]);wrote=True
                    elif a.ndim==3:
                        nt,ns,nx=a.shape;xx,ss,th=_grid_coords(summary,(ns,nx))
                        factor=lfmap.get(step,step);actual=funits.load_axis([factor],load_definition,system)[0][0]
                        for tau in range(nt):
                            for js in range(ns):
                                for ix in range(nx):wr.write([step,factor,actual,name,tau,_resultant_unit(name,tau,labels),ix,js,xx[ix],ss[js],th[js],a[tau,js,ix]]);wrote=True
        except Exception as exc:
            book.warn("Resultants", dp.name, exc)
            continue
    return wrote


def _write_standard_z_displacements(book: _Book, root: Path, summary: dict):
    system,labels,load_definition,load_label,_=_display_context(root,summary)
    lfmap=_load_factor_map(root)
    wr=_SplitLongWriter(book,"CUF_Standard_Z",["Шаг [-]","Коэффициент нагрузки λ [-]",load_label,f"z [{labels['length']}]","Индекс ix [-]","Индекс is [-]",f"x [{labels['length']}]",f"s [{labels['length']}]","θ [rad]",f"u(x,θ,z) — axial displacement [{labels['displacement']}]",f"v(x,θ,z) — circumferential displacement [{labels['displacement']}]",f"w(x,θ,z) — radial displacement [{labels['displacement']}]"] ,[10,16,18,14,10,10,14,14,14,18,18,18])
    wrote=False
    for dp in sorted((root/"NUMERIC").glob("STEP_*_DERIVED.npz")):
        step=_step_number(dp)
        try:
            with np.load(dp,allow_pickle=False) as z:
                if not {"standard_z","disp_standard_z"}.issubset(set(z.files)):continue
                zz=np.asarray(z["standard_z"],float);d=np.asarray(z["disp_standard_z"],float);nz,_,ns,nx=d.shape;xx,ss,th=_grid_coords(summary,(ns,nx))
                factor=lfmap.get(step,step);actual=funits.load_axis([factor],load_definition,system)[0][0]
                for g in range(nz):
                    for js in range(ns):
                        for ix in range(nx):wr.write([step,factor,actual,zz[g],ix,js,xx[ix],ss[js],th[js],d[g,0,js,ix],d[g,1,js,ix],d[g,2,js,ix]]);wrote=True
        except Exception as exc:
            book.warn("CUF_Standard_Z", dp.name, exc)
            continue
    return wrote




def _write_field_diagnostics(book: _Book, root: Path, summary: dict):
    """Quantify whether final stress/strain fields are truly spatially flat.

    Arrays use z-s-x ordering.  The report separates circumferential (theta/s),
    axial x and through-thickness z variation, which is especially useful for
    axisymmetric closed-cylinder benchmarks.
    """
    files=sorted((root/"NUMERIC").glob("STEP_*_DERIVED.npz"))
    if not files:return False
    dp=files[-1]; step=_step_number(dp)
    _,labels,_,_,_=_display_context(root,summary)
    rows=[]
    def add(code,a):
        a=np.asarray(a,float)
        if a.ndim!=3 or a.size==0:return
        finite=a[np.isfinite(a)]
        if finite.size==0:return
        mn=float(np.min(finite));mx=float(np.max(finite));rng=float(mx-mn)
        theta=float(np.nanmax(np.ptp(a,axis=1))) if a.shape[1]>1 else 0.0
        xvar=float(np.nanmax(np.ptp(a,axis=2))) if a.shape[2]>1 else 0.0
        zvar=float(np.nanmax(np.ptp(a,axis=0))) if a.shape[0]>1 else 0.0
        scale=max(abs(mn),abs(mx),abs(rng),1e-30)
        rel=theta/scale
        absmax=max(abs(mn),abs(mx))
        low=str(code).lower()
        near_zero=(absmax<=1e-12) if (low.startswith("e_") or low.startswith("eps") or low=="epsilon_i") else ((absmax<=1e-8) if (low.startswith("s_") or low.startswith("sig") or low.startswith("tau")) else False)
        if near_zero:
            status="Практически нулевая компонента (уровень численного шума)"
        elif rng<=1e-12*scale:
            status="Практически постоянное поле"
        elif rel<=1e-8:
            status="Осесимметрично по θ/s: окружная вариация ≈ 0"
        else:
            status="Есть заметная вариация по θ/s"
        name,unit,_meaning=_field_meta(code,summary,labels)
        rows.append([step,code,name,unit,mn,mx,rng,theta,xvar,zvar,rel,status])
    try:
        with np.load(dp,allow_pickle=False) as zf:
            keys=set(zf.files)
            for code in ("epsilon_i","sigma_i","detF"):
                if code in keys:add(code,zf[code])
            if _is_birger_mode(summary) and "GB" in keys:
                add("GB",zf["GB"])
            if "E" in keys:
                a=np.asarray(zf["E"],float)
                for i,c in enumerate(("E_xx","E_ss","E_rr","E_xs","E_xr","E_sr")):
                    if a.ndim==4 and i<a.shape[1]:add(c,a[:,i])
            if "S" in keys:
                a=np.asarray(zf["S"],float)
                for i,c in enumerate(("S_xx","S_ss","S_rr","S_xs","S_xr","S_sr")):
                    if a.ndim==4 and i<a.shape[1]:add(c,a[:,i])
            if "strain" in keys:
                a=np.asarray(zf["strain"],float)
                for i,c in enumerate(("eps_xx","eps_ss","eps_zz","gamma_xs")):
                    if a.ndim==4 and i<a.shape[1]:add(c,a[:,i])
            if "stress" in keys:
                a=np.asarray(zf["stress"],float)
                for i,c in enumerate(("sig_xx","sig_ss","tau_xs")):
                    if a.ndim==4 and i<a.shape[1]:add(c,a[:,i])
    except Exception as exc:
        book.warn("Диагностика_полей",dp.name,exc);return False
    if not rows:return False
    headers=[
        "Шаг [-]","Код поля","Полное название","Единицы","min","max","Размах max-min",
        "max Δ по θ/s","max Δ по x","max Δ по z","Отн. вариация θ/s [-]","Интерпретация",
    ]
    ws,_=book.simple_sheet("Диагностика_полей",headers,rows,[10,18,54,14,18,18,18,18,18,18,20,48])
    ws.set_column(2,2,54,book.wrap);ws.set_column(11,11,48,book.wrap)
    return True

def _write_section_exports(book: _Book, root: Path, summary: dict):
    """Collect FULL checkpoint sections with readable field names and units."""
    base=root/"NUMERIC"/"SECTIONS"
    if not base.exists():return False
    _,labels,_,_,_=_display_context(root,summary)
    writers={
        "XZ":_SplitLongWriter(book,"Sections_XZ",["Шаг [-]","Код поля","Полное название поля","Единицы",f"z [{labels['length']}]",f"x [{labels['length']}]","is [-]","Значение"],[10,18,54,16,14,14,10,18]),
        "SZ":_SplitLongWriter(book,"Sections_SZ",["Шаг [-]","Код поля","Полное название поля","Единицы",f"z [{labels['length']}]","θ [rad]",f"s [{labels['length']}]","ix [-]","Значение"],[10,18,54,16,14,14,14,10,18]),
        "Z":_SplitLongWriter(book,"Profiles_Z",["Шаг [-]","Код поля","Полное название поля","Единицы",f"z [{labels['length']}]","is [-]","ix [-]","Значение"],[10,18,54,16,14,10,10,18]),
    }
    wrote=False
    for step_dir in sorted(p for p in base.glob("STEP_*") if p.is_dir()):
        m=re.search(r"STEP_(\d+)",step_dir.name); step=int(m.group(1)) if m else None
        for cp in sorted(step_dir.glob("*.csv")):
            rows=_read_csv(cp)
            if not rows:continue
            kind="XZ" if cp.name.startswith("XZ_") else ("SZ" if cp.name.startswith("SZ_") else ("Z" if cp.name.startswith("Z_") else None))
            if kind is None:continue
            for r in rows:
                code=r.get("field"); readable,unit,_meaning=_field_meta(code,summary,labels)
                if kind=="XZ":writers[kind].write([step,code,readable,unit,r.get("z"),r.get("x"),r.get("is"),r.get("value")])
                elif kind=="SZ":writers[kind].write([step,code,readable,unit,r.get("z"),r.get("theta"),r.get("s"),r.get("ix"),r.get("value")])
                else:writers[kind].write([step,code,readable,unit,r.get("z"),r.get("is"),r.get("ix"),r.get("value")])
                wrote=True
    return wrote


def _write_export_warnings(book: _Book):
    if not book.export_warnings:
        return
    ws=book.wb.add_worksheet("EXPORT_WARNINGS")
    headers=["sheet","item","error"]
    for c,h in enumerate(headers):
        ws.write(0,c,h,book.header)
    for r,item in enumerate(book.export_warnings,1):
        ws.write_row(r,0,[item["sheet"],item["item"],item["error"]])
    ws.freeze_panes(1,0)
    ws.autofilter(0,0,len(book.export_warnings),len(headers)-1)
    ws.set_column(0,0,22); ws.set_column(1,1,38); ws.set_column(2,2,90)


def export_calculation_excel(output_dir: str | Path) -> Path:
    root=Path(output_dir).resolve(); root.mkdir(parents=True,exist_ok=True)
    run_name=_run_name(root); summary=_summary(root); case=_case(root)
    excel_dir=root/"EXCEL";excel_dir.mkdir(parents=True,exist_ok=True)
    for old in excel_dir.glob("*__FULL_RESULTS.xlsx"):
        try:old.unlink()
        except OSError:pass
    filename=f"{_file_prefix(root)}__FULL_RESULTS.xlsx"
    path=excel_dir/filename
    book=_Book(path)
    try:
        _write_readme(book,root,run_name,summary,case)
        _,_labels,_,_,_=_display_context(root,summary,case)
        _write_field_legend(book,summary,_labels)
        _write_parameters(book,case,summary)
        _write_step_tables(book,root)
        _write_progress(book,root)
        _write_control_points(book,root)
        _write_surface_fields(book,root,summary)
        _write_thickness_fields(book,root,summary)
        _write_field_diagnostics(book,root,summary)
        _write_cuf_coefficients(book,root,summary)
        _write_standard_z_displacements(book,root,summary)
        _write_resultants(book,root,summary)
        _write_section_exports(book,root,summary)
        _write_export_warnings(book)
    finally:
        book.close()
    return path



def _write_moments(book, root, summary):
    """Export classical KL bending/twisting moments Mx, Ms, Mxs only."""
    return _write_resultants(book, root, summary, family="moments")


def _write_theory_comparison(book, root, summary):
    """Basic theory comparison information."""
    rows=[{
        "theory": summary.get("theory"),
        "cuf_model": summary.get("cuf_model"),
        "method": summary.get("method"),
        "mode": summary.get("mode")
    }]
    book.simple_sheet(
        "Theory_Comparison",
        ["theory","cuf_model","method","mode"],
        ([r.get(k) for k in ["theory","cuf_model","method","mode"]] for r in rows),
        [20,20,20,25]
    )

def export_named_excels(output_dir: str | Path, include=None) -> dict[str,Path]:
    """Human-readable category workbooks in addition to the full workbook."""
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    excel_dir=root/"EXCEL";excel_dir.mkdir(parents=True,exist_ok=True)
    run_name=_run_name(root);summary=_summary(root);case=_case(root);prefix=_file_prefix(root)
    include_set=None if include is None else {str(x).upper() for x in include}
    theory=str(summary.get("theory",case.get("theory",""))).upper()
    specs=[
        ("LOAD_DEFLECTION",lambda b:_write_step_tables(b,root)),
        ("DISPLACEMENTS",lambda b:(_write_surface_fields(b,root,summary),_write_standard_z_displacements(b,root,summary))),
        ("STRAINS_STRESSES",lambda b:(_write_thickness_fields(b,root,summary),_write_field_diagnostics(b,root,summary))),
        ("THEORY_COMPARISON",lambda b:_write_theory_comparison(b,root,summary)),
        ("CONVERGENCE",lambda b:_write_progress(b,root)),
        ("CONTROL_POINTS",lambda b:_write_control_points(b,root)),
    ]
    if theory=="KL":
        specs.extend([
            ("RESULTANTS",lambda b:_write_resultants(b,root,summary,family="membrane")),
            ("MOMENTS",lambda b:_write_moments(b,root,summary)),
        ])
    else:
        # CUF generalized resultants (Au/Av/Aw/C*) are not classical KL
        # bending moments.  Export them once under RESULTANTS; do not create a
        # misleading duplicate MOMENTS workbook.
        specs.append(("RESULTANTS",lambda b:_write_resultants(b,root,summary,family="all")))
    if theory=="CUF":
        specs.append(("CUF_COEFFICIENTS",lambda b:_write_cuf_coefficients(b,root,summary)))
        # Remove a stale duplicate MOMENTS workbook created by older releases.
        for old in excel_dir.glob("*__MOMENTS.xlsx"):
            try: old.unlink()
            except OSError: pass
    out={}
    for label,writer in specs:
        if include_set is not None and label not in include_set:
            continue
        for old in excel_dir.glob(f"*__{label}.xlsx"):
            try:old.unlink()
            except OSError:pass
        path=excel_dir/f"{prefix}__{label}.xlsx"
        book=_Book(path)
        try:
            _write_readme(book,root,run_name,summary,case)
            _,_labels,_,_,_=_display_context(root,summary,case)
            _write_field_legend(book,summary,_labels)
            _write_parameters(book,case,summary)
            writer(book)
            _write_export_warnings(book)
        finally:book.close()
        out[label]=path
    return out


def main(argv=None):
    import sys
    args=sys.argv[1:] if argv is None else list(argv)
    if len(args)!=1: raise SystemExit("Usage: python final_excel_export.py CALCULATION_OUTPUT_DIR")
    p=export_calculation_excel(args[0]); print(f"EXCEL EXPORT PASS: {p}")


if __name__=="__main__":
    main()
