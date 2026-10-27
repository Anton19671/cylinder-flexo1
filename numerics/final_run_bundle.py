#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output-only helper for self-contained calculation run folders.

This module does not implement or alter shell mechanics.  It only creates a
unique run directory, snapshots file-based inputs, and writes immutable input
metadata for reproducibility/replay.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import csv
import json
from pathlib import Path


SOLVER_REVISION = "2026-08-22_V13_PLATE_STYLE_MVI_EXPERIMENTAL"
import re
import shutil
from typing import Any


def _slug(text: Any, max_len: int = 32) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._+-]+", "_", s)
    s = s.strip("._-+")
    return (s[:max_len] or "RUN")


def _num(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return _slug(value, 16)
    s = f"{x:.8g}"
    return s.replace("-", "m").replace("+", "p").replace(".", "p")


def _load_token(cfg: dict) -> str:
    ld = cfg.get("load_definition") or {}
    typ = str(ld.get("type") or "LOAD").upper()
    def amp(*keys):
        for key in keys:
            if ld.get(key) is not None:
                try:return _num(abs(float(ld.get(key))))
                except Exception:return _num(ld.get(key))
        return "X"
    if typ == "HARMONIC_RADIAL_FACE":
        return f"HARM_m{ld.get('m')}_n{ld.get('n')}_q{amp('q_magnitude','q')}"
    if typ == "UNIFORM_RADIAL_FACE":
        return f"Q{amp('q_magnitude','q')}"
    if typ == "PATCH_RADIAL_FACE":
        return f"PATCH_Q{amp('q_magnitude','q')}"
    if typ == "LINE_RADIAL_GENERATOR":
        return f"LINE_P{amp('p_line_magnitude','p_line')}_b{_num(ld.get('band_width'))}"
    if typ == "EXTERNAL_NPZ":
        return "EXTLOAD"
    return _slug(typ, 18)


def _theory_token(cfg: dict) -> str:
    theory = str(cfg.get("theory") or "THEORY").upper()
    if theory == "CUF":
        return str(cfg.get("cuf_model") or "CUF").upper()
    return theory


def _method_token(cfg: dict) -> str:
    method = str(cfg.get("method") or "METHOD").upper()
    if method == "MVI":
        return f"MVI{cfg.get('rank') or ''}"
    return method


def _mode_token(cfg: dict) -> str:
    mode = str(cfg.get("mode") or "MODE").upper()
    return {
        "LINEAR_ELASTIC":"LE",
        "LINEAR_BIRGER":"LB",
        "NONLINEAR_ELASTIC":"NL",
        "NONLINEAR_BIRGER":"NB",
    }.get(mode,_slug(mode,10))


def _bc_token(cfg: dict) -> str:
    def short(kind):
        k=str(kind or "X").upper()
        return {
            "CLAMPED":"C",
            "FREE":"F",
            "PINNED_S1":"S1",
            "PINNED_S2":"S2",
            "PINNED_S3":"S3",
            "PINNED_S4":"S4",
        }.get(k,_slug(k,6))
    bc = cfg.get("boundary_conditions") or {}
    if isinstance(bc,dict) and bc.get("advanced") is None:
        left=short(bc.get("left"))
        right=short(bc.get("right"))
        return f"BC{left}-{right}"
    return "BCADV"


def _grid_token(cfg: dict) -> str:
    grid = (cfg.get("mvi_grid") if str(cfg.get("method")).upper() == "MVI" else cfg.get("fdm_grid")) or cfg.get("grid") or {}
    return f"g{grid.get('nx','x')}x{grid.get('ns','x')}"


def _full_variant_description(cfg: dict) -> str:
    g = cfg.get("geometry") or {}
    mat = cfg.get("material") or {}
    parts=(
        _theory_token(cfg),_method_token(cfg),str(cfg.get("mode") or "MODE").upper(),
        f"L{_num(g.get('L'))}_R{_num(g.get('R'))}_h{_num(g.get('h'))}",
        f"E{_num(mat.get('E0'))}_nu{_num(mat.get('nu0'))}",
        _bc_token(cfg),_load_token(cfg),_grid_token(cfg),
    )
    return "__".join(_slug(p,32) for p in parts)[:180].rstrip("._ ")


def build_run_folder_name(cfg: dict, run_label: str = "", now: datetime | None = None) -> str:
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    theory = str(cfg.get("theory") or "THEORY").upper()
    if theory == "CUF":
        theory = f"CUF_{str(cfg.get('cuf_model') or 'TE').upper()}"
    method = str(cfg.get("method") or "METHOD").upper()
    if method == "MVI":
        method = f"MVI{cfg.get('rank') or ''}"
    mode = str(cfg.get("mode") or "MODE").upper()
    g = cfg.get("geometry") or {}
    mat = cfg.get("material") or {}
    bc = cfg.get("boundary_conditions") or {}
    grid = (cfg.get("mvi_grid") if str(cfg.get("method")).upper() == "MVI" else cfg.get("fdm_grid")) or cfg.get("grid") or {}
    geom = f"L{_num(g.get('L'))}_R{_num(g.get('R'))}_h{_num(g.get('h'))}"
    material = f"E{_num(mat.get('E0'))}_nu{_num(mat.get('nu0'))}"
    if isinstance(bc,dict) and bc.get("advanced") is None:
        bctok=f"BC{_slug(bc.get('left') or 'X',8)}-{_slug(bc.get('right') or 'X',8)}"
    else:
        bctok="BC_ADV"
    grd = f"g{grid.get('nx','x')}x{grid.get('ns','x')}"
    parts = [stamp]
    if str(run_label or "").strip():
        parts.append(_slug(run_label, 28))
    parts += [_slug(theory, 18), _slug(method, 12), _slug(mode, 24), geom, material, bctok, _load_token(cfg), grd]
    name = "__".join(parts)
    # Stay well below common Windows path-component limits.
    return name[:180].rstrip("._ ")


def build_calculation_folder_name(cfg: dict, run_label: str = "") -> str:
    """Permanent calculation folder shared by all of its RUN variants."""
    if str(run_label or "").strip():
        return _slug(run_label, 48)
    return "__".join((
        "CASE",
        _slug(_theory_token(cfg),10),
        _slug(_mode_token(cfg),8),
        _slug(_bc_token(cfg),14),
        _slug(_load_token(cfg),18),
    ))[:72].rstrip("._ ")


def build_variant_folder_name(cfg: dict) -> str:
    """Short variant identity written after RUN_XXX; full details go to RUN_INFO."""
    parts=(
        _slug(_theory_token(cfg),10),
        _slug(_method_token(cfg),10),
        _slug(_mode_token(cfg),8),
        _slug(_load_token(cfg),18),
        _slug(_grid_token(cfg),10),
    )
    return "_".join(p for p in parts if p)[:64].rstrip("._ ")


def _next_run_number(calculation_dir: Path) -> int:
    used=[]
    for p in calculation_dir.iterdir() if calculation_dir.exists() else ():
        if not p.is_dir():continue
        m=re.match(r"^RUN_(\d{3,})",p.name,re.IGNORECASE)
        if m:used.append(int(m.group(1)))
    return max(used,default=0)+1


def _write_calculation_readme(calculation_dir: Path, calculation_name: str) -> None:
    text=(
        "RESULTS ROOT DIRECTORY / КОРНЕВОЙ КАТАЛОГ РЕЗУЛЬТАТОВ\n"
        "========================================================\n\n"
        "ENGLISH\n"
        "-------\n"
        "Each solver execution is stored here as RUN_001, RUN_002, ... .\n"
        "Intermediate CASE directories are not created.\n"
        "Previous runs are never overwritten.\n\n"
        "What to open first:\n"
        "1. RUNS_INDEX.csv — index of all runs.\n"
        "2. The required RUN_XXX__<short_variant_label> folder.\n"
        "3. Inside a RUN, open the Excel workbooks and README_RESULTS_EN_RU.txt after post-processing.\n"
        "4. The complete run identity is stored in RUN_INFO.json and PARAMETERS.txt.\n"
        "5. Numerical CSV/NPZ data are in NUMERIC, 2D plots are in GRAPHS, and three-dimensional results are in 3D.\n\n"
        "РУССКИЙ\n"
        "-------\n"
        "Каждый запуск решателя сохраняется здесь как RUN_001, RUN_002, ... .\n"
        "Промежуточные каталоги CASE не создаются.\n"
        "Предыдущие запуски никогда не перезаписываются.\n\n"
        "Что открыть первым:\n"
        "1. RUNS_INDEX.csv — перечень всех запусков.\n"
        "2. Нужную папку RUN_XXX__<короткая_метка_варианта>.\n"
        "3. Внутри RUN после постобработки откройте книги Excel и README_RESULTS_EN_RU.txt.\n"
        "4. Полный паспорт варианта записан в RUN_INFO.json и PARAMETERS.txt.\n"
        "5. Числовые CSV/NPZ находятся в NUMERIC, 2D-графики — в GRAPHS, трёхмерные результаты — в 3D.\n"
    )
    canonical=calculation_dir/"README_RESULTS_EN_RU.txt"
    legacy=calculation_dir/"README_RESULTS_RU.txt"
    canonical.write_text(text,encoding="utf-8")
    # Legacy filename retained for old scripts/tests; content is intentionally bilingual.
    legacy.write_text(text,encoding="utf-8")


def _append_run_index(calculation_dir: Path, calculation_name: str, info: dict, cfg: dict) -> Path:
    p=calculation_dir/"RUNS_INDEX.csv"
    fields=("run_number","run_name","variant","created_at","theory","cuf_model","method","mode","load_type","load_value","bc_left","bc_right","L","R","h","E0","nu0","nx","ns","path")
    grid=(cfg.get("mvi_grid") if str(cfg.get("method")).upper()=="MVI" else cfg.get("fdm_grid")) or cfg.get("grid") or {}
    g=cfg.get("geometry") or {}; mat=cfg.get("material") or {}; ld=cfg.get("load_definition") or {}; bc=cfg.get("boundary_conditions") or {}
    load_value=ld.get("q_magnitude",ld.get("q",ld.get("p_line_magnitude",ld.get("p_line"))))
    row={
        "run_number":info.get("run_number"),"run_name":info.get("run_name"),"variant":info.get("variant_name"),"created_at":info.get("created_at"),
        "theory":cfg.get("theory"),"cuf_model":cfg.get("cuf_model"),"method":cfg.get("method"),"mode":cfg.get("mode"),
        "load_type":ld.get("type"),"load_value":load_value,
        "bc_left":bc.get("left") if isinstance(bc,dict) else None,"bc_right":bc.get("right") if isinstance(bc,dict) else None,
        "L":g.get("L"),"R":g.get("R"),"h":g.get("h"),"E0":mat.get("E0"),"nu0":mat.get("nu0"),
        "nx":grid.get("nx"),"ns":grid.get("ns"),"path":info.get("run_dir"),
    }
    exists=p.exists()
    with p.open("a",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if not exists:w.writeheader()
        w.writerow(row)
    return p


def _unique_dir(root: Path, name: str) -> Path:
    p = root / name
    if not p.exists():
        return p
    for i in range(2, 1000):
        q = root / f"{name}__{i:02d}"
        if not q.exists():
            return q
    raise RuntimeError(f"Cannot allocate unique run directory under {root}")


def _resolve_path(value: str | None, source_base: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = source_base / p
    return p.resolve()


def _copy_input(value: str | None, role: str, files_dir: Path, source_base: Path, copied: dict) -> str | None:
    src = _resolve_path(value, source_base)
    if src is None:
        return value
    if not src.exists() or not src.is_file():
        # Keep the original reference; the normal CASE validator will issue the
        # authoritative input error.  Do not invent replacement data.
        return str(value)
    suffix = "".join(src.suffixes[-2:]) if len(src.suffixes) > 1 else src.suffix
    stem = _slug(src.stem, 45)
    dst = files_dir / f"{_slug(role,24)}__{stem}{suffix}"
    k = 2
    while dst.exists() and dst.resolve() != src.resolve():
        dst = files_dir / f"{_slug(role,24)}__{stem}__{k}{suffix}"
        k += 1
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)
    rel = Path("FILES") / dst.name
    copied[role] = {"source": str(src), "snapshot": str(dst.resolve()), "relative_to_case": rel.as_posix()}
    return rel.as_posix()


def snapshot_file_inputs(cfg: dict, input_dir: Path, source_base: Path) -> tuple[dict, dict]:
    c = deepcopy(cfg)
    files_dir = input_dir / "FILES"
    files_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict] = {}

    c["loads_npz"] = _copy_input(c.get("loads_npz"), "loads_npz", files_dir, source_base, copied)
    ld = c.get("load_definition")
    if isinstance(ld, dict) and str(ld.get("type") or "").upper() == "EXTERNAL_NPZ":
        ld["npz"] = _copy_input(ld.get("npz"), "load_definition_npz", files_dir, source_base, copied)

    c["initial_profiles_npz"] = _copy_input(c.get("initial_profiles_npz"), "initial_profiles", files_dir, source_base, copied)
    st = c.get("initial_state")
    if isinstance(st, dict):
        st["npz"] = _copy_input(st.get("npz"), "initial_state", files_dir, source_base, copied)
    gb = c.get("initial_GB")
    if isinstance(gb, dict):
        gb["npz"] = _copy_input(gb.get("npz"), "initial_GB", files_dir, source_base, copied)
    bj = c.get("birger")
    if isinstance(bj, dict):
        curve = bj.get("curve")
        if isinstance(curve, dict) and str(curve.get("mode") or "").upper() == "CSV":
            curve["path"] = _copy_input(curve.get("path"), "birger_curve_csv", files_dir, source_base, copied)
    return c, copied


def _summary_text(cfg: dict, run_name: str, copied: dict) -> str:
    g = cfg.get("geometry") or {}
    mat = cfg.get("material") or {}
    grid = (cfg.get("mvi_grid") if str(cfg.get("method")).upper() == "MVI" else cfg.get("fdm_grid")) or cfg.get("grid") or {}
    ld = cfg.get("load_definition") or {}
    lines = [
        f"RUN: {run_name}",
        f"Variant: {_full_variant_description(cfg)}",
        f"Theory: {cfg.get('theory')} {cfg.get('cuf_model') or ''}",
        f"Method: {cfg.get('method')}" + (f" rank={cfg.get('rank')}" if str(cfg.get('method')).upper()=="MVI" else ""),
        f"Mode: {cfg.get('mode')}",
        f"Units: {(cfg.get('units') or {}).get('system')}",
        f"Geometry: L={g.get('L')}, R={g.get('R')}, h={g.get('h')}",
        f"Material: E0={mat.get('E0')}, nu0={mat.get('nu0')}",
        f"Grid: nx={grid.get('nx')}, ns={grid.get('ns')}, stencil={grid.get('stencil')}",
        (
            "MVI initial profiles: external NPZ"
            if str(cfg.get("initial_profiles_npz") or "").strip()
            else "MVI initial profiles: automatic deterministic seed"
        ) if str(cfg.get("method")).upper()=="MVI" else "MVI initial profiles: not applicable",
        f"Load: {ld.get('type')}",
        f"Output: {cfg.get('output_dir')}",
        "",
        "Boundary conditions:",
        json.dumps(cfg.get("boundary_conditions"), ensure_ascii=False, indent=2),
        "",
        "Load definition:",
        json.dumps(ld, ensure_ascii=False, indent=2),
    ]
    if copied:
        lines += ["", "Snapshotted external inputs:", json.dumps(copied, ensure_ascii=False, indent=2)]
    lines += ["", "Full validated/replay input is stored in INPUT/FINAL_CASE.json."]
    return "\n".join(lines) + "\n"


def prepare_run_bundle(
    cfg: dict,
    results_root: str | Path,
    *,
    run_label: str = "",
    source_base: str | Path | None = None,
    now: datetime | None = None,
) -> dict:
    root = Path(results_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    calculation_name=build_calculation_folder_name(cfg,run_label=run_label)
    # Keep the user-selected result root flat: RUN folders live directly here.
    # The full calculation identity remains in RUN_INFO/PARAMETERS/RUNS_INDEX.
    calculation_dir=root
    _write_calculation_readme(calculation_dir,calculation_name)
    run_number=_next_run_number(calculation_dir)
    variant=build_variant_folder_name(cfg)
    full_variant=_full_variant_description(cfg)
    name=f"RUN_{run_number:03d}__{variant}"
    run_dir = _unique_dir(calculation_dir, name)
    input_dir = run_dir / "INPUT"
    input_dir.mkdir(parents=True, exist_ok=False)
    source_base_p = Path(source_base).resolve() if source_base is not None else Path.cwd().resolve()

    patched, copied = snapshot_file_inputs(cfg, input_dir, source_base_p)
    patched["output_dir"] = str(run_dir.resolve())
    case_path = input_dir / "FINAL_CASE.json"
    case_path.write_text(json.dumps(patched, ensure_ascii=False, indent=2), encoding="utf-8")
    (input_dir / "RUN_PARAMETERS.json").write_text(json.dumps(patched, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "format": "CYLINDER_RUN_BUNDLE_V1",
        "solver_revision": SOLVER_REVISION,
        "created_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        "full_run_name": run_dir.name,
        "run_number":run_number,
        "run_label": str(run_label or ""),
        "variant_name": variant,
        "full_variant_name": full_variant,
        "calculation_name":calculation_name,
        "calculation_dir":str(calculation_dir.resolve()),
        "results_root": str(root),
        "run_dir": str(run_dir.resolve()),
        "case_path": str(case_path.resolve()),
        "copied_inputs": copied,
    }
    index_path=_append_run_index(calculation_dir,calculation_name,info,patched)
    info["runs_index_csv"]=str(index_path.resolve())
    (run_dir / "RUN_INFO.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "PARAMETERS.txt").write_text(_summary_text(patched, run_dir.name, copied), encoding="utf-8")
    return {**info, "cfg": patched, "input_dir": input_dir, "copied_inputs": copied}
