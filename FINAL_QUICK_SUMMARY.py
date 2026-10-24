#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a compact bilingual quick-analysis PDF from an existing CYL_SOLVER RUN.

Output-only module: never calls the solver and never modifies accepted states.
It reads already generated JSON/PNG results and writes REPORT/Quick Summary PDFs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _read_json(path: Path) -> dict:
    try:
        obj=json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj,dict) else {}
    except Exception:
        return {}


def _load_meta(run: Path) -> dict:
    # Merge from generic input to authoritative final output. Later files win.
    candidates=[
        run/"INPUT"/"FINAL_CASE.json",
        run/"SUMMARY.json",
        run/"RUN_SUMMARY.json",
        run/"FINAL_RUN_SUMMARY.json",
        run/"REPORT"/"CALCULATION_SUMMARY.json",
    ]
    merged={}
    for p in candidates:
        if p.exists():
            merged={**merged,**_read_json(p)}
    return merged


def _num(v, default=None):
    try:return float(v)
    except Exception:return default


def _fmt(v, unit="", digits=5):
    if v is None:return "-"
    try:
        x=float(v)
        if not np.isfinite(x):return "-"
        if abs(x)>=1e5 or (0<abs(x)<1e-3):s=f"{x:.3e}"
        else:s=f"{x:.{digits}g}"
        return s+(f" {unit}" if unit else "")
    except Exception:return str(v)


def _deep(d,*keys,default=None):
    cur=d
    for k in keys:
        if not isinstance(cur,dict) or k not in cur:return default
        cur=cur[k]
    return cur


def _all_png(run: Path) -> list[Path]:
    return [p for p in run.rglob("*.png") if "QUICK_SUMMARY" not in p.name.upper()]


def _find_first(run: Path, exact: Iterable[str]=(), contains: Iterable[str]=(), exclude: Iterable[str]=()) -> Path|None:
    allp=_all_png(run);ex=[x.upper() for x in exclude];exactu=[x.upper() for x in exact];cont=[x.upper() for x in contains]
    for name in exactu:
        for p in allp:
            if p.name.upper()==name and not any(e in str(p).upper() for e in ex):return p
    # Plain tokens match the FILE NAME, not a parent folder. Tokens containing
    # '/' intentionally match the normalized full path (used for XZ/SZ folders).
    for token in cont:
        for p in allp:
            full=str(p).replace('\\','/').upper();target=full if '/' in token else p.name.upper()
            if token in target and not any(e in full for e in ex):return p
    return None

def _find_all_tokens(run: Path, required: Iterable[str], exclude: Iterable[str]=()) -> Path|None:
    req=[str(x).upper() for x in required];ex=[str(x).upper() for x in exclude]
    for p in _all_png(run):
        u=str(p).replace("\\","/").upper()
        if all(t in u for t in req) and not any(e in u for e in ex):return p
    return None

def _image_choices(run: Path) -> dict[str,Path|None]:
    return {
        "loaddef":_find_first(run, exact=("LOAD_DEFLECTION.png",), contains=("__LOAD_DEFLECTION.PNG","LOAD_DEFLECTION.PNG"), exclude=("COMPARE","DIAGNOSTIC")),
        "birger_compare":_find_first(run, exact=("GL_BIRGER_VS_GL_HOOKE.png","BIRGER_VS_HOOKE_LOAD_DEFLECTION.png"), contains=("BIRGER_VS_HOOKE","GL_BIRGER_VS_GL_HOOKE","LOAD_DEFLECTION_COMPARE_BIRGER")),
        "birger_evolution":_find_first(run, exact=("BIRGER_NONLINEAR_VOLUME_FRACTION.png","GB_RATIO_MIN.png"), contains=("BIRGER_NONLINEAR_VOLUME_FRACTION","GB_RATIO_MIN","BIRGER_EVOLUTION")),
        "deformed":_find_first(run, exact=("FINAL_DEFORMED_SHELL.png",), contains=("DEFORMED_SHELL",)),
        "stress3d":_find_first(run, exact=("FINAL_SIGMA_I_SURFACE_3D.png",), contains=("SIGMA_I_SURFACE_3D","STRESS_FIELDS")),
        "strain3d":_find_first(run, exact=("FINAL_EPS_I_SURFACE_3D.png",), contains=("EPS_I_SURFACE_3D",)),
        "xz":_find_first(run, contains=("/XZ/","_XZ_","XZ_AT_WMAX","XZ_AT","SECTION_XZ")),
        "sz":_find_first(run, contains=("/SZ/","_SZ_","SZ_AT_WMAX","SZ_AT","SECTION_SZ")),
        "thickness":_find_first(run, contains=("/Z_PROFILES/","_Z_AT_WMAX","THROUGH_THICKNESS","THICKNESS_PROFILE")),
        "birger3d":_find_first(run, exact=("FINAL_GB_RATIO_3D.png","FINAL_UNLOADING_MASK_3D.png"), contains=("GB_RATIO_3D","UNLOADING_MASK_3D","BIRGER_ZONES")),
        "birger_xz":_find_first(run, contains=("GB_RATIO_XZ_AT_WMAX","BIRGER_NONLINEAR_XZ_AT_WMAX")) or _find_all_tokens(run,("/XZ/","GB_RATIO")) or _find_all_tokens(run,("/XZ/","BIRGER_NONLINEAR")),
        "birger_sz":_find_first(run, contains=("GB_RATIO_SZ_AT_WMAX","BIRGER_NONLINEAR_SZ_AT_WMAX")) or _find_all_tokens(run,("/SZ/","GB_RATIO")) or _find_all_tokens(run,("/SZ/","BIRGER_NONLINEAR")),
        "birger_z":_find_first(run, contains=("GB_RATIO_Z_AT_WMAX","EPS_I_Z_AT_WMAX","SIGMA_I_Z_AT_WMAX")) or _find_all_tokens(run,("/Z_PROFILES/","GB_RATIO")),
        "unloading":_find_first(run, contains=("UNLOADING_MASK_3D","UNLOADING_MASK","UNLOADING")),
        "nonlinear2d":_find_first(run, contains=("BIRGER_NONLINEAR_THICKNESS_FRACTION","BIRGER_NONLINEAR_ZONE_2D_ANY","GB_RATIO_MIN")),
        "field2d":_find_first(run, exact=("FINAL_W.png",), contains=("FINAL_SIGMA_I_G","FINAL_EPS_I_G","FINAL_W")),
    }


def _has_birger(meta: dict, imgs: dict) -> bool:
    mm=str(meta.get("material_model") or "").upper()
    mode=str(meta.get("mode") or "").upper()
    if "BIRGER" in mm or "BIRGER" in mode:return True
    return any(imgs.get(k) is not None for k in ("birger3d","birger_xz","birger_sz","birger_z","unloading","nonlinear2d"))


def _safe_imread(path):
    if path is None:return None
    try:
        import matplotlib.image as mpimg
        return mpimg.imread(path)
    except Exception:return None


def _image_panel(ax,path,title,subtitle=None):
    ax.set_facecolor("#f8fafc")
    for sp in ax.spines.values():sp.set_visible(True);sp.set_color("#c3ced8");sp.set_linewidth(.9)
    ax.set_xticks([]);ax.set_yticks([]);img=_safe_imread(path)
    if img is not None:
        try:
            arr=np.asarray(img)
            rgb=arr[...,:3] if arr.ndim==3 else arr
            if rgb.ndim==3:
                # Layout-only trim of near-white margins; no physical data processing.
                mask=np.any(rgb<0.985,axis=2);ys,xs=np.where(mask)
                if xs.size and ys.size:
                    pad=8;y0=max(0,int(ys.min())-pad);y1=min(arr.shape[0],int(ys.max())+pad+1);x0=max(0,int(xs.min())-pad);x1=min(arr.shape[1],int(xs.max())+pad+1);arr=arr[y0:y1,x0:x1]
            ax.imshow(arr,aspect="auto")
        except Exception:ax.imshow(img,aspect="auto")
    else:
        ax.text(.5,.52,"Нет данных / No data",ha="center",va="center",fontsize=11,color="#7b8794",transform=ax.transAxes)
        ax.text(.018,.975,title,ha="left",va="top",fontsize=11.2,fontweight="bold",color="#172b4d",transform=ax.transAxes)
    if subtitle:ax.text(.018,.045,subtitle,ha="left",va="bottom",fontsize=8.2,color="#5e6c84",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.18",facecolor=(1,1,1,.84),edgecolor="none"))

def _status(meta):
    mvi=str(meta.get("method") or "").upper()=="MVI"
    suffix=" — MVI projected equations converged" if mvi and meta.get("converged") is True else ""
    if meta.get("status"):return str(meta.get("status"))+suffix
    if meta.get("converged") is True:return "CONVERGED"+suffix
    if meta.get("converged") is False:return "NOT CONVERGED"
    accepted=meta.get("accepted_steps") or meta.get("steps")
    requested=meta.get("requested_steps")
    try:
        if accepted is not None and requested is not None and int(accepted)>=int(requested)>0:return "COMPLETED"
        if accepted is not None and int(accepted)>0:return "PARTIAL / ACCEPTED STEPS SAVED"
    except Exception:pass
    return "UNKNOWN"


def _header(fig,meta,page_title,en_only=False):
    theory=str(meta.get("theory") or "-");gm=str(meta.get("geometry_model") or "-");mm=str(meta.get("material_model") or "-");method=str(meta.get("method") or "-");title=("QUICK RESULT SUMMARY" if en_only else "ЭКСПРЕСС-АНАЛИЗ / QUICK RESULT SUMMARY")
    fig.text(.03,.958,title,fontsize=19,fontweight="bold",color="#15385b",va="top");fig.text(.03,.913,page_title,fontsize=11,color="#506070",va="top");fig.text(.97,.952,f"{theory} | {gm} | {mm} | {method}",fontsize=10,color="#334e68",va="top",ha="right");fig.lines.append(__import__('matplotlib').lines.Line2D([.03,.97],[.89,.89],transform=fig.transFigure,color="#ccd6e0",lw=1.0))

def _key_card(fig,meta,en_only=False):
    g=meta.get("geometry") or {};mat=meta.get("material") or {};grid=meta.get("grid") or {};units=meta.get("units") or {};lu=units.get("length") or units.get("length_unit") or "mm";su=units.get("stress") or units.get("stress_unit") or "N/mm²";status=_status(meta)
    final_load=meta.get("final_load_value",meta.get("final_load",meta.get("load_value_final",meta.get("last_load_value"))));final_load_label=str(meta.get("final_load_label") or "").strip()
    rows=[
        (("Status" if en_only else "Статус / Status"),status),
        (("Geometry" if en_only else "Геометрия / Geometry"),f"L={_fmt(g.get('L'),lu)}  R={_fmt(g.get('R'),lu)}  h={_fmt(g.get('h'),lu)}"),
        (("Grid" if en_only else "Сетка / Grid"),f"Nx={grid.get('Nx',grid.get('nx','N/A'))}  Ns={grid.get('Ns',grid.get('ns','N/A'))}"),
        (("Material" if en_only else "Материал / Material"),f"E={_fmt(mat.get('E',mat.get('E0')),su)}  ν={_fmt(mat.get('nu',mat.get('nu0')))}"),
        (("Accepted steps" if en_only else "Принято шагов / Accepted steps"),str(meta.get("accepted_steps") or meta.get("steps") or "N/A")),
        (("Final load" if en_only else "Конечная нагрузка / Final load"),((_fmt(final_load)+("  "+final_load_label if final_load_label else "")) if final_load is not None else "N/A")),
        (("Max deflection" if en_only else "Макс. прогиб / Max deflection"),_fmt(meta.get("w_abs_max_final",meta.get("max_w_abs")),lu)),
        (("Max stress" if en_only else "Макс. напряжение / Max stress"),_fmt(_deep(meta,"solver_diagnostics","sigma_max"),su) if _deep(meta,"solver_diagnostics","sigma_max") is not None else "N/A"),
        (("Wall time" if en_only else "Время / Wall time"),_fmt(_deep(meta,"solver_diagnostics","wall_time",default=meta.get("wall_time")),"s") if _deep(meta,"solver_diagnostics","wall_time",default=meta.get("wall_time")) is not None else "N/A"),
    ]
    ax=fig.add_axes([.03,.085,.18,.78]);ax.axis("off");ax.add_patch(__import__('matplotlib').patches.FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.012,rounding_size=.015",facecolor="#16283d",edgecolor="#0c1b2a",lw=.8,transform=ax.transAxes));ax.text(.065,.955,("CALCULATION" if en_only else "РАСЧЁТ / CALCULATION"),color="white",fontweight="bold",fontsize=10.5,va="top")
    y=.89
    for lab,val in rows:
        ax.text(.065,y,lab,color="#9fb7cc",fontsize=7.5,va="top");ax.text(.065,y-.032,str(val),color="white",fontsize=8.3,va="top",wrap=True);y-=.094
    return ax

def _build_page1(meta,imgs,en_only=False):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,9),facecolor="#f4f7fa");_header(fig,meta,("Page 1 - global response" if en_only else "Страница 1 - общий результат / Page 1 - global response"),en_only);_key_card(fig,meta,en_only)
    # Give the two primary results the largest panels.
    ax1=fig.add_axes([.225,.515,.365,.35]);_image_panel(ax1,imgs["loaddef"],"Load-deflection" if en_only else "Нагрузка-прогиб / Load-deflection")
    ax2=fig.add_axes([.605,.515,.365,.35]);_image_panel(ax2,imgs["deformed"],"Deformed shell" if en_only else "Деформированная оболочка / Deformed shell")
    ax3=fig.add_axes([.225,.09,.365,.36]);_image_panel(ax3,imgs["stress3d"] or imgs["field2d"],"Stress / primary field" if en_only else "Напряжения / основное поле / Stress / primary field")
    ax4=fig.add_axes([.605,.09,.365,.36]);_image_panel(ax4,imgs["strain3d"] or imgs["field2d"],"Strain / secondary field" if en_only else "Деформации / дополнительное поле / Strain / secondary field")
    return fig

def _build_page2(meta,imgs,en_only=False):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,9),facecolor="#f4f7fa");_header(fig,meta,("Page 2 - sections and through-thickness analysis" if en_only else "Страница 2 - разрезы и анализ по толщине / Page 2 - sections and through-thickness analysis"),en_only)
    positions=[(.03,.49,.46,.375),(.51,.49,.46,.375),(.03,.065,.46,.375),(.51,.065,.46,.375)];specs=[(imgs["xz"],"Longitudinal x-z section" if en_only else "Продольный разрез x-z / Longitudinal x-z section"),(imgs["sz"],"Circumferential s-z section" if en_only else "Поперечный разрез s-z / Circumferential s-z section"),(imgs["thickness"],"Through-thickness profile" if en_only else "Профиль по толщине / Through-thickness profile"),(imgs.get("nonlinear2d") or imgs.get("stress3d") or imgs.get("deformed"),"Additional nonlinear / 3D field" if en_only else "Дополнительное нелинейное / 3D поле / Additional nonlinear / 3D field")]
    for pos,(path,title) in zip(positions,specs):ax=fig.add_axes(list(pos));_image_panel(ax,path,title)
    return fig

def _build_page3(meta,imgs,en_only=False):
    """Birger page: pure layout of already-created final PNGs only."""
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,9),facecolor="#f4f7fa");_header(fig,meta,("Page 3 - Birger final plots" if en_only else "Страница 3 - готовые графики Биргера / Page 3 - Birger final plots"),en_only)
    top=[(.03,.50,.30,.35),(.35,.50,.30,.35),(.67,.50,.30,.35)];comparison=imgs.get("birger_compare") or imgs.get("loaddef");evolution=imgs.get("birger_evolution")
    for pos,path,title in [(top[0],comparison,"GL+Birger vs GL+Hooke" if en_only else "GL+Birger / GL+Hooke - готовый график"),(top[1],imgs.get("birger3d"),"3D GB/G0 / Birger zone" if en_only else "3D GB/G0 / зона Биргера"),(top[2],imgs.get("unloading") or evolution,"Unloading / Birger evolution" if en_only else "Unloading / развитие Биргера")]:ax=fig.add_axes(list(pos));_image_panel(ax,path,title)
    lower=[(.03,.07,.30,.35),(.35,.07,.30,.35),(.67,.07,.30,.35)];specs=[(imgs.get("birger_xz"),"Birger x-z section" if en_only else "Разрез Биргера x-z"),(imgs.get("birger_sz"),"Birger s-z section" if en_only else "Разрез Биргера s-z"),(imgs.get("birger_z"),"Through-thickness Birger profile" if en_only else "Профиль Биргера по толщине")]
    for pos,(path,title) in zip(lower,specs):ax=fig.add_axes(list(pos));_image_panel(ax,path,title)
    return fig

def build(run_dir: str|Path) -> tuple[Path,Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    run=Path(run_dir).resolve()
    if not run.exists():raise FileNotFoundError(run)
    meta=_load_meta(run);imgs=_image_choices(run)
    report=run/"REPORT";report.mkdir(parents=True,exist_ok=True)
    bilingual=report/"QUICK_SUMMARY_RU_EN.pdf"
    english=report/"QUICK_SUMMARY_EN.pdf"
    for path,en_only in ((bilingual,False),(english,True)):
        with PdfPages(path) as pdf:
            makers=[_build_page1,_build_page2]
            if _has_birger(meta,imgs):makers.append(_build_page3)
            for maker in makers:
                fig=maker(meta,imgs,en_only=en_only);pdf.savefig(fig,bbox_inches="tight",pad_inches=.08);plt.close(fig)
    return bilingual,english


def main(argv=None):
    ap=argparse.ArgumentParser(description="Build bilingual quick analysis PDF from an existing CYL_SOLVER run")
    ap.add_argument("run_dir")
    ns=ap.parse_args(argv)
    a,b=build(ns.run_dir)
    print(f"QUICK SUMMARY RU/EN: {a}",flush=True)
    print(f"QUICK SUMMARY EN: {b}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
