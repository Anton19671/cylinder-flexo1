#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import csv,json,re,shutil,sys,zipfile
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage3a_connector as c3
import cylinder_mkr_fd2_operators as mkrfd2
import cylinder_te1_stage2b as s2
import cylinder_te1_local_kernel as lk
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7f_general_linear as gl
import cylinder_cuf_stage7h_general_vk as gvk
import cylinder_cuf_stage10a_strong_fdm_linear_elastic as c10a
import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8c1_birger_material as klb
import final_te_basis_provider as tebasis
import final_units as funits
import final_boundary_registry as bcreg
from FINAL_VTK_EXPORT import export_cylinder_surface_vtk
import FINAL_VIDEO_EXPORT as fvideo


_PLOT_CONTEXT={
    "L":None,"R":None,"length":"length unit","displacement":"length unit",
    "stress":"stress unit","pressure":"pressure unit","line_load":"line-load unit","force":"force unit",
    "summary":{},
}


def _configure_plot_context(summary,L=None,R=None):
    units=(summary.get("units") or {}) if isinstance(summary,dict) else {}
    system=str(units.get("system","N_MM_MPA"))
    labels=funits.unit_labels(system)
    _PLOT_CONTEXT.update(labels)
    _PLOT_CONTEXT["L"]=None if L is None else float(L)
    _PLOT_CONTEXT["R"]=None if R is None else float(R)
    _PLOT_CONTEXT["summary"]=summary if isinstance(summary,dict) else {}


def _field_unit_from_title(title):
    t=str(title).lower()
    if any(k in t for k in ("gb/g0","detf","j-1","fraction","zone","mask")):
        return "[-]"
    if any(k in t for k in ("stress","sigma","second piola","sig_","s_xx","s_ss","s_rr","s_xs","s_xr","s_sr")):
        return f"[{_PLOT_CONTEXT['stress']}]"
    if any(k in t for k in ("strain","epsilon","green/linear","eps_","e_xx","e_ss","e_rr","e_xs","e_xr","e_sr")):
        return "[-]"
    if re.match(r"^\s*[uvw](?:\s|—|-)",t):
        return f"[{_PLOT_CONTEXT['displacement']}]"
    if re.search(r"\b(?:nx|ns|nxs)\b",t):
        return f"[{_PLOT_CONTEXT['line_load']}]"
    if re.search(r"\b(?:mx|ms|mxs)\b",t):
        return f"[{_PLOT_CONTEXT['force']}]"
    return "[coherent units]"



def _surface_variation_note(a):
    """Compact diagnostic for 2-D s-x maps; output-only, no mechanics."""
    arr=np.asarray(a,float)
    finite=arr[np.isfinite(arr)]
    if finite.size==0:return "нет конечных значений"
    mn=float(np.min(finite));mx=float(np.max(finite));rng=mx-mn
    note=f"min={mn:.6g}; max={mx:.6g}; Δ={rng:.3g}"
    if arr.ndim==2 and arr.shape[0]>1:
        circ=float(np.nanmax(np.ptp(arr,axis=0)))
        scale=max(abs(mn),abs(mx),abs(rng),1e-30)
        if circ/scale<=1e-8:
            note+=f"; осесимметрично по θ/s (max Δθ={circ:.3g})"
    return note

def _mkdirs(out):
    roots={
        "numeric":out/"NUMERIC",
        "disp":out/"GRAPHS"/"DISPLACEMENTS",
        "strain":out/"GRAPHS"/"STRAINS",
        "stress":out/"GRAPHS"/"STRESSES",
        "resultants":out/"GRAPHS"/"RESULTANTS",
        "loaddef":out/"GRAPHS"/"LOAD_DEFLECTION",
        "thickness":out/"GRAPHS"/"THROUGH_THICKNESS",
        "sections":out/"GRAPHS"/"SECTIONS",
        "conv":out/"GRAPHS"/"CONVERGENCE",
        "birger":out/"GRAPHS"/"BIRGER",
        "plasticity":out/"GRAPHS"/"PLASTICITY",
        "cufdiag":out/"GRAPHS"/"CUF_DIAGNOSTICS",
        "deformed":out/"3D"/"DEFORMED_SHELL",
        "loadbc":out/"3D"/"LOAD_AND_BC",
        "stress3d":out/"3D"/"STRESS_FIELDS",
        "overview3d":out/"3D"/"RESULTS_OVERVIEW",
        "birger3d":out/"3D"/"BIRGER_ZONES",
        "vtk":out/"VTK",
        "report":out/"REPORT",
    }
    for p in roots.values(): p.mkdir(parents=True,exist_ok=True)
    return roots


def _cleanup_full_last_step_artifacts(roots,last_step):
    """Remove stale heavy postprocess artifacts, never solver STEP states.

    A RUN folder may be postprocessed more than once.  FULL is defined as a
    complete package for the LAST accepted state only, so old per-step graphs
    or derived files from an earlier postprocess must not survive and masquerade
    as current results.
    """
    last_step=int(last_step)
    # Per-step graph trees created only by postprocessing.
    for key in ("disp","strain","stress","resultants","sections","plasticity","cufdiag","deformed"):
        d=roots[key]/"STEPS"
        if d.exists():shutil.rmtree(d,ignore_errors=True)
    # THROUGH_THICKNESS is a FULL snapshot of the last accepted state.  Clear
    # stale profiles from an earlier postprocess so the folder never mixes
    # different accepted steps.  Raw solver STEP states are untouched.
    tdir=roots.get("thickness")
    if tdir is not None and tdir.exists():
        for q in tdir.iterdir():
            try:
                if q.is_dir():shutil.rmtree(q,ignore_errors=True)
                else:q.unlink()
            except OSError:pass
    coeff=roots["cufdiag"]/"COEFFICIENTS"
    if coeff.exists():shutil.rmtree(coeff,ignore_errors=True)
    nsec=roots["numeric"]/"SECTIONS"
    if nsec.exists():shutil.rmtree(nsec,ignore_errors=True)
    ntt=roots["numeric"]/"THROUGH_THICKNESS"
    if ntt.exists():shutil.rmtree(ntt,ignore_errors=True)
    # Keep only the derived package that will correspond to the current last
    # accepted state.  Raw STEP_*.npz files in the RUN root are untouched.
    for q in roots["numeric"].glob("STEP_*_DERIVED.npz"):
        m=re.search(r"STEP_(\d{4})_DERIVED",q.name)
        if not m or int(m.group(1))!=last_step:
            try:q.unlink()
            except OSError:pass
    for key in ("birger3d","stress3d"):
        for q in roots[key].glob("STEP_*.png"):
            try:q.unlink()
            except OSError:pass


def _save_png(fig,path,dpi=160):
    """Save every engineering figure in preview + publication formats.

    PNG is the fast preview used by the GUI.  SVG and PDF are publication/
    partner formats; text, axes and line art remain vector.  This function is
    output-only and never changes calculated arrays.
    """
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=dpi,pil_kwargs={"compress_level":1},bbox_inches="tight")
    # Vector companions are intentionally next to the PNG so every result has
    # an immediately discoverable publication-quality counterpart.
    for ext in (".svg",".pdf"):
        try:
            fig.savefig(path.with_suffix(ext),bbox_inches="tight",pad_inches=.06)
        except Exception as exc:
            print(f"VECTOR EXPORT WARNING: {path.with_suffix(ext).name}: {type(exc).__name__}: {exc}",flush=True)


def _bilingual_plot_text(text):
    """English-first bilingual text for partner-facing plots."""
    t=str(text or "")
    replacements={
        "FINAL u":"Axial displacement u / Продольное перемещение u",
        "FINAL v":"Circumferential displacement v / Окружное перемещение v",
        "FINAL w":"Radial displacement w / Радиальный прогиб w",
        "accepted step":"accepted step / принятый шаг",
        "through thickness":"through thickness / по толщине",
        "strain intensity":"strain intensity / интенсивность деформаций",
        "stress intensity":"stress intensity / интенсивность напряжений",
        "Second Piola stress":"Second Piola stress / 2-й тензор Пиолы–Кирхгофа",
        "Green/linear strain":"Green/linear strain / деформация Green/линейная",
        "KL strain":"KL strain / деформация KL",
        "KL stress":"KL stress / напряжение KL",
        "min GB/G0":"min GB/G0 / минимум GB/G0",
        "max GB/G0":"max GB/G0 / максимум GB/G0",
        "Plastic/nonlinear zone":"Plastic/nonlinear zone / пластическая/нелинейная зона",
        "Physical-nonlinearity zone":"Physical-nonlinearity zone / зона физической нелинейности",
    }
    for a,b in replacements.items():
        t=t.replace(a,b)
    common={
        "Plastic volume fraction vs load":"Plastic volume fraction vs load / Доля пластического объёма от нагрузки",
        "Maximum plastic penetration through thickness":"Maximum plastic penetration through thickness / Максимальное проникновение пластики по толщине",
        "Plastic area fraction — top integration layer":"Plastic area fraction — top integration layer / Доля пластической площади — верхний слой",
        "Plastic area fraction — bottom integration layer":"Plastic area fraction — bottom integration layer / Доля пластической площади — нижний слой",
        "Birger nonlinear volume fraction vs load":"Birger nonlinear volume fraction vs load / Доля нелинейного объёма Биргера от нагрузки",
        "Maximum Birger nonlinear fraction through thickness":"Maximum Birger nonlinear fraction through thickness / Максимальная нелинейная доля Биргера по толщине",
        "Birger nonlinear area fraction — top integration layer":"Birger nonlinear area fraction — top layer / Нелинейная площадь Биргера — верхний слой",
        "Birger nonlinear area fraction — bottom integration layer":"Birger nonlinear area fraction — bottom layer / Нелинейная площадь Биргера — нижний слой",
        "Minimum GB/G0 vs load":"Minimum GB/G0 vs load / Минимум GB/G0 от нагрузки",
        "Maximum GB/G0 vs load":"Maximum GB/G0 vs load / Максимум GB/G0 от нагрузки",
        "Minimum detF vs load":"Minimum detF vs load / Минимум detF от нагрузки",
        "through thickness at |w|max":"through thickness at |w|max / по толщине в точке |w|max",
    }
    for a,b in common.items():
        if t==a:t=b
    # Reorder simple Russian-first bilingual phrases into English-first form.
    if " / " in t:
        left,right=t.split(" / ",1)
        has_ru_left=bool(re.search(r"[А-Яа-яЁё]",left));has_ru_right=bool(re.search(r"[А-Яа-яЁё]",right))
        if has_ru_left and not has_ru_right:
            t=f"{right} / {left}"
    return t

def _field_limits(a):
    """Shared finite min/max for every displayed integration layer of one field."""
    finite=np.asarray(a,float)[np.isfinite(a)]
    if not finite.size:return (-0.5,0.5)
    lo=float(np.min(finite)); hi=float(np.max(finite))
    if hi<=lo:
        pad=max(abs(lo)*1.0e-6,1.0e-12); lo-=pad; hi+=pad
    return lo,hi


def _zone_cmap():
    """Semantic binary palette: 0=no zone (grey), 1=zone (red)."""
    return matplotlib.colors.ListedColormap(("#d9dfe5","#c62828"),name="zone_binary")


def save_map(a,title,path,xlabel=None,ylabel=None,colorbar_label=None,*,cmap=None,vmin=None,vmax=None,binary_zone=False):
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    arr=np.asarray(a,float)
    L=_PLOT_CONTEXT.get("L"); R=_PLOT_CONTEXT.get("R")
    kw={"shading":"auto","cmap":_zone_cmap() if binary_zone else cmap}
    if binary_zone:
        kw.update({"vmin":-0.5,"vmax":1.5})
    else:
        if vmin is not None:kw["vmin"]=vmin
        if vmax is not None:kw["vmax"]=vmax
    if L is not None and R is not None and arr.ndim==2:
        xx=np.linspace(0.0,float(L),arr.shape[1]); ss=np.linspace(0.0,2.0*np.pi*float(R),arr.shape[0])
        im=ax.pcolormesh(xx,ss,arr,**kw)
        xlabel=xlabel or f"Axial coordinate x / Продольная координата x [{_PLOT_CONTEXT['length']}]"
        ylabel=ylabel or f"Circumferential coordinate s / Окружная координата s [{_PLOT_CONTEXT['length']}]"
    else:
        xx=np.arange(arr.shape[1] if arr.ndim>1 else arr.size); yy=np.arange(arr.shape[0] if arr.ndim>1 else 1)
        im=ax.pcolormesh(xx,yy,arr if arr.ndim==2 else np.atleast_2d(arr),**kw)
        xlabel=xlabel or "x index / индекс x [-]"; ylabel=ylabel or "s index / индекс s [-]"
    ax.set_title(_bilingual_plot_text(title)); ax.set_xlabel(_bilingual_plot_text(xlabel)); ax.set_ylabel(_bilingual_plot_text(ylabel))
    cb=fig.colorbar(im,ax=ax,ticks=(0,1) if binary_zone else None)
    if binary_zone:
        cb.set_label("0 = no zone / нет зоны; 1 = zone / зона");cb.ax.set_yticklabels(("0 — no / нет","1 — zone / зона"))
    else:cb.set_label(colorbar_label or _field_unit_from_title(title))
    fig.text(0.5,0.012,_surface_variation_note(arr),ha="center",va="bottom",fontsize=8)
    fig.tight_layout(rect=(0,0.045,1,1)); _save_png(fig,path,dpi=160); plt.close(fig)


def _profile_grid_note(xx):
    xx=np.asarray(xx,float)
    if xx.size<2:return f"nx={xx.size}"
    dx=np.diff(xx)
    dx0=float(np.median(dx))
    return f"nx={xx.size}; Δx≈{dx0:.6g} {_PLOT_CONTEXT['length']}; markers = calculation nodes"


def _display_vertical_exaggeration(fig,ax):
    """Rendered vertical magnification relative to a 1:1 data-unit scale."""
    try:
        fig.canvas.draw()
        bbox=ax.get_window_extent()
        x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
        xs=abs(float(x1)-float(x0)); ys=abs(float(y1)-float(y0))
        if xs<=0.0 or ys<=0.0 or bbox.width<=0.0 or bbox.height<=0.0:return None
        return (float(bbox.height)/ys)/(float(bbox.width)/xs)
    except Exception:
        return None


def save_profiles(w,title,path,x=None,extra_rows=None):
    """Engineering w(x): autoscaled vertically, with explicit scale disclosure."""
    w=np.asarray(w,float); nx=w.shape[1]
    xx=np.arange(nx,dtype=float) if x is None or len(x)!=nx else np.asarray(x,float)
    xlabel="x index [-]" if x is None else f"x [{_PLOT_CONTEXT['length']}]"
    base_rows=[0,w.shape[0]//4,w.shape[0]//2,3*w.shape[0]//4]
    rows=sorted(set(base_rows + ([] if extra_rows is None else [int(r) for r in extra_rows])))
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for is_ in rows:
        theta_deg=360.0*float(is_)/max(int(w.shape[0]),1)
        tags=[]
        if is_==0: tags.append("reference")
        if is_==w.shape[0]//2: tags.append("index-center generator")
        if extra_rows is not None and is_ in [int(r) for r in extra_rows]: tags.append("contains global |w|max")
        suffix=("; "+", ".join(tags)) if tags else ""
        ax.plot(xx,w[is_],marker="o",ms=2,label=f"is={is_}, theta={theta_deg:.1f} deg{suffix}")
    ax.set_title(_bilingual_plot_text(title)); ax.set_xlabel(_bilingual_plot_text(xlabel)); ax.set_ylabel(f"Radial displacement w / Радиальный прогиб w [{_PLOT_CONTEXT['displacement']}]"); ax.grid(True,alpha=.3); ax.legend()
    fig.tight_layout(rect=(0,0.085,1,1))
    exaggeration=_display_vertical_exaggeration(fig,ax)
    scale_txt=(f"vertical scale exaggerated ≈ ×{exaggeration:.3g} vs 1:1" if exaggeration is not None and np.isfinite(exaggeration) else "vertical axis autoscaled (not 1:1)")
    fig.text(0.5,0.041,_profile_grid_note(xx),ha="center",va="center",fontsize=8.2)
    fig.text(0.5,0.014,scale_txt,ha="center",va="center",fontsize=8.2,fontweight="semibold")
    _save_png(fig,path,dpi=160); plt.close(fig)


def save_profiles_true_scale(w,title,path,x=None,extra_rows=None):
    """True geometric w(x): one data-unit on x equals one data-unit on w."""
    w=np.asarray(w,float); nx=w.shape[1]
    xx=np.arange(nx,dtype=float) if x is None or len(x)!=nx else np.asarray(x,float)
    xlabel="x index [-]" if x is None else f"x [{_PLOT_CONTEXT['length']}]"
    base_rows=[0,w.shape[0]//4,w.shape[0]//2,3*w.shape[0]//4]
    rows=sorted(set(base_rows + ([] if extra_rows is None else [int(r) for r in extra_rows])))
    fig,ax=plt.subplots(figsize=(10.0,6.0))
    for is_ in rows:
        theta_deg=360.0*float(is_)/max(int(w.shape[0]),1)
        tags=[]
        if is_==0: tags.append("reference")
        if is_==w.shape[0]//2: tags.append("index-center generator")
        if extra_rows is not None and is_ in [int(r) for r in extra_rows]: tags.append("contains global |w|max")
        suffix=("; "+", ".join(tags)) if tags else ""
        ax.plot(xx,w[is_],lw=1.5,label=f"is={is_}, theta={theta_deg:.1f} deg{suffix}")
    ax.axhline(0.0,lw=.8,alpha=.45)
    ax.set_title(_bilingual_plot_text(title)+"\nTRUE GEOMETRIC SCALE 1:1 / ИСТИННЫЙ ГЕОМЕТРИЧЕСКИЙ МАСШТАБ 1:1")
    ax.set_xlabel(_bilingual_plot_text(xlabel)); ax.set_ylabel(f"Radial displacement w / Радиальный прогиб w [{_PLOT_CONTEXT['displacement']}]")
    ax.grid(True,alpha=.18); ax.legend(loc="upper right",fontsize=8)
    if xx.size>1:
        xmin=float(np.min(xx)); xmax=float(np.max(xx)); span=max(xmax-xmin,1e-12)
        ax.set_xlim(xmin,xmax)
        # A 10×6 canvas has width/height≈1.667.  This symmetric y-window,
        # together with aspect='equal', fills the canvas while preserving 1:1 units.
        yhalf=0.30*span
        ax.set_ylim(-yhalf,yhalf)
        ax.set_aspect("equal",adjustable="box")
    maxw=float(np.nanmax(np.abs(w))) if w.size else 0.0
    fig.text(0.5,0.035,_profile_grid_note(xx),ha="center",va="center",fontsize=8.2)
    fig.text(0.5,0.012,f"true scale: 1 {_PLOT_CONTEXT['length']} horizontally = 1 {_PLOT_CONTEXT['displacement']} vertically; max |w|={maxw:.6g} {_PLOT_CONTEXT['displacement']}",ha="center",va="center",fontsize=8.2,fontweight="semibold")
    fig.tight_layout(rect=(0,0.07,1,1)); _save_png(fig,path,dpi=160); plt.close(fig)


def save_history(xs,ys,title,xlabel,ylabel,path):
    if len(xs)==0:return
    fig,ax=plt.subplots(figsize=(8.8,5.6),facecolor="#f7f9fc");ax.set_facecolor("#fbfcfd");ax.plot(xs,ys,marker="o",markersize=5.2,linewidth=2.0)
    ax.set_title(_bilingual_plot_text(title),fontsize=14,fontweight="semibold",pad=10);ax.set_xlabel(_bilingual_plot_text(xlabel),fontsize=11.5);ax.set_ylabel(_bilingual_plot_text(ylabel),fontsize=11.5);ax.tick_params(labelsize=10);ax.grid(True,alpha=.22,linewidth=.75)
    for sp in ax.spines.values():sp.set_color("#aebbc6");sp.set_linewidth(.8)
    fig.tight_layout(pad=1.0);_save_png(fig,path,dpi=180);plt.close(fig)

def save_history_multi(series,title,xlabel,ylabel,path):
    kept=[(xs,ys,label) for xs,ys,label in series if len(xs)>0 and len(ys)>0]
    if not kept:return
    fig,ax=plt.subplots(figsize=(8.8,5.6),facecolor="#f7f9fc");ax.set_facecolor("#fbfcfd")
    markers=("o","x","s","^")
    for k,(xs,ys,label) in enumerate(kept):ax.plot(xs,ys,marker=markers[k%len(markers)],markersize=5.4,linewidth=2.0,label=label)
    ax.set_title(_bilingual_plot_text(title),fontsize=14,fontweight="semibold",pad=10);ax.set_xlabel(_bilingual_plot_text(xlabel),fontsize=11.5);ax.set_ylabel(_bilingual_plot_text(ylabel),fontsize=11.5);ax.tick_params(labelsize=10);ax.grid(True,alpha=.22,linewidth=.75);ax.legend(fontsize=10.2,framealpha=.92)
    for sp in ax.spines.values():sp.set_color("#aebbc6");sp.set_linewidth(.8)
    fig.tight_layout(pad=1.0);_save_png(fig,path,dpi=180);plt.close(fig)

def _visual_compare_signature(summary):
    """Strict signature for pairing Birger and Hooke postprocessing results."""
    def clean(d,keys):
        d=d or {};return tuple((k,d.get(k)) for k in keys)
    return (
        str(summary.get("theory") or "").upper(),str(summary.get("cuf_model") or "").upper(),str(summary.get("method") or "").upper(),str(summary.get("geometry_model") or "").upper(),
        clean(summary.get("geometry"),("L","R","h")),clean(summary.get("grid"),("nx","ns","stencil")),
        json.dumps(summary.get("boundary_conditions") or {},sort_keys=True,ensure_ascii=True),json.dumps(summary.get("load_definition") or {},sort_keys=True,ensure_ascii=True),
    )


def _read_load_wmax_csv(run):
    fp=Path(run)/"NUMERIC"/"LOAD_DEFLECTION.csv"
    if not fp.exists():return None
    try:
        with fp.open("r",newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
        if not rows:return None
        keys=list(rows[0]);wk=next((k for k in keys if k.startswith("w_abs_max")),None);lk=next((k for k in reversed(keys) if "load" in k.lower() or "q [" in k.lower() or "p [" in k.lower()),keys[-1] if keys else None)
        if wk is None or lk is None:return None
        x=[];y=[]
        for r in rows:
            try:x.append(float(r[wk]));y.append(float(r[lk]))
            except Exception:pass
        return (x,y) if x else None
    except Exception:return None


def _save_birger_hooke_comparison(out,summary,roots,birger_w,birger_load,load_axis_label):
    """Create a READY PNG only when a strictly matching GL+Hooke RUN exists."""
    if "BIRGER" not in (str(summary.get("material_model") or "")+str(summary.get("mode") or "")).upper():return None
    if "GREEN" not in str(summary.get("geometry_model") or "").upper():return None
    sig=_visual_compare_signature(summary);best=None
    for cand in sorted(Path(out).parent.iterdir()):
        if not cand.is_dir() or cand.resolve()==Path(out).resolve():continue
        try:sm=_summary(cand)
        except Exception:continue
        if str(sm.get("material_model") or "").upper() not in ("HOOKE","ELASTIC","LINEAR_ELASTIC"):continue
        if _visual_compare_signature(sm)!=sig:continue
        xy=_read_load_wmax_csv(cand)
        if xy is not None:best=(cand,xy);break
    if best is None:return None
    target=roots["loaddef"]/"GL_BIRGER_VS_GL_HOOKE.png"
    save_history_multi([
        ([0.0]+list(birger_w),[0.0]+list(birger_load),"Green-Lagrange + Birger"),
        (best[1][0],best[1][1],"Green-Lagrange + Hooke"),
    ],"Green-Lagrange: Birger vs Hooke",f"|w|max [{_PLOT_CONTEXT['displacement']}]",load_axis_label,target)
    return target


def _read_json(path):
    try: return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception: return None


def _case(out):
    for name in ("INPUT/FINAL_CASE.json","FINAL_CASE.json","RUN_PARAMETERS.json","INPUT/RUN_PARAMETERS.json"):
        p=Path(out)/name
        if p.exists():
            j=_read_json(p)
            if isinstance(j,dict): return j
    return {}


def _is_missing_summary_value(value):
    return value is None or value=={}


def _summary_with_case_fallback(summary,case):
    """Fill postprocessor metadata from CASE when run summary is absent/incomplete."""
    merged=dict(summary) if isinstance(summary,dict) else {}
    if not isinstance(case,dict): return merged
    for key in (
        "theory","cuf_model","method","mode","geometry_model","material_model",
        "rank","units","geometry","material",
        "grid","fdm_grid","mvi_grid","quadrature","boundary_conditions",
        "load_definition","linear","newton","birger","postprocess",
    ):
        if _is_missing_summary_value(merged.get(key)) and not _is_missing_summary_value(case.get(key)):
            merged[key]=case.get(key)
    if _is_missing_summary_value(merged.get("grid")):
        method=str(merged.get("method") or "").upper()
        grid_key="mvi_grid" if method=="MVI" else "fdm_grid"
        if not _is_missing_summary_value(case.get(grid_key)):
            merged["grid"]=case.get(grid_key)
    return merged


def _summary(out):
    case=_case(out)
    for name in ("FINAL_RUN_SUMMARY.json","RECOVERY_CHECKPOINT.json","CASE_RUN_SUMMARY.json","RUN_RESULT.json","SERIES_SUMMARY.json"):
        p=out/name
        if p.exists():
            j=_read_json(p)
            if isinstance(j,dict): return _summary_with_case_fallback(j,case)
    return _summary_with_case_fallback({},case)


def _geometry(summary):
    g=summary.get("geometry",{}) if isinstance(summary,dict) else {}
    return g.get("L"),g.get("R"),g.get("h")


def _resolved_bc_pair(summary):
    """Exact end-support codes for plot/report captions, never mechanics."""
    if not isinstance(summary,dict):
        return ("—","—")
    bc=summary.get("boundary_conditions") or {}
    if not isinstance(bc,dict):
        return ("—","—")
    if bc.get("advanced") is not None:
        return ("CUSTOM","CUSTOM")
    theory=str(summary.get("theory") or "").upper().strip()
    vals=[]
    for side in ("left","right"):
        code=str(bc.get(side) or "").upper().strip()
        if not code:
            vals.append("—"); continue
        try: vals.append(bcreg.resolved_preset_code(code,theory))
        except Exception: vals.append(code)
    return tuple(vals)


def _run_identity_caption(summary):
    if not isinstance(summary,dict):
        return ""
    theory=str(summary.get("theory") or "—").upper()
    geom=str(summary.get("geometry_model") or "—").upper().replace("_","-")
    method=str(summary.get("method") or "—").upper()
    model=str(summary.get("cuf_model") or "").upper()
    model_part=(f" {model}" if theory=="CUF" and model else "")
    left,right=_resolved_bc_pair(summary)
    return f"{theory}{model_part} · {geom} · {method} · {left}–{right}"


def _material(summary):
    m=summary.get("material",{}) if isinstance(summary,dict) else {}
    return m.get("E0"),m.get("nu0")


def _state_from_npz(path):
    with np.load(path,allow_pickle=False) as z: data={k:np.asarray(z[k]) for k in z.files}
    files=set(data); state={"arrays":sorted(files),"raw":data}
    if "q" in files:
        q=np.asarray(data["q"],float); state["kind"]="CUF_FDM"; state["q"]=q; state["w"]=q[2,0]; state["u"]=q[0,0]; state["v"]=q[1,0]
    elif "U" in files:
        U=np.asarray(data["U"],float); state["kind"]="KL_FDM"; state["U"]=U; state["w"]=U[2]; state["u"]=U[0]; state["v"]=U[1]
    elif {"X","Y"}.issubset(files):
        X=np.asarray(data["X"],float); Y=np.asarray(data["Y"],float)
        if X.ndim==3:
            U=np.einsum("crx,crs->csx",X,Y); state["kind"]="KL_MVI"; state["U"]=U; state["w"]=U[2]; state["u"]=U[0]; state["v"]=U[1]
        elif X.ndim==4:
            q=np.einsum("cmrx,cmrs->cmsx",X,Y); state["kind"]="CUF_MVI"; state["q"]=q; state["w"]=q[2,0]; state["u"]=q[0,0]; state["v"]=q[1,0]
        else: state["kind"]="UNKNOWN"
    else: state["kind"]="UNKNOWN"
    if "GB" in files: state["GB"]=np.asarray(data["GB"],float)
    for key in ("epsilon_i","sigma_i","detF","unloading_mask","quadrature_z","quadrature_w"):
        if key in files: state[key]=np.asarray(data[key])
    return state


def _state_files(out):
    """Return accepted state files even after an abrupt GUI STOP.

    Accepted FDM steps are uniquely named STEP_xxxx_STATE.npz; failed trial
    states use *_NOT_CONVERGED.npz.  Therefore the accepted state filename is
    authoritative and safer than a summary/checkpoint that may lag by one file
    if Windows terminates the process between the atomic state save and the
    checkpoint update.  MVI accepted states use STEP_xxxx.npz.
    """
    fdm=sorted(out.glob("STEP_[0-9][0-9][0-9][0-9]_STATE.npz"))
    if fdm:
        return fdm
    return sorted(
        p for p in out.glob("STEP_[0-9][0-9][0-9][0-9].npz")
        if "STATE" not in p.name and "NOT_CONVERGED" not in p.name
    )


def _step_factor(out,index):
    """Recover the physical load factor robustly for a saved accepted step.

    Normal path is STEP_xxxx.json.  During a user STOP there is a very small
    window where the atomic state exists but its JSON/checkpoint has not yet
    been published.  In that case recover the factor from the saved load series
    rather than silently using the step number (which is wrong for e.g. 0.05,
    0.10, ... load histories).
    """
    p=out/f"STEP_{index:04d}.json"; j=_read_json(p)
    if isinstance(j,dict) and j.get("load_factor") is not None:
        try:return float(j["load_factor"])
        except Exception:pass

    gp=out/"GENERATED_LOADS.npz"
    if gp.exists():
        try:
            with np.load(gp,allow_pickle=False) as z:
                if "load_factors" in z.files:
                    a=np.asarray(z["load_factors"],float).reshape(-1)
                    if 1 <= int(index) <= a.size:return float(a[int(index)-1])
        except Exception:pass

    case=_case(out); ld=case.get("load_definition") if isinstance(case,dict) else None
    if isinstance(ld,dict):
        raw=ld.get("definition") if isinstance(ld.get("definition"),dict) else ld
        vals=raw.get("load_factors") if isinstance(raw,dict) else None
        if isinstance(vals,(list,tuple)) and 1 <= int(index) <= len(vals):
            try:return float(vals[int(index)-1])
            except Exception:pass
        # EXTERNAL_NPZ is snapshotted into INPUT by the run bundle.
        npz_value=raw.get("npz") if isinstance(raw,dict) else None
        if npz_value:
            try:
                lp=Path(str(npz_value))
                if not lp.is_absolute():lp=(out/"INPUT"/lp.name)
                if lp.exists():
                    with np.load(lp,allow_pickle=False) as z:
                        if "load_factors" in z.files:
                            a=np.asarray(z["load_factors"],float).reshape(-1)
                            if 1 <= int(index) <= a.size:return float(a[int(index)-1])
            except Exception:pass
    return float(index)


def _deformed_xyz(u,v,w,L,R,scale=1.0):
    ns,nx=w.shape; x=np.linspace(0,float(L),nx); th=np.linspace(0,2*np.pi,ns,endpoint=False)
    X0,T=np.meshgrid(x,th); c=np.cos(T); s=np.sin(T)
    X=X0+scale*u
    Y=(float(R)+scale*w)*c-scale*v*s
    Z=(float(R)+scale*w)*s+scale*v*c
    return X,Y,Z


def _close_circumferential_rows(*arrays):
    """Duplicate theta=0 as theta=2*pi for plotting only."""
    return tuple(np.concatenate((np.asarray(a),np.asarray(a)[:1,...]),axis=0) for a in arrays)


def _static_step_from_title(title):
    m=re.search(r"(?:STEP|step|accepted step)\s*[_-]?(\d+)",str(title))
    return int(m.group(1)) if m else 0


def _static_video_dict(L,R,field,label,unit,*,h=None,step=0,zvals=None):
    sm=_PLOT_CONTEXT.get("summary") or {};geo=sm.get("geometry") or {};grid=sm.get("grid") or {}
    if h is None:
        try:h=float(geo.get("h",0.0) or 0.0)
        except Exception:h=0.0
    if zvals is None:zvals=np.asarray([-0.5*h,0.5*h] if h else [0.0],float)
    d={
        "L":float(L),"R":float(R),"h":float(h or 0.0),"length_unit":_PLOT_CONTEXT.get("length","mm"),
        "field":str(label),"field_label":str(label),"field_unit":str(unit).strip("[]") or "-",
        "color_values":np.asarray(field,float),"zvals":np.asarray(zvals,float),"step":int(step),
        "theory":sm.get("theory","—"),"cuf_model":sm.get("cuf_model","") or "",
        "geometry_model":sm.get("geometry_model",sm.get("geometry","—") if isinstance(sm.get("geometry"),str) else "—"),
        "material_model":sm.get("material_model","—"),"method":sm.get("method","—"),"rank":sm.get("rank"),
        "load_spec":sm.get("load_definition") or {},
    }
    # Keep grid information concise without changing the shared HUD contract.
    if isinstance(grid,dict):d["grid"]=grid
    return d


def _static_3d_axis(fig):
    ax=fig.add_axes([.135,.055,.765,.845],projection="3d");ax.set_facecolor("#f7f9fc");return ax

def _static_colorbar(fig,norm,cmap,label):
    cax=fig.add_axes([.918,.205,.022,.56]);sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap);sm.set_array([]);cb=fig.colorbar(sm,cax=cax);cb.ax.tick_params(labelsize=9.5);cb.set_label(label,fontsize=10.3,labelpad=8);return cb

def _static_title(fig,ru,en,detail=""):
    """Partner-facing 3-D title: English first, Russian second."""
    txt=f"{en} / {ru}"+(f" · {detail}" if detail else "")
    fig.text(.53,.963,txt,ha="center",va="top",fontsize=13.2,fontweight="semibold",color="#14212c")

def _display_upsample_2d(a,ns_target=None,nx_target=None):
    """Bilinear/periodic display interpolation only; calculation arrays are untouched."""
    a=np.asarray(a,float)
    if a.ndim!=2:return a
    ns,nx=a.shape
    ns_target=int(ns_target or min(max(ns,96),180))
    nx_target=int(nx_target or min(max(nx,96),180))
    if ns>=ns_target and nx>=nx_target:return a.copy()
    xo=np.linspace(0.0,1.0,nx);xn=np.linspace(0.0,1.0,nx_target)
    tmp=np.empty((ns,nx_target),float)
    for j in range(ns):tmp[j]=np.interp(xn,xo,a[j])
    so=np.linspace(0.0,1.0,ns+1);sn=np.linspace(0.0,1.0,ns_target,endpoint=False)
    ext=np.vstack((tmp,tmp[:1]));out=np.empty((ns_target,nx_target),float)
    for i in range(nx_target):out[:,i]=np.interp(sn,so,ext[:,i])
    return out


def _display_interpolate_shell_dict(d):
    """Upsample shell render arrays while preserving original extrema metadata."""
    try:raw_info=fvideo._wmax_info(d)
    except Exception:raw_info=None
    for key in ("field3d","uvol","vvol","wvol"):
        a=np.asarray(d.get(key),float)
        if a.ndim==3:
            d[key]=np.stack([_display_upsample_2d(a[g]) for g in range(a.shape[0])],axis=0)
    cv=np.asarray(d.get("color_values"),float)
    if cv.ndim==2:d["color_values"]=_display_upsample_2d(cv)
    elif cv.ndim==3:d["color_values"]=np.stack([_display_upsample_2d(cv[g]) for g in range(cv.shape[0])],axis=0)
    if raw_info is not None:d["_wmax_info_override"]=raw_info
    d["visual_interpolation_note"]="display only / только визуально"
    return d


def save_deformed_shell(u,v,w,L,R,path,title,scale=1.0,load_field=None):
    u=np.asarray(u,float);v=np.asarray(v,float);w=np.asarray(w,float);fig=plt.figure(figsize=(12.8,7.2),facecolor="#f7f9fc");ax=_static_3d_axis(fig);mn=float(np.nanmin(w));mx=float(np.nanmax(w));norm,cmap=fvideo._engineering_norm_cmap(mn,mx)
    d=_static_video_dict(L,R,w,"Transverse displacement w / Радиальный прогиб w",_PLOT_CONTEXT['displacement'],step=_static_step_from_title(title));d.update({"scale":float(scale),"global_max_displacement":float(np.max(np.abs(w))),"global_zmax":0.0,"field3d":w[None,...],"color_values":w,"uvol":u[None,...],"vvol":v[None,...],"wvol":w[None,...],"layer_index":0,"_render_mode":"3D deformation / 3D деформация"})
    d=_display_interpolate_shell_dict(d)
    fvideo._plot_shell(ax,d,norm,cmap);_static_colorbar(fig,norm,cmap,f"w [{_PLOT_CONTEXT['displacement']}]");_static_title(fig,"Деформированная оболочка","Deformed Shell",f"scale / масштаб = {scale:g}");_save_png(fig,path,dpi=180);plt.close(fig)

def save_volume_scatter(field,L,R,h,path,title,label,z_coords=None):
    """Presentation through-thickness 3-D field using translucent shell layers.

    Historical point-scatter output looked like disconnected stripes.  This is
    still the same z-s-x data, but rendered as its physical concentric shell
    layers with an open sector so the through-thickness stack is visible.
    """
    a=np.asarray(field,float)
    if a.ndim!=3:return
    nz,ns,nx=a.shape
    if z_coords is None:
        xi,_=np.polynomial.legendre.leggauss(nz);zz=.5*float(h)*xi
    else:
        zz=np.asarray(z_coords,float)
        if zz.shape!=(nz,):raise ValueError("z_coords shape mismatch in save_volume_scatter")
    finite=a[np.isfinite(a)]
    if not finite.size:return
    norm,cmap=fvideo._engineering_norm_cmap(float(np.min(finite)),float(np.max(finite)))
    fig=plt.figure(figsize=(12.8,7.2),facecolor="#f7f9fc");ax=_static_3d_axis(fig)
    # Remove a sector facing the observer; otherwise the outer layer hides all
    # interior z-layers and the picture ceases to be a volume visualization.
    opening=np.deg2rad(72.0);theta0=0.0;th=np.linspace(theta0+opening/2,theta0-opening/2+2*np.pi,ns)
    jj=np.mod(np.round((np.mod(th,2*np.pi))/(2*np.pi)*ns).astype(int),ns)
    xx=np.linspace(0,float(L),nx);X,T=np.meshgrid(xx,th)
    for g,zg in enumerate(zz):
        C=a[g,jj,:];rr=float(R)+float(zg);Y=rr*np.cos(T);Z=rr*np.sin(T)
        edge=(g in (0,nz-1));alpha=.42 if edge else max(.10,.24/(max(nz-2,1)**.25))
        ax.plot_surface(X,Y,Z,facecolors=cmap(norm(C)),linewidth=0,antialiased=True,shade=False,alpha=alpha)
    # Exposed radial faces connect all z layers and make the wall thickness explicit.
    first_face=None
    for tb in (theta0+opening/2,theta0-opening/2):
        j=int(np.round((tb%(2*np.pi))/(2*np.pi)*ns))%ns
        Xf=np.tile(xx,(nz,1));Yf=np.empty_like(Xf);Zf=np.empty_like(Xf);Cf=np.empty_like(Xf)
        for g,zg in enumerate(zz):
            rr=float(R)+float(zg);Yf[g]=rr*np.cos(tb);Zf[g]=rr*np.sin(tb);Cf[g]=a[g,j,:]
        ax.plot_surface(Xf,Yf,Zf,facecolors=cmap(norm(Cf)),linewidth=.28,edgecolor=(.05,.07,.09,.44),antialiased=True,shade=False,alpha=.98)
        ax.plot(Xf[0],Yf[0],Zf[0],color="#17212b",linewidth=1.35,alpha=.92);ax.plot(Xf[-1],Yf[-1],Zf[-1],color="#17212b",linewidth=1.35,alpha=.92)
        if first_face is None:first_face=(Xf.copy(),Yf.copy(),Zf.copy())
    unit=_field_unit_from_title(title).strip("[]");d=_static_video_dict(L,R,a,label,unit,h=h,zvals=zz);d["global_zmax"]=float(np.max(np.abs(zz))) if len(zz) else 0.0;d["visual_thickness_scale"]=1.0;d["field3d"]=a;d["color_values"]=a
    fvideo._set_3d_axes(ax,d,view="cutaway",show_axes=False);fvideo._draw_dimensions(ax,d,kind="cutaway",theta_cut=theta0+opening/2);fvideo._add_fixed_triad(fig)
    if first_face is not None:
        Xf,Yf,Zf=first_face;m=nx//2
        try:ax.text(Xf[-1,m],Yf[-1,m],Zf[-1,m],f" uppermost integration layer / крайний верхний интеграционный слой, z={zz[-1]:.6g}",fontsize=7.0,color="#253849",fontweight="bold");ax.text(Xf[0,m],Yf[0,m],Zf[0,m],f" lowermost integration layer / крайний нижний интеграционный слой, z={zz[0]:.6g}",fontsize=7.0,color="#253849",fontweight="bold")
        except Exception:pass
    fvideo._hud(ax,d,0,"VOLUME",theta_cut=theta0,thickness_note=True);fvideo._add_section_schematic(fig,d,"cutaway",theta_cut=theta0)
    _static_colorbar(fig,norm,cmap,f"{label} [{unit}]");_static_title(fig,"Объёмное поле по толщине","Through-thickness Volume Field",title)
    _save_png(fig,path,dpi=170);plt.close(fig)

def save_surface_field_3d(field,L,R,z_offset,path,title,label,u=None,v=None,w=None,scale=1.0):
    a=np.asarray(field,float)
    if a.ndim!=2:return
    if u is None:u=np.zeros_like(a)
    if v is None:v=np.zeros_like(a)
    if w is None:w=np.zeros_like(a)
    u=np.asarray(u,float);v=np.asarray(v,float);w=np.asarray(w,float)
    fig=plt.figure(figsize=(12.8,7.2),facecolor="#f7f9fc");ax=_static_3d_axis(fig);mn=float(np.nanmin(a));mx=float(np.nanmax(a));norm,cmap=fvideo._engineering_norm_cmap(mn,mx);unit=_field_unit_from_title(title).strip("[]")
    d=_static_video_dict(L,R,a,label,unit,step=_static_step_from_title(title),zvals=[z_offset]);d.update({"scale":float(scale),"global_zmax":abs(float(z_offset)),"global_max_displacement":float(max(np.max(np.abs(u)),np.max(np.abs(v)),np.max(np.abs(w)))),"field3d":a[None,...],"color_values":a,"uvol":u[None,...],"vvol":v[None,...],"wvol":w[None,...],"layer_index":0,"_render_mode":"3D selected field / 3D выбранное поле"})
    d=_display_interpolate_shell_dict(d)
    fvideo._plot_shell(ax,d,norm,cmap);_static_colorbar(fig,norm,cmap,f"{label} [{unit}]");_static_title(fig,"Поле на поверхности оболочки","Shell Surface Field",title);_save_png(fig,path,dpi=180);plt.close(fig)

def save_cylinder_three_view_overview(u,v,w,L,R,path,derived,scale=1.0):
    """Save one partner-facing 3x3 cylinder result sheet.

    Rows are: total displacement/deformed shape, stress intensity, strain
    intensity.  Columns are: isometric, side and end views.  One shared
    physical normalization is used per row, so the same colour means exactly
    the same calculated value in all three views.  Display interpolation is
    visual only; min/max and colour limits come from the original arrays.
    """
    from matplotlib.collections import LineCollection

    u=np.asarray(u,float);v=np.asarray(v,float);w=np.asarray(w,float)
    if u.ndim!=2 or v.shape!=u.shape or w.shape!=u.shape:return
    if not isinstance(derived,dict):return
    z=np.asarray(derived.get("z",[]),float)
    sig=np.asarray(derived.get("sigma_i",[]),float)
    eps=np.asarray(derived.get("epsilon_i",[]),float)
    if not z.size or sig.ndim!=3 or eps.ndim!=3:return
    g=int(np.argmax(z))
    if sig.shape[1:]!=u.shape or eps.shape[1:]!=u.shape:return

    total=np.sqrt(u*u+v*v+w*w)
    rows=[
        ("1. DEFORMED SHAPE (TOTAL DISPLACEMENT) / ДЕФОРМИРОВАННАЯ ФОРМА (ПОЛНОЕ ПЕРЕМЕЩЕНИЕ)",total,f"Disp. total / Перемещение [{_PLOT_CONTEXT['displacement']}]"),
        ("2. STRESS INTENSITY (VON MISES) / ИНТЕНСИВНОСТЬ НАПРЯЖЕНИЙ (МИЗЕС)",sig[g],f"von Mises / Мизес [{_PLOT_CONTEXT['stress']}]"),
        ("3. STRAIN INTENSITY (VON MISES) / ИНТЕНСИВНОСТЬ ДЕФОРМАЦИЙ (МИЗЕС)",eps[g],"von Mises strain / Деформация Мизеса [-]"),
    ]

    # Smooth geometry/colour display only.  Physical limits below are computed
    # from original row arrays before this interpolation.
    ur=_display_upsample_2d(u);vr=_display_upsample_2d(v);wr=_display_upsample_2d(w)
    ns,nx=ur.shape
    xx=np.linspace(0.0,float(L),nx);th=np.linspace(0.0,2*np.pi,ns,endpoint=False)
    X0,T=np.meshgrid(xx,th);cc=np.cos(T);ss=np.sin(T)
    rr=float(R)+float(scale)*wr
    X=X0+float(scale)*ur;Y=rr*cc-float(scale)*vr*ss;Z=rr*ss+float(scale)*vr*cc
    Xc=np.concatenate((X,X[:1]),axis=0);Yc=np.concatenate((Y,Y[:1]),axis=0);Zc=np.concatenate((Z,Z[:1]),axis=0)

    fig=plt.figure(figsize=(16.5,11.0),facecolor="#f7f9fc")
    fig.text(.5,.982,"CYLINDER SHELL — 3D RESULTS / ЦИЛИНДРИЧЕСКАЯ ОБОЛОЧКА — 3D РЕЗУЛЬТАТЫ",ha="center",va="top",fontsize=15.5,fontweight="bold",color="#14213d")
    y0s=(.665,.355,.045);hax=.235
    x0s=(.035,.338,.641);wax=.245
    view_titles=("Isometric view / Аксонометрия","Side view / Боковой вид","End view / Вид на торец")

    for ir,(row_title,field_raw,cbar_label) in enumerate(rows):
        finite=np.asarray(field_raw,float);finite=finite[np.isfinite(finite)]
        if not finite.size:continue
        mn=float(np.min(finite));mx=float(np.max(finite));norm,cmap=fvideo._engineering_norm_cmap(mn,mx)
        field=_display_upsample_2d(np.asarray(field_raw,float));Fc=np.concatenate((field,field[:1]),axis=0)
        y0=y0s[ir]
        fig.text(.038,y0+hax+.016,row_title,ha="left",va="bottom",fontsize=10.6,fontweight="bold",color="#172b4d")

        for ic in (0,1):
            ax=fig.add_axes([x0s[ic],y0,wax,hax],projection="3d")
            ax.set_facecolor("#fbfcfd")
            ax.plot_surface(Xc,Yc,Zc,facecolors=cmap(norm(Fc)),linewidth=0,antialiased=True,shade=False,alpha=.998)
            # Thin end outlines make both shell edges readable without changing data.
            ax.plot(Xc[:,0],Yc[:,0],Zc[:,0],color="#162536",linewidth=.85,alpha=.78)
            ax.plot(Xc[:,-1],Yc[:,-1],Zc[:,-1],color="#162536",linewidth=.85,alpha=.78)
            rad=max(float(np.nanmax(np.hypot(Yc,Zc))),float(R),1e-12);pad=.035*max(float(L),2*rad,1e-12)
            ax.set_xlim(float(np.nanmin(Xc))-pad,float(np.nanmax(Xc))+pad);ax.set_ylim(-rad-pad,rad+pad);ax.set_zlim(-rad-pad,rad+pad)
            try:ax.set_box_aspect((max(float(np.ptp(Xc)),1e-12),2*rad,2*rad))
            except Exception:pass
            ax.view_init(elev=22 if ic==0 else 0,azim=-58 if ic==0 else -90)
            ax.set_axis_off();ax.set_title(view_titles[ic],fontsize=8.3,fontweight="semibold",loc="left",pad=1.5,color="#243b53")

        # End view is a physical coloured rim at x=L; same norm/cmap as the
        # other two columns.  It deliberately remains open, not a filled disk.
        ax=fig.add_axes([x0s[2]+.01,y0+.006,wax-.025,hax-.012]);ax.set_aspect("equal");ax.set_facecolor("#fbfcfd")
        fend=field[:,-1];uend=ur[:,-1];vend=vr[:,-1];wend=wr[:,-1]
        rend=float(R)+float(scale)*wend;te=np.linspace(0.0,2*np.pi,ns,endpoint=False)
        ye=rend*np.cos(te)-float(scale)*vend*np.sin(te);ze=rend*np.sin(te)+float(scale)*vend*np.cos(te)
        pts=np.column_stack([ye,ze]);pts2=np.vstack([pts,pts[:1]]);segments=np.stack([pts2[:-1],pts2[1:]],axis=1)
        lc=LineCollection(segments,cmap=cmap,norm=norm,linewidths=5.0,capstyle="round");lc.set_array(fend);ax.add_collection(lc)
        rmax=max(float(np.max(np.hypot(ye,ze))),float(R),1e-12)*1.16;ax.set_xlim(-rmax,rmax);ax.set_ylim(-rmax,rmax)
        ax.axhline(0,color="#758394",lw=.55,ls="--",alpha=.55);ax.axvline(0,color="#758394",lw=.55,ls="--",alpha=.55)
        ax.set_xticks([]);ax.set_yticks([]);[sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_title(view_titles[2],fontsize=8.3,fontweight="semibold",loc="left",pad=1.5,color="#243b53")

        cax=fig.add_axes([.908,y0+.028,.012,hax-.055]);sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap);sm.set_array([]);cb=fig.colorbar(sm,cax=cax);cb.ax.tick_params(labelsize=7.7);cb.set_label(cbar_label,fontsize=8.2,labelpad=6)

    fig.text(.5,.012,"Colour mapping / Цветовая привязка: physical min–max from calculated arrays; one common scale per row / физические min–max из расчётных массивов; единая шкала на строку.\nDisplay smoothing / Сглаживание: visualization only; calculated values and extrema are not modified / только визуализация; расчётные значения и экстремумы не изменяются.",fontsize=7.2,color="#425466",ha="center",va="bottom")
    _save_png(fig,path,dpi=190);plt.close(fig)


def save_load_bc_shell(L,R,path,load_field=None,boundary_conditions=None):
    ns=56;nx=22;xx=np.linspace(0,float(L),nx);th=np.linspace(0,2*np.pi,ns+1,endpoint=True);X,T=np.meshgrid(xx,th);Y=float(R)*np.cos(T);Z=float(R)*np.sin(T)
    fig=plt.figure(figsize=(12.8,7.2),facecolor="#f7f9fc");ax=_static_3d_axis(fig)
    ax.plot_wireframe(X,Y,Z,rstride=4,cstride=3,linewidth=.52,alpha=.24)
    tt=np.linspace(0,2*np.pi,180);ax.plot(np.zeros_like(tt),float(R)*np.cos(tt),float(R)*np.sin(tt),linewidth=1.6);ax.plot(np.full_like(tt,float(L)),float(R)*np.cos(tt),float(R)*np.sin(tt),linewidth=1.6)
    if load_field is not None:
        lf=np.asarray(load_field,float);ids=np.argwhere(np.abs(lf)>0)
        if ids.size:
            stride=max(1,len(ids)//50);xs=np.linspace(0,float(L),lf.shape[1])
            for js,ix in ids[::stride]:
                ang=2*np.pi*js/lf.shape[0];mag=np.sign(lf[js,ix]) or 1.0
                ax.quiver(xs[ix],float(R)*np.cos(ang),float(R)*np.sin(ang),0,-mag*np.cos(ang),-mag*np.sin(ang),length=.065*max(float(R),float(L)),normalize=True,alpha=.55,linewidth=.9)
    d=_static_video_dict(L,R,np.asarray(load_field,float) if load_field is not None else np.zeros((1,1)),"Нагрузка / Load","-")
    fvideo._set_3d_axes(ax,d,view="shell",show_axes=True);fvideo._hud(ax,d,0,"LOAD")
    bc=str(boundary_conditions or {})
    _static_title(fig,"Нагрузка и граничные условия","Load and Boundary Conditions",bc)
    _save_png(fig,path,dpi=165);plt.close(fig)

def _load_radial_field(out):
    p=out/"GENERATED_LOADS.npz"
    if p.exists():
        try:
            with np.load(p,allow_pickle=False) as z:
                if "radial_surface_field" in z.files:return np.asarray(z["radial_surface_field"],float)
        except:pass
    return None


def _quadrature_from_summary(summary,h):
    qj=summary.get("quadrature",{}) or {}
    qtype=str(qj.get("type","GAUSS")).upper()
    if qtype=="GAUSS":
        ng=int(qj.get("n_gauss",3))
        return s2.gauss_thickness_quadrature(float(h),ng)
    if qtype in ("LAYERED","SUBLAYERS"):
        bounds=qj.get("layer_bounds",qj.get("bounds"))
        orders=qj.get("n_gauss_per_layer",qj.get("n_gauss"))
        if bounds is None or orders is None:
            raise ValueError("LAYERED quadrature in summary requires layer_bounds and Gauss order")
        return s2.layered_thickness_quadrature(float(h),bounds,orders)
    raise ValueError(f"unsupported quadrature.type in summary: {qtype}")


def _weighted_volume_fraction(mask,z,w,R):
    """V7 physical volume fraction using thickness weights and H=1+z/R.

    The strong grid is uniform in midsurface x,s, so equal surface-node weights
    are used here; thickness integration uses the actual quadrature weights and
    cylindrical Jacobian multiplier.
    """
    m=np.asarray(mask,bool); z=np.asarray(z,float); w=np.asarray(w,float)
    if m.ndim!=3 or m.shape[0]!=z.size or w.shape!=(z.size,):
        raise ValueError("mask/z/w shape mismatch for weighted volume fraction")
    H=1.0+z/float(R)
    if np.any(H<=0.0) or not np.all(np.isfinite(H)):
        raise ValueError("invalid cylindrical thickness metric H=1+z/R")
    wz=w*H
    denom=float(np.sum(wz))*m.shape[1]*m.shape[2]
    if denom<=0.0:
        raise ValueError("non-positive volume quadrature denominator")
    numer=float(np.sum(m.astype(float)*wz[:,None,None]))
    return numer/denom



def _j2_saved_derived(st,summary):
    """Read accepted J2 fields exactly as written by the solver.

    J2 stresses and internal variables are never reconstructed with Hooke in
    postprocessing.  That would erase plastic history and give false fields.
    """
    raw=st.get("raw") if isinstance(st,dict) else None
    if not isinstance(raw,dict):
        raise ValueError("J2 postprocess requires saved STEP raw fields")
    z=np.asarray(raw.get("quadrature_z"),float).reshape(-1)
    qw=np.asarray(raw.get("quadrature_w"),float).reshape(-1)
    if z.size==0 or qw.shape!=z.shape:
        raise ValueError("J2 STEP lacks exact quadrature provenance")
    stress=raw.get("j2_second_piola",raw.get("j2_stress"))
    strain=raw.get("j2_green_strain",raw.get("j2_total_strain"))
    if stress is None:
        raise ValueError("J2 STEP lacks accepted stress output")
    stress=np.asarray(stress,float)
    if strain is None and raw.get("j2_deformation_gradient") is not None:
        # Kinematic output only: recover E=1/2(F^T F-I) from the accepted F.
        F=np.asarray(raw["j2_deformation_gradient"],float)
        if F.shape[-2:]!=(3,3):
            raise ValueError("J2 deformation gradient must end in (3,3)")
        strain=0.5*(
            np.einsum("...ki,...kj->...ij",F,F,optimize=False)
            -np.eye(3)
        )
    if strain is None:
        raise ValueError("J2 STEP lacks accepted stress/strain output")
    strain=np.asarray(strain,float)
    surface_shape=tuple(np.asarray(st["w"],float).shape)
    expected=(z.size,)+surface_shape+(3,3)
    if stress.shape!=expected or strain.shape!=expected:
        raise ValueError(f"J2 tensors must have shape {expected}")

    def components(a,indices):
        return np.stack([a[...,i,j] for i,j in indices],axis=1)

    dev=strain-np.trace(strain,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3.0
    epsilon_i=np.sqrt(np.maximum(
        0.0,(2.0/3.0)*np.sum(dev*dev,axis=(-2,-1))
    ))
    sigma_i=np.asarray(
        raw.get("j2_equivalent_kirchhoff_stress",raw.get("j2_equivalent_stress")),
        float
    )
    if sigma_i.shape!=(z.size,)+surface_shape:
        raise ValueError("J2 equivalent stress shape does not match quadrature")
    finite_measure = "j2_second_piola" in raw

    required=(
        "j2_yield_stress","j2_plastic_multiplier","j2_plastic_mask",
        "j2_unloading_mask","j2_reloading_mask","j2_kappa",
    )
    missing=[name for name in required if name not in raw]
    if missing:
        raise ValueError("J2 STEP lacks material fields: "+", ".join(missing))

    out={
        "epsilon_i":epsilon_i,
        "sigma_i":sigma_i,
        "kappa":np.asarray(raw["j2_kappa"],float),
        "yield_stress":np.asarray(raw["j2_yield_stress"],float),
        "stress_measure":"SECOND_PIOLA" if finite_measure else "SMALL_STRAIN_CAUCHY",
        "stress_label":"Second Piola stress S" if finite_measure else "J2 Cauchy stress",
        "sigma_i_label":"equivalent Kirchhoff stress" if finite_measure else "von Mises Cauchy stress",
        "plastic_multiplier":np.asarray(raw["j2_plastic_multiplier"],float),
        "plastic_mask":np.asarray(raw["j2_plastic_mask"],np.uint8),
        "unloading_mask":np.asarray(raw["j2_unloading_mask"],np.uint8),
        "reloading_mask":np.asarray(raw["j2_reloading_mask"],np.uint8),
        "z":z,
        "quadrature_weights":qw,
        "resultants":{
            key[len("j2_resultant_"):]:np.asarray(value,float)
            for key,value in raw.items() if key.startswith("j2_resultant_")
        },
    }
    theory=str(summary.get("theory") or "").upper()
    if theory=="CUF":
        out["E"]=components(strain,((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)))
        out["S"]=components(stress,((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)))
    elif theory=="KL":
        out["strain"]=components(strain,((0,0),(1,1),(2,2),(0,1)))
        out["stress"]=components(stress,((0,0),(1,1),(0,1)))
    else:
        raise ValueError("J2 postprocess supports CUF or KL only")
    return out

def _postprocess_cuf_ops(summary,grid):
    """Use the same spatial operator family that produced the saved state."""
    gr=summary.get("grid",{}) or {}
    method=str(summary.get("method") or "").upper().strip()
    spatial=summary.get("spatial_discretisation") or {}
    scheme=str(spatial.get("scheme") or "").upper().strip()
    if method=="FDM" and scheme=="MKR_FDM_INDEPENDENT_FD2":
        return mkrfd2.MKRSpatialOperators.build(grid)
    return c3.CylinderSpatialOperators.build(grid,int(gr.get("stencil",7)))


def _cuf_local_kernel_for_geometry(geometry,qpt,qxpt,qspt,z,R,K0,GB,basis):
    geometry=str(geometry or "").upper().strip()
    if geometry=="LINEAR":
        return gl.general_linear_local_kernel(qpt,qxpt,qspt,z,R,K0,GB,basis)
    if geometry=="VON_KARMAN":
        return gvk.general_vk_local_kernel(qpt,qxpt,qspt,z,R,K0,GB,basis)
    if geometry=="GREEN_LAGRANGE":
        return g7.general_local_kernel(qpt,qxpt,qspt,z,R,K0,GB,basis)
    raise ValueError(f"unsupported CUF geometry_model in diagnostic: {geometry}")


def _safe_ratio(num,den):
    num=float(num); den=float(den)
    if not np.isfinite(num) or not np.isfinite(den) or abs(den)<=1.0e-30:
        return None
    return num/den


def _cuf_center_membrane_diagnostic(st,summary,load_factor=None,load_value=None):
    """
    Lightweight point diagnostic at x=L/2, theta=0 for accepted CUF states.

    This is output-only. It does not alter the solver state or convergence.
    It reports the finite-strain E/S/P tensors at z=0 plus two independent
    through-thickness integrations:
      N_S_xx/N_S_ss = integral(S_xx dz)/integral(S_ss dz),
      A_P_x/A_P_s   = integral(H P_xx dz)/integral(P_ss dz).
    The latter matches the tau=0 generalized first-Piola work resultants used
    by the production CUF equilibrium equations.
    """
    if "q" not in st:
        return None
    q=np.asarray(st["q"],float)
    L,R,h=_geometry(summary); E0,nu0=_material(summary)
    if None in (L,R,h,E0,nu0):
        return None
    gr=summary.get("grid",{}) or {}
    nx=int(gr["nx"]); ns=int(gr["ns"])
    grid=mvi.CylinderGrid(float(L),float(R),nx,ns)
    ops=_postprocess_cuf_ops(summary,grid)
    model=summary.get("cuf_model") or "TE1"
    basis=tebasis.get_basis(model)
    qx,qs=c10a.surface_first_derivatives(q,ops,basis)
    ix=int(np.argmin(np.abs(np.asarray(grid.x,float)-0.5*float(L))))
    is0=0
    qpt=q[:,:,is0,ix]; qxpt=qx[:,:,is0,ix]; qspt=qs[:,:,is0,ix]
    ec=lk.ElasticConstants(float(E0),float(nu0))
    quad=_quadrature_from_summary(summary,h)
    ng=int(quad.z.size)
    gb_raw=np.asarray(st.get("GB",ec.G0),float)

    def gb_at(g):
        if gb_raw.ndim==0:
            return float(gb_raw)
        if gb_raw.shape==(ns,nx):
            return float(gb_raw[is0,ix])
        if gb_raw.shape==(ng,ns,nx):
            return float(gb_raw[g,is0,ix])
        return float(ec.G0)

    mode=str(summary.get("mode","LINEAR_ELASTIC")).upper()
    geometry=str(summary.get("geometry_model") or "").upper().strip()
    if not geometry:
        geometry="LINEAR" if mode.startswith("LINEAR") else "GREEN_LAGRANGE"

    g0=int(np.argmin(np.abs(np.asarray(quad.z,float)))) if ng else 0
    rr0=_cuf_local_kernel_for_geometry(
        geometry,qpt,qxpt,qspt,0.0,float(R),ec.K0,gb_at(g0),basis
    )

    NSx=0.0; NSs=0.0; APx=0.0; APs=0.0
    Szz_abs_max=0.0; Sss_abs_max=0.0
    for g,(z,wz) in enumerate(zip(np.asarray(quad.z,float),np.asarray(quad.w,float))):
        rr=_cuf_local_kernel_for_geometry(
            geometry,qpt,qxpt,qspt,float(z),float(R),ec.K0,gb_at(g),basis
        )
        H=1.0+float(z)/float(R)
        NSx += float(wz)*float(rr.S[0,0])
        NSs += float(wz)*float(rr.S[1,1])
        APx += float(wz)*H*float(rr.P[0,0])
        APs += float(wz)*float(rr.P[1,1])
        Szz_abs_max=max(Szz_abs_max,abs(float(rr.S[2,2])))
        Sss_abs_max=max(Sss_abs_max,abs(float(rr.S[1,1])))

    F0=np.asarray(basis.values(0.0),float)
    u_mid=float(F0@q[0,:,is0,ix])
    ux_mid=float(F0@qx[0,:,is0,ix])
    w_mid=float(F0@q[2,:,is0,ix])
    Exx=float(rr0.E[0,0]); Ess=float(rr0.E[1,1]); Ezz=float(rr0.E[2,2])
    Sxx=float(rr0.S[0,0]); Sss=float(rr0.S[1,1]); Szz=float(rr0.S[2,2])
    Pxx=float(rr0.P[0,0]); Pss=float(rr0.P[1,1]); Pzz=float(rr0.P[2,2])
    qR_over_h=None; Sss_to_qR_over_h=None
    try:
        if load_value is not None and abs(float(load_value))>0.0:
            qR_over_h=abs(float(load_value))*float(R)/float(h)
            Sss_to_qR_over_h=abs(Sss)/qR_over_h if qR_over_h>0.0 else None
    except Exception:
        pass

    return {
        "definition":"x=L/2, theta=0, z=0 unless noted",
        "x":float(grid.x[ix]),"theta":0.0,"s":0.0,"ix":ix,"is":is0,
        "load_factor":None if load_factor is None else float(load_factor),
        "load_value":None if load_value is None else float(load_value),
        "u_mid":u_mid,"du_dx_mid":ux_mid,"w_mid":w_mid,
        "E_xx":Exx,"E_ss":Ess,"E_zz":Ezz,
        "S_xx":Sxx,"S_ss":Sss,"S_zz":Szz,
        "P_xx":Pxx,"P_ss":Pss,"P_zz":Pzz,
        "E_xx_over_E_ss":_safe_ratio(Exx,Ess),
        "E_zz_over_E_ss":_safe_ratio(Ezz,Ess),
        "S_xx_over_S_ss":_safe_ratio(Sxx,Sss),
        "S_zz_over_S_ss":_safe_ratio(Szz,Sss),
        "P_xx_over_P_ss":_safe_ratio(Pxx,Pss),
        "N_S_xx":float(NSx),"N_S_ss":float(NSs),
        "N_S_xx_over_N_S_ss":_safe_ratio(NSx,NSs),
        "A_P_x_tau0":float(APx),"A_P_s_tau0":float(APs),
        "A_P_x_over_A_P_s":_safe_ratio(APx,APs),
        "S_zz_abs_max_through_thickness":float(Szz_abs_max),
        "S_ss_abs_max_through_thickness":float(Sss_abs_max),
        "Szz_absmax_over_Sss_absmax":_safe_ratio(Szz_abs_max,Sss_abs_max),
        "qR_over_h_abs_assuming_pressure":qR_over_h,
        "S_ss_abs_over_qR_over_h":Sss_to_qR_over_h,
        "nu":float(nu0),
        "target_Ezz_over_Ess_if_Exx0_Szz0":float(-float(nu0)/(1.0-float(nu0))),
        "target_Exx_over_Ess_if_Sxx0_Szz0":float(-float(nu0)),
        "spatial_operator_scheme":str((summary.get("spatial_discretisation") or {}).get("scheme") or ("MVI_HIGH_ORDER" if str(summary.get("method") or "").upper()=="MVI" else "LEGACY")),
    }


def _save_cuf_axial_reference_profile(st,summary,path):
    """Save u, du/dx and w at theta=0, z=0 for the final CUF state."""
    if "q" not in st:
        return None
    q=np.asarray(st["q"],float)
    L,R,_h=_geometry(summary)
    gr=summary.get("grid",{}) or {}
    grid=mvi.CylinderGrid(float(L),float(R),int(gr["nx"]),int(gr["ns"]))
    ops=_postprocess_cuf_ops(summary,grid)
    basis=tebasis.get_basis(summary.get("cuf_model") or "TE1")
    qx,_qs=c10a.surface_first_derivatives(q,ops,basis)
    F0=np.asarray(basis.values(0.0),float)
    u=np.einsum('t,tx->x',F0,q[0,:,0,:],optimize=False)
    ux=np.einsum('t,tx->x',F0,qx[0,:,0,:],optimize=False)
    w=np.einsum('t,tx->x',F0,q[2,:,0,:],optimize=False)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f)
        wr.writerow(["x","u_z0_theta0","du_dx_z0_theta0","w_z0_theta0"])
        for xv,uv,uxv,wv in zip(np.asarray(grid.x,float),u,ux,w):
            wr.writerow([float(xv),float(uv),float(uxv),float(wv)])
    return str(path)

def _cuf_derived(st,summary):
    q=np.asarray(st["q"],float); L,R,h=_geometry(summary); E0,nu0=_material(summary)
    gr=summary.get("grid",{}); nx=int(gr["nx"]);ns=int(gr["ns"]);stencil=int(gr["stencil"])
    model=summary.get("cuf_model") or "TE1"
    grid=mvi.CylinderGrid(float(L),float(R),nx,ns); ops=_postprocess_cuf_ops(summary,grid); quad=_quadrature_from_summary(summary,h); ng=int(quad.z.size)
    basis=tebasis.get_basis(model); qx,qs=c10a.surface_first_derivatives(q,ops,basis)
    ec=lk.ElasticConstants(float(E0),float(nu0)); GB=np.asarray(st.get("GB",np.full((ng,ns,nx),ec.G0)),float)
    if GB.ndim==0: GB=np.full((ng,ns,nx),float(GB))
    mode=str(summary.get("mode","LINEAR_ELASTIC")).upper()
    geometry=str(summary.get("geometry_model") or "").upper().strip()
    if not geometry:
        # Exact backward compatibility: historic nonlinear CUF files are the
        # full Green--Lagrange branch.  Only new files explicitly select VK.
        geometry="LINEAR" if mode.startswith("LINEAR") else "GREEN_LAGRANGE"
    E=np.empty((ng,6,ns,nx)); S=np.empty_like(E); ei=np.empty((ng,ns,nx)); si=np.empty_like(ei); det=np.ones_like(ei)
    comps=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))
    for idx in np.ndindex((ns,nx)):
        qpt=q[(slice(None),slice(None))+idx]; qxpt=qx[(slice(None),slice(None))+idx]; qspt=qs[(slice(None),slice(None))+idx]
        for g,z in enumerate(quad.z):
            gb=float(GB[(g,)+idx]) if GB.shape==(ng,ns,nx) else ec.G0
            if geometry=="LINEAR":
                rr=gl.general_linear_local_kernel(qpt,qxpt,qspt,z,R,ec.K0,gb,basis)
            elif geometry=="VON_KARMAN":
                rr=gvk.general_vk_local_kernel(qpt,qxpt,qspt,z,R,ec.K0,gb,basis)
            elif geometry=="GREEN_LAGRANGE":
                rr=g7.general_local_kernel(qpt,qxpt,qspt,z,R,ec.K0,gb,basis)
            else:
                raise ValueError(f"unsupported CUF geometry_model in postprocess: {geometry}")
            E[g,:,idx[0],idx[1]]=[rr.E[a,b] for a,b in comps]; S[g,:,idx[0],idx[1]]=[rr.S[a,b] for a,b in comps]
            ei[g,idx[0],idx[1]]=rr.strain_intensity; si[g,idx[0],idx[1]]=rr.stress_intensity
            if hasattr(rr,"detF"):det[g,idx[0],idx[1]]=rr.detF
    if geometry=="LINEAR":
        resultants=gl.general_linear_thickness_resultants(q,qx,qs,R=R,K0=ec.K0,GB=GB,quadrature=quad,basis=basis)
    elif geometry=="VON_KARMAN":
        resultants=gvk.general_vk_thickness_resultants(q,qx,qs,R=R,K0=ec.K0,GB=GB,quadrature=quad,basis=basis)
    else:
        resultants=g7.general_thickness_resultants(q,qx,qs,R=R,K0=ec.K0,GB=GB,quadrature=quad,basis=basis)
    return {"E":E,"S":S,"epsilon_i":ei,"sigma_i":si,"detF":det,"GB":GB,"resultants":resultants.as_dict(),"z":np.asarray(quad.z),"quadrature_weights":np.asarray(quad.w),"basis":model}


def _kl_derived(st,summary):
    U=np.asarray(st["U"],float); L,R,h=_geometry(summary); E0,nu0=_material(summary); gr=summary.get("grid",{})
    grid=mvi.CylinderGrid(float(L),float(R),int(gr["nx"]),int(gr["ns"])); ops=c3.CylinderSpatialOperators.build(grid,int(gr["stencil"])); quad=_quadrature_from_summary(summary,h); ng=int(quad.z.size)
    mode=str(summary.get("mode","LINEAR_ELASTIC")).upper()
    geometry=str(summary.get("geometry_model") or ("LINEAR" if mode.startswith("LINEAR") else "VON_KARMAN")).upper()
    kin=kl.field_kinematics(U,ops,geometry=geometry)
    G0=float(E0)/(2*(1+float(nu0))); K0=float(E0)/(3*(1-2*float(nu0))); GB=np.asarray(st.get("GB",np.full((ng,grid.ns,grid.nx),G0)),float)
    if GB.ndim==0:GB=np.full((ng,grid.ns,grid.nx),float(GB))
    ex=np.empty((ng,grid.ns,grid.nx));es=np.empty_like(ex);gam=np.empty_like(ex);ezz=np.empty_like(ex);sx=np.empty_like(ex);ss=np.empty_like(ex);tau=np.empty_like(ex);ei=np.empty_like(ex);si=np.empty_like(ex)
    birger=mode.endswith("BIRGER")
    for g,z in enumerate(quad.z):
        ex[g],es[g],gam[g]=kl.through_thickness_strains(kin.state,z)
        if birger:
            sx[g],ss[g],tau[g],_Eb,nub=klb.plane_stress_stress_from_KG(ex[g],es[g],gam[g],K0=K0,GB=GB[g]); ei[g],ezz[g]=klb.strain_intensity_plane_stress(ex[g],es[g],gam[g],nub); si[g]=klb.stress_intensity_plane_stress(sx[g],ss[g],tau[g])
        else:
            sx[g],ss[g],tau[g]=kl.plane_stress(ex[g],es[g],gam[g],E=E0,nu=nu0); ezz[g]=-(float(nu0)/(1-float(nu0)))*(ex[g]+es[g]); ei[g],_=klb.strain_intensity_plane_stress(ex[g],es[g],gam[g],float(nu0)); si[g]=klb.stress_intensity_plane_stress(sx[g],ss[g],tau[g])
    if birger: res=klb.frozen_birger_resultants_from_kinematics(kin,K0=K0,GB=GB,quadrature=quad).resultants
    else: res=kl.quadrature_elastic_resultants(kin.state,kl.KLElasticConstants(E0,nu0,h),quad)
    return {"strain":np.stack([ex,es,ezz,gam],axis=1),"stress":np.stack([sx,ss,tau],axis=1),"epsilon_i":ei,"sigma_i":si,"GB":GB,"resultants":{k:np.asarray(getattr(res,k)) for k in ("Nx","Ns","Nxs","Mx","Ms","Mxs")},"z":np.asarray(quad.z),"quadrature_weights":np.asarray(quad.w)}



def _cuf_displacements_from_coefficients(q,basis,zvals):
    q=np.asarray(q,float); zvals=np.asarray(zvals,float).reshape(-1)
    F=np.asarray([basis.values(float(z)) for z in zvals],float)
    return np.einsum("gt,ctsx->gcsx",F,q,optimize=False)


def _birger_zone_arrays(d,summary,R,E0,nu0):
    """Return zone arrays/metrics without changing solver state.

    Plastic mask uses the explicit user-supplied yield criterion when present.
    Independently, Birger nonlinear/degraded zone is GB/G0 < 1-1e-10.
    Material variables are defined at the actual integration points, so surface
    maps use the nearest bottom/middle/top integration levels and record z.
    """
    if d is None or "GB" not in d: return {},{}
    G0=float(E0)/(2.0*(1.0+float(nu0)))
    GB=np.asarray(d["GB"],float); ratio=GB/G0
    z=np.asarray(d["z"],float); qw=np.asarray(d["quadrature_weights"],float)
    nonlinear=ratio < (1.0-1.0e-10)
    H=1.0+z/float(R); wz=qw*H; den=float(np.sum(wz))
    if den<=0.0:
        raise ValueError("non-positive local through-thickness quadrature weight")
    nonlinear_frac=np.sum(nonlinear.astype(float)*wz[:,None,None],axis=0)/den
    gbot=int(np.argmin(z)); gtop=int(np.argmax(z)); gmid=int(np.argmin(np.abs(z)))
    arr={
        "GB_ratio":ratio,
        "birger_nonlinear_mask":nonlinear.astype(np.uint8),
        "birger_nonlinear_thickness_fraction":nonlinear_frac,
        "birger_nonlinear_bottom_layer":nonlinear[gbot].astype(np.uint8),
        "birger_nonlinear_mid_layer":nonlinear[gmid].astype(np.uint8),
        "birger_nonlinear_top_layer":nonlinear[gtop].astype(np.uint8),
    }
    info={
        "GB_ratio_min":float(np.min(ratio)),"GB_ratio_max":float(np.max(ratio)),
        "birger_nonlinear_volume_fraction":float(_weighted_volume_fraction(nonlinear,z,qw,R)),
        "birger_nonlinear_thickness_fraction_max":float(np.max(nonlinear_frac)),
        "birger_nonlinear_area_fraction_bottom_layer":float(np.mean(nonlinear[gbot])),
        "birger_nonlinear_area_fraction_mid_layer":float(np.mean(nonlinear[gmid])),
        "birger_nonlinear_area_fraction_top_layer":float(np.mean(nonlinear[gtop])),
        "birger_bottom_z":float(z[gbot]),"birger_mid_z":float(z[gmid]),"birger_top_z":float(z[gtop]),
    }
    # Coordinates of the most degraded material point.  These are output-only
    # diagnostics and use the same uniform x/s grid as the solver.
    try:
        L=float((summary.get("geometry") or {}).get("L"))
        gg,jj,ii=np.unravel_index(int(np.argmin(ratio)),ratio.shape)
        nx=ratio.shape[2]; ns=ratio.shape[1]
        th=2*np.pi*jj/ns
        info.update({
            "GB_ratio_min_z":float(z[gg]),
            "GB_ratio_min_ix":int(ii),"GB_ratio_min_is":int(jj),
            "GB_ratio_min_x":float(L*ii/(nx-1)) if nx>1 else 0.0,
            "GB_ratio_min_theta":float(th),"GB_ratio_min_s":float(R*th),
        })
    except Exception:
        pass
    bj=summary.get("birger_definition") or {}; curve=bj.get("curve",{}) if isinstance(bj,dict) else {}
    ys=curve.get("yield_stress") if isinstance(curve,dict) else None
    ye=curve.get("yield_strain") if isinstance(curve,dict) else None
    mask=None; criterion=None
    if ys not in (None,""):
        val=float(ys); mask=np.asarray(d["sigma_i"],float)>=val; criterion=f"sigma_i >= {val:g}"
    elif ye not in (None,""):
        val=float(ye); mask=np.asarray(d["epsilon_i"],float)>=val; criterion=f"epsilon_i >= {val:g}"
    if mask is not None:
        frac=np.sum(mask.astype(float)*wz[:,None,None],axis=0)/den
        arr.update({
            "plastic_mask":mask.astype(np.uint8),
            "plastic_thickness_fraction":frac,
            "plastic_bottom_layer":mask[gbot].astype(np.uint8),
            "plastic_mid_layer":mask[gmid].astype(np.uint8),
            "plastic_top_layer":mask[gtop].astype(np.uint8),
        })
        info.update({
            "plastic_zone_criterion":criterion,
            "plastic_volume_fraction":float(_weighted_volume_fraction(mask,z,qw,R)),
            "plastic_thickness_fraction_max":float(np.max(frac)),
            "plastic_area_fraction_bottom_layer":float(np.mean(mask[gbot])),
            "plastic_area_fraction_mid_layer":float(np.mean(mask[gmid])),
            "plastic_area_fraction_top_layer":float(np.mean(mask[gtop])),
            "plastic_bottom_z":float(z[gbot]),"plastic_mid_z":float(z[gmid]),"plastic_top_z":float(z[gtop]),
        })
        try:
            pts=np.argwhere(mask)
            if pts.size:
                gg,jj,ii=(int(x) for x in pts[0])
                L=float((summary.get("geometry") or {}).get("L")); nx=mask.shape[2];ns=mask.shape[1]
                th=2*np.pi*jj/ns
                info.update({
                    "plastic_first_z":float(z[gg]),"plastic_first_ix":ii,"plastic_first_is":jj,
                    "plastic_first_x":float(L*ii/(nx-1)) if nx>1 else 0.0,
                    "plastic_first_theta":float(th),"plastic_first_s":float(R*th),
                })
        except Exception:
            pass
    return arr,info


def _save_cross_section(a,z,L,R,js,ix,path,title,kind="x-z",*,binary_zone=False):
    a=np.asarray(a,float);z=np.asarray(z,float)
    if a.ndim!=3:return
    finite=a[np.isfinite(a)]
    if not finite.size:return
    if binary_zone:
        cmap=_zone_cmap(); norm=matplotlib.colors.BoundaryNorm((-0.5,0.5,1.5),cmap.N)
    else:
        norm,cmap=fvideo._engineering_norm_cmap(float(np.min(finite)),float(np.max(finite)))
    fig,ax=plt.subplots(figsize=(9.2,5.8),facecolor="#f7f9fc");ax.set_facecolor("#fbfcfd");js=max(0,min(int(js),a.shape[1]-1));ix=max(0,min(int(ix),a.shape[2]-1))
    if kind=="x-z":
        sec=a[:,js,:];xx=np.linspace(0,float(L),sec.shape[1]);im=ax.pcolormesh(xx,z,sec,shading="auto",cmap=cmap,norm=norm);xcrit=xx[ix];ax.axvline(xcrit,color="#7b3342",lw=1.3,ls="--",alpha=.8);ax.text(xcrit,float(np.max(z)),"  x(|w|max)",fontsize=9,color="#7b3342",va="top");ax.set_xlabel(f"x / axial coordinate [{_PLOT_CONTEXT['length']}]\nSection B–B / Сечение B–B",fontsize=11);subtitle=f"θ={360.0*js/max(a.shape[1],1):.1f}° · through |w|max"
    else:
        sec=a[:,:,ix];th=np.linspace(0,2*np.pi,sec.shape[1],endpoint=False);im=ax.pcolormesh(th,z,sec,shading="auto",cmap=cmap,norm=norm);tc=th[js];ax.axvline(tc,color="#7b3342",lw=1.3,ls="--",alpha=.8);ax.text(tc,float(np.max(z)),"  θ(|w|max)",fontsize=9,color="#7b3342",va="top");xcut=float(L)*ix/max(a.shape[2]-1,1);ax.set_xlabel("θ / circumferential coordinate [rad]\nSection A–A / Сечение A–A",fontsize=11);subtitle=f"x={xcut:.4g} {_PLOT_CONTEXT['length']} · through |w|max"
    ax.set_ylabel(f"Thickness coordinate z / Координата по толщине z [{_PLOT_CONTEXT['length']}]",fontsize=11);ax.set_title(_bilingual_plot_text(title),fontsize=13.5,fontweight="semibold",pad=9);ax.tick_params(labelsize=9.5);cb=fig.colorbar(im,ax=ax,pad=.025,ticks=(0,1) if binary_zone else None);cb.ax.tick_params(labelsize=9.2)
    if binary_zone:
        cb.set_label("0 = no zone / нет зоны; 1 = zone / зона",fontsize=10);cb.ax.set_yticklabels(("0 — no / нет","1 — zone / зона"))
    else:cb.set_label(_field_unit_from_title(title),fontsize=10)
    ax.grid(False);ax.text(.012,.985,subtitle,transform=ax.transAxes,ha='left',va='top',fontsize=8.5,color='#53677b',bbox=dict(boxstyle='round,pad=.25',facecolor=(1,1,1,.88),edgecolor='none'))
    # large simple section map
    import matplotlib.patches as patches
    iax=fig.add_axes([.72,.70,.20,.20]);iax.set_facecolor('#fff');iax.set_xticks([]);iax.set_yticks([]);iax.set_xlim(0,1);iax.set_ylim(0,1)
    for sp in iax.spines.values():sp.set_color('#bdc9d3');sp.set_linewidth(.8)
    iax.add_patch(patches.Rectangle((.10,.32),.72,.34,fill=False,ec='#66798a',lw=1.1));iax.plot([.10,.82],[.49,.49],color='#d6dde4',lw=.8)
    if kind=="x-z":iax.plot([.10,.82],[.49,.49],color='#2968a2',lw=1.7);iax.text(.66,.55,'B–B',fontsize=7.5,color='#295c95',fontweight='bold')
    else:
        xp=.10+.72*ix/max(a.shape[2]-1,1);iax.plot([xp,xp],[.22,.76],color='#b93434',lw=1.7);iax.text(xp,.79,'A–A',ha='center',fontsize=7.5,color='#8e2020',fontweight='bold')
    iax.text(.03,.04,'Схема сечения / Section map',fontsize=6.8,color='#23384d');fig.subplots_adjust(left=.09,right=.94,bottom=.14,top=.90);_save_png(fig,path,dpi=180);plt.close(fig)

def _save_thickness_profile(z,series,path,title,ylabel):
    fig,ax=plt.subplots(figsize=(8.2,5.4),facecolor="#f7f9fc");ax.set_facecolor("#fbfcfd");ax.plot(np.asarray(series,float),np.asarray(z,float),marker="o",linewidth=2.0,markersize=5.2);ax.set_xlabel(f"{ylabel} {_field_unit_from_title(title)}",fontsize=11);ax.set_ylabel(f"Thickness coordinate z / Координата по толщине z [{_PLOT_CONTEXT['length']}]",fontsize=11);ax.set_title(_bilingual_plot_text(title),fontsize=13.5,fontweight="semibold",pad=9);ax.tick_params(labelsize=9.5);ax.grid(True,alpha=.18,linewidth=.7)
    for sp in ax.spines.values():sp.set_color("#aebbc6");sp.set_linewidth(.8)
    fig.tight_layout();_save_png(fig,path,dpi=180);plt.close(fig)

_FIELD_DISPLAY = {
    "u": "u(x, θ, z) — axial displacement",
    "v": "v(x, θ, z) — circumferential displacement",
    "w": "w(x, θ, z) — radial displacement",
    "S_xx": "σ_x = S_xx — axial normal stress",
    "S_ss": "σ_θ = S_θθ — circumferential normal stress",
    "S_rr": "σ_r = S_rr — radial normal stress",
    "S_xs": "τ_xθ = S_xθ — in-plane shear stress",
    "S_xr": "τ_xr = S_xr — transverse shear stress",
    "S_sr": "τ_θr = S_θr — transverse shear stress",
    "E_xx": "E_xx — axial strain",
    "E_ss": "E_θθ — circumferential strain",
    "E_rr": "E_rr — radial strain",
    "E_xs": "E_xθ — tensor shear strain",
    "E_xr": "E_xr — transverse tensor shear strain",
    "E_sr": "E_θr — transverse tensor shear strain",
}

def _field_display_name(name):
    return _FIELD_DISPLAY.get(str(name), str(name))

def _safe_field_name(name):
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё._+-]+","_",str(name)).strip("._") or "FIELD"


def _save_numeric_section_csv(a,z,L,R,js,ix,base_dir,field_name):
    """Persist x-z, s-z and local z-profile values for one 3-D field."""
    a=np.asarray(a,float); z=np.asarray(z,float)
    if a.ndim!=3 or z.ndim!=1 or a.shape[0]!=z.size:return
    base_dir=Path(base_dir); base_dir.mkdir(parents=True,exist_ok=True)
    safe=_safe_field_name(field_name)
    # x-z at the circumferential index of |w|max
    sec=a[:,int(js),:]; xx=np.linspace(0,float(L),sec.shape[1])
    with (base_dir/f"XZ_{safe}.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f);wr.writerow(["field","z", "x", "is", "value"])
        for g,zz in enumerate(z):
            for i,x in enumerate(xx):wr.writerow([field_name,float(zz),float(x),int(js),float(sec[g,i])])
    # s-z / theta-z at the longitudinal index of |w|max
    sec=a[:,:,int(ix)]; th=np.linspace(0,2*np.pi,sec.shape[1],endpoint=False); ss=float(R)*th
    with (base_dir/f"SZ_{safe}.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f);wr.writerow(["field","z","theta","s","ix","value"])
        for g,zz in enumerate(z):
            for j,t in enumerate(th):wr.writerow([field_name,float(zz),float(t),float(ss[j]),int(ix),float(sec[g,j])])
    with (base_dir/f"Z_{safe}.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f);wr.writerow(["field","z","is","ix","value"])
        for g,zz in enumerate(z):wr.writerow([field_name,float(zz),int(js),int(ix),float(a[g,int(js),int(ix)])])


def _save_standard_sections(d,st,roots,idx,L,R,summary=None,lite=False):
    """Publish x-z / s-z / z profiles for the last accepted state.

    STANDARD uses a compact engineering subset so it remains materially lighter
    than FULL. FULL keeps every available physical 3-D field.
    """
    if d is None:return
    z=np.asarray(d.get("z",[]),float)
    if z.size==0:return
    w=np.asarray(st.get("w"),float)
    if w.ndim!=2:return
    js,ix=np.unravel_index(int(np.argmax(np.abs(w))),w.shape)
    step_root=roots["sections"] / "STEPS" / f"STEP_{int(idx):04d}"
    xz=step_root/"XZ"; sz=step_root/"SZ"
    for q in (xz,sz):q.mkdir(parents=True,exist_ok=True)
    nroot=roots["numeric"] / "SECTIONS" / f"STEP_{int(idx):04d}"
    nroot.mkdir(parents=True,exist_ok=True)

    is_birger=str((summary or {}).get("mode","")).upper().endswith("BIRGER")

    fields=[]
    def add(name,a):
        q=np.asarray(a,float)
        if q.ndim==3 and q.shape[0]==z.size:fields.append((name,q))
    add("epsilon_i",d.get("epsilon_i"))
    add("sigma_i",d.get("sigma_i"))
    if d.get("detF") is not None:add("detF",d.get("detF"))
    if is_birger and d.get("GB") is not None:
        add("GB",d.get("GB"))
        try:
            summary_mat=d.get("_material_G0")
            if summary_mat is not None:add("GB_ratio",np.asarray(d.get("GB"),float)/float(summary_mat))
        except Exception:pass
    if d.get("E") is not None:
        a=np.asarray(d["E"],float)
        for c,name in enumerate(("E_xx","E_ss","E_rr","E_xs","E_xr","E_sr")):
            if a.ndim==4 and c<a.shape[1]:add(name,a[:,c])
    if d.get("S") is not None:
        a=np.asarray(d["S"],float)
        for c,name in enumerate(("S_xx","S_ss","S_rr","S_xs","S_xr","S_sr")):
            if a.ndim==4 and c<a.shape[1]:add(name,a[:,c])
    if d.get("strain") is not None:
        a=np.asarray(d["strain"],float)
        for c,name in enumerate(("eps_xx","eps_ss","eps_zz","gamma_xs")):
            if a.ndim==4 and c<a.shape[1]:add(name,a[:,c])
    if d.get("stress") is not None:
        a=np.asarray(d["stress"],float)
        for c,name in enumerate(("sig_xx","sig_ss","tau_xs")):
            if a.ndim==4 and c<a.shape[1]:add(name,a[:,c])
    if is_birger and d.get("GB_ratio") is not None:add("GB_ratio",d.get("GB_ratio"))
    for name in ("birger_nonlinear_mask","plastic_mask"):
        if d.get(name) is not None:add(name,d.get(name))

    # CUF physical displacements through the actual integration points.
    if st.get("q") is not None:
        try:
            basis=tebasis.get_basis(d.get("_cuf_model") or "TE1")
            disp=_cuf_displacements_from_coefficients(np.asarray(st["q"],float),basis,z)
            for c,name in enumerate(("u","v","w")):add(name,disp[:,c])
        except Exception:pass

    if lite:
        # STANDARD: only the fields most useful for day-to-day engineering
        # inspection. The complete component set remains exclusive to FULL.
        priority=(
            "w","sigma_i","epsilon_i","sig_xx","S_xx","eps_xx","E_xx",
            "detF","GB_ratio","GB","plastic_mask","birger_nonlinear_mask",
        )
        selected=[]
        by_name={name:a for name,a in fields}
        for name in priority:
            if name in by_name and all(name!=n for n,_ in selected):
                selected.append((name,by_name[name]))
        fields=selected

    # The legacy GRAPHS/THROUGH_THICKNESS directory is a public result folder
    # and must be useful even when no optional control points are configured.
    # Publish the same physical z-profiles generated for the last FULL step
    # there, rather than leaving it empty and relying on a GUI alias.
    troot=roots["thickness"]
    troot.mkdir(parents=True,exist_ok=True)
    tnroot=roots["numeric"] / "THROUGH_THICKNESS"
    tnroot.mkdir(parents=True,exist_ok=True)
    profile_index=[]
    for name,a in fields:
        safe=_safe_field_name(name)
        _save_cross_section(a,z,L,R,js,ix,xz/f"{safe}.png",f"STEP {idx}: {_field_display_name(name)} | x-z section at s(wmax)","x-z")
        _save_cross_section(a,z,L,R,js,ix,sz/f"{safe}.png",f"STEP {idx}: {_field_display_name(name)} | s-z section at x(wmax)","s-z")
        # One physical z-profile, one PNG.  Older releases wrote the same image
        # twice (SECTIONS/.../Z_PROFILES and THROUGH_THICKNESS).  Keep the
        # public THROUGH_THICKNESS location as the canonical graph and retain
        # XZ/SZ under SECTIONS.
        public_png=troot/f"STEP_{int(idx):04d}__{safe}.png"
        _save_thickness_profile(z,a[:,js,ix],public_png,f"STEP {idx}: {_field_display_name(name)} | through thickness at |w|max",_field_display_name(name))
        _save_numeric_section_csv(a,z,L,R,js,ix,nroot,name)
        src_csv=nroot/f"Z_{safe}.csv"
        public_csv=tnroot/f"STEP_{int(idx):04d}__{safe}.csv"
        if src_csv.exists():shutil.copy2(src_csv,public_csv)
        profile_index.append({
            "step":int(idx),"field":str(name),"graph":public_png.name,
            "csv":public_csv.name if public_csv.exists() else None,
            "is":int(js),"ix":int(ix),
        })
    with (tnroot/"PROFILE_INDEX.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=["step","field","graph","csv","is","ix"]);wr.writeheader();wr.writerows(profile_index)
    (troot/"README.txt").write_text(
        f"Through-thickness profiles / Профили по толщине: STEP_{int(idx):04d}\n"
        f"Location / Положение: |w|max node / узел, is={int(js)}, ix={int(ix)}.\n"
        "STANDARD contains the key engineering subset; FULL contains all available physical fields. / "
        "STANDARD содержит ключевой инженерный набор; FULL содержит все доступные физические поля.\n",
        encoding="utf-8"
    )
    (nroot/"SECTION_LOCATION.json").write_text(json.dumps({
        "step":int(idx),"wmax_is":int(js),"wmax_ix":int(ix),
        "section_definition":"XZ at s-index of |w|max; SZ at x-index of |w|max; Z profile at |w|max node",
    },indent=2,ensure_ascii=False),encoding="utf-8")

def _save_step_derived_maps(d,roots,idx):
    """Save the full practical 2-D field set for an accepted load step.

    Material fields live at through-thickness integration levels, so surface
    plots are intentionally generated at bottom/middle/top integration levels.
    Resultants are thickness-integrated already and are saved directly.
    """
    if d is None:return
    z=np.asarray(d.get("z",[]),float)
    if z.size==0:return
    gbot=int(np.argmin(z)); gtop=int(np.argmax(z)); gmid=int(np.argmin(np.abs(z)))
    levels=((gbot,"BOTTOM"),(gmid,"MID"),(gtop,"TOP"))

    sdir=roots["strain"] / "STEPS" / f"STEP_{idx:04d}"
    tdir=roots["stress"] / "STEPS" / f"STEP_{idx:04d}"
    rdir=roots["resultants"] / "STEPS" / f"STEP_{idx:04d}"
    cdir=roots["cufdiag"] / "STEPS" / f"STEP_{idx:04d}"
    for q in (sdir,tdir,rdir,cdir):q.mkdir(parents=True,exist_ok=True)

    def save_components(arr,names,outdir,prefix):
        a=np.asarray(arr,float)
        if a.ndim!=4:return
        for ci,name in enumerate(names):
            if ci>=a.shape[1]:break
            for g,label in levels:
                caption=d.get("stress_label") if outdir==tdir else None
                field_name=(f"{caption}: {name}" if caption else _field_display_name(name))
                save_map(a[g,ci],f"STEP {idx}: {field_name} — {label} integration level, z={z[g]:.6g}",outdir/f"{prefix}_{name}_{label}.png")

    if "E" in d:
        save_components(d["E"],("E_xx","E_ss","E_rr","E_xs","E_xr","E_sr"),sdir,"CUF")
    if "S" in d:
        save_components(d["S"],("S_xx","S_ss","S_rr","S_xs","S_xr","S_sr"),tdir,"CUF")
    if "strain" in d:
        save_components(d["strain"],("eps_xx","eps_ss","eps_zz","gamma_xs"),sdir,"KL")
    if "stress" in d:
        save_components(d["stress"],("sig_xx","sig_ss","tau_xs"),tdir,"KL")

    for key,outdir,title in (("epsilon_i",sdir,"epsilon_i"),("sigma_i",tdir,"sigma_i")):
        if key in d:
            a=np.asarray(d[key],float)
            if a.ndim==3:
                for g,label in levels:
                    intensity_label=d.get(f"{key}_label",title)
                    save_map(a[g],f"STEP {idx}: {intensity_label} — {label.lower()} integration layer, z={z[g]:.6g}",outdir/f"{title.upper()}_{label}.png")
                save_map(np.max(a,axis=0),f"STEP {idx}: max {title} through thickness",outdir/f"{title.upper()}_MAX_THROUGH_THICKNESS.png")

    if "detF" in d:
        a=np.asarray(d["detF"],float)
        if a.ndim==3:
            for g,label in levels:
                save_map(a[g],f"STEP {idx}: detF — {label.lower()} integration layer, z={z[g]:.6g}",cdir/f"DETF_{label}.png")
            save_map(np.min(a,axis=0),f"STEP {idx}: min detF through thickness",cdir/"DETF_MIN_THROUGH_THICKNESS.png")
            save_map(np.max(np.abs(a-1.0),axis=0),f"STEP {idx}: max |J-1| through thickness",cdir/"ABS_J_MINUS_1_MAX_THROUGH_THICKNESS.png")

    for key,title in (
        ("plastic_mask","J2 plastic zone"),
        ("unloading_mask","J2 unloading"),
        ("reloading_mask","J2 reloading"),
    ):
        a=d.get(key)
        if a is None:
            continue
        a=np.asarray(a,float)
        if a.ndim==3:
            for g,label in levels:
                save_map(
                    a[g],
                    f"STEP {idx}: {title} — {label.lower()} integration layer, z={z[g]:.6g}",
                    cdir/f"{key.upper()}_{label}.png",
                    binary_zone=True,
                )
            save_map(
                np.any(a>0.5,axis=0).astype(float),
                f"STEP {idx}: {title} anywhere through thickness",
                cdir/f"{key.upper()}_ANY.png",
                binary_zone=True,
            )

    for name,a in (d.get("resultants") or {}).items():
        q=np.asarray(a,float)
        safe=str(name).replace("/","_").replace("\\","_")
        if q.ndim==2:
            save_map(q,f"STEP {idx}: resultant {name}",rdir/f"{safe}.png")
        elif q.ndim==3:
            for tau in range(q.shape[0]):
                save_map(q[tau],f"STEP {idx}: resultant {name}, tau={tau}",rdir/f"{safe}_TAU{tau}.png")

def _derive_step_and_save(st,summary,roots,idx,L,R,h,E0,nu0,load_factor=1.0,radial_surface_field=None,make_graphs=True,lite_sections=False):
    j2=str(summary.get("material_model") or "").upper()=="J2_INCREMENTAL"
    if j2:
        d=_j2_saved_derived(st,summary)
        theory=str(summary.get("theory") or "").upper()
        if theory not in ("CUF","KL"):
            return None,{}
        if theory=="CUF":
            d["_cuf_model"]=summary.get("cuf_model") or "TE1"
    elif "q" in st:
        d=_cuf_derived(st,summary); theory="CUF"
        d["_cuf_model"]=summary.get("cuf_model") or "TE1"
    elif "U" in st:
        d=_kl_derived(st,summary); theory="KL"
    else:
        return None,{}
    arrays={k:v for k,v in d.items() if isinstance(v,np.ndarray)}
    if st.get("unloading_mask") is not None:
        arrays["unloading_mask"]=np.asarray(st["unloading_mask"],np.uint8)
    for k,v in d.get("resultants",{}).items(): arrays["resultant_"+k]=np.asarray(v)
    if theory=="CUF":
        q=np.asarray(st["q"],float); basis=tebasis.get_basis(summary.get("cuf_model") or "TE1")
        arrays["q_coefficients"]=q
        arrays["disp_gauss"]=_cuf_displacements_from_coefficients(q,basis,d["z"])
        zstd=np.asarray([-0.5*float(h),-0.25*float(h),0.0,0.25*float(h),0.5*float(h)])
        arrays["standard_z"]=zstd
        arrays["disp_standard_z"]=_cuf_displacements_from_coefficients(q,basis,zstd)

    # Persist the physical applied radial surface field for this accepted step
    # when it comes from the built-in load generator.  EXTERNAL_NPZ can contain
    # arbitrary generalized loading, so no radial field is invented for it.
    if radial_surface_field is not None:
        arrays["radial_surface_load"]=float(load_factor)*np.asarray(radial_surface_field,float)

    # Save the undeformed-scale physical midsurface coordinates of the accepted
    # state.  This is output-only redundancy that lets the GUI/export tools
    # inspect the deformed shell without rebuilding it from the solver.
    try:
        u0=np.asarray(st["u"],float); v0=np.asarray(st["v"],float); w0=np.asarray(st["w"],float)
        ns0,nx0=w0.shape
        xx=np.linspace(0.0,float(L),nx0); th=np.linspace(0.0,2*np.pi,ns0,endpoint=False)
        X0,T=np.meshgrid(xx,th); ct=np.cos(T); stn=np.sin(T)
        arrays["deformed_midsurface_xyz"]=np.stack([
            X0+u0,
            (float(R)+w0)*ct-v0*stn,
            (float(R)+w0)*stn+v0*ct,
        ],axis=0)
    except Exception:
        pass
    zone_info={}
    try:d["_material_G0"]=float(E0)/(2.0*(1.0+float(nu0)))
    except Exception:pass
    if str(summary.get("mode","")).upper().endswith("BIRGER"):
        za,zone_info=_birger_zone_arrays(d,summary,R,E0,nu0); arrays.update(za); d.update(za)
    np.savez_compressed(roots["numeric"]/f"STEP_{idx:04d}_DERIVED.npz",**arrays)
    # Every accepted step gets a complete numeric package so the GUI can
    # inspect any step.  Heavy PNG generation remains optional; FAST therefore
    # stays lightweight while its direct numeric viewer is still complete.
    if make_graphs:
        _save_step_derived_maps(d,roots,idx)
        _save_standard_sections(d,st,roots,idx,L,R,summary=summary,lite=lite_sections)

    # Compact per-step plasticity package: maps + two thickness cuts + profile.
    # These are intentionally generated only when heavy graphics are requested.
    if make_graphs and str(summary.get("mode","")).upper().endswith("BIRGER"):
        pdir=roots["plasticity"]/"STEPS"/f"STEP_{idx:04d}"; pdir.mkdir(parents=True,exist_ok=True)
        ratio=np.asarray(arrays.get("GB_ratio"),float)
        save_map(np.min(ratio,axis=0),f"STEP {idx}: min GB/G0 through thickness",pdir/"GB_RATIO_MIN.png")
        save_map(np.max(ratio,axis=0),f"STEP {idx}: max GB/G0 through thickness",pdir/"GB_RATIO_MAX.png")
        _save_unloading_mask_3d(
            st.get("unloading_mask"),L,R,h,d.get("z"),
            roots["birger3d"]/f"STEP_{idx:04d}_UNLOADING_MASK_3D.png",
            f"STEP {idx}: Birger unloading mask, Delta epsilon_i < 0",
        )
        w=np.asarray(st["w"],float); js,ix=np.unravel_index(int(np.argmax(np.abs(w))),w.shape)
        z=np.asarray(d["z"],float)
        gbot=int(np.argmin(z)); gtop=int(np.argmax(z)); gmid=int(np.argmin(np.abs(z)))
        # The material variables live at the actual integration levels.  These
        # three maps are intentionally labelled as integration layers rather
        # than geometric surfaces.
        eps_i=np.asarray(d["epsilon_i"],float); sigma_i=np.asarray(d["sigma_i"],float)
        ratio_lo,ratio_hi=_field_limits(ratio); eps_lo,eps_hi=_field_limits(eps_i); sig_lo,sig_hi=_field_limits(sigma_i)
        for g,name in ((gbot,"BOTTOM"),(gmid,"MID"),(gtop,"TOP")):
            # A field has one scale through thickness: identical values have identical colours.
            save_map(ratio[g],f"STEP {idx}: GB/G0 — {name.lower()} integration layer, z={z[g]:.6g}",pdir/f"GB_RATIO_{name}.png",vmin=ratio_lo,vmax=ratio_hi)
            save_map(eps_i[g],f"STEP {idx}: epsilon_i — {name.lower()} integration layer, z={z[g]:.6g}",pdir/f"EPS_I_{name}.png",vmin=eps_lo,vmax=eps_hi)
            save_map(sigma_i[g],f"STEP {idx}: sigma_i — {name.lower()} integration layer, z={z[g]:.6g}",pdir/f"SIGMA_I_{name}.png",vmin=sig_lo,vmax=sig_hi)
        if "birger_nonlinear_thickness_fraction" in arrays:
            save_map(arrays["birger_nonlinear_thickness_fraction"],f"STEP {idx}: Birger nonlinear fraction through thickness",pdir/"BIRGER_NONLINEAR_THICKNESS_FRACTION.png")
            save_map(np.any(np.asarray(arrays["birger_nonlinear_mask"],bool),axis=0).astype(float),f"STEP {idx}: 2D Birger nonlinear zone anywhere through thickness",pdir/"BIRGER_NONLINEAR_ZONE_2D_ANY.png",binary_zone=True)
            for key,title in (("birger_nonlinear_bottom_layer","bottom integration layer"),("birger_nonlinear_mid_layer","middle integration layer"),("birger_nonlinear_top_layer","top integration layer")):
                save_map(arrays[key],f"STEP {idx}: Birger nonlinear zone — {title}",pdir/(key.upper()+".png"),binary_zone=True)
        _save_cross_section(ratio,z,L,R,js,ix,pdir/"GB_RATIO_XZ_AT_WMAX.png",f"STEP {idx}: GB/G0 x-z at s(wmax)","x-z")
        _save_cross_section(ratio,z,L,R,js,ix,pdir/"GB_RATIO_SZ_AT_WMAX.png",f"STEP {idx}: GB/G0 theta-z at x(wmax)","s-z")
        _save_thickness_profile(z,ratio[:,js,ix],pdir/"GB_RATIO_Z_AT_WMAX.png",f"STEP {idx}: GB/G0 through thickness at wmax","GB/G0")
        _save_thickness_profile(z,d["epsilon_i"][:,js,ix],pdir/"EPS_I_Z_AT_WMAX.png",f"STEP {idx}: epsilon_i through thickness at wmax","epsilon_i")
        _save_thickness_profile(z,d["sigma_i"][:,js,ix],pdir/"SIGMA_I_Z_AT_WMAX.png",f"STEP {idx}: sigma_i through thickness at wmax","sigma_i")
        _save_cross_section(np.asarray(arrays["birger_nonlinear_mask"],float),z,L,R,js,ix,pdir/"BIRGER_NONLINEAR_XZ_AT_WMAX.png",f"STEP {idx}: Birger nonlinear zone x-z at s(wmax)","x-z",binary_zone=True)
        _save_cross_section(np.asarray(arrays["birger_nonlinear_mask"],float),z,L,R,js,ix,pdir/"BIRGER_NONLINEAR_SZ_AT_WMAX.png",f"STEP {idx}: Birger nonlinear zone theta-z at x(wmax)","s-z",binary_zone=True)
        if "plastic_mask" in arrays:
            pm=np.asarray(arrays["plastic_mask"],float)
            save_map(arrays["plastic_thickness_fraction"],f"STEP {idx}: plastic thickness fraction hp/h",pdir/"PLASTIC_THICKNESS_FRACTION.png")
            save_map(np.any(pm>0.5,axis=0).astype(float),f"STEP {idx}: 2D plastic zone anywhere through thickness",pdir/"PLASTIC_ZONE_2D_ANY.png",binary_zone=True)
            for key,title in (("plastic_bottom_layer","bottom integration layer"),("plastic_mid_layer","middle integration layer"),("plastic_top_layer","top integration layer")):
                save_map(arrays[key],f"STEP {idx}: plastic zone — {title}",pdir/(key.upper()+".png"),binary_zone=True)
            _save_cross_section(pm,z,L,R,js,ix,pdir/"PLASTIC_XZ_AT_WMAX.png",f"STEP {idx}: plastic zone x-z at s(wmax)","x-z",binary_zone=True)
            _save_cross_section(pm,z,L,R,js,ix,pdir/"PLASTIC_SZ_AT_WMAX.png",f"STEP {idx}: plastic zone theta-z at x(wmax)","s-z",binary_zone=True)
    return d,zone_info


def _progress_events(out):
    p=Path(out)/"LIVE_PROGRESS.jsonl"; rows=[]
    if not p.exists():return rows
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        try:
            j=json.loads(line)
            if isinstance(j,dict):rows.append(j)
        except Exception:pass
    return rows


def _save_progress_history(out,roots):
    rows=_progress_events(out)
    if not rows:return {}
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (roots["numeric"]/"ONLINE_HISTORY.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows([{k:r.get(k) for k in keys} for r in rows])
    newton=[r for r in rows if r.get("event")=="newton_iteration"]
    if newton:
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        for step in sorted(set(int(r.get("step_index",0)) for r in newton)):
            q=[r for r in newton if int(r.get("step_index",0))==step]
            ax.semilogy([r.get("iteration",0) for r in q],[max(float(r.get("residual_convergence_relative",r.get("residual_relative",0)) or 0),1e-300) for r in q],marker="o",label=f"step {step}")
        ax.set_xlabel("Newton iteration / Итерация Newton [-]");ax.set_ylabel("Relative convergence residual / Относительная невязка [-]");ax.set_title("Newton convergence / Сходимость Newton");ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();_save_png(fig,roots["conv"]/"NEWTON_RESIDUAL_HISTORY.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        for step in sorted(set(int(r.get("step_index",0)) for r in newton)):
            q=[r for r in newton if int(r.get("step_index",0))==step]
            ax.semilogy([r.get("iteration",0) for r in q],[max(float(r.get("increment_inf",0) or 0),1e-300) for r in q],marker="o",label=f"step {step}")
        ax.set_xlabel("Newton iteration / Итерация Newton [-]");ax.set_ylabel("Increment infinity norm / Норма приращения inf [solver units]");ax.set_title("Newton increment history / История приращений Newton");ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();_save_png(fig,roots["conv"]/"NEWTON_INCREMENT_HISTORY.png",dpi=160);plt.close(fig)
    mvirows=[r for r in rows if r.get("event")=="mvi_cycle"]
    if mvirows:
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        for step in sorted(set(int(r.get("step_index",0)) for r in mvirows)):
            q=[r for r in mvirows if int(r.get("step_index",0))==step]
            ax.semilogy([r.get("cycle",0) for r in q],[max(float(r.get("projected_relative",r.get("projected_inf",0)) or 0),1e-300) for r in q],marker="o",label=f"step {step}")
        ax.set_xlabel("X↔Y cycle / Цикл X↔Y [-]");ax.set_ylabel("Relative projected residual Rproj / Относительная Rproj [-]");ax.set_title("MVI projected residual / Проекционная невязка МВИ");ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();_save_png(fig,roots["conv"]/"MVI_RPROJ_HISTORY.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        for step in sorted(set(int(r.get("step_index",0)) for r in mvirows)):
            q=[r for r in mvirows if int(r.get("step_index",0))==step]
            ax.semilogy([r.get("cycle",0) for r in q],[max(float(r.get("field_increment_rel",0) or 0),1e-300) for r in q],marker="o",label=f"step {step}")
        ax.set_xlabel("X↔Y cycle / Цикл X↔Y [-]");ax.set_ylabel("Relative field change / Относительное изменение поля [-]");ax.set_title("MVI field-change history / История изменения поля МВИ");ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();_save_png(fig,roots["conv"]/"MVI_FIELD_CHANGE_HISTORY.png",dpi=160);plt.close(fig)
    brows=[r for r in rows if r.get("event")=="birger_outer"]
    if brows:
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        for step in sorted(set(int(r.get("step_index",0)) for r in brows)):
            q=[r for r in brows if int(r.get("step_index",0))==step]; x=[int(r.get("outer_iteration",i+1)) for i,r in enumerate(q)]
            ax.semilogy(x,[max(float(r.get("eta_G",0) or 0),1e-300) for r in q],marker="o",label=f"eta_G step {step}")
            ax.semilogy(x,[max(float(r.get("eta_sigma",0) or 0),1e-300) for r in q],marker="x",label=f"eta_sigma step {step}")
        ax.set_xlabel("Birger outer iteration / Внешняя итерация Биргера [-]");ax.set_ylabel("Convergence metric η / Метрика сходимости η [-]");ax.set_title("Birger outer convergence / Внешняя сходимость Биргера");ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();_save_png(fig,roots["conv"]/"BIRGER_ETA_HISTORY.png",dpi=160);plt.close(fig)
    return {"events":len(rows),"newton_events":len(newton),"mvi_cycle_events":len(mvirows),"birger_outer_events":len(brows)}


def _save_final_derived(st,summary,roots,d=None,all_levels=False):
    """Publish the final accepted derived state without redundant PNG work.

    All through-thickness integration levels remain in DERIVED_FINAL_FIELDS.npz
    and Excel/CSV.  For the pre-rendered 2-D map gallery we publish the
    physically representative bottom/middle/top integration layers.  Any
    other z level is still rendered on demand by the GUI from the same full
    numeric package.
    """
    if d is None:
        if str(summary.get("material_model") or "").upper()=="J2_INCREMENTAL":
            d=_j2_saved_derived(st,summary)
            theory=str(summary.get("theory") or "").upper()
            if theory not in ("CUF","KL"): return None
        elif "q" in st: d=_cuf_derived(st,summary); theory="CUF"
        elif "U" in st: d=_kl_derived(st,summary); theory="KL"
        else:return None
    else:
        theory="CUF" if "q" in st else ("KL" if "U" in st else None)
        if theory is None:return None
    npz={k:v for k,v in d.items() if isinstance(v,np.ndarray)}
    for k,v in d.get("resultants",{}).items():npz["resultant_"+k]=np.asarray(v)
    np.savez_compressed(roots["numeric"]/"DERIVED_FINAL_FIELDS.npz",**npz)
    try:
        (roots["report"]/"FULL_GRAPH_POLICY.txt").write_text(
            "Numerical fields contain every saved through-thickness integration level.\n"
            + ("FULL pre-renders every saved z level.\n" if all_levels else "STANDARD pre-renders bottom/middle/top z levels.\n")
            + "XZ, SZ and THROUGH_THICKNESS profiles use the complete z array; any z level can also be rendered on demand in the GUI.\n",
            encoding="utf-8"
        )
    except Exception:
        pass

    z=np.asarray(d.get("z",[]),float)
    if z.size:
        if all_levels:
            levels=list(range(int(z.size)))
        else:
            levels=[]
            for g in (int(np.argmin(z)),int(np.argmin(np.abs(z))),int(np.argmax(z))):
                if g not in levels:levels.append(g)
    else:
        levels=[]
    if theory=="CUF":
        names=("xx","ss","rr","xs","xr","sr")
        for g in levels:
            zz=z[g]
            for j,n in enumerate(names):
                save_map(d["E"][g,j],f"Green/linear strain E_{n}; z={zz:.6g}",roots["strain"]/f"FINAL_E_{n}_G{g+1:02d}.png")
                save_map(d["S"][g,j],f"Second Piola stress S_{n}; z={zz:.6g}",roots["stress"]/f"FINAL_S_{n}_G{g+1:02d}.png")
    else:
        en=("xx","ss","zz","gamma_xs"); sn=("xx","ss","xs")
        for g in levels:
            zz=z[g]
            for j,n in enumerate(en):save_map(d["strain"][g,j],f"KL strain {n}; z={zz:.6g}",roots["strain"]/f"FINAL_E_{n}_G{g+1:02d}.png")
            for j,n in enumerate(sn):
                label=d.get("stress_label","KL stress")
                save_map(d["stress"][g,j],f"{label} {n}; z={zz:.6g}",roots["stress"]/f"FINAL_S_{n}_G{g+1:02d}.png")
    eps_lo,eps_hi=_field_limits(d["epsilon_i"]); sig_lo,sig_hi=_field_limits(d["sigma_i"])
    for g in levels:
        zz=z[g]
        save_map(d["epsilon_i"][g],f"strain intensity; z={zz:.6g}",roots["strain"]/f"FINAL_EPS_I_G{g+1:02d}.png",vmin=eps_lo,vmax=eps_hi)
        intensity_label=d.get("sigma_i_label","stress intensity")
        save_map(d["sigma_i"][g],f"{intensity_label}; z={zz:.6g}",roots["stress"]/f"FINAL_SIGMA_I_G{g+1:02d}.png",vmin=sig_lo,vmax=sig_hi)
    for k,a in d["resultants"].items():
        a=np.asarray(a,float)
        if a.ndim==2: save_map(a,f"{k}",roots["resultants"]/f"FINAL_{k}.png")
        elif a.ndim==3:
            for tau in range(a.shape[0]):save_map(a[tau],f"{k}, tau={tau}",roots["resultants"]/f"FINAL_{k}_TAU{tau}.png")
    return d


def _save_unloading_mask_3d(unloading_mask,L,R,h,z_coords,path,title):
    """J-13 output-only visualization of the already-computed unloading mask."""
    if unloading_mask is None:
        return False
    mask=np.asarray(unloading_mask,bool)
    if mask.ndim != 3 or mask.size == 0 or not np.any(mask):
        return False
    save_volume_scatter(
        mask.astype(float),L,R,h,path,title,"unloading 0/1",z_coords=z_coords
    )
    return True


def _birger_outputs(d,summary,roots,L,R,h,G0,unloading_mask=None):
    if d is None or "GB" not in d:return {}
    GB=np.asarray(d["GB"],float); ratio=GB/float(G0); ei=np.asarray(d["epsilon_i"],float);si=np.asarray(d["sigma_i"],float)
    save_map(np.min(ratio,axis=0),"min GB/G0 through thickness",roots["birger"]/"FINAL_GB_RATIO_MIN.png")
    save_map(np.max(ratio,axis=0),"max GB/G0 through thickness",roots["birger"]/"FINAL_GB_RATIO_MAX.png")
    save_volume_scatter(ratio,L,R,h,roots["birger3d"]/"FINAL_GB_RATIO_3D.png","GB/G0 through shell","GB/G0",z_coords=d.get("z"))
    _save_unloading_mask_3d(
        unloading_mask,L,R,h,d.get("z"),
        roots["birger3d"]/"FINAL_UNLOADING_MASK_3D.png",
        "Birger unloading mask: Delta epsilon_i < 0",
    )
    bj=summary.get("birger_definition") or {}; curve=bj.get("curve",{}) if isinstance(bj,dict) else {}; ys=curve.get("yield_stress") if isinstance(curve,dict) else None; ye=curve.get("yield_strain") if isinstance(curve,dict) else None
    info={"GB_ratio_min":float(np.min(ratio)),"GB_ratio_max":float(np.max(ratio))}
    mask=None; criterion=None
    if ys not in (None,""):
        val=float(ys);mask=si>=val;criterion=f"sigma_i >= {val:g}";_field=si
    elif ye not in (None,""):
        val=float(ye);mask=ei>=val;criterion=f"epsilon_i >= {val:g}";_field=ei
    if mask is not None:
        z=np.asarray(d.get("z"),float); qw=np.asarray(d.get("quadrature_weights"),float)
        vf=_weighted_volume_fraction(mask,z,qw,R)
        legacy=float(np.mean(mask))
        info.update({
            "plastic_zone_criterion":criterion,
            "plastic_volume_fraction":float(vf),
            "plastic_point_fraction_unweighted":legacy,
            "plastic_volume_weighting":"quadrature_w * (1 + z/R); uniform midsurface grid",
        })
        save_volume_scatter(mask.astype(float),L,R,h,roots["birger3d"]/"FINAL_PLASTIC_ZONE_3D.png",f"Plastic/nonlinear zone: {criterion}","zone 0/1",z_coords=z)
        (roots["report"]/"PLASTIC_ZONE.json").write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding="utf-8")
    # Explicit GB thresholds supplied by user: physical-nonlinearity zones, not automatically called plastic.
    pp=summary.get("postprocess",{}) or {}; thresholds=pp.get("gb_ratio_thresholds",[])
    for t in thresholds:
        tv=float(t); mask=ratio<tv; save_volume_scatter(mask.astype(float),L,R,h,roots["birger3d"]/f"GB_RATIO_LT_{tv:g}.png",f"Physical-nonlinearity zone GB/G0 < {tv:g}","zone 0/1",z_coords=d.get("z"))
    return info


def _control_points(d,summary,roots,L,R):
    if d is None:return []
    pp=summary.get("postprocess",{}) or {}; cps=pp.get("control_points",[])
    if not isinstance(cps,list) or not cps:return []
    arr_e=d["epsilon_i"]; arr_s=d["sigma_i"]; z=np.asarray(d["z"]);ns,nx=arr_e.shape[1:];xx=np.linspace(0,L,nx);th=np.linspace(0,2*np.pi,ns,endpoint=False)
    rows=[]
    for k,cp in enumerate(cps,1):
        try:x0=float(cp["x"]);t0=float(cp["theta"])%(2*np.pi); ix=int(np.argmin(abs(xx-x0))); ds=np.abs(th-t0);ds=np.minimum(ds,2*np.pi-ds);js=int(np.argmin(ds)); label=str(cp.get("label") or f"P{k}")
        except Exception:continue
        rows.append({"label":label,"x_requested":x0,"theta_requested":t0,"x_node":float(xx[ix]),"theta_node":float(th[js]),"epsilon_i_max_thickness":float(arr_e[:,js,ix].max()),"sigma_i_max_thickness":float(arr_s[:,js,ix].max())})
        fig,ax=plt.subplots(figsize=(6.6,4.6));ax.plot(z,arr_e[:,js,ix],marker="o");ax.set_xlabel(f"z [{_PLOT_CONTEXT['length']}]");ax.set_ylabel("epsilon_i [-]");ax.set_title(f"{label}: epsilon_i through thickness");ax.grid(True,alpha=.3);fig.tight_layout();_save_png(fig,roots["thickness"]/f"{label}_EPS_I.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6.6,4.6));ax.plot(z,arr_s[:,js,ix],marker="o");ax.set_xlabel(f"z [{_PLOT_CONTEXT['length']}]");ax.set_ylabel(f"sigma_i [{_PLOT_CONTEXT['stress']}]");ax.set_title(f"{label}: sigma_i through thickness");ax.grid(True,alpha=.3);fig.tight_layout();_save_png(fig,roots["thickness"]/f"{label}_SIGMA_I.png",dpi=160);plt.close(fig)
    if rows:
        with (roots["numeric"]/"CONTROL_POINTS.csv").open("w",newline="",encoding="utf-8") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    return rows


def numeric_zip(out,graphs_root=None):
    zpath=out/"NUMERIC_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if not p.is_file() or p==zpath:continue
            if p.suffix.lower() in (".npz",".json",".csv",".txt",".xlsx"):
                zf.write(p,arcname=p.relative_to(out))
    return zpath


def _readable_prefix(out,summary):
    info=_read_json(out/"RUN_INFO.json") or {}
    def slug(value,n):
        s=re.sub(r"\s+","_",str(value or "").strip())
        s=re.sub(r"[^0-9A-Za-zА-Яа-яЁё._+-]+","_",s).strip("._-+")
        return (s[:n] or "CALCULATION")
    try:run=f"RUN_{int(info.get('run_number')):03d}"
    except Exception:
        m=re.match(r"^(RUN_\d+)",str(info.get("run_name") or out.name),re.IGNORECASE)
        run=m.group(1).upper() if m else slug(out.name,18)
    variant=slug(info.get("variant_name") or info.get("run_name") or out.name,54)
    if variant.upper().startswith(run.upper()):
        return variant[:64].rstrip("._ ")
    return f"{run}__{variant}"[:64].rstrip("._ ")


def publish_readable_exports(out,summary):
    """Create human-facing CSV exports and the run README.

    PNG graph names are deliberately kept canonical/stable.  Older releases
    renamed every PNG to a RUN-prefixed alias at the very end of postprocess;
    if FULL was interrupted and then re-run, both the old prefixed file and a
    newly regenerated canonical file could coexist and look like duplicate
    results.  Excel/CSV filenames retain the run prefix; PNGs do not.
    """
    prefix=_readable_prefix(out,summary)
    csv_dir=out/"CSV";csv_dir.mkdir(parents=True,exist_ok=True)
    for old in csv_dir.glob("*.csv"):
        try:old.unlink()
        except OSError:pass
    csv_files=[]
    numeric=out/"NUMERIC"
    if numeric.exists():
        for src in sorted(numeric.glob("*.csv")):
            dst=csv_dir/f"{prefix}__{src.name}"
            shutil.copy2(src,dst);csv_files.append(str(dst))

    graph_files=[]
    for base in (out/"GRAPHS",out/"3D"):
        if not base.exists():continue
        # Keep canonical PNG names.  If an older release left a RUN-prefixed
        # alias next to a freshly generated canonical file, remove only that
        # proven duplicate alias.  Unique legacy files are left untouched.
        for alias in sorted(base.rglob(f"{prefix}__*.png")):
            canonical=alias.with_name(alias.name[len(prefix)+2:])
            if canonical.exists():
                try: alias.unlink()
                except OSError: pass
        graph_files.extend(str(src) for src in sorted(base.rglob("*.png")))

    excel_dir=out/"EXCEL"
    excel_files=sorted(excel_dir.glob("*.xlsx")) if excel_dir.exists() else []
    readme_text=(
        "RUN RESULTS / РЕЗУЛЬТАТЫ ОДНОГО ЗАПУСКА\n"
        "=========================================\n\n"
        "ENGLISH\n"
        "-------\n"
        f"Run folder: {out.name}\n"
        f"File prefix: {prefix}\n\n"
        "WHAT TO OPEN FIRST\n"
        "1. EXCEL/*__FULL_RESULTS.xlsx — consolidated workbook with the main numerical results.\n"
        "2. EXCEL/*__LOAD_DEFLECTION.xlsx — load-deflection history and accepted load steps.\n"
        "3. Other EXCEL workbooks are grouped by content: displacements, stresses/strains, resultants, convergence, control points and CUF coefficients.\n\n"
        "FOLDERS\n"
        "EXCEL — labelled Excel workbooks for inspection and independent plotting.\n"
        "CSV — principal numerical tables in portable CSV format.\n"
        "GRAPHS — 2D plots grouped by engineering topic.\n"
        "3D — deformed shell, load/boundary-condition views, field surfaces and Birger zones where applicable.\n"
        "VTK — closed-shell VTK data and deformed geometry for ParaView.\n"
        "NUMERIC — machine-readable NPZ/JSON data and source numerical tables.\n"
        "REPORT — calculation summary and machine-readable report files.\n"
        "INPUT — exact input parameters and copies of external input files used by the run.\n\n"
        "Excel and CSV filenames include the calculation/RUN prefix. Plot files keep stable canonical names to avoid duplicate RUN-prefixed copies.\n\n"
        "РУССКИЙ\n"
        "-------\n"
        f"Папка запуска: {out.name}\n"
        f"Префикс файлов: {prefix}\n\n"
        "ЧТО ОТКРЫТЬ ПЕРВЫМ\n"
        "1. EXCEL/*__FULL_RESULTS.xlsx — единая книга с основными численными результатами.\n"
        "2. EXCEL/*__LOAD_DEFLECTION.xlsx — история нагрузка–прогиб и принятые шаги нагружения.\n"
        "3. Остальные книги EXCEL сгруппированы по содержанию: перемещения, напряжения/деформации, результанты, сходимость, контрольные точки и коэффициенты CUF.\n\n"
        "ПАПКИ\n"
        "EXCEL — подписанные книги Excel для проверки и самостоятельного построения графиков.\n"
        "CSV — основные численные таблицы в переносимом формате CSV.\n"
        "GRAPHS — 2D-графики по инженерным тематическим подпапкам.\n"
        "3D — деформированная оболочка, нагрузка/граничные условия, поверхности полей и зоны Биргера при их наличии.\n"
        "VTK — данные замкнутой оболочки и деформированной геометрии для ParaView.\n"
        "NUMERIC — машинные NPZ/JSON и исходные числовые таблицы.\n"
        "REPORT — сводка расчёта и машиночитаемые отчётные файлы.\n"
        "INPUT — точные входные параметры и копии внешних файлов, использованных в расчёте.\n\n"
        "Excel и CSV содержат префикс расчёта/RUN. Файлы графиков сохраняют стабильные канонические имена, чтобы не создавать дублирующие RUN-копии.\n"
    )
    readme=out/"README_RESULTS_EN_RU.txt"
    readme.write_text(readme_text,encoding="utf-8")
    # Legacy filename is kept for compatibility, but its content is bilingual.
    legacy_readme=out/"README_RESULTS_RU.txt"
    legacy_readme.write_text(readme_text,encoding="utf-8")
    return {"prefix":prefix,"readme":str(readme),"readme_legacy":str(legacy_readme),"excel_files":[str(p) for p in excel_files],"csv_files":csv_files,"graph_files":graph_files}


def process_output(out_dir,make_zip=True,fast=False,mode=None):
    mode=(str(mode).upper().strip() if mode is not None else ("FAST" if fast else "FULL"))
    if mode not in ("FAST","STANDARD","FULL"):
        raise ValueError(f"Unsupported postprocess mode: {mode}")
    is_fast=(mode=="FAST"); is_standard=(mode=="STANDARD"); is_full=(mode=="FULL")
    fast=is_fast  # backward compatibility for older callers/tests
    out=Path(out_dir).resolve()
    if not out.exists():raise RuntimeError(f"Output directory does not exist: {out}")
    roots=_mkdirs(out); summary=_summary(out); L,R,h=_geometry(summary); E0,nu0=_material(summary)
    _configure_plot_context(summary,L,R)
    unit_system=str((summary.get("units") or {}).get("system","N_MM_MPA"))
    load_definition=summary.get("load_definition") or {}
    _,load_axis_label,load_value_key=funits.load_axis([0.0],load_definition,unit_system)
    states=_state_files(out); manifest={"output_dir":str(out),"states":[],"folders":{k:str(v) for k,v in roots.items()}}
    if states:
        print(f"POSTPROCESS {mode}: selected accepted state {states[-1].name} ({len(states)} accepted step file(s))", flush=True)
    if states and not is_fast:
        mm=re.search(r"STEP_(\d{4})",states[-1].name)
        last_idx=int(mm.group(1)) if mm else len(states)
        _cleanup_full_last_step_artifacts(roots,last_idx)
    history_step=[];history_factor=[];history_load=[];history_w=[];history_w_center=[];history_w_center_abs=[];history_w_ref=[];final_st=None;load_field=_load_radial_field(out)
    # Three different displacement metrics are intentionally kept separate:
    #   |w|max      = global maximum over the whole midsurface;
    #   w_center    = diagnostic array-center value w[Ns//2, Nx//2];
    #   w_reference = fixed physical control point x=L/2, theta=0.
    # The reference point NEVER follows the location of |w|max, so a two-hump
    # axial profile cannot silently change the meaning of the load-deflection curve.
    ref_is=0;ref_ix=None;ref_theta=0.0
    scale=float((summary.get("postprocess",{}) or {}).get("deformation_scale",1.0) or 1.0)
    step_rows=[]; result_steps=[]; last_step_zone={}; final_derived=None; center_diag_rows=[]
    progress_rows=_progress_events(out)
    completion_elapsed={}
    for r in progress_rows:
        if r.get("event")=="step_complete" and r.get("step_index") is not None:
            try: completion_elapsed[int(r["step_index"])]=float(r.get("elapsed_s",0.0))
            except Exception: pass

    for ordinal,p in enumerate(states,1):
        m=re.search(r"STEP_(\d{4})",p.name)
        idx=int(m.group(1)) if m else int(ordinal)
        st=_state_from_npz(p);final_st=st;w=np.asarray(st.get("w"),float);u=np.asarray(st.get("u"),float);v=np.asarray(st.get("v"),float);factor=_step_factor(out,idx)
        js,ix=np.unravel_index(int(np.argmax(np.abs(w))),w.shape)
        center_is=int(w.shape[0]//2); center_ix=int(w.shape[1]//2)
        w_center=float(w[center_is,center_ix])
        rjs=max(0,min(int(ref_is),w.shape[0]-1));rix=w.shape[1]//2 if ref_ix is None else max(0,min(int(ref_ix),w.shape[1]-1));w_ref=float(w[rjs,rix])
        x_ref=(float(L)*rix/(w.shape[1]-1)) if L is not None and w.shape[1]>1 else float(rix)
        theta_ref=float(2*np.pi*rjs/max(w.shape[0],1)); s_ref=(float(R)*theta_ref) if R is not None else float(rjs)
        xw=(float(L)*ix/(w.shape[1]-1)) if L is not None and w.shape[1]>1 else float(ix)
        thw=(2*np.pi*js/w.shape[0])
        sw=(float(R)*thw) if R is not None else float(js)
        x_center=(float(L)*center_ix/(w.shape[1]-1)) if L is not None and w.shape[1]>1 else float(center_ix)
        th_center=(2*np.pi*center_is/w.shape[0])
        s_center=(float(R)*th_center) if R is not None else float(center_is)
        actual_load=float(funits.load_axis([factor],load_definition,unit_system)[0][0])
        entry={"step":idx,"file":p.name,"kind":st.get("kind"),"load_factor":factor,"load_value":actual_load,"load_quantity":load_value_key,"load_unit":load_axis_label.split("[",1)[-1].rstrip("]"),"arrays":st.get("arrays",[]),"u_abs_max":float(np.max(np.abs(u))),"v_abs_max":float(np.max(np.abs(v))),"w_abs_max":float(np.max(np.abs(w))),"w_at_abs_max":float(w[js,ix]),"w_center_mid":w_center,"w_abs_center_mid":abs(w_center),"center_ix":center_ix,"center_is":center_is,"center_x":x_center,"center_theta":float(th_center),"center_s":s_center,"w_min":float(w.min()),"w_max":float(w.max()),"wmax_ix":int(ix),"wmax_is":int(js),"wmax_x":xw,"wmax_theta":float(thw),"wmax_s":sw}
        entry.update({"reference_ix":int(rix),"reference_is":int(rjs),"reference_x":x_ref,"reference_theta":theta_ref,"reference_s":s_ref,"w_reference":w_ref,"w_abs_reference":abs(w_ref)})
        history_step.append(idx);history_factor.append(factor);history_load.append(actual_load);history_w.append(entry["w_abs_max"]);history_w_center.append(w_center);history_w_center_abs.append(abs(w_center));history_w_ref.append(abs(w_ref))
        if str(summary.get("theory") or "").upper()=="CUF" and "q" in st:
            try:
                diag=_cuf_center_membrane_diagnostic(st,summary,load_factor=factor,load_value=actual_load)
                if diag is not None:
                    diag["step"]=idx; diag["state_file"]=p.name
                    center_diag_rows.append(diag)
            except Exception as exc:
                entry["center_membrane_diagnostic_error"]=f"{type(exc).__name__}: {exc}"
        # Keep the accepted-step history compact.  The expensive physical
        # reconstruction (derived stresses/strains, sections and 3-D graphics)
        # is intentionally performed ONLY for the last accepted state.  This
        # rule is the same for normal completion, GUI STOP and non-standard
        # termination: _state_files() contains only fully published accepted
        # states, therefore its last item is the recovery point.
        is_last=(ordinal==len(states))
        heavy_step=(not is_fast) and is_last
        if heavy_step:
            print(f"POSTPROCESS {mode}: final displacement maps for {p.name}...", flush=True)
            # FULL is a complete package for the LAST accepted state only.
            # Do not duplicate identical maps under both STEPS/STEP_xxxx and
            # FINAL_* names.
            save_map(u,f"FINAL u — accepted step {idx}",roots["disp"]/"FINAL_U.png")
            save_map(v,f"FINAL v — accepted step {idx}",roots["disp"]/"FINAL_V.png")
            save_map(w,f"FINAL w — accepted step {idx}",roots["disp"]/"FINAL_W.png")
            if L is not None and R is not None:
                save_deformed_shell(u,v,w,L,R,roots["deformed"]/"FINAL_DEFORMED_SHELL.png",f"Final deformed shell — accepted step {idx}",scale=scale,load_field=load_field)
            if "q" in st:
                q=st["q"]
                cdir=roots["cufdiag"]/"COEFFICIENTS"; cdir.mkdir(parents=True,exist_ok=True)
                for cidx,name in enumerate(("u","v","w")):
                    for tau in range(q.shape[1]):save_map(q[cidx,tau],f"{name}{tau} — accepted step {idx}",cdir/f"LAST_STEP_{idx:04d}_{name}{tau}.png")

        zone_info={}; derived_error=None; d=None
        # Cheap diagnostics for every accepted step come directly from the
        # solver-published state and therefore do not trigger reconstruction.
        try:
            if st.get("detF") is not None:
                det=np.asarray(st["detF"],float)
                if det.size:
                    entry["detF_min"]=float(np.min(det)); entry["detF_max"]=float(np.max(det))
            if st.get("GB") is not None and E0 is not None and nu0 is not None:
                G0=float(E0)/(2*(1+float(nu0))); ratio=np.asarray(st["GB"],float)/G0
                if ratio.size:
                    entry["GB_ratio_min"]=float(np.min(ratio)); entry["GB_ratio_max"]=float(np.max(ratio))
            if st.get("epsilon_i") is not None:
                eps=np.asarray(st["epsilon_i"],float)
                if eps.size:entry["epsilon_i_max"]=float(np.max(eps))
            if st.get("sigma_i") is not None:
                sig=np.asarray(st["sigma_i"],float)
                if sig.size:entry["sigma_i_max"]=float(np.max(sig))
        except Exception:
            pass

        # Full physical derived fields are reconstructed only for the last
        # accepted state.  FAST remains a history/preview operation; FULL adds
        # all sections, field maps and the final-step numerical package.
        if (not is_fast) and is_last and L is not None and R is not None and h is not None and E0 is not None and nu0 is not None:
            try:
                print(f"POSTPROCESS {mode}: reconstructing stresses/strains for {p.name}...", flush=True)
                d,zone_info=_derive_step_and_save(
                    st,summary,roots,idx,L,R,h,E0,nu0,
                    load_factor=factor,radial_surface_field=load_field,
                    make_graphs=False
                )
                last_step_zone=zone_info
                if d is not None:
                    final_derived=d
                    if heavy_step:
                        print(f"POSTPROCESS {mode}: sections and through-thickness profiles for {p.name}...", flush=True)
                        # Sections are unique information; publish them once for
                        # the last accepted state.  Field maps themselves are
                        # emitted later by _save_final_derived().
                        _save_standard_sections(d,st,roots,idx,L,R,summary=summary,lite=is_standard)
                    if "detF" in d:
                        entry["detF_min"]=float(np.min(d["detF"])); entry["detF_max"]=float(np.max(d["detF"]))
                    if "GB" in d:
                        G0=float(E0)/(2*(1+float(nu0))); ratio=np.asarray(d["GB"],float)/G0
                        entry["GB_ratio_min"]=float(np.min(ratio)); entry["GB_ratio_max"]=float(np.max(ratio))
                    entry.update(zone_info)
            except Exception as exc:
                derived_error=f"{type(exc).__name__}: {exc}"
                entry["derived_error"]=derived_error
                print(f"POSTPROCESS_ERROR step={idx:04d} file={p.name}: {derived_error}", flush=True)

        sj=_read_json(out/f"STEP_{idx:04d}.json") or {}
        for key in ("status","solver_status","converged","iterations","solver_iterations","outer_iterations","residual_inf","residual_relative","residual_convergence_inf","residual_convergence_relative","increment_inf","max_epsilon_i","max_abs_J_minus_1"):
            if key in sj: entry[key]=sj.get(key)
        if idx in completion_elapsed:
            entry["elapsed_cumulative_s"]=completion_elapsed[idx]
            prev=completion_elapsed.get(idx-1,0.0); entry["elapsed_step_s"]=completion_elapsed[idx]-prev
        manifest["states"].append(entry)
        step_rows.append(dict(entry))
        result_steps.append({"step":idx,"load_factor":factor,"state_file":p.name,"derived_file":f"NUMERIC/STEP_{idx:04d}_DERIVED.npz" if d is not None else None,"kind":st.get("kind")})

    center_diag_csv=None; center_diag_json=None; axial_profile_csv=None
    if center_diag_rows:
        center_diag_csv=roots["numeric"]/"CENTER_MEMBRANE_DIAGNOSTICS.csv"
        center_diag_json=roots["numeric"]/"CENTER_MEMBRANE_DIAGNOSTICS.json"
        fieldnames=[]
        for row in center_diag_rows:
            for key in row.keys():
                if key not in fieldnames: fieldnames.append(key)
        with center_diag_csv.open("w",newline="",encoding="utf-8") as f:
            wr=csv.DictWriter(f,fieldnames=fieldnames)
            wr.writeheader()
            for row in center_diag_rows: wr.writerow(row)
        center_diag_json.write_text(json.dumps(center_diag_rows,indent=2,ensure_ascii=False),encoding="utf-8")
    if final_st is not None and str(summary.get("theory") or "").upper()=="CUF" and "q" in final_st:
        try:
            axial_profile_csv=_save_cuf_axial_reference_profile(final_st,summary,roots["numeric"]/"AXIAL_U_PROFILE_THETA0_FINAL.csv")
        except Exception as exc:
            manifest["axial_u_profile_error"]=f"{type(exc).__name__}: {exc}"
    if final_st is not None:
        x=np.linspace(0,float(L),final_st["w"].shape[1]) if L is not None else None
        profile_title="Final w profiles"
        identity=_run_identity_caption(summary)
        if identity:
            profile_title+=f"\n{identity}"
        _fjs,_fix=np.unravel_index(int(np.nanargmax(np.abs(np.asarray(final_st["w"],float)))),np.asarray(final_st["w"],float).shape)
        save_profiles(final_st["w"],profile_title,roots["disp"]/"FINAL_W_PROFILES.png",x,extra_rows=[int(_fjs)])
        save_profiles_true_scale(final_st["w"],profile_title,roots["disp"]/"FINAL_W_PROFILES_TRUE_SCALE.png",x,extra_rows=[int(_fjs)])
        if is_full and L is not None and R is not None:
            try:
                vtk_scalars={}
                if final_derived is not None and isinstance(final_derived,dict):
                    zf=np.asarray(final_derived.get("z",[]),float)
                    if zf.size and final_derived.get("sigma_i") is not None:
                        vtk_scalars["sigma_i_uppermost_integration_layer"]=np.asarray(final_derived["sigma_i"],float)[int(np.argmax(zf))]
                    if zf.size and final_derived.get("epsilon_i") is not None:
                        vtk_scalars["epsilon_i_uppermost_integration_layer"]=np.asarray(final_derived["epsilon_i"],float)[int(np.argmax(zf))]
                export_cylinder_surface_vtk(
                    roots["vtk"]/"FINAL_CYLINDER.vtk",
                    final_st["u"],final_st["v"],final_st["w"],L,R,
                    scalars=vtk_scalars,deformed_points=False,
                    title="Final closed cylinder midsurface"
                )
                export_cylinder_surface_vtk(
                    roots["vtk"]/"FINAL_CYLINDER_DEFORMED.vtk",
                    final_st["u"],final_st["v"],final_st["w"],L,R,
                    scalars=vtk_scalars,deformed_points=True,deformation_scale=scale,
                    title=f"Final closed deformed cylinder; scale={scale:g}"
                )
            except Exception as exc:
                manifest["vtk_error"]=f"{type(exc).__name__}: {exc}"
    # Required convention: X = deflection, Y = load; include undeformed origin.
    for pat in (
        "*LOAD_W_CENTER.png","*LOAD_WMAX.png","*LOAD_WMAX_VS_CENTER.png",
        "*LOAD_DEFLECTION.png","*LOAD_WMAX_DIAGNOSTIC.png","*LOAD_COMPARE_CENTER_WMAX.png",
    ):
        for old in roots["loaddef"].glob(pat):
            try:old.unlink()
            except OSError:pass
    ref_label=f"|w|(x=L/2, θ={np.degrees(ref_theta):.1f}°) / контрольная точка"
    save_history_multi(
        [([0.0]+history_w,[0.0]+history_load,"|w|max"),([0.0]+history_w_ref,[0.0]+history_load,ref_label)],
        "Нагрузка–прогиб / Load–Deflection",
        f"Прогиб / Deflection |w| [{_PLOT_CONTEXT['displacement']}]",
        load_axis_label,
        roots["loaddef"]/"LOAD_DEFLECTION.png",
    )
    save_history([0.0]+history_w,[0.0]+history_load,"Диагностика: нагрузка–|w|max",f"|w|max [{_PLOT_CONTEXT['displacement']}]",load_axis_label,roots["loaddef"]/"LOAD_WMAX_DIAGNOSTIC.png")
    save_history_multi(
        [([0.0]+history_w,[0.0]+history_load,"|w|max"),([0.0]+history_w_center_abs,[0.0]+history_load,"|w(Ns/2,Nx/2)| — diagnostic")],
        "Диагностика: максимум и индексный центр",
        f"deflection [{_PLOT_CONTEXT['displacement']}]",load_axis_label,roots["loaddef"]/"LOAD_COMPARE_CENTER_WMAX.png",
    )
    _save_birger_hooke_comparison(out,summary,roots,history_w,history_load,load_axis_label)
    with (roots["numeric"]/"LOAD_DEFLECTION.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f); wr.writerow(["step [-]",f"w_abs_max [{_PLOT_CONTEXT['displacement']}]",f"w_reference_abs [{_PLOT_CONTEXT['displacement']}]",f"w_center_mid [{_PLOT_CONTEXT['displacement']}]",f"w_abs_center_mid [{_PLOT_CONTEXT['displacement']}]","reference_theta [rad]","load_factor [-]",load_axis_label]); wr.writerow([0,0.0,0.0,0.0,0.0,float(ref_theta),0.0,0.0])
        for step_no,ww,wrf,wc,wca,lf,ll in zip(history_step,history_w,history_w_ref,history_w_center,history_w_center_abs,history_factor,history_load):wr.writerow([step_no,ww,wrf,wc,wca,float(ref_theta),lf,ll])

    # One machine-readable row per accepted/completed step.
    if step_rows:
        fields=sorted(set().union(*(r.keys() for r in step_rows)))
        summary_rows=[{k:r.get(k) for k in fields} for r in step_rows]
        for name in ("STEP_SUMMARY.csv","RESULTS_SUMMARY.csv"):
            with (roots["numeric"]/name).open("w",newline="",encoding="utf-8") as f:
                wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(summary_rows)
    with (roots["numeric"]/"FIELD_HISTORY.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f);wr.writerow(["step [-]","load_factor [-]",load_axis_label,f"w_abs_max [{_PLOT_CONTEXT['displacement']}]",f"w_reference_abs [{_PLOT_CONTEXT['displacement']}]",f"w_center_mid [{_PLOT_CONTEXT['displacement']}]",f"w_abs_center_mid [{_PLOT_CONTEXT['displacement']}]"]);[wr.writerow([step_no,lf,ll,ww,wrf,wc,wca]) for step_no,lf,ll,ww,wrf,wc,wca in zip(history_step,history_factor,history_load,history_w,history_w_ref,history_w_center,history_w_center_abs)]

    # Evolution plots for Birger/plasticity and nonlinear admissibility.
    def series_plot(key,title,ylabel,filename):
        vals=[r.get(key) for r in step_rows]
        if not vals or all(v is None or v=="" for v in vals):return
        xs=[r.get("load_value",r.get("load_factor",i+1)) for i,r in enumerate(step_rows)]; ys=[np.nan if v is None or v=="" else float(v) for v in vals]
        save_history(xs,ys,title,load_axis_label,ylabel,roots["plasticity"]/filename)
    series_plot("plastic_volume_fraction","Plastic volume fraction vs load","Vp/V [-]","PLASTIC_VOLUME_FRACTION.png")
    series_plot("plastic_thickness_fraction_max","Maximum plastic penetration through thickness","max hp/h [-]","PLASTIC_THICKNESS_MAX.png")
    series_plot("plastic_area_fraction_top_layer","Plastic area fraction — top integration layer","Ap/A [-]","PLASTIC_AREA_TOP.png")
    series_plot("plastic_area_fraction_bottom_layer","Plastic area fraction — bottom integration layer","Ap/A [-]","PLASTIC_AREA_BOTTOM.png")
    series_plot("birger_nonlinear_volume_fraction","Birger nonlinear volume fraction vs load","Vnl/V [-]","BIRGER_NONLINEAR_VOLUME_FRACTION.png")
    series_plot("birger_nonlinear_thickness_fraction_max","Maximum Birger nonlinear fraction through thickness","max hnl/h [-]","BIRGER_NONLINEAR_THICKNESS_MAX.png")
    series_plot("birger_nonlinear_area_fraction_top_layer","Birger nonlinear area fraction — top integration layer","Anl/A [-]","BIRGER_NONLINEAR_AREA_TOP.png")
    series_plot("birger_nonlinear_area_fraction_bottom_layer","Birger nonlinear area fraction — bottom integration layer","Anl/A [-]","BIRGER_NONLINEAR_AREA_BOTTOM.png")
    series_plot("GB_ratio_min","Minimum GB/G0 vs load","min GB/G0 [-]","GB_RATIO_MIN_HISTORY.png")
    series_plot("GB_ratio_max","Maximum GB/G0 vs load","max GB/G0 [-]","GB_RATIO_MAX_HISTORY.png")
    series_plot("detF_min","Minimum detF vs load","min detF [-]","DETF_MIN_HISTORY.png")

    progress_info=_save_progress_history(out,roots)

    derived=None;birger_info={};cp_rows=[]
    if final_st is not None and L is not None and R is not None and h is not None and E0 is not None and nu0 is not None:
        try:
            if is_fast:
                # FAST: the final step was already derived once inside the step
                # loop. Reuse it verbatim; do not call _save_final_derived(),
                # _birger_outputs(), control-point plotting or 3-D surface plots.
                # This keeps FAST truly fast for KL/CUF, Elastic/Birger, MVI/FDM.
                derived=final_derived
                if derived is not None and str(summary.get("mode","")).upper().endswith("BIRGER"):
                    G0=float(E0)/(2*(1+float(nu0)))
                    gb=np.asarray(derived.get("GB",[]),float)
                    if gb.size:
                        ratio=gb/G0
                        birger_info={
                            "GB_ratio_min":float(np.min(ratio)),
                            "GB_ratio_max":float(np.max(ratio)),
                        }
                    birger_info.update(last_step_zone)
            else:
                print(f"POSTPROCESS {mode}: final field maps, 3D views and control points...", flush=True)
                # FULL: the final accepted state is reconstructed once in the
                # state loop; reuse it here while publishing the complete final
                # graph suite.
                derived=_save_final_derived(final_st,summary,roots,d=final_derived,all_levels=is_full)
                G0=float(E0)/(2*(1+float(nu0)))
                if str(summary.get("mode","")).upper().endswith("BIRGER"):
                    birger_info=_birger_outputs(
                        derived,summary,roots,L,R,h,G0,
                        unloading_mask=final_st.get("unloading_mask"),
                    )
                    birger_info.update(last_step_zone)
                cp_rows=_control_points(derived,summary,roots,L,R)
                if derived is not None:
                    z=np.asarray(derived.get("z",[]),float)
                    if z.size:
                        g=int(np.argmax(z))
                        save_surface_field_3d(
                            derived["sigma_i"][g],L,R,z[g],
                            roots["stress3d"]/("FINAL_SIGMA_I_SURFACE_3D.png"),
                            f"Stress intensity — uppermost integration layer / крайний верхний интеграционный слой, z={z[g]:.6g}","sigma_i",
                            u=np.asarray(final_st.get("u"),float),v=np.asarray(final_st.get("v"),float),w=np.asarray(final_st.get("w"),float),scale=scale
                        )
                        save_surface_field_3d(
                            derived["epsilon_i"][g],L,R,z[g],
                            roots["stress3d"]/("FINAL_EPS_I_SURFACE_3D.png"),
                            f"Strain intensity — uppermost integration layer / крайний верхний интеграционный слой, z={z[g]:.6g}","epsilon_i",
                            u=np.asarray(final_st.get("u"),float),v=np.asarray(final_st.get("v"),float),w=np.asarray(final_st.get("w"),float),scale=scale
                        )
                        save_cylinder_three_view_overview(
                            np.asarray(final_st.get("u"),float),
                            np.asarray(final_st.get("v"),float),
                            np.asarray(final_st.get("w"),float),
                            L,R,roots["overview3d"]/("CYLINDER_SHELL_3D_RESULTS.png"),
                            derived,scale=scale
                        )
                save_load_bc_shell(
                    L,R,roots["loadbc"]/("FINAL_LOAD_AND_BC_3D.png"),
                    load_field=load_field,
                    boundary_conditions=summary.get("boundary_conditions")
                )
        except Exception as exc:
            manifest["derived_error"]=f"{type(exc).__name__}: {exc}"

    first_plastic=next((r for r in step_rows if float(r.get("plastic_volume_fraction",0.0) or 0.0)>0.0),None)
    first_nonlinear=next((r for r in step_rows if float(r.get("birger_nonlinear_volume_fraction",0.0) or 0.0)>0.0),None)
    result_index={
        "format":"CYLINDER_RESULTS_V2","theory":summary.get("theory"),"cuf_model":summary.get("cuf_model"),"mode":summary.get("mode"),
        "geometry_model":summary.get("geometry_model"),"material_model":summary.get("material_model"),"method":summary.get("method"),
        "steps":result_steps,
        "selected_state_file":(states[-1].name if states else None),
        "selected_step":(history_step[-1] if history_step else None),
        "numeric_fields":{
            "common":["epsilon_i","sigma_i","z","quadrature_weights"],
            "CUF":["q_coefficients","disp_gauss","standard_z","disp_standard_z","E","S","detF"],
            "KL":["strain","stress"],
            "Birger":["GB","GB_ratio","birger_nonlinear_mask","birger_nonlinear_thickness_fraction","birger_nonlinear_bottom_layer","birger_nonlinear_mid_layer","birger_nonlinear_top_layer","plastic_mask","plastic_thickness_fraction","plastic_bottom_layer","plastic_mid_layer","plastic_top_layer","unloading_mask"],
        },
        "progress":progress_info,
        "center_membrane_diagnostics_csv":None if center_diag_csv is None else str(center_diag_csv),
        "center_membrane_diagnostics_json":None if center_diag_json is None else str(center_diag_json),
        "axial_u_profile_theta0_final_csv":axial_profile_csv,
        "first_plastic_step":None if first_plastic is None else {k:first_plastic.get(k) for k in ("step","load_factor","plastic_first_x","plastic_first_s","plastic_first_theta","plastic_first_z")},
        "first_birger_nonlinear_step":None if first_nonlinear is None else {k:first_nonlinear.get(k) for k in ("step","load_factor","GB_ratio_min_x","GB_ratio_min_s","GB_ratio_min_theta","GB_ratio_min_z")},
    }
    (roots["numeric"]/"RESULTS_INDEX.json").write_text(json.dumps(result_index,indent=2,ensure_ascii=False),encoding="utf-8")
    manifest["selected_state_file"]=(states[-1].name if states else None)
    manifest["selected_step"]=(history_step[-1] if history_step else None)
    manifest["postprocess_mode"]=mode
    (out/"GRAPHS"/"GRAPHS_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    load_def=summary.get("load_definition") or {}
    if isinstance(load_def,dict) and "generated_meta" in load_def:
        load_kinematics=(load_def.get("generated_meta") or {}).get("load_kinematics","DEAD_REFERENCE")
    else:
        load_kinematics=load_def.get("load_kinematics","DEAD_REFERENCE") if isinstance(load_def,dict) else "DEAD_REFERENCE"
    center_diag_first=(center_diag_rows[0] if center_diag_rows else None); center_diag_final=(center_diag_rows[-1] if center_diag_rows else None)
    report={"theory":summary.get("theory"),"cuf_model":summary.get("cuf_model"),"method":summary.get("method"),"mode":summary.get("mode"),"geometry_model":summary.get("geometry_model"),"material_model":summary.get("material_model"),"units":summary.get("units"),"converged":summary.get("converged"),"status":summary.get("status"),"geometry":summary.get("geometry"),"material":summary.get("material"),"grid":summary.get("grid"),"quadrature":summary.get("quadrature"),"load_kinematics":load_kinematics,"load_definition":load_def,"validation_warnings":summary.get("validation_warnings",[]),"steps":len(states),"postprocess_mode":mode,"selected_state_file":(states[-1].name if states else None),"selected_step":(history_step[-1] if history_step else None),"reference_point":{"definition":"x=L/2, theta=0","reference_ix":(step_rows[-1].get("reference_ix") if step_rows else None),"reference_is":(step_rows[-1].get("reference_is") if step_rows else None),"reference_theta":(step_rows[-1].get("reference_theta") if step_rows else None)},"final_load_value":(float(history_load[-1]) if history_load else None),"final_load_label":load_axis_label,"w_abs_max_final":None if final_st is None else float(np.max(np.abs(final_st["w"]))),"w_center_final":(step_rows[-1].get("w_center_mid") if step_rows else None),"w_abs_center_final":(step_rows[-1].get("w_abs_center_mid") if step_rows else None),"w_reference_final":(step_rows[-1].get("w_reference") if step_rows else None),"w_abs_reference_final":(step_rows[-1].get("w_abs_reference") if step_rows else None),"birger":birger_info,"control_points":cp_rows,"progress":progress_info,"solver_diagnostics":{"mvi_rank":summary.get("rank"),"accepted_steps":summary.get("accepted_steps"),"requested_steps":summary.get("requested_steps"),"failed_step":summary.get("failed_step_index"),"grid_aspect_ratio":None if not summary.get("geometry") else float((summary.get("geometry") or {}).get("h",0))/float((summary.get("geometry") or {}).get("R",1)),"cpu_time":summary.get("cpu_time"),"wall_time":summary.get("wall_time"),"mvi_iterations_total":summary.get("mvi_iterations_total"),"mvi_iterations_max":summary.get("mvi_iterations_max"),"field_change_max":summary.get("field_change_max"),"residual_max":summary.get("residual_max"),"mvi_convergence_semantics":summary.get("convergence_semantics"),"mx_max":summary.get("mx_max"),"my_max":summary.get("my_max"),"mxy_max":summary.get("mxy_max"),"sigma_max":(summary.get("sigma_max") if summary.get("sigma_max") is not None else (float(np.nanmax(np.asarray(final_derived.get("sigma_i"),float))) if isinstance(final_derived,dict) and final_derived.get("sigma_i") is not None else None))},"center_membrane_diagnostic_first":center_diag_first,"center_membrane_diagnostic_final":center_diag_final,"center_membrane_diagnostics_csv":None if center_diag_csv is None else str(center_diag_csv),"axial_u_profile_theta0_final_csv":axial_profile_csv,"results_index":str(roots["numeric"]/"RESULTS_INDEX.json"),"derived_error":manifest.get("derived_error")}
    report_path=roots["report"]/"CALCULATION_SUMMARY.json"
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")

    # One independent Excel workbook per calculation folder.  This is an
    # output-only export: it reads already-saved JSON/CSV/NPZ results and never
    # calls or modifies the mechanics.  MVI rank-series postprocessing invokes
    # process_output() for each rank OUT folder, so every rank receives its own
    # workbook as well.
    excel_path=None
    named_excel_files={}
    excel_error=None
    try:
        import final_excel_export
        if is_fast:
            print("POSTPROCESS FAST: creating lightweight Excel...", flush=True)
            named_excel_files=final_excel_export.export_named_excels(
                out, include=("LOAD_DEFLECTION","CONVERGENCE")
            )
        elif is_standard:
            print("POSTPROCESS STANDARD: creating category Excel workbooks...", flush=True)
            named_excel_files=final_excel_export.export_named_excels(out)
        else:
            print("POSTPROCESS FULL: creating complete Excel package...", flush=True)
            excel_path=final_excel_export.export_calculation_excel(out)
            named_excel_files=final_excel_export.export_named_excels(out)
        print(f"POSTPROCESS {mode}: Excel ready: {excel_path}", flush=True)
        report["excel_file"]=None if excel_path is None else str(excel_path)
        report["named_excel_files"]={k:str(v) for k,v in named_excel_files.items()}
        report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    except Exception as exc:
        excel_error=f"{type(exc).__name__}: {exc}"
        report["excel_error"]=excel_error
        report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")

    readable=publish_readable_exports(out,summary)
    report["readable_exports"]=readable
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"POSTPROCESS {mode}: packing numeric ZIP...", flush=True)
    zpath=numeric_zip(out) if make_zip else None
    partner_path=None
    try:
        print(f"POSTPROCESS {mode}: publishing bilingual partner results package...", flush=True)
        import final_partner_export
        partner_path=final_partner_export.publish_partner_package(out,mode=mode)
        report["partner_results_package"]=str(partner_path)
        report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    except Exception as exc:
        report["partner_results_error"]=f"{type(exc).__name__}: {exc}"
        report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return {"states":len(states),"graphs_dir":str(out/"GRAPHS"),"numeric_zip":None if zpath is None else str(zpath),"report":str(report_path),"excel_file":None if excel_path is None else str(excel_path),"named_excel_files":{k:str(v) for k,v in named_excel_files.items()},"excel_error":excel_error,"derived_error":manifest.get("derived_error"),"partner_results":None if partner_path is None else str(partner_path)}


def main(out_dir,mode="FULL"):
    mode=str(mode).upper().strip()
    r=process_output(out_dir,make_zip=True,mode=mode)
    print(f"POSTPROCESS {mode} PASS: {r['states']} state file(s); graphs={r['graphs_dir']}; numeric_zip={r['numeric_zip']}; derived_error={r['derived_error']}")

if __name__=="__main__":
    if len(sys.argv) not in (2,3):
        raise SystemExit("Usage: python FINAL_POSTPROCESS.py OUTPUT_DIR [--fast|--standard|--full]")
    arg=(sys.argv[2].lower() if len(sys.argv)==3 else "--full")
    modes={"--fast":"FAST","--standard":"STANDARD","--full":"FULL"}
    if arg not in modes:
        raise SystemExit("Second argument must be --fast, --standard or --full")
    main(sys.argv[1],mode=modes[arg])
