#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional MP4 export for CYL_SOLVER accepted-step results.

This module is deliberately output-only.  It never calls the solver, mutates
accepted STEP files, runs FULL postprocessing, or writes into existing GRAPH /
NUMERIC / EXCEL trees.  The only user artifact it creates is VIDEO/*.mp4.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


BASE_VIDEO_MODES = (
    "3D деформация",
    "3D напряжения",
    "3D деформации",
    "3D выбранное поле",
    "3D разрез по толщине",
    "3D продольный разрез",
    "3D поперечный разрез",
    "3D комбинированный разрез",
    "3D + нагрузка–прогиб",
    "Разрез x-z",
    "Разрез s-z",
    "Профиль по толщине",
)
BIRGER_VIDEO_MODE = "Birger GB/G0"
ALL_VIDEO_MODE = "ВСЕ основные видео"
# Modes accepted by the low-level single-MP4 encoder.  ALL_VIDEO_MODE is a
# GUI orchestration option and is intentionally not passed to export_mp4().
VIDEO_MODES = BASE_VIDEO_MODES + (BIRGER_VIDEO_MODE,)

MODE_FILE_TOKENS = {
    "3D деформация": "DEFORMATION_3D",
    "3D напряжения": "STRESS_3D",
    "3D деформации": "STRAIN_3D",
    "3D выбранное поле": "FIELD_3D",
    "3D разрез по толщине": "CUTAWAY_3D",
    "3D продольный разрез": "LONGITUDINAL_CUT_3D",
    "3D поперечный разрез": "TRANSVERSE_CUT_3D",
    "3D комбинированный разрез": "COMBINED_CUT_3D",
    "3D + нагрузка–прогиб": "FIELD_LOAD_3D",
    "Разрез x-z": "SECTION_XZ",
    "Разрез s-z": "SECTION_SZ",
    "Профиль по толщине": "THROUGH_THICKNESS",
    BIRGER_VIDEO_MODE: "BIRGER_GB_RATIO",
}

def available_modes(*, has_birger: bool = False, include_all: bool = True) -> tuple[str, ...]:
    vals=list(BASE_VIDEO_MODES)
    if has_birger:
        vals.append(BIRGER_VIDEO_MODE)
    if include_all:
        vals.append(ALL_VIDEO_MODE)
    return tuple(vals)



def _ffmpeg_module():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg
    except Exception as exc:
        raise RuntimeError(
            "Для MP4 требуется пакет imageio-ffmpeg. Запустите START_PROGRAM.cmd — "
            "он установит зависимости из requirements.txt."
        ) from exc


def _safe_token(text: str) -> str:
    keep=[]
    for ch in str(text):
        if ch.isalnum() or ch in ("-","_","."):
            keep.append(ch)
        elif ch in (" ","/","\\",":","+"):
            keep.append("_")
    s="".join(keep).strip("_.")
    return s or "video"


def output_name(mode: str, field: str, first_step: int, last_step: int) -> str:
    token=MODE_FILE_TOKENS.get(str(mode),_safe_token(mode))
    return f"VIDEO_{token}__{_safe_token(field)}__STEP_{int(first_step):04d}_{int(last_step):04d}.mp4"


def select_steps(steps: Iterable[int], every: int = 1) -> list[int]:
    vals=sorted({int(x) for x in steps})
    if not vals:
        return []
    every=max(1,int(every))
    chosen=vals[::every]
    if vals[-1] not in chosen:
        chosen.append(vals[-1])
    return chosen


def _finite_minmax(arr) -> tuple[float,float] | None:
    a=np.asarray(arr,float)
    f=a[np.isfinite(a)]
    if not f.size:
        return None
    return float(np.min(f)),float(np.max(f))


def _expand_limits(mn: float, mx: float) -> tuple[float,float]:
    if not (math.isfinite(mn) and math.isfinite(mx)):
        return -1.0,1.0
    if mx==mn:
        d=max(abs(mn),1.0)*1e-12
        return mn-d,mx+d
    return mn,mx


def _global_scan(steps, provider, mode, field, progress=None, cancel_event=None):
    mn=float("inf");mx=float("-inf");max_disp=0.0;zmax=0.0
    n=len(steps)
    for i,step in enumerate(steps,1):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("VIDEO export cancelled")
        d=provider(int(step),mode,field,True)
        mm=_finite_minmax(d.get("color_values",[]))
        if mm is not None:
            mn=min(mn,mm[0]);mx=max(mx,mm[1])
        try:max_disp=max(max_disp,float(d.get("max_displacement",0.0) or 0.0))
        except Exception:pass
        try:zmax=max(zmax,float(d.get("zmax",0.0) or 0.0))
        except Exception:pass
        if progress:progress("scan",i,n,int(step))
    if not math.isfinite(mn):mn,mx=-1.0,1.0
    return (*_expand_limits(mn,mx),max_disp,zmax)


def _figure_rgb(fig, width: int, height: int) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas=FigureCanvasAgg(fig)
    canvas.draw()
    a=np.asarray(canvas.buffer_rgba(),dtype=np.uint8)[...,:3]
    if a.shape[1]!=width or a.shape[0]!=height:
        # Normally impossible because figsize*dpi is fixed, but keep the encoder
        # contract exact on platforms with unusual DPI handling.
        from PIL import Image
        a=np.asarray(Image.fromarray(a).resize((width,height)))
    return np.ascontiguousarray(a)


def _surface_xyz(d, zindex=None):
    L=float(d["L"]);R=float(d["R"]);scale=float(d.get("scale",1.0))
    field=np.asarray(d["field3d"],float);zvals=np.asarray(d["zvals"],float)
    if field.ndim!=3:raise ValueError("VIDEO: field3d must have shape [z,s,x]")
    zi=int(d.get("layer_index",len(zvals)//2) if zindex is None else zindex)
    zi=max(0,min(zi,len(zvals)-1));z=float(zvals[zi])
    u=np.asarray(d["uvol"],float)[zi];v=np.asarray(d["vvol"],float)[zi];w=np.asarray(d["wvol"],float)[zi]
    layer=field[zi]
    ns,nx=layer.shape
    xx=np.linspace(0.0,L,nx);th=np.linspace(0.0,2*np.pi,ns,endpoint=False);X0,T=np.meshgrid(xx,th)
    c=np.cos(T);s=np.sin(T);rr=R+z+scale*w
    X=X0+scale*u;Y=rr*c-scale*v*s;Z=rr*s+scale*v*c
    return X,Y,Z,layer,z


def _wmax_info(d):
    """Return the fixed midsurface |w|max point for display annotations.

    A postprocessor may visually interpolate the rendered surface. In that case
    _wmax_info_override stores the extrema from the original calculation grid.
    """
    override=d.get("_wmax_info_override")
    if isinstance(override,dict):
        return override
    try:
        wv=np.asarray(d.get("wvol"),float)
        if wv.ndim!=3 or not wv.size:return None
        g=max(0,min(wv.shape[0]//2,wv.shape[0]-1));w=wv[g]
        js,ix=np.unravel_index(int(np.nanargmax(np.abs(w))),w.shape)
        L=float(d.get("L",0.0));R=float(d.get("R",0.0));scale=float(d.get("scale",1.0))
        u=np.asarray(d.get("uvol"),float)[g];v=np.asarray(d.get("vvol"),float)[g]
        th=2*math.pi*js/max(w.shape[0],1);x0=L*ix/max(w.shape[1]-1,1)
        zvals=np.asarray(d.get("zvals",[0.0]),float);zg=float(zvals[g]) if g<len(zvals) else 0.0
        rr=R+zg+scale*w[js,ix];c=math.cos(th);ss=math.sin(th)
        X=x0+scale*u[js,ix];Y=rr*c-scale*v[js,ix]*ss;Z=rr*ss+scale*v[js,ix]*c
        return {"g":g,"is":int(js),"ix":int(ix),"x":float(x0),"theta":float(th),"w":float(w[js,ix]),"wabs":float(abs(w[js,ix])),"X":float(X),"Y":float(Y),"Z":float(Z)}
    except Exception:return None


def _add_fixed_triad(fig):
    """Small screen-fixed XYZ triad, independent of 3-D camera/geometry."""
    if getattr(fig,"_cyl_fixed_triad",False):return
    fig._cyl_fixed_triad=True
    ax=fig.add_axes([.158,.055,.068,.105]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    o=(.28,.25)
    ax.annotate("",xy=(.88,.25),xytext=o,arrowprops=dict(arrowstyle="-|>",lw=1.25,color="#a83a32"))
    ax.annotate("",xy=(.28,.86),xytext=o,arrowprops=dict(arrowstyle="-|>",lw=1.25,color="#2e7847"))
    ax.annotate("",xy=(.68,.63),xytext=o,arrowprops=dict(arrowstyle="-|>",lw=1.25,color="#285e9a"))
    ax.text(.91,.22,"x",fontsize=8.5,fontweight="bold",color="#a83a32")
    ax.text(.22,.89,"y",fontsize=8.5,fontweight="bold",color="#2e7847")
    ax.text(.70,.65,"z",fontsize=8.5,fontweight="bold",color="#285e9a")


def _mark_wmax(ax,d):
    info=_wmax_info(d)
    if not info:return
    try:
        ax.scatter([info["X"]],[info["Y"]],[info["Z"]],s=38,marker="o",facecolor="#f4f7fa",edgecolor="#b64d32",linewidth=1.25,depthshade=False,zorder=20)
        ax.text(info["X"],info["Y"],info["Z"],"  |w|max",fontsize=7.8,color="#8f3527",fontweight="bold",zorder=21)
    except Exception:pass


def _plot_undeformed_reference(ax,d):
    try:
        mode=str(d.get("_render_mode","")).lower()
        if "деформац" not in mode and "deformation" not in mode and "нагрузка" not in mode:return
        field=np.asarray(d.get("field3d"),float)
        if field.ndim!=3:return
        ns,nx=field.shape[1],field.shape[2];L=float(d["L"]);R=float(d["R"])
        xx=np.linspace(0,L,nx);th=np.linspace(0,2*np.pi,ns+1);X,T=np.meshgrid(xx,th);Y=R*np.cos(T);Z=R*np.sin(T)
        ax.plot_wireframe(X,Y,Z,rstride=max(1,ns//10),cstride=max(1,nx//12),color="#5d6b78",linewidth=.45,alpha=.10)
    except Exception:pass

def _plot_shell(ax,d,norm,cmap):
    X,Y,Z,C,z=_surface_xyz(d)
    X=np.concatenate((X,X[:1]),axis=0);Y=np.concatenate((Y,Y[:1]),axis=0);Z=np.concatenate((Z,Z[:1]),axis=0);C=np.concatenate((C,C[:1]),axis=0)
    ax.plot_surface(X,Y,Z,facecolors=cmap(norm(C)),linewidth=0,antialiased=True,shade=False,alpha=.998)
    try:
        rs=max(1,(X.shape[0]-1)//13);cs=max(1,(X.shape[1]-1)//17)
        ax.plot_wireframe(X,Y,Z,rstride=rs,cstride=cs,color="#18222c",linewidth=.22,alpha=.055)
        ax.plot(X[:,0],Y[:,0],Z[:,0],color="#17212b",linewidth=.95,alpha=.78)
        ax.plot(X[:,-1],Y[:,-1],Z[:,-1],color="#17212b",linewidth=.95,alpha=.78)
    except Exception:pass
    _plot_undeformed_reference(ax,d)
    _set_3d_axes(ax,d,z,view="shell",show_axes=False)
    _draw_dimensions(ax,d,kind="shell")
    _draw_load_arrows(ax,d)
    _mark_wmax(ax,d)
    _add_fixed_triad(ax.figure)
    _hud(ax,d,d.get("step",0),"3D")

def _set_3d_axes(ax,d,z=0.0,*,view="shell",show_axes=True):
    """Tight, data-aware 3-D framing; avoids huge empty margins for inward w."""
    L=float(d["L"]);R=float(d["R"]);scale=float(d.get("scale",1.0));lu=str(d.get("length_unit","mm"))
    xlo=0.0;xhi=L;rad=R
    try:
        u=np.asarray(d.get("uvol"),float);v=np.asarray(d.get("vvol"),float);w=np.asarray(d.get("wvol"),float)
        if u.ndim==3 and w.shape==u.shape:
            xlo=min(0.0,float(np.nanmin(scale*u)));xhi=max(L,float(L+np.nanmax(scale*u)))
            zz=_display_zvals(d) if view in ("cutaway","longitudinal","transverse","combined") else np.asarray(d.get("zvals",[z]),float)
            if zz.size!=w.shape[0]:zz=np.linspace(float(np.min(zz)) if zz.size else 0.0,float(np.max(zz)) if zz.size else 0.0,w.shape[0])
            rr=R+zz[:,None,None]+scale*w
            rad=max(R,float(np.nanmax(np.abs(rr)))+float(np.nanmax(np.abs(scale*v))))
    except Exception:pass
    span=max(xhi-xlo,2*rad,1e-12);pad=.035*span
    ax.set_xlim(xlo-pad,xhi+pad);ax.set_ylim(-(rad+.035*span),rad+.035*span);ax.set_zlim(-(rad+.035*span),rad+.035*span)
    try:ax.set_box_aspect((max(xhi-xlo+2*pad,1e-12),max(2*(rad+.035*span),1e-12),max(2*(rad+.035*span),1e-12)))
    except Exception:pass
    views={"shell":(22,-58),"cutaway":(20,-55),"longitudinal":(19,-112),"transverse":(0,0),"combined":(21,-52)}
    elev,azim=views.get(view,views["shell"]);ax.view_init(elev=elev,azim=azim)
    try:ax.set_proj_type("ortho" if view=="transverse" else "persp")
    except Exception:pass
    if show_axes:
        ax.set_xlabel(f"x [{lu}]",labelpad=7);ax.set_ylabel(f"y [{lu}]",labelpad=7);ax.set_zlabel(f"z [{lu}]",labelpad=7);ax.tick_params(labelsize=8,pad=1)
    else:ax.set_axis_off()
    try:
        ax.xaxis.pane.set_alpha(.015);ax.yaxis.pane.set_alpha(.015);ax.zaxis.pane.set_alpha(.015);ax.grid(False)
    except Exception:pass

def _field_range_text(d):
    """Bilingual current-frame/global colour-range text for the engineering HUD."""
    a=np.asarray(d.get("color_values",d.get("field3d",[])),float)
    f=a[np.isfinite(a)]
    if not f.size:return ""
    cur=f"Frame / Кадр: {float(np.min(f)):.4g} … {float(np.max(f)):.4g}"
    try:
        g0=float(d.get("_video_global_min"));g1=float(d.get("_video_global_max"))
        if math.isfinite(g0) and math.isfinite(g1):
            return cur+f"\nVideo scale / Шкала ролика: {g0:.4g} … {g1:.4g}"
    except Exception:pass
    return cur


def _meta_value(d,*keys,default="—"):
    for key in keys:
        val=d.get(key)
        if val is not None and str(val).strip():return str(val)
    return default


def _hud(ax,d,step,mode,*,xcut=None,theta_cut=None,thickness_note=False):
    """Compact bilingual engineering HUD; omit unknown metadata instead of dashes."""
    fig=ax.figure;lu=str(d.get("length_unit","mm"));L=float(d.get("L",0.0));R=float(d.get("R",0.0))
    zvals=np.asarray(d.get("zvals",[]),float)
    try:h=float(d.get("h",0.0) or 0.0)
    except Exception:h=0.0
    if h<=0 and zvals.size>1:
        try:h=float(np.ptp(zvals))
        except Exception:h=0.0
    field=str(d.get("field_label",d.get("field","field")));unit=str(d.get("field_unit","-"));load=d.get("load_value");load_label=str(d.get("load_label","Load"))
    theory=str(d.get("theory") or "");cuf=str(d.get("cuf_model") or "");geometry=str(d.get("geometry_model") or "");material=str(d.get("material_model") or "");solver=str(d.get("method") or "");rank=d.get("rank")
    model=(theory+(f" {cuf}" if cuf and theory.upper()=="CUF" else "")).strip()
    if rank is not None and str(rank).strip() not in ("","None"):
        solver+=(f"-{rank}" if solver.upper()=="MVI" else f" · rank {rank}")
    first=f"STEP / ШАГ {int(step)}"
    if load is not None:
        try:first+=f"   {load_label} = {float(load):.5g}"
        except Exception:pass
    lines=[first]
    if model or geometry:lines.append("Model / Модель: "+" · ".join(x for x in (model,geometry) if x))
    if material or solver:lines.append("Material / Solver / Материал / Решатель: "+" · ".join(x for x in (material,solver) if x))
    lines.append(f"L={L:.4g} {lu}   R={R:.4g} {lu}   h={h:.4g} {lu}")
    lines.append(f"Field / Поле: {field} [{unit}]")
    rng=_field_range_text(d)
    if rng:lines.extend(rng.splitlines())
    loc=[]
    if xcut is not None:loc.append(f"A–A: x={float(xcut):.4g} {lu}")
    if theta_cut is not None:loc.append(f"B–B: θ={math.degrees(float(theta_cut)):.1f}°")
    if d.get("section_note"):loc.append(str(d.get("section_note")))
    if loc:lines.append(" · ".join(loc))
    if thickness_note and zvals.size>1:
        vt=_visual_thickness_scale(d)
        if abs(vt-1)>1e-12:lines.append(f"h ×{vt:g} — display only / только визуально")
        aa=np.asarray(d.get("field3d",[]),float)
        if aa.ndim==3 and aa.shape[0]>1 and np.isfinite(aa).any():
            try:
                spread=float(np.nanmax(np.nanmax(aa,axis=0)-np.nanmin(aa,axis=0)));span=max(float(np.nanmax(aa)-np.nanmin(aa)),1e-30)
                if spread/span<1e-5:lines.append("Through-thickness ≈ const / По толщине ≈ const")
            except Exception:pass
    info=_wmax_info(d)
    if info and not thickness_note:
        lines.append(f"|w|max={info['wabs']:.5g} {lu}   x={info['x']:.4g} {lu}   θ={math.degrees(info['theta']):.1f}°")
    text="\n".join(lines)
    anchor=d.get("_hud_anchor",(.018,.875));font=float(d.get("_hud_font",7.35))
    fig.text(float(anchor[0]),float(anchor[1]),text,ha="left",va="top",fontsize=font,linespacing=1.16,color="#f5f7fa",family="DejaVu Sans",
             bbox=dict(boxstyle="round,pad=.55",facecolor="#0d1822",edgecolor="#597084",linewidth=.85,alpha=.955),zorder=50)

def _visual_thickness_scale(d):
    """Display-only thickness exaggeration; never changes solver geometry/data."""
    try:
        v=float(d.get("visual_thickness_scale",1.0) or 1.0)
    except Exception:
        v=1.0
    return max(0.1,min(v,50.0))



def _display_zvals(d):
    """Display-only thickness coordinates, preserving order of saved field layers.

    If the physical h is known, map the saved through-thickness sampling span to
    the real wall thickness before applying the user's visual exaggeration.  No
    field value or solver coordinate is modified.
    """
    z=np.asarray(d.get("zvals",[]),float)
    if z.size<2:return z
    try:h=abs(float(d.get("h")))
    except Exception:h=0.0
    zmin=float(np.nanmin(z));zmax=float(np.nanmax(z));mid=.5*(zmin+zmax);span=zmax-zmin
    if h>0 and span>0:
        zn=(z-mid)/span
        base=zn*h
    else:
        base=z-mid
    return base*_visual_thickness_scale(d)


def _plot_cutaway(ax,d,norm,cmap):
    """Open-wall 3/4 cutaway through the saved integration layers."""
    a=np.asarray(d["field3d"],float);zd=_display_zvals(d);u=np.asarray(d["uvol"],float);v=np.asarray(d["vvol"],float);w=np.asarray(d["wvol"],float)
    nz,ns,nx=a.shape;L=float(d["L"]);R=float(d["R"]);scale=float(d.get("scale",1.0));js=int(d.get("point_is",0))%max(ns,1);theta=2*math.pi*js/ns
    if nz<2:raise ValueError("3D разрез по толщине требует минимум два уровня z")
    half=math.radians(58.0);start=theta+half;end=theta-half+2*math.pi;nth=min(max(ns,42),108);thv=np.linspace(start,end,nth);jv=np.mod(np.round((np.mod(thv,2*np.pi))/(2*np.pi)*ns).astype(int),ns)
    ixv=np.arange(nx);xx=np.linspace(0,L,nx);X0,T=np.meshgrid(xx,thv)
    for g,alpha in ((0,.90),(nz-1,.995)):
        zg=float(zd[g]);ug=u[g][np.ix_(jv,ixv)];vg=v[g][np.ix_(jv,ixv)];wg=w[g][np.ix_(jv,ixv)];cg=a[g][np.ix_(jv,ixv)]
        rr=R+zg+scale*wg;X=X0+scale*ug;Y=rr*np.cos(T)-scale*vg*np.sin(T);Z=rr*np.sin(T)+scale*vg*np.cos(T)
        ax.plot_surface(X,Y,Z,facecolors=cmap(norm(cg)),linewidth=0,antialiased=True,shade=False,alpha=alpha)
    faces=[]
    for th in (start,end):
        j=int(np.round((th%(2*math.pi))/(2*np.pi)*ns))%ns;c=math.cos(th);ss=math.sin(th);Xf=np.empty((nz,nx));Yf=np.empty_like(Xf);Zf=np.empty_like(Xf);Cf=np.empty_like(Xf)
        for g,zg in enumerate(zd):
            rr=R+float(zg)+scale*w[g,j,:];Xf[g]=xx+scale*u[g,j,:];Yf[g]=rr*c-scale*v[g,j,:]*ss;Zf[g]=rr*ss+scale*v[g,j,:]*c;Cf[g]=a[g,j,:]
        ax.plot_surface(Xf,Yf,Zf,facecolors=cmap(norm(Cf)),linewidth=.30,edgecolor=(.04,.06,.08,.48),antialiased=True,shade=False,alpha=1.0)
        ax.plot(Xf[0],Yf[0],Zf[0],color="#17212b",linewidth=1.45,alpha=.95);ax.plot(Xf[-1],Yf[-1],Zf[-1],color="#17212b",linewidth=1.45,alpha=.95);faces.append((Xf,Yf,Zf))
    _set_3d_axes(ax,d,view="cutaway",show_axes=False);ax.view_init(elev=20,azim=-55)
    _draw_dimensions(ax,d,kind="cutaway",theta_cut=start)
    try:
        Xf,Yf,Zf=faces[0];m=nx//2
        ax.text(Xf[-1,m],Yf[-1,m],Zf[-1,m],f"  uppermost integration layer / крайний верхний интеграционный слой, z={zd[-1]:.6g}",fontsize=7.0,color="#253849",fontweight="bold")
        ax.text(Xf[0,m],Yf[0,m],Zf[0,m],f"  lowermost integration layer / крайний нижний интеграционный слой, z={zd[0]:.6g}",fontsize=7.0,color="#253849",fontweight="bold")
    except Exception:pass
    _draw_load_arrows(ax,d,cut=True);_add_fixed_triad(ax.figure);_hud(ax,d,d.get("step",0),"3D разрез по толщине",thickness_note=True)

def _nested_load_spec(d):
    spec=d.get("load_spec") or {}
    if isinstance(spec,dict) and isinstance(spec.get("definition"),dict):
        spec=spec.get("definition") or {}
    return spec if isinstance(spec,dict) else {}


def _draw_load_arrows(ax,d,*,cut=False):
    """Sparse readable load arrows; display-only, never used by the solver."""
    spec=_nested_load_spec(d);typ=str(spec.get("type","")).upper()
    if not typ or typ=="EXTERNAL_NPZ":return
    L=float(d["L"]);R=float(d["R"]);zmax=float(d.get("global_zmax",d.get("zmax",0.0)) or 0.0);r0=R+zmax
    direction=str(spec.get("pressure_direction") or "").upper();sign=1.0 if direction!="EXTERNAL" else -1.0
    for key in ("q","p_line","P"):
        try:
            if float(spec.get(key))<0.0:sign=-1.0
        except Exception:pass
    alen=max(.14*R,.035*max(L,2*R,1.0))
    def arrow(x,theta):
        c=math.cos(theta);ss=math.sin(theta);y=r0*c;z=r0*ss;dy=sign*alen*c;dz=sign*alen*ss
        if sign<0:y-=dy;z-=dz
        ax.quiver([x],[y],[z],[0.0],[dy],[dz],length=1.0,normalize=False,arrow_length_ratio=.30,linewidth=1.25,color="#376f9f",alpha=.78)
    if typ in ("UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE"):
        for x in np.linspace(.20*L,.80*L,3):
            for th in np.linspace(0,2*math.pi,6,endpoint=False)[::2]:arrow(float(x),float(th))
    elif typ=="PATCH_RADIAL_FACE":
        x1=float(spec.get("x1",.25*L));x2=float(spec.get("x2",.75*L))
        if bool(spec.get("full_circumference",False)):ths=np.linspace(0,2*math.pi,4,endpoint=False)
        else:
            c1=float(spec.get("c1",0.0));c2=float(spec.get("c2",math.pi/2));cm=str(spec.get("circumferential_coordinate","THETA")).upper()
            if cm!="THETA":c1/=max(R,1e-300);c2/=max(R,1e-300)
            if c2<c1:c2+=2*math.pi
            ths=np.linspace(c1,c2,3)
        for x in np.linspace(x1,x2,2):
            for th in ths:arrow(float(x),float(th))
    elif typ=="LINE_RADIAL_GENERATOR":
        th=float(spec.get("c0",0.0));
        if str(spec.get("circumferential_coordinate","THETA")).upper()!="THETA":th/=max(R,1e-300)
        for x in np.linspace(.18*L,.82*L,5):arrow(float(x),th)
    elif typ=="POINT_RADIAL_FORCE":arrow(float(spec.get("x0",.5*L)),float(spec.get("theta0",0.0)))

def _draw_dimensions(ax,d,*,kind="shell",xcut=None,theta_cut=None):
    """Clean engineering dimensions; wall h is shown only on true cuts."""
    L=float(d["L"]);R=float(d["R"]);lu=str(d.get("length_unit","mm"));zvals=np.asarray(d.get("zvals",[]),float)
    zd=_display_zvals(d) if zvals.size>1 else zvals;rin=R+(float(np.min(zd)) if zd.size else 0.0);rout=R+(float(np.max(zd)) if zd.size else 0.0);rdim=max(R,abs(rout),abs(rin))+0.11*max(R,1e-12)
    if kind=="shell":
        # L below the cylinder axis, R on the visible end plane. h intentionally omitted.
        y=-rdim;z=-.12*R
        ax.plot([0,L],[y,y],[z,z],color="#4b5e70",linewidth=1.0,alpha=.82)
        ax.plot([0,0],[y-.025*R,y+.025*R],[z,z],color="#4b5e70",linewidth=.9);ax.plot([L,L],[y-.025*R,y+.025*R],[z,z],color="#4b5e70",linewidth=.9)
        ax.text(.5*L,y,z,f"L = {L:.4g} {lu}",fontsize=8.2,color="#334e68",ha="center",va="bottom",fontweight="semibold")
        xr=L;ax.plot([xr,xr],[0,R],[0,0],color="#2f5f87",linewidth=1.15,alpha=.88);ax.scatter([xr],[0],[0],s=8,color="#2f5f87")
        ax.text(xr,.55*R,0,f"R = {R:.4g} {lu}",fontsize=8.0,color="#2f5f87",ha="left",va="center",fontweight="semibold")
    elif kind in ("longitudinal","cutaway"):
        th=float(theta_cut or 0.0);c=math.cos(th);ss=math.sin(th);rr=rout+.07*R
        ax.plot([0,L],[rr*c,rr*c],[rr*ss,rr*ss],color="#4b5e70",linewidth=.95,alpha=.75)
        ax.text(.5*L,rr*c,rr*ss,f"L={L:.4g} {lu}",fontsize=14,color="#334e68",ha="center",fontweight="semibold")
        if zd.size>1:
            x=.90*L;ax.plot([x,x],[rin*c,rout*c],[rin*ss,rout*ss],color="#8c5a22",linewidth=1.5,alpha=.95)
            vt=_visual_thickness_scale(d);tag=f"h={float(d.get('h',np.ptp(zvals))):.4g} {lu}"+(f" ×{vt:g} vis." if abs(vt-1)>1e-12 else "")
            ax.text(x,rout*c,rout*ss,tag,fontsize=14,color="#8c5a22",fontweight="bold")

def _plot_longitudinal_cut(ax,d,norm,cmap):
    """Diametral longitudinal B–B section in a stable 3/4 axonometric view."""
    a=np.asarray(d["field3d"],float);zd=_display_zvals(d);u=np.asarray(d["uvol"],float);v=np.asarray(d["vvol"],float);w=np.asarray(d["wvol"],float)
    nz,ns,nx=a.shape;L=float(d["L"]);R=float(d["R"]);scale=float(d.get("scale",1.0));js=int(d.get("point_is",0))%max(ns,1);theta0=2*math.pi*js/ns
    if nz<2:raise ValueError("3D продольный разрез требует минимум два уровня z")
    th_keep=np.linspace(theta0+math.pi/2,theta0+3*math.pi/2,min(max(ns//2+1,34),92));jk=np.mod(np.round((np.mod(th_keep,2*np.pi))/(2*np.pi)*ns).astype(int),ns);xx=np.linspace(0,L,nx);X0,T=np.meshgrid(xx,th_keep)
    for g,alpha in ((0,.88),(nz-1,.995)):
        zg=float(zd[g]);ug=u[g][np.ix_(jk,np.arange(nx))];vg=v[g][np.ix_(jk,np.arange(nx))];wg=w[g][np.ix_(jk,np.arange(nx))];cg=a[g][np.ix_(jk,np.arange(nx))]
        rr=R+zg+scale*wg;X=X0+scale*ug;Y=rr*np.cos(T)-scale*vg*np.sin(T);Z=rr*np.sin(T)+scale*vg*np.cos(T);ax.plot_surface(X,Y,Z,facecolors=cmap(norm(cg)),linewidth=0,antialiased=True,shade=False,alpha=alpha)
    cutfaces=[]
    for th in (theta0+math.pi/2,theta0+3*math.pi/2):
        j=int(np.round((th%(2*math.pi))/(2*np.pi)*ns))%ns;c=math.cos(th);ss=math.sin(th);Xf=np.empty((nz,nx));Yf=np.empty_like(Xf);Zf=np.empty_like(Xf);Cf=np.empty_like(Xf)
        for g,zg in enumerate(zd):
            rr=R+float(zg)+scale*w[g,j,:];Xf[g]=xx+scale*u[g,j,:];Yf[g]=rr*c-scale*v[g,j,:]*ss;Zf[g]=rr*ss+scale*v[g,j,:]*c;Cf[g]=a[g,j,:]
        ax.plot_surface(Xf,Yf,Zf,facecolors=cmap(norm(Cf)),linewidth=.30,edgecolor=(.04,.06,.08,.45),antialiased=True,shade=False,alpha=1.0);cutfaces.append((Xf,Yf,Zf))
    # Removed half: very faint context only.
    try:
        gm=nz//2;thg=np.linspace(theta0-math.pi/2,theta0+math.pi/2,11);jg=np.mod(np.round((np.mod(thg,2*np.pi))/(2*np.pi)*ns).astype(int),ns);ixg=np.unique(np.linspace(0,nx-1,11).round().astype(int));xg=np.linspace(0,L,nx)[ixg];Xg,Tg=np.meshgrid(xg,thg);ug=u[gm][np.ix_(jg,ixg)];vg=v[gm][np.ix_(jg,ixg)];wg=w[gm][np.ix_(jg,ixg)];rr=R+float(zd[gm])+scale*wg;Yg=rr*np.cos(Tg)-scale*vg*np.sin(Tg);Zg=rr*np.sin(Tg)+scale*vg*np.cos(Tg);Xg=Xg+scale*ug;ax.plot_wireframe(Xg,Yg,Zg,rstride=2,cstride=2,color="#5b6975",linewidth=.32,alpha=.08)
    except Exception:pass
    _set_3d_axes(ax,d,view="longitudinal",show_axes=False);ax.view_init(elev=19,azim=-112);_draw_dimensions(ax,d,kind="longitudinal",theta_cut=theta0+math.pi/2)
    try:
        Xf,Yf,Zf=cutfaces[0];m=max(1,nx//3);ax.text(Xf[-1,m],Yf[-1,m],Zf[-1,m],"  B–B",fontsize=14,color="#295c95",fontweight="bold")
    except Exception:pass
    _add_fixed_triad(ax.figure);_hud(ax,d,d.get("step",0),"3D продольный разрез",theta_cut=theta0,thickness_note=True)

def _plot_transverse_cut(ax,d,norm,cmap):
    """Large annular A–A section with R, h, O and local x-r-theta orientation."""
    a=np.asarray(d["field3d"],float);zd=_display_zvals(d);u=np.asarray(d["uvol"],float);v=np.asarray(d["vvol"],float);w=np.asarray(d["wvol"],float)
    nz,ns,nx=a.shape;L=float(d["L"]);R=float(d["R"]);scale=float(d.get("scale",1.0));ix=max(0,min(int(d.get("point_ix",nx//2)),nx-1));xcut=L*ix/max(nx-1,1)
    if nz<2:raise ValueError("3D поперечный разрез требует минимум два уровня z")
    th=np.linspace(0,2*math.pi,ns+1);jv=np.r_[np.arange(ns),0];Xf=np.empty((nz,ns+1));Yf=np.empty_like(Xf);Zf=np.empty_like(Xf);Cf=np.empty_like(Xf)
    for g,zg in enumerate(zd):
        uu=u[g,jv,ix];vv=v[g,jv,ix];ww=w[g,jv,ix];rr=R+float(zg)+scale*ww;Xf[g]=xcut+scale*uu;Yf[g]=rr*np.cos(th)-scale*vv*np.sin(th);Zf[g]=rr*np.sin(th)+scale*vv*np.cos(th);Cf[g]=a[g,jv,ix]
    ax.plot_surface(Xf,Yf,Zf,facecolors=cmap(norm(Cf)),linewidth=.28,edgecolor=(.04,.06,.08,.42),antialiased=True,shade=False,alpha=1.0);ax.plot(Xf[0],Yf[0],Zf[0],color="#17212b",linewidth=1.55,alpha=.96);ax.plot(Xf[-1],Yf[-1],Zf[-1],color="#17212b",linewidth=1.55,alpha=.96)
    # undeformed reference circles
    try:
        for rr in (R-.5*float(d.get("h",0.0)),R+.5*float(d.get("h",0.0))):ax.plot(np.full_like(th,xcut),rr*np.cos(th),rr*np.sin(th),color="#6f7c87",linewidth=.65,alpha=.12,linestyle="--")
    except Exception:pass
    _set_3d_axes(ax,d,view="transverse",show_axes=False)
    span=max(R+float(np.max(np.abs(zd))) if zd.size else R,1e-9)*1.12;ax.set_xlim(xcut-.045*max(L,2*R),xcut+.045*max(L,2*R));ax.set_ylim(-span,span);ax.set_zlim(-span,span)
    try:ax.set_box_aspect((.13,2,2));ax.set_proj_type("ortho");ax.view_init(elev=0,azim=0)
    except Exception:pass
    _annotate_transverse_geometry(ax,d,xcut,zd);_hud(ax,d,d.get("step",0),"3D поперечный разрез",xcut=xcut,thickness_note=True)

def _plot_combined_cut(ax,d,norm,cmap):
    """Combined A–A/B–B section: transverse face forward, longitudinal cut receding."""
    a=np.asarray(d["field3d"],float);zd=_display_zvals(d);u=np.asarray(d["uvol"],float);v=np.asarray(d["vvol"],float);w=np.asarray(d["wvol"],float)
    nz,ns,nx=a.shape;L=float(d["L"]);R=float(d["R"]);scale=float(d.get("scale",1.0));js=int(d.get("point_is",0))%max(ns,1);ix=max(1,min(int(d.get("point_ix",nx//2)),nx-2));theta=2*math.pi*js/ns
    if nz<2:raise ValueError("3D комбинированный разрез требует минимум два уровня z")
    half=math.radians(48);start=theta+half;end=theta-half+2*math.pi;thv=np.linspace(start,end,min(max(ns,40),104));jv=np.mod(np.round((np.mod(thv,2*np.pi))/(2*np.pi)*ns).astype(int),ns);gap=max(2,nx//18);left=max(0,ix-gap);right=min(nx-1,ix+gap)
    for indices in (np.arange(0,left+1),np.arange(right,nx)):
        if len(indices)<2:continue
        xx=np.linspace(0,L,nx)[indices];X0,T=np.meshgrid(xx,thv)
        for g,alpha in ((0,.82),(nz-1,.98)):
            ug=u[g][np.ix_(jv,indices)];vg=v[g][np.ix_(jv,indices)];wg=w[g][np.ix_(jv,indices)];cg=a[g][np.ix_(jv,indices)];rr=R+float(zd[g])+scale*wg;X=X0+scale*ug;Y=rr*np.cos(T)-scale*vg*np.sin(T);Z=rr*np.sin(T)+scale*vg*np.cos(T);ax.plot_surface(X,Y,Z,facecolors=cmap(norm(cg)),linewidth=0,antialiased=True,shade=False,alpha=alpha)
    for th in (start,end):
        j=int(np.round((th%(2*math.pi))/(2*np.pi)*ns))%ns;c=math.cos(th);ss=math.sin(th)
        for indices in (np.arange(0,left+1),np.arange(right,nx)):
            if len(indices)<2:continue
            xx=np.linspace(0,L,nx)[indices];Xl=np.empty((nz,len(indices)));Yl=np.empty_like(Xl);Zl=np.empty_like(Xl);Cl=np.empty_like(Xl)
            for g,zg in enumerate(zd):rr=R+float(zg)+scale*w[g,j,indices];Xl[g]=xx+scale*u[g,j,indices];Yl[g]=rr*c-scale*v[g,j,indices]*ss;Zl[g]=rr*ss+scale*v[g,j,indices]*c;Cl[g]=a[g,j,indices]
            ax.plot_surface(Xl,Yl,Zl,facecolors=cmap(norm(Cl)),linewidth=.28,edgecolor=(.04,.06,.08,.42),shade=False,alpha=1.0)
    xcut=L*ix/max(nx-1,1);Xt=np.empty((nz,len(thv)));Yt=np.empty_like(Xt);Zt=np.empty_like(Xt);Ct=np.empty_like(Xt)
    for g,zg in enumerate(zd):rr=R+float(zg)+scale*w[g,jv,ix];Xt[g]=xcut+scale*u[g,jv,ix];Yt[g]=rr*np.cos(thv)-scale*v[g,jv,ix]*np.sin(thv);Zt[g]=rr*np.sin(thv)+scale*v[g,jv,ix]*np.cos(thv);Ct[g]=a[g,jv,ix]
    ax.plot_surface(Xt,Yt,Zt,facecolors=cmap(norm(Ct)),linewidth=.32,edgecolor=(.03,.05,.07,.50),antialiased=True,shade=False,alpha=1.0)
    _set_3d_axes(ax,d,view="combined",show_axes=False);ax.view_init(elev=21,azim=-52)
    try:
        mid=len(thv)//2;ax.text(Xt[-1,mid],Yt[-1,mid],Zt[-1,mid],"  A–A",fontsize=9.2,color="#8e2020",fontweight="bold")
        j=int(np.round(((start%(2*math.pi))/(2*np.pi))*ns))%ns;c=math.cos(start);ss=math.sin(start);rr=R+float(zd[-1])+scale*w[-1,j,ix];xp=xcut+scale*u[-1,j,ix];yp=rr*c-scale*v[-1,j,ix]*ss;zp=rr*ss+scale*v[-1,j,ix]*c;ax.scatter([xp],[yp],[zp],s=42,facecolor="#f5f7fa",edgecolor="#6d2942",linewidth=1.2,depthshade=False);ax.text(xp,yp,zp,"  P=A–A∩B–B",fontsize=8.2,color="#6d2942",fontweight="bold")
        ax.text(.55*L,(R+zd[-1])*math.cos(start),(R+zd[-1])*math.sin(start),"B–B",fontsize=9,color="#295c95",fontweight="bold")
    except Exception:pass
    _add_fixed_triad(ax.figure);_hud(ax,d,d.get("step",0),"3D комбинированный разрез",xcut=xcut,theta_cut=theta,thickness_note=True)

def _plot_xz(ax,d,norm,cmap):
    a=np.asarray(d["field3d"],float);z=np.asarray(d["zvals"],float);js=max(0,min(int(d.get("point_is",0)),a.shape[1]-1));ix=max(0,min(int(d.get("point_ix",a.shape[2]//2)),a.shape[2]-1));x=np.linspace(0,float(d["L"]),a.shape[2]);sec=a[:,js,:]
    im=ax.pcolormesh(x,z,sec,shading="auto",cmap=cmap,norm=norm);xcrit=x[ix];ax.axvline(xcrit,color="#7b3342",linewidth=1.25,linestyle="--",alpha=.78);ax.text(xcrit,float(np.max(z)),"  x(|w|max)",fontsize=8.5,color="#7b3342",va="top")
    ax.set_xlabel(f"x — осевая координата / axial coordinate [{d.get('length_unit','mm')}]\nSection B–B / Сечение B–B");ax.set_ylabel(f"z — толщина / thickness [{d.get('length_unit','mm')}]");ax.grid(False);return im

def _plot_sz(ax,d,norm,cmap):
    a=np.asarray(d["field3d"],float);z=np.asarray(d["zvals"],float);ix=max(0,min(int(d.get("point_ix",0)),a.shape[2]-1));js=max(0,min(int(d.get("point_is",0)),a.shape[1]-1));th=np.linspace(0,2*np.pi,a.shape[1],endpoint=False);sec=a[:,:,ix]
    im=ax.pcolormesh(th,z,sec,shading="auto",cmap=cmap,norm=norm);tc=th[js];ax.axvline(tc,color="#7b3342",linewidth=1.25,linestyle="--",alpha=.78);ax.text(tc,float(np.max(z)),"  θ(|w|max)",fontsize=8.5,color="#7b3342",va="top")
    ax.set_xlabel("θ — окружная координата / circumferential coordinate [rad]\nSection A–A / Сечение A–A");ax.set_ylabel(f"z — толщина / thickness [{d.get('length_unit','mm')}]");ax.grid(False);return im

def _plot_thickness(ax,d,mn,mx):
    a=np.asarray(d["field3d"],float);z=np.asarray(d["zvals"],float);ix=max(0,min(int(d.get("point_ix",0)),a.shape[2]-1));js=max(0,min(int(d.get("point_is",0)),a.shape[1]-1));series=a[:,js,ix];L=float(d.get("L",0.0));x=L*ix/max(a.shape[2]-1,1);th=2*math.pi*js/max(a.shape[1],1)
    ax.plot(series,z,marker="o",linewidth=2.0,markersize=5.2);ax.set_xlabel(f"{d.get('field_label',d.get('field','field'))} [{d.get('field_unit','-')}]");ax.set_ylabel(f"z — толщина / thickness [{d.get('length_unit','mm')}]");ax.set_title(f"x={x:.4g} {d.get('length_unit','mm')}, θ={math.degrees(th):.1f}° · at |w|max",fontsize=10.5,fontweight="semibold");ax.grid(True,alpha=.18,linewidth=.7)
    if math.isfinite(mn) and math.isfinite(mx) and mx>mn:pad=.05*(mx-mn);ax.set_xlim(mn-pad,mx+pad)

def _section_location_text(d, *, xcut=None, theta_cut=None):
    lu=str(d.get("length_unit","mm"))
    parts=[]
    if xcut is not None:
        parts.append(f"A–A: x = {float(xcut):.4g} {lu}")
    if theta_cut is not None:
        parts.append(f"B–B: θ = {math.degrees(float(theta_cut)):.1f}°")
    # Most section plots are taken at the selected critical cut (often |w|max).
    if d.get('section_note'):
        parts.append(str(d.get('section_note')))
    return "   ·   ".join(parts)


def _add_section_schematic(fig,d,kind,*,xcut=None,theta_cut=None):
    """Large readable section map, kept clear of HUD and main geometry."""
    import matplotlib.patches as patches
    ax=fig.add_axes([.018,.045,.17,.155]);ax.set_facecolor('#ffffff')
    for sp in ax.spines.values():sp.set_color('#b9c6d1');sp.set_linewidth(.8)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_xlim(0,1);ax.set_ylim(0,1);L=float(d.get('L',1.0))
    ax.add_patch(patches.Rectangle((.09,.32),.76,.34,fill=False,ec='#66798a',lw=1.15));ax.plot([.09,.85],[.49,.49],color='#d5dde5',lw=.8)
    if kind in ('transverse','sz'):
        xc=.09+.76*(0.5 if xcut is None or L<=0 else max(0,min(1,float(xcut)/L)));ax.plot([xc,xc],[.22,.76],color='#b93434',lw=1.8);ax.text(xc,.79,'A–A',ha='center',fontsize=7.8,color='#8e2020',fontweight='bold')
    elif kind=='cutaway':
        ax.add_patch(patches.Rectangle((.41,.31),.13,.36,facecolor=(.75,.20,.20,.10),edgecolor='#b93434',lw=1.2));ax.text(.475,.16,'удалённый сектор / open sector',ha='center',fontsize=6.6,color='#8e2020')
    elif kind in ('longitudinal','xz'):
        ax.plot([.09,.85],[.49,.49],color='#2968a2',lw=1.8);ax.text(.69,.54,'B–B',fontsize=7.8,color='#295c95',fontweight='bold')
    else:
        xc=.09+.76*(0.5 if xcut is None or L<=0 else max(0,min(1,float(xcut)/L)));ax.plot([xc,xc],[.22,.76],color='#b93434',lw=1.7);ax.text(xc,.79,'A–A',ha='center',fontsize=7.6,color='#8e2020',fontweight='bold');ax.plot([.09,.85],[.49,.49],color='#2968a2',lw=1.7);ax.text(.69,.54,'B–B',fontsize=7.6,color='#295c95',fontweight='bold')
    loc=_section_location_text(d,xcut=xcut,theta_cut=theta_cut)
    if loc:ax.text(.03,.035,loc,ha='left',va='bottom',fontsize=6.6,color='#44576a',wrap=True)
    ax.text(.03,.96,'Схема сечения / Section map',ha='left',va='top',fontsize=7.3,color='#23384d',fontweight='bold');return ax

def _annotate_transverse_geometry(ax,d,xcut,zd):
    try:
        R=float(d['R']);rin=R+float(np.min(zd));rout=R+float(np.max(zd));rmid=R;span=max(abs(rout),abs(rin),R,1e-9)*1.18;lu=str(d.get("length_unit","mm"));h=float(d.get("h",np.ptp(np.asarray(d.get("zvals",[]),float))))
        ax.scatter([xcut],[0],[0],s=12,color="#263746",depthshade=False);ax.text(xcut,0,0," O",fontsize=8.5,color="#263746",fontweight="bold")
        ax.plot([xcut,xcut],[0,rmid],[0,0],color="#263746",linewidth=1.25,alpha=.9);ax.text(xcut,.54*rmid,0,f" R={R:.4g} {lu}",color="#263746",fontsize=8.7,fontweight="bold")
        # radial wall dimension near +y
        ang=math.radians(30);c=math.cos(ang);ss=math.sin(ang);ax.plot([xcut,xcut],[rin*c,rout*c],[rin*ss,rout*ss],color="#8e5a21",linewidth=1.6,alpha=.95);ax.text(xcut,(rout+.035*span)*c,(rout+.035*span)*ss,f"h={h:.4g} {lu}",color="#8e5a21",fontsize=8.4,fontweight="bold")
        # compact local axes in the section plane + axial out-of-plane mark
        oY=-.42*span;oZ=-.42*span;l=.18*span;ax.plot([xcut,xcut],[oY,oY+l],[oZ,oZ],color="#b23a2a",linewidth=1.15);ax.plot([xcut,xcut],[oY,oY],[oZ,oZ+l],color="#267a3a",linewidth=1.15);ax.text(xcut,oY+l*1.08,oZ,"r",color="#b23a2a",fontsize=8,fontweight="bold");ax.text(xcut,oY,oZ+l*1.08,"z",color="#267a3a",fontsize=8,fontweight="bold");ax.text(xcut,oY-.04*span,oZ-.04*span,"x ⊙",color="#275ea5",fontsize=8,fontweight="bold")
        zvals=np.asarray(d.get("zvals",[]),float)
        if zvals.size:
            ax.text2D(.50,.82,f"uppermost integration layer / крайний верхний интеграционный слой, z={zvals[-1]:.6g}",transform=ax.transAxes,fontsize=7.0,color="#334e68",fontweight="semibold")
            ax.text2D(.50,.73,f"lowermost integration layer / крайний нижний интеграционный слой, z={zvals[0]:.6g}",transform=ax.transAxes,fontsize=7.0,color="#334e68",fontweight="semibold")
    except Exception:pass

def _mode_bilingual(mode):
    mp={
        "3D деформация":"3D деформация / 3D Deformation",
        "3D напряжения":"3D напряжения / 3D Stress",
        "3D деформации":"3D деформации / 3D Strain",
        "3D выбранное поле":"3D выбранное поле / 3D Selected Field",
        "Birger GB/G0":"Биргер GB/G0 / Birger GB/G0",
        "3D разрез по толщине":"Разрез по толщине / Through-thickness Cut",
        "3D продольный разрез":"Продольный разрез / Longitudinal Section",
        "3D поперечный разрез":"Поперечный разрез / Transverse Section",
        "3D комбинированный разрез":"Комбинированный разрез / Combined Section",
        "3D + нагрузка–прогиб":"3D + нагрузка–прогиб / 3D + Load–Deflection",
        "Разрез x-z":"Разрез x–z / x–z Section",
        "Разрез s-z":"Разрез s–z / s–z Section",
        "Профиль по толщине":"Профиль по толщине / Through-thickness Profile",
    }
    return mp.get(str(mode),str(mode))


def _title(d,step,mode):
    field=d.get("field_label",d.get("field",""));load=d.get("load_value");load_label=d.get("load_label","Load")
    suffix="" if load is None else f" · {load_label}={float(load):.6g}"
    return f"{_mode_bilingual(mode)} · STEP {int(step)} · {field}{suffix}"

def _presentation_3d_axis(fig):
    ax=fig.add_axes([.135,.055,.765,.845],projection="3d");ax.set_facecolor("#f7f9fc");return ax

def _presentation_colorbar(fig,norm,cmap,d,*,x=.918,y=.205,h=.56):
    import matplotlib as mpl
    cax=fig.add_axes([x,y,.022,h]);sm=mpl.cm.ScalarMappable(norm=norm,cmap=cmap);sm.set_array([]);cb=fig.colorbar(sm,cax=cax);cb.ax.tick_params(labelsize=9.5,width=.7);cb.set_label(f"{d.get('field_label',d.get('field',''))} [{d.get('field_unit','-')}]",fontsize=10.3,labelpad=8);return cb

def _presentation_title(fig,d,step,mode):
    fig.text(.53,.963,_title(d,step,mode),ha="center",va="top",fontsize=13.2,fontweight="semibold",color="#14212c")

def _style_2d_axes(ax):
    ax.set_facecolor("#fbfcfd");ax.grid(True,alpha=.16,linewidth=.7)
    for sp in ax.spines.values():sp.set_color("#aebbc6");sp.set_linewidth(.8)
    ax.tick_params(labelsize=9.3);ax.xaxis.label.set_size(10.2);ax.yaxis.label.set_size(10.2)

def _engineering_norm_cmap(mn,mx):
    """Final agreed presentation palette "E".

    The palette stays cold and насыщенная overall, but highlights maxima with a
    restrained warm tail.  Signed fields still use symmetric physical limits so
    zero keeps the same colour across frames.  No clipping, percentile tricks
    or non-physical remapping are used.
    """
    import matplotlib as mpl
    mn=float(mn);mx=float(mx)
    if not math.isfinite(mn) or not math.isfinite(mx):mn,mx=-1.0,1.0
    if mx<=mn:mx=mn+max(abs(mn),1.0)*1e-12
    span=mx-mn
    # COMSOL-like engineering rainbow agreed for static 3-D and video:
    # deep blue -> blue -> cyan -> green -> yellow -> orange -> red.
    # This changes display colours only; physical values and normalization are untouched.
    colors=("#0015A8","#003DDB","#0078F0","#00BFEA","#20D57A","#87DE32","#E6E52B","#FFD22A","#FF8B13","#E3180A")
    cmap=mpl.colors.LinearSegmentedColormap.from_list("cyl_comsol_like",colors,N=256)
    # Treat a field as genuinely signed only when both signs occupy a meaningful
    # part of its range; tiny numerical noise around zero must not flip palette.
    if mn<0.0<mx and min(abs(mn),abs(mx))>=0.035*span:
        lim=max(abs(mn),abs(mx),1e-30)
        return mpl.colors.Normalize(-lim,lim),cmap
    return mpl.colors.Normalize(mn,mx),cmap


def _render_frame(d,step,mode,mn,mx,width,height,history=None):
    from matplotlib.figure import Figure
    dpi=100;d=dict(d);d["_video_global_min"]=float(mn);d["_video_global_max"]=float(mx);d["_render_mode"]=str(mode)
    fig=Figure(figsize=(width/dpi,height/dpi),dpi=dpi,facecolor="#f7f9fc");norm,cmap=_engineering_norm_cmap(mn,mx);single3d=("3D деформация","3D напряжения","3D деформации","3D выбранное поле","Birger GB/G0")
    if mode=="3D + нагрузка–прогиб":
        d["_hud_anchor"]=(.012,.875);d["_hud_font"]=6.5
        ax=fig.add_axes([.02,.10,.50,.76],projection="3d");_plot_shell(ax,d,norm,cmap)
        ax2=fig.add_axes([.59,.15,.38,.69]);_style_2d_axes(ax2)
        if history:
            wref=np.asarray(history.get("wref",history.get("wcenter",[])),float);wm=np.asarray(history.get("wmax",[]),float);loads=np.asarray(history.get("loads",[]),float);hst=np.asarray(history.get("steps",[]),int)
            n=min(len(wref),len(wm),len(loads),len(hst));wref=wref[:n];wm=wm[:n];loads=loads[:n];hst=hst[:n];mask=hst<=int(step);wref=wref[mask];wm=wm[mask];loads=loads[mask];hst=hst[mask]
            ax2.plot(wref,loads,marker="o",markersize=4.2,linewidth=1.8,label=str(history.get("reference_label") or "|w| в контрольной точке / reference point"));ax2.plot(wm,loads,marker="x",markersize=5.0,linewidth=1.9,label="|w|max")
            pos=np.where(hst==int(step))[0]
            if pos.size:
                k=int(pos[-1]);ax2.scatter([wref[k]],[loads[k]],s=105,zorder=6,facecolor="#174f8a",edgecolor="white",linewidth=1.0);ax2.scatter([wm[k]],[loads[k]],s=125,zorder=7,facecolor="#d96d3e",edgecolor="white",linewidth=1.1);ax2.axhline(loads[k],color="#71879a",linewidth=1.0,alpha=.52,linestyle="--",zorder=1);ax2.axvline(wm[k],color="#b48a58",linewidth=1.0,alpha=.48,linestyle=":",zorder=1)
                txt=f"STEP {int(step)}\n{history.get('load_label','q')} = {loads[k]:.5g}\n|w|max = {wm[k]:.5g} {history.get('displacement_unit','mm')}";ax2.annotate(txt,(wm[k],loads[k]),xytext=(-112,14),textcoords="offset points",fontsize=9.5,fontweight="bold",color="#243b53",bbox=dict(boxstyle="round,pad=.35",facecolor=(1,1,1,.92),edgecolor="#9aa8b5",linewidth=.8));ax2.margins(x=.10,y=.10)
            ax2.set_xlabel(f"Прогиб / Deflection |w| [{history.get('displacement_unit','mm')}]");ax2.set_ylabel(history.get("load_label","Нагрузка / Load"));ax2.legend(fontsize=8.8,loc="best");ax2.set_title("Нагрузка–прогиб / Load–Deflection",fontsize=11.5,fontweight="semibold")
        _presentation_title(fig,d,step,mode)
    elif mode in single3d:
        ax=_presentation_3d_axis(fig);_plot_shell(ax,d,norm,cmap);_presentation_colorbar(fig,norm,cmap,d);_presentation_title(fig,d,step,mode)
    elif mode=="3D разрез по толщине":
        ax=_presentation_3d_axis(fig);_plot_cutaway(ax,d,norm,cmap);_presentation_colorbar(fig,norm,cmap,d);_presentation_title(fig,d,step,mode);_add_section_schematic(fig,d,"cutaway")
    elif mode=="3D продольный разрез":
        ax=_presentation_3d_axis(fig);_plot_longitudinal_cut(ax,d,norm,cmap);_presentation_colorbar(fig,norm,cmap,d);_presentation_title(fig,d,step,mode);_add_section_schematic(fig,d,"longitudinal",theta_cut=2*math.pi*int(d.get("point_is",0))/max(np.asarray(d.get("field3d",[])).shape[1],1))
    elif mode=="3D поперечный разрез":
        ax=_presentation_3d_axis(fig);_plot_transverse_cut(ax,d,norm,cmap);_presentation_colorbar(fig,norm,cmap,d);_presentation_title(fig,d,step,mode);nx=np.asarray(d.get("field3d",[])).shape[2];xcut=float(d.get("L",0))*int(d.get("point_ix",0))/max(nx-1,1);_add_section_schematic(fig,d,"transverse",xcut=xcut)
    elif mode=="3D комбинированный разрез":
        ax=_presentation_3d_axis(fig);_plot_combined_cut(ax,d,norm,cmap);_presentation_colorbar(fig,norm,cmap,d);_presentation_title(fig,d,step,mode);arr=np.asarray(d.get("field3d",[]));nx=arr.shape[2];ns=arr.shape[1];xcut=float(d.get("L",0))*int(d.get("point_ix",0))/max(nx-1,1);theta=2*math.pi*int(d.get("point_is",0))/max(ns,1);_add_section_schematic(fig,d,"combined",xcut=xcut,theta_cut=theta)
    elif mode=="Разрез x-z":
        ax=fig.add_axes([.11,.15,.73,.70]);_style_2d_axes(ax);im=_plot_xz(ax,d,norm,cmap);_presentation_title(fig,d,step,mode);theta=2*math.pi*int(d.get("point_is",0))/max(np.asarray(d["field3d"]).shape[1],1);_add_section_schematic(fig,d,"xz",theta_cut=theta);cax=fig.add_axes([.89,.20,.022,.55]);cb=fig.colorbar(im,cax=cax);cb.ax.tick_params(labelsize=9);cb.set_label(f"{d.get('field_label',d.get('field',''))} [{d.get('field_unit','-')}]",fontsize=10)
    elif mode=="Разрез s-z":
        ax=fig.add_axes([.11,.15,.73,.70]);_style_2d_axes(ax);im=_plot_sz(ax,d,norm,cmap);_presentation_title(fig,d,step,mode);arr=np.asarray(d["field3d"]);xcut=float(d.get("L",0))*int(d.get("point_ix",0))/max(arr.shape[2]-1,1);_add_section_schematic(fig,d,"sz",xcut=xcut);cax=fig.add_axes([.89,.20,.022,.55]);cb=fig.colorbar(im,cax=cax);cb.ax.tick_params(labelsize=9);cb.set_label(f"{d.get('field_label',d.get('field',''))} [{d.get('field_unit','-')}]",fontsize=10)
    elif mode=="Профиль по толщине":
        ax=fig.add_axes([.13,.16,.75,.68]);_style_2d_axes(ax);_plot_thickness(ax,d,mn,mx);_presentation_title(fig,d,step,mode)
    else:raise ValueError(f"Неизвестный VIDEO режим: {mode}")
    rgb=_figure_rgb(fig,width,height);fig.clear();return rgb

def export_mp4(
    output_path: str | Path,
    steps: Iterable[int],
    provider: Callable[[int,str,str,bool],dict],
    *,
    mode: str,
    field: str,
    fps: int = 10,
    seconds_per_step: float = .35,
    width: int = 1280,
    height: int = 720,
    history: dict | None = None,
    progress: Callable[[str,int,int,int],None] | None = None,
    cancel_event=None,
) -> Path:
    """Render an MP4 from already accepted steps.

    provider(step, mode, field, scan_only) must return numeric data.  The module
    never asks the solver for a new state and never writes anything besides the
    requested MP4.
    """
    steps=[int(s) for s in steps]
    if not steps:raise ValueError("Нет принятых шагов для VIDEO")
    if mode not in VIDEO_MODES:raise ValueError(f"Неизвестный VIDEO режим: {mode}")
    fps=max(1,min(int(fps),60));seconds_per_step=max(.05,float(seconds_per_step));width=int(width);height=int(height)
    if width%2:width+=1
    if height%2:height+=1
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    # Crash/cancel safety: encode into a temporary MP4 first.  The public
    # filename appears only after FFmpeg has closed and the file has passed a
    # frame/duration check.  This keeps RUN/VIDEO free of half-written files.
    tmp_path=output_path.with_name(output_path.stem+".__writing__.mp4")
    if tmp_path.exists():
        try:tmp_path.unlink()
        except Exception:pass
    mn,mx,max_disp,zmax=_global_scan(steps,provider,mode,field,progress,cancel_event)
    ff=_ffmpeg_module();writer=ff.write_frames(
        str(tmp_path),(width,height),fps=fps,codec="libx264",pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=1,
        output_params=["-movflags","+faststart","-crf","20","-preset","medium"],
    )
    writer.send(None)
    repeats=max(1,int(round(fps*seconds_per_step)));total=len(steps)*repeats;done=0
    try:
        for step in steps:
            if cancel_event is not None and cancel_event.is_set():raise InterruptedError("VIDEO export cancelled")
            d=provider(int(step),mode,field,False);d=dict(d);d["global_max_displacement"]=max_disp;d["global_zmax"]=zmax;d["step"]=int(step)
            frame=_render_frame(d,step,mode,mn,mx,width,height,history)
            raw=frame.tobytes()
            for _ in range(repeats):
                if cancel_event is not None and cancel_event.is_set():raise InterruptedError("VIDEO export cancelled")
                writer.send(raw);done+=1
                if progress:progress("encode",done,total,int(step))
    except Exception:
        try:writer.close()
        finally:
            if tmp_path.exists():
                try:tmp_path.unlink()
                except Exception:pass
        raise
    writer.close()
    try:
        nframes,duration=ff.count_frames_and_secs(str(tmp_path))
        if int(nframes)<int(total) or not math.isfinite(float(duration)) or float(duration)<=0.0:
            raise RuntimeError(
                f"FFmpeg создал неполное VIDEO: frames={nframes}/{total}, duration={duration}"
            )
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            try:tmp_path.unlink()
            except Exception:pass
        raise
    if progress:progress("done",total,total,int(steps[-1]))
    return output_path
