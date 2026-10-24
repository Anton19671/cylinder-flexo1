#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage11B Russian GUI for the final cylinder solver dispatcher.

GUI is input/routing only. It does not implement shell mechanics.
No physical geometry/material/load defaults are embedded.
"""
from __future__ import annotations

import json
import os
import copy
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage7a_general_mechanics as g7
from cylinder_final_stage11a_dispatcher import (
    THEORIES, CUF_MODELS, UNIT_SYSTEMS, METHODS, MODES, validate_final_case
)
import final_load_builder as loadbuilder
import final_birger_curve as birgercurve
import final_units as funits
import final_capability_registry as capreg
import flexo_scale_policy as flexoscale
import final_te_basis_provider as tebasis
import final_run_bundle as runbundle
import final_boundary_registry as bcreg

APP_DIR=Path(__file__).resolve().parent


def _default_results_root():
    """Shortest stable user-facing results root, independent of the ZIP path."""
    if os.name == "nt":
        drive = str(os.environ.get("SystemDrive") or APP_DIR.drive or "C:").rstrip("\\/")
        return Path(drive + "\\") / "CYL_RESULTS"
    return (APP_DIR.parent / "CYL_RESULTS").resolve()


def _is_legacy_program_results_root(value):
    """Recognise old solver-owned RESULTS roots and migrate them automatically."""
    text=str(value or "").strip()
    if not text:return True
    norm=text.replace("\\","/").rstrip("/").lower()
    if not norm.endswith("/results"):return False
    return any(token in norm for token in ("cyl_solver", "cylinder_solver", "/solver_v"))


class GreenStepProgressbar(tk.Canvas):
    """Deterministic green progress bar independent of the Windows ttk theme."""
    def __init__(self, master, length=250, mode="determinate", maximum=100, value=0, **kwargs):
        self._maximum=max(float(maximum),1.0)
        self._value=max(0.0,float(value))
        self._mode=str(mode)
        height=int(kwargs.pop("height",18))
        super().__init__(
            master,width=int(length),height=height,bd=0,highlightthickness=1,
            highlightbackground="#9a9a9a",background="#e8e8e8",**kwargs
        )
        self.bind("<Configure>",lambda _e:self._redraw())
        self._animating=False
        self._animation_after=None
        self._phase=0.0
        self._animation_interval=90
        self._display_text=""
        self._redraw()

    def configure(self, cnf=None, **kwargs):
        if isinstance(cnf,dict):
            kwargs={**cnf,**kwargs}
            cnf=None
        changed=False
        if "mode" in kwargs:
            self._mode=str(kwargs.pop("mode")); changed=True
        if "maximum" in kwargs:
            self._maximum=max(float(kwargs.pop("maximum")),1.0); changed=True
        if "value" in kwargs:
            self._value=max(0.0,float(kwargs.pop("value"))); changed=True
        result=super().configure(**kwargs) if kwargs else None
        if changed:self._redraw()
        return result

    config=configure

    def start(self, interval=None):
        # Keep the accepted-step fraction determinate, but animate a short green
        # segment in the still-unfinished part so a long first MVI step does not
        # look like a frozen grey control.  The numeric label remains the source
        # of truth for completed steps; this animation never invents progress.
        if interval is not None:
            try:self._animation_interval=max(30,int(interval))
            except Exception:pass
        if self._animating:return
        self._animating=True
        self._schedule_animation()

    def _schedule_animation(self):
        if not self._animating:return
        self._phase=(self._phase+0.055)%1.0
        self._redraw()
        try:self._animation_after=self.after(self._animation_interval,self._schedule_animation)
        except Exception:self._animation_after=None

    def stop(self):
        self._animating=False
        if self._animation_after is not None:
            try:self.after_cancel(self._animation_after)
            except Exception:pass
            self._animation_after=None
        self._redraw()

    def step(self, amount=1.0):
        self._value=min(self._maximum,self._value+float(amount)); self._redraw()

    def set_text(self,text):
        self._display_text=str(text or "")
        self._redraw()

    def _redraw(self):
        try:
            self.delete("progress_fill")
            self.delete("progress_text")
            w=max(int(self.winfo_width()),1); h=max(int(self.winfo_height()),1)
            ratio=min(max(self._value/self._maximum,0.0),1.0)
            fill=int(round(w*ratio))
            if fill>0:
                self.create_rectangle(0,0,fill,h,fill="#20a847",outline="",tags="progress_fill")
            if self._animating and ratio<1.0 and w>4:
                # Activity segment is restricted to the unfinished part.
                left=max(fill,0); remain=max(w-left,0)
                if remain>2:
                    seg=max(12,min(int(w*0.16),max(12,remain//3)))
                    travel=max(remain+seg,1)
                    x0=left+int(self._phase*travel)-seg
                    x1=x0+seg
                    x0=max(left,x0); x1=min(w,x1)
                    if x1>x0:
                        self.create_rectangle(x0,0,x1,h,fill="#43c463",outline="",tags="progress_fill")
            if self._display_text and w>70 and h>10:
                # Keep accepted step/percent visible even when neighbouring
                # toolbar labels are clipped by Windows/Tk layout.
                self.create_text(w//2,h//2,text=self._display_text,fill="#102016",
                                 font=("Segoe UI",9,"bold"),tags="progress_text")
        except Exception:
            pass


def _float(text,name,positive=False,nonnegative=False):
    s=str(text).strip()
    if not s:
        raise ValueError(f"Не заполнено поле: {name}")
    v=float(s)
    if not np.isfinite(v):
        raise ValueError(f"{name}: требуется конечное число")
    if positive and v<=0:
        raise ValueError(f"{name}: требуется число > 0")
    if nonnegative and v<0:
        raise ValueError(f"{name}: требуется число >= 0")
    return v


def _int(text,name,minv=1):
    s=str(text).strip()
    if not s:
        raise ValueError(f"Не заполнено поле: {name}")
    v=int(s)
    if str(v)!=s and float(s)!=float(v):
        raise ValueError(f"{name}: требуется целое число")
    if v<minv:
        raise ValueError(f"{name}: требуется целое >= {minv}")
    return v


class SolverGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Замкнутая цилиндрическая оболочка — CUF / Kirchhoff–Love")
        self.geometry("1120x780")
        self.minsize(980,680)

        self.proc=None
        self.post_proc=None
        self.summary_proc=None
        self.log_queue=queue.Queue()
        self.run_output_dir=None
        self.expected_steps=None
        self.run_kind="CASE"
        self.advanced_bc=None
        self.engineering_bc=None

        self.vars={}
        self._build_ui()
        self._install_global_edit_bindings()
        self._refresh_visibility()
        self.after(150,self._poll_log_queue)
        self.after(500,self._poll_progress_files)

    def _install_global_edit_bindings(self):
        """Windows-friendly edit shortcuts + right-click menu for all input widgets.

        Tk themes/platforms differ in their default Ctrl+C/Ctrl+V behavior.  The
        solver GUI therefore owns these edit bindings explicitly.  Class bindings
        also cover Entry/Text/Combobox/Spinbox widgets created later in dialogs.
        """
        classes=("TEntry","Entry","Text","TCombobox","TSpinbox","Spinbox")
        keymap={
            "<Control-a>":"select_all", "<Control-A>":"select_all",
            "<Control-c>":"copy", "<Control-C>":"copy",
            "<Control-x>":"cut", "<Control-X>":"cut",
            "<Control-v>":"paste", "<Control-V>":"paste",
            "<Control-Insert>":"copy", "<Shift-Insert>":"paste",
            "<Shift-Delete>":"cut",
        }
        for cls in classes:
            for seq,action in keymap.items():
                self.bind_class(cls,seq,lambda e,a=action:self._edit_event(e,a))
            self.bind_class(cls,"<Button-3>",self._show_edit_context_menu)

    @staticmethod
    def _widget_editable(widget):
        try:
            state=str(widget.cget("state")).lower()
            return state not in ("disabled","readonly")
        except Exception:
            return True

    def _edit_selected_text(self,widget):
        try:
            if isinstance(widget,tk.Text):
                ranges=widget.tag_ranges("sel")
                if len(ranges)==2:return widget.get(ranges[0],ranges[1])
                return ""
            if hasattr(widget,"selection_present") and widget.selection_present():
                a=widget.index("sel.first"); b=widget.index("sel.last")
                return str(widget.get())[int(a):int(b)]
        except Exception:
            pass
        return ""

    def _delete_edit_selection(self,widget):
        try:
            if isinstance(widget,tk.Text):
                ranges=widget.tag_ranges("sel")
                if len(ranges)==2:widget.delete(ranges[0],ranges[1])
            elif hasattr(widget,"selection_present") and widget.selection_present():
                widget.delete("sel.first","sel.last")
        except Exception:
            pass

    @staticmethod
    def _normalized_paste_text(widget,text):
        text=str(text).replace("\r\n","\n").replace("\r","\n")
        # List-valued single-line entries accept an Excel column/row as a
        # comma-separated list.  Ordinary scalar entries keep one line.
        if not isinstance(widget,tk.Text):
            if getattr(widget,"_clipboard_list_mode",False):
                import re
                tokens=[q.strip() for q in re.split(r"[\t\n]+",text.strip()) if q.strip()]
                return ", ".join(tokens)
            if "\n" in text:
                text=" ".join(q.strip() for q in text.splitlines() if q.strip())
            text=text.replace("\t"," ")
        return text

    def _edit_action(self,widget,action):
        try:
            if action=="select_all":
                widget.focus_set()
                if isinstance(widget,tk.Text):
                    widget.tag_add("sel","1.0","end-1c"); widget.mark_set("insert","1.0")
                else:
                    widget.selection_range(0,"end"); widget.icursor("end")
                return
            if action=="copy":
                txt=self._edit_selected_text(widget)
                if txt:
                    self.clipboard_clear(); self.clipboard_append(txt); self.update_idletasks()
                return
            if action in ("cut","paste") and not self._widget_editable(widget):return
            if action=="cut":
                txt=self._edit_selected_text(widget)
                if txt:
                    self.clipboard_clear(); self.clipboard_append(txt); self.update_idletasks(); self._delete_edit_selection(widget)
                return
            if action=="paste":
                try:txt=self.clipboard_get()
                except Exception:return
                txt=self._normalized_paste_text(widget,txt)
                self._delete_edit_selection(widget)
                if isinstance(widget,tk.Text):widget.insert("insert",txt)
                else:widget.insert("insert",txt)
        except Exception:
            # Clipboard support is convenience-only and must never break input.
            pass

    def _edit_event(self,event,action):
        self._edit_action(event.widget,action)
        return "break"

    def _show_edit_context_menu(self,event):
        w=event.widget
        try:w.focus_set()
        except Exception:pass
        menu=tk.Menu(self,tearoff=0)
        menu.add_command(label="Вырезать",command=lambda:self._edit_action(w,"cut"))
        menu.add_command(label="Копировать",command=lambda:self._edit_action(w,"copy"))
        menu.add_command(label="Вставить",command=lambda:self._edit_action(w,"paste"))
        menu.add_separator()
        menu.add_command(label="Выделить всё",command=lambda:self._edit_action(w,"select_all"))
        if not self._widget_editable(w):
            menu.entryconfigure("Вырезать",state="disabled"); menu.entryconfigure("Вставить",state="disabled")
        try:menu.tk_popup(event.x_root,event.y_root)
        finally:
            try:menu.grab_release()
            except Exception:pass
        return "break"

    def _v(self,name,default=""):
        # Reuse an existing Tk variable. Replacing it would disconnect widgets
        # already bound to the previous StringVar and lose GUI state.
        if name in self.vars:
            return self.vars[name]
        v=tk.StringVar(value=default)
        self.vars[name]=v
        return v

    def _build_ui(self):
        top=ttk.Frame(self,padding=8)
        top.pack(fill="x")

        ttk.Label(top,text="Теория").grid(row=0,column=0,sticky="w")
        theory=ttk.Combobox(top,textvariable=self._v("theory","CUF_TE1"),
                            values=THEORIES,state="readonly",width=14)
        theory.grid(row=0,column=1,padx=4)
        theory.bind("<<ComboboxSelected>>",lambda e:self._refresh_visibility())

        ttk.Label(top,text="Метод").grid(row=0,column=2,sticky="w")
        method=ttk.Combobox(top,textvariable=self._v("method","FDM"),
                            values=METHODS,state="readonly",width=10)
        method.grid(row=0,column=3,padx=4)
        method.bind("<<ComboboxSelected>>",lambda e:self._refresh_visibility())

        ttk.Label(top,text="Режим").grid(row=0,column=4,sticky="w")
        mode=ttk.Combobox(top,textvariable=self._v("mode","LINEAR_ELASTIC"),
                          values=MODES,state="readonly",width=23)
        mode.grid(row=0,column=5,padx=4)
        mode.bind("<<ComboboxSelected>>",lambda e:self._refresh_visibility())

        ttk.Label(top,text="МВИ-N").grid(row=0,column=6,sticky="w")
        self.rank_combo=ttk.Combobox(
            top,textvariable=self._v("rank",""),
            values=[str(i) for i in range(1,9)],state="readonly",width=5)
        self.rank_combo.grid(row=0,column=7,padx=4)

        notebook=ttk.Notebook(self)
        notebook.pack(fill="both",expand=True,padx=8,pady=(0,8))

        self.tab_problem=ttk.Frame(notebook,padding=10)
        self.tab_numeric=ttk.Frame(notebook,padding=10)
        self.tab_birger=ttk.Frame(notebook,padding=10)
        self.tab_files=ttk.Frame(notebook,padding=10)
        self.tab_run=ttk.Frame(notebook,padding=10)

        notebook.add(self.tab_problem,text="Постановка")
        notebook.add(self.tab_numeric,text="Численные параметры")
        notebook.add(self.tab_birger,text="Биргер")
        notebook.add(self.tab_files,text="Файлы и результаты")
        notebook.add(self.tab_run,text="Расчёт / лог")

        self._build_problem_tab()
        self._build_numeric_tab()
        self._build_birger_tab()
        self._build_files_tab()
        self._build_run_tab()

    def _entry_row(self,parent,row,label,key,width=18):
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky="w",pady=3)
        e=ttk.Entry(parent,textvariable=self._v(key),width=width)
        e.grid(row=row,column=1,sticky="w",padx=6,pady=3)
        return e

    def _build_problem_tab(self):
        f=self.tab_problem
        geom=ttk.LabelFrame(f,text="Геометрия — вводится пользователем",padding=8)
        geom.grid(row=0,column=0,sticky="nsew",padx=(0,8),pady=4)
        self._entry_row(geom,0,"L","L")
        self._entry_row(geom,1,"R","R")
        self._entry_row(geom,2,"h","h")

        mat=ttk.LabelFrame(f,text="Материал",padding=8)
        mat.grid(row=0,column=1,sticky="nsew",pady=4)
        self._entry_row(mat,0,"E0","E0")
        self._entry_row(mat,1,"ν0","nu0")

        grid=ttk.LabelFrame(f,text="Глобальная сетка",padding=8)
        grid.grid(row=1,column=0,sticky="nsew",padx=(0,8),pady=4)
        self._entry_row(grid,0,"nx","nx")
        self._entry_row(grid,1,"ns","ns")
        self._entry_row(grid,2,"Stencil","stencil")
        self._entry_row(grid,3,"Gauss по толщине","n_gauss")

        bc=ttk.LabelFrame(f,text="Краевые условия по торцам x=0 / x=L",padding=8)
        bc.grid(row=1,column=1,sticky="nsew",pady=4)
        ttk.Label(bc,text="Режим КУ").grid(row=0,column=0,sticky="w")
        bcm=ttk.Combobox(bc,textvariable=self._v("bc_mode","SIMPLE"),
                         values=("SIMPLE","INDIVIDUAL"),state="readonly",width=14)
        bcm.grid(row=0,column=1,padx=6,sticky="w")
        bcm.bind("<<ComboboxSelected>>",lambda e:self._refresh_visibility())

        ttk.Label(bc,text="Левый торец").grid(row=1,column=0,sticky="w")
        self.bc_left_combo=ttk.Combobox(bc,textvariable=self._v("bc_left",""),
                     values=("","CLAMPED","FREE"),state="readonly",width=14)
        self.bc_left_combo.grid(row=1,column=1,padx=6,sticky="w")
        ttk.Label(bc,text="Правый торец").grid(row=2,column=0,sticky="w")
        self.bc_right_combo=ttk.Combobox(bc,textvariable=self._v("bc_right",""),
                     values=("","CLAMPED","FREE"),state="readonly",width=14)
        self.bc_right_combo.grid(row=2,column=1,padx=6,sticky="w")

        self.bc_edit_btn=ttk.Button(bc,text="Редактор индивидуальных КУ…",
                                    command=self._edit_individual_bc)
        self.bc_edit_btn.grid(row=3,column=0,columnspan=2,sticky="w",pady=(8,4))
        self.bc_summary=ttk.Label(bc,text="SIMPLE: требуется явный выбор",justify="left")
        self.bc_summary.grid(row=4,column=0,columnspan=2,sticky="w")
        ttk.Label(
            bc,
            text="FREE–FREE без отдельной обработки жёстких мод явно отклоняется.",
            justify="left"
        ).grid(row=5,column=0,columnspan=2,sticky="w",pady=(8,0))

        f.columnconfigure(0,weight=1)
        f.columnconfigure(1,weight=1)

    def _build_numeric_tab(self):
        f=self.tab_numeric

        self.mvi_frame=ttk.LabelFrame(f,text="МВИ — только при method=MVI",padding=8)
        self.mvi_frame.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        self._entry_row(self.mvi_frame,0,"Projected tolerance (relative)","mvi_proj")
        self._entry_row(self.mvi_frame,1,"Increment tolerance","mvi_inc")
        self._entry_row(self.mvi_frame,2,"Tangent FD step","mvi_fd")
        self._entry_row(self.mvi_frame,3,"Max halfstep iterations","mvi_half")
        self._entry_row(self.mvi_frame,4,"Max X↔Y cycles","mvi_cycles")
        ttk.Label(self.mvi_frame,text="Step predictor").grid(row=5,column=0,sticky="w")
        ttk.Combobox(
            self.mvi_frame,textvariable=self._v("step_predictor","ERROR"),
            values=("ERROR","LINEAR"),state="readonly",width=12
        ).grid(row=5,column=1,sticky="w",padx=6)
        ttk.Label(
            self.mvi_frame,
            text="Начальные X/Y не генерируются автоматически.",
        ).grid(row=6,column=0,columnspan=2,sticky="w",pady=(8,0))
        ttk.Label(
            self.mvi_frame,
            text=("Критерий принятия МВИ: RprojX/RprojY и приращение поля. "
                  "Полная невязка МКР Rfull — диагностическая величина и для "
                  "низкого ранга не обязана быть нулевой."),
            justify="left",wraplength=430,
        ).grid(row=7,column=0,columnspan=2,sticky="w",pady=(8,0))

        self.newton_frame=ttk.LabelFrame(f,text="Newton — только nonlinear FDM",padding=8)
        self.newton_frame.grid(row=0,column=1,sticky="nsew")
        self._v("newton_res","1e-9")
        self._entry_row(self.newton_frame,0,"Relative residual tolerance (V2)","newton_res")
        self._entry_row(self.newton_frame,1,"Increment tolerance","newton_inc")
        self._entry_row(self.newton_frame,2,"Relative FD Jacobian step (V3)","newton_fd")
        self._entry_row(self.newton_frame,3,"Max iterations","newton_max")

        self.linear_frame=ttk.LabelFrame(f,text="Линейный FDM — проверка К-2",padding=8)
        self.linear_frame.grid(row=1,column=1,sticky="new",pady=(10,0))
        self._v("linear_res","1e-10")
        self._entry_row(self.linear_frame,0,"Relative residual tolerance","linear_res")
        ttk.Label(
            self.linear_frame,
            text="После np.linalg.solve невязка обязательно проверяется; рекомендуемо 1e-10.",
            wraplength=430,
        ).grid(row=1,column=0,columnspan=2,sticky="w",pady=(7,0))

        f.columnconfigure(0,weight=1)
        f.columnconfigure(1,weight=1)

    def _build_birger_tab(self):
        f=self.tab_birger
        self.birger_frame=ttk.LabelFrame(f,text="Внешний цикл Биргера",padding=8)
        self.birger_frame.pack(fill="both",expand=True)

        labels=[
            ("epsilon_tol","ε_i tolerance"),
            ("omega","ω"),
            ("G_ref","G_ref"),
            ("sigma_ref","σ_ref"),
            ("tol_G","tol_G"),
            ("tol_sigma","tol_σ"),
            ("tol_u","tol_u (изменение поля, В-6)"),
            ("max_outer","Max outer iterations"),
            ("eps_limit","ε_i limit"),
            ("dJ_limit","|J−1| limit"),
        ]
        for r,(key,label) in enumerate(labels):
            self._entry_row(self.birger_frame,r,label,key)

        ttk.Separator(self.birger_frame,orient="horizontal").grid(
            row=len(labels),column=0,columnspan=3,sticky="ew",pady=8)

        r=len(labels)+1
        ttk.Label(self.birger_frame,text="Модуль Φ(ε_i)").grid(row=r,column=0,sticky="w")
        ttk.Entry(self.birger_frame,textvariable=self._v("phi_module"),width=28).grid(row=r,column=1,sticky="w",padx=6)
        r+=1
        ttk.Label(self.birger_frame,text="Callable").grid(row=r,column=0,sticky="w")
        ttk.Entry(self.birger_frame,textvariable=self._v("phi_callable"),width=28).grid(row=r,column=1,sticky="w",padx=6)
        r+=1
        ttk.Label(self.birger_frame,text="Python path").grid(row=r,column=0,sticky="w")
        ttk.Entry(self.birger_frame,textvariable=self._v("phi_path"),width=48).grid(row=r,column=1,sticky="ew",padx=6)
        ttk.Button(self.birger_frame,text="…",command=lambda:self._choose_dir("phi_path")).grid(row=r,column=2)

    def _file_row(self,parent,row,label,key,kind="file"):
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky="w",pady=3)
        ttk.Entry(parent,textvariable=self._v(key),width=72).grid(row=row,column=1,sticky="ew",padx=6,pady=3)
        cmd=(lambda:self._choose_dir(key)) if kind=="dir" else (lambda:self._choose_file(key))
        ttk.Button(parent,text="…",width=3,command=cmd).grid(row=row,column=2,pady=3)

    def _build_files_tab(self):
        f=self.tab_files
        f.columnconfigure(1,weight=1)

        self._file_row(f,0,"Loads NPZ","loads_npz")
        ttk.Button(f,text="Создать простую радиальную нагрузку",
                   command=self._create_simple_radial_load).grid(row=1,column=1,sticky="w",pady=(0,8))

        self.mvi_profile_row=ttk.Frame(f)
        self.mvi_profile_row.grid(row=2,column=0,columnspan=3,sticky="ew")
        self.mvi_profile_row.columnconfigure(1,weight=1)
        self._file_row(self.mvi_profile_row,0,"Initial X/Y NPZ","profiles_npz")

        self.initial_state_frame=ttk.LabelFrame(f,text="Начальное поле nonlinear FDM",padding=8)
        self.initial_state_frame.grid(row=3,column=0,columnspan=3,sticky="ew",pady=6)
        ttk.Label(self.initial_state_frame,text="Source").grid(row=0,column=0,sticky="w")
        ttk.Combobox(self.initial_state_frame,textvariable=self._v("state_source","ZERO"),
                     values=("ZERO","NPZ"),state="readonly",width=12).grid(row=0,column=1,sticky="w",padx=6)
        self._file_row(self.initial_state_frame,1,"State NPZ","state_npz")

        self.gb_frame=ttk.LabelFrame(f,text="Начальное GB для Birger",padding=8)
        self.gb_frame.grid(row=4,column=0,columnspan=3,sticky="ew",pady=6)
        ttk.Label(self.gb_frame,text="Source").grid(row=0,column=0,sticky="w")
        ttk.Combobox(self.gb_frame,textvariable=self._v("gb_source","ELASTIC_G0"),
                     values=("ELASTIC_G0","NPZ"),state="readonly",width=14).grid(row=0,column=1,sticky="w",padx=6)
        self._file_row(self.gb_frame,1,"GB NPZ","gb_npz")

        self._file_row(f,5,"Папка результатов","output_dir",kind="dir")
        self._file_row(f,6,"FINAL_CASE.json","case_path")

        buttons=ttk.Frame(f)
        buttons.grid(row=7,column=0,columnspan=3,sticky="w",pady=10)
        ttk.Button(buttons,text="Сохранить CASE",command=self.save_case).pack(side="left",padx=(0,6))
        ttk.Button(buttons,text="Загрузить CASE",command=self.load_case).pack(side="left")

    def _build_run_tab(self):
        f=self.tab_run
        controls=ttk.Frame(f)
        controls.pack(fill="x")
        self.run_btn=ttk.Button(controls,text="РАССЧИТАТЬ",command=self.run_case)
        self.run_btn.pack(side="left")
        self.series_btn=ttk.Button(controls,text="МВИ-1…8",command=self.run_mvi_series)
        self.series_btn.pack(side="left",padx=(6,0))
        ttk.Button(controls,text="Проверить входные данные",command=self.validate_case).pack(side="left",padx=6)
        ttk.Button(controls,text="Открыть результаты",command=self.open_results).pack(side="left")

        self.progress=GreenStepProgressbar(controls,mode="determinate",length=250)
        self.progress.pack(side="left",padx=12)
        self.progress_label=ttk.Label(controls,text="Готово")
        self.progress_label.pack(side="left")

        self.log=tk.Text(f,wrap="word",height=30)
        self.log.pack(fill="both",expand=True,pady=(8,0))

    def _progress_reset(self,total):
        total=max(int(total or 0),1)
        self.progress.stop()
        self.progress.configure(mode="determinate",maximum=total,value=0)
        shown_total=int(self.expected_steps or total)
        text=f"Принято 0/{shown_total} · 0%"
        self.progress_label.configure(text=text)
        try:self.progress.set_text(text)
        except Exception:pass
        if hasattr(self,"progress_status_label"):
            self.progress_status_label.configure(text="Готово")

    def _progress_update(self,completed,total=None,label_prefix=None):
        total=int(total or self.expected_steps or 0)
        completed=max(int(completed or 0),0)
        if total>0:
            shown=min(completed,total)
            pct=int(round(100.0*shown/total))
            self.progress.configure(mode="determinate",maximum=total,value=shown)
            text=f"Принято {shown}/{total} · {pct}%"
        else:
            self.progress.configure(mode="determinate",maximum=1,value=0)
            text=f"Принято шагов: {completed}"
        self.progress_label.configure(text=text)
        try:self.progress.set_text(text)
        except Exception:pass
        if hasattr(self,"progress_status_label"):
            if label_prefix:
                self.progress_status_label.configure(text=str(label_prefix))
        elif label_prefix:
            self.progress_label.configure(text=f"{label_prefix}: {text}")

    def _refresh_visibility(self):
        method=self.vars.get("method",tk.StringVar(value="FDM")).get()
        mode=self.vars.get("mode",tk.StringVar(value="LINEAR_ELASTIC")).get()
        nonlinear=mode.startswith("NONLINEAR")
        birger=mode.endswith("BIRGER")

        try:
            self.rank_combo.configure(state="readonly" if method=="MVI" else "disabled")
            if hasattr(self,"series_btn"):
                self.series_btn.configure(state="normal" if method=="MVI" else "disabled")
        except Exception:
            pass

        for child in getattr(self,"mvi_frame",ttk.Frame()).winfo_children():
            try: child.configure(state="normal" if method=="MVI" else "disabled")
            except: pass
        for child in getattr(self,"newton_frame",ttk.Frame()).winfo_children():
            try: child.configure(state="normal" if (method=="FDM" and nonlinear) else "disabled")
            except: pass
        for child in getattr(self,"linear_frame",ttk.Frame()).winfo_children():
            try: child.configure(state="normal" if (method=="FDM" and mode=="LINEAR_ELASTIC") else "disabled")
            except: pass
        for child in getattr(self,"birger_frame",ttk.Frame()).winfo_children():
            try: child.configure(state="normal" if birger else "disabled")
            except: pass

        bc_mode=self.vars.get("bc_mode",tk.StringVar(value="SIMPLE")).get()
        try:
            self.bc_left_combo.configure(state="readonly" if bc_mode=="SIMPLE" else "disabled")
            self.bc_right_combo.configure(state="readonly" if bc_mode=="SIMPLE" else "disabled")
            self.bc_edit_btn.configure(state="normal")
            if bc_mode=="SIMPLE":
                self.bc_summary.configure(
                    text=f"SIMPLE: {self.vars['bc_left'].get()} / {self.vars['bc_right'].get()}"
                )
            else:
                self.bc_summary.configure(text=self._advanced_bc_summary())
        except Exception:
            pass

    def _advanced_bc_summary(self):
        if not self.advanced_bc:
            return "INDIVIDUAL: не задано"
        theory=self.vars["theory"].get()
        try:
            if theory=="KL":
                parts=[]
                for side in ("left","right"):
                    vals=[self.advanced_bc[side][f]["kind"][0] for f in ("u","v","w","slope")]
                    parts.append(f"{side}: "+"/".join(vals))
                return "INDIVIDUAL: " + " ; ".join(parts)
            parts=[]
            for side in ("left","right"):
                ks=[]
                for comp in ("u","v","w"):
                    ks.extend(item["kind"][0] for item in self.advanced_bc[side][comp])
                parts.append(f"{side}: "+"/".join(ks))
            return "INDIVIDUAL: " + " ; ".join(parts)
        except Exception:
            return "INDIVIDUAL: задано через JSON"

    def _default_individual_bc(self,theory):
        def from_simple(value):
            if value=="CLAMPED": return "GEOMETRIC"
            if value=="FREE": return "NATURAL"
            return ""
        left_kind=from_simple(self.vars["bc_left"].get())
        right_kind=from_simple(self.vars["bc_right"].get())
        if theory=="KL":
            return {
                "left":{f:{"kind":left_kind,"value":0.0} for f in ("u","v","w","slope")},
                "right":{f:{"kind":right_kind,"value":0.0} for f in ("u","v","w","slope")},
            }
        def end(kind):
            return {c:[{"kind":kind,"data":0.0},{"kind":kind,"data":0.0}]
                    for c in ("u","v","w")}
        return {"left":end(left_kind),"right":end(right_kind)}

    def _edit_individual_bc(self):
        theory=self.vars["theory"].get()
        # The editor is always reachable.  When opened from SIMPLE, use the
        # currently selected end presets as the initial individual BC set and
        # switch the GUI to INDIVIDUAL mode before showing the editor.
        if self.vars.get("bc_mode", tk.StringVar(value="SIMPLE")).get() != "INDIVIDUAL":
            self.advanced_bc=self._default_individual_bc(theory)
            self.vars["bc_mode"].set("INDIVIDUAL")
            self._refresh_visibility()
        elif self.advanced_bc is None:
            self.advanced_bc=self._default_individual_bc(theory)

        win=tk.Toplevel(self)
        win.title(f"Индивидуальные КУ — {theory}")
        win.transient(self)
        vars_map={}

        headers=("Торец","Поле","Тип","Значение")
        for c,h in enumerate(headers):
            ttk.Label(win,text=h).grid(row=0,column=c,padx=5,pady=4,sticky="w")

        rows=[]
        if theory=="KL":
            for side in ("left","right"):
                for field in ("u","v","w","slope"):
                    item=self.advanced_bc.get(side,{}).get(field,{"kind":"","value":0.0})
                    rows.append((side,field,item.get("kind",""),item.get("value",0.0)))
        else:
            for side in ("left","right"):
                for comp in ("u","v","w"):
                    items=self.advanced_bc.get(side,{}).get(comp,[
                        {"kind":"","data":0.0},{"kind":"","data":0.0}])
                    for tau in range(2):
                        item=items[tau]
                        rows.append((side,f"{comp}{tau}",item.get("kind",""),item.get("data",0.0)))

        for r,(side,field,kind,value) in enumerate(rows,1):
            ttk.Label(win,text="x=0" if side=="left" else "x=L").grid(row=r,column=0,padx=5,pady=2,sticky="w")
            ttk.Label(win,text=field).grid(row=r,column=1,padx=5,pady=2,sticky="w")
            kv=tk.StringVar(value=str(kind).upper())
            vv=tk.StringVar(value=str(value))
            ttk.Combobox(win,textvariable=kv,values=("","GEOMETRIC","NATURAL"),
                         state="readonly",width=12).grid(row=r,column=2,padx=5,pady=2)
            ttk.Entry(win,textvariable=vv,width=18).grid(row=r,column=3,padx=5,pady=2)
            vars_map[(side,field)]=(kv,vv)

        def save():
            if theory=="KL":
                adv={"left":{},"right":{}}
                for side in ("left","right"):
                    for field in ("u","v","w","slope"):
                        kv,vv=vars_map[(side,field)]
                        if kv.get() not in ("GEOMETRIC","NATURAL"):
                            raise ValueError(f"Не выбран тип КУ: {side}.{field}")
                        adv[side][field]={"kind":kv.get(),"value":_float(vv.get(),f"{side}.{field}")}
            else:
                adv={"left":{c:[None,None] for c in ("u","v","w")},
                     "right":{c:[None,None] for c in ("u","v","w")}}
                for side in ("left","right"):
                    for comp in ("u","v","w"):
                        for tau in range(2):
                            kv,vv=vars_map[(side,f"{comp}{tau}")]
                            if kv.get() not in ("GEOMETRIC","NATURAL"):
                                raise ValueError(f"Не выбран тип КУ: {side}.{comp}{tau}")
                            adv[side][comp][tau]={"kind":kv.get(),"data":_float(vv.get(),f"{side}.{comp}{tau}")}
            self.advanced_bc=adv
            self.vars["bc_mode"].set("INDIVIDUAL")
            self._refresh_visibility()
            win.destroy()

        ttk.Button(win,text="Сохранить КУ",command=lambda:self._safe_dialog_call(save)).grid(
            row=len(rows)+1,column=0,columnspan=4,pady=10)

    def _choose_file(self,key):
        p=filedialog.askopenfilename()
        if p:
            self.vars[key].set(p)

    def _choose_dir(self,key):
        p=filedialog.askdirectory()
        if p:
            self.vars[key].set(p)

    def _case_dict(self):
        theory=self.vars["theory"].get()
        method=self.vars["method"].get()
        mode=self.vars["mode"].get()

        cfg={
            "theory":theory,
            "method":method,
            "mode":mode,
            "rank":int(self.vars["rank"].get()) if method=="MVI" else None,
            "geometry":{
                "L":_float(self.vars["L"].get(),"L",positive=True),
                "R":_float(self.vars["R"].get(),"R",positive=True),
                "h":_float(self.vars["h"].get(),"h",positive=True),
            },
            "material":{
                "E0":_float(self.vars["E0"].get(),"E0",positive=True),
                "nu0":_float(self.vars["nu0"].get(),"ν0"),
            },
            "grid":{
                "nx":_int(self.vars["nx"].get(),"nx",7),
                "ns":_int(self.vars["ns"].get(),"ns",7),
                "stencil":_int(self.vars["stencil"].get(),"stencil",3),
            },
            "quadrature":{
                "type":"GAUSS",
                "n_gauss":_int(self.vars["n_gauss"].get(),"n_gauss",1),
            },
            "boundary_conditions":({
                "left":self.vars["bc_left"].get(),
                "right":self.vars["bc_right"].get(),
                "advanced":None,
            } if self.vars["bc_mode"].get()=="SIMPLE" else {
                "left":None,
                "right":None,
                "advanced":self.advanced_bc,
            }),
            "loads_npz":self.vars["loads_npz"].get().strip() or None,
            "initial_profiles_npz":self.vars["profiles_npz"].get().strip() or None,
            "initial_state":{
                "source":self.vars["state_source"].get() if (method=="FDM" and mode.startswith("NONLINEAR")) else None,
                "npz":self.vars["state_npz"].get().strip() or None,
            },
            "initial_GB":{
                "source":self.vars["gb_source"].get() if mode.endswith("BIRGER") else None,
                "npz":self.vars["gb_npz"].get().strip() or None,
            },
            "inner_mvi":{
                "projected_tolerance":_float(self.vars["mvi_proj"].get(),"MVI projected tolerance",positive=True) if method=="MVI" else None,
                "increment_tolerance":_float(self.vars["mvi_inc"].get(),"MVI increment tolerance",positive=True) if method=="MVI" else None,
                "tangent_fd_step":_float(self.vars["mvi_fd"].get(),"MVI tangent FD step",positive=True) if method=="MVI" else None,
                "max_halfstep_iterations":_int(self.vars["mvi_half"].get(),"MVI max halfstep",1) if method=="MVI" else None,
                "max_xy_cycles":_int(self.vars["mvi_cycles"].get(),"MVI max cycles",1) if method=="MVI" else None,
                "detf_guarded_correction":True if method=="MVI" else None,
                "max_correction_halvings":24 if method=="MVI" else None,
            },
            "step_predictor":{"mode":self._v("step_predictor","ERROR").get() if method=="MVI" else "ERROR"},
            "linear":{
                "residual_tolerance_relative":_float(self.vars["linear_res"].get(),"Linear relative residual tolerance",positive=True) if (method=="FDM" and mode=="LINEAR_ELASTIC") else None,
            },
            "newton":{
                "residual_tolerance_relative":_float(self.vars["newton_res"].get(),"Newton relative residual tolerance",positive=True) if (method=="FDM" and mode.startswith("NONLINEAR")) else None,
                "increment_tolerance":_float(self.vars["newton_inc"].get(),"Newton increment tolerance",positive=True) if (method=="FDM" and mode.startswith("NONLINEAR")) else None,
                "fd_step":_float(self.vars["newton_fd"].get(),"Newton relative FD step",positive=True) if (method=="FDM" and mode.startswith("NONLINEAR")) else None,
                "max_iterations":_int(self.vars["newton_max"].get(),"Newton max iterations",1) if (method=="FDM" and mode.startswith("NONLINEAR")) else None,
            },
            "birger":{
                "epsilon_tol":_float(self.vars["epsilon_tol"].get(),"epsilon_tol",nonnegative=True) if mode.endswith("BIRGER") else None,
                "omega":_float(self.vars["omega"].get(),"omega",positive=True) if mode.endswith("BIRGER") else None,
                "G_ref":_float(self.vars["G_ref"].get(),"G_ref",positive=True) if mode.endswith("BIRGER") else None,
                "sigma_ref":_float(self.vars["sigma_ref"].get(),"sigma_ref",positive=True) if mode.endswith("BIRGER") else None,
                "tol_G":_float(self.vars["tol_G"].get(),"tol_G",positive=True) if mode.endswith("BIRGER") else None,
                "tol_sigma":_float(self.vars["tol_sigma"].get(),"tol_sigma",positive=True) if mode.endswith("BIRGER") else None,
                "tol_u":_float(self.vars["tol_u"].get(),"tol_u",positive=True) if mode.endswith("BIRGER") else None,
                "max_outer_iterations":_int(self.vars["max_outer"].get(),"max_outer_iterations",1) if mode.endswith("BIRGER") else None,
                "epsilon_i_limit":_float(self.vars["eps_limit"].get(),"epsilon_i_limit",positive=True) if mode.endswith("BIRGER") else None,
                "delta_J_limit":_float(self.vars["dJ_limit"].get(),"delta_J_limit",positive=True) if mode.endswith("BIRGER") else None,
                "reject_local_unloading":True if mode.endswith("BIRGER") else None,
                "unloading_tolerance":1.0e-12 if mode.endswith("BIRGER") else None,
                "material_diagram":{
                    "module":self.vars["phi_module"].get().strip() or None,
                    "callable":self.vars["phi_callable"].get().strip() or None,
                    "python_path":self.vars["phi_path"].get().strip() or None,
                },
            },
            "output_dir":self.vars["output_dir"].get().strip() or None,
        }
        return cfg

    def _case_path_for_save(self):
        p=self.vars["case_path"].get().strip()
        if p:
            return Path(p)
        p=filedialog.asksaveasfilename(
            title="Сохранить FINAL_CASE.json",
            defaultextension=".json",
            filetypes=[("JSON","*.json")])
        if not p:
            return None
        self.vars["case_path"].set(p)
        return Path(p)

    def save_case(self):
        try:
            cfg=self._case_dict()
            p=self._case_path_for_save()
            if p is None:
                return
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8")
            self._log(f"CASE сохранён: {p}\n")
        except Exception as exc:
            messagebox.showerror("Ошибка CASE",str(exc))

    def load_case(self):
        p=filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not p:
            return
        try:
            cfg=json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(cfg.get("newton"),dict) and "residual_tolerance" in cfg["newton"]:
                raise ValueError("V2: старое поле newton.residual_tolerance (абсолютное) запрещено; используйте residual_tolerance_relative, рекомендуемо 1e-9")
            self.vars["case_path"].set(p)
            scalar_paths={
                "theory":("theory",),"method":("method",),"mode":("mode",),"rank":("rank",),
                "L":("geometry","L"),"R":("geometry","R"),"h":("geometry","h"),
                "E0":("material","E0"),"nu0":("material","nu0"),
                "nx":("grid","nx"),"ns":("grid","ns"),"stencil":("grid","stencil"),
                "n_gauss":("quadrature","n_gauss"),
                "loads_npz":("loads_npz",),"profiles_npz":("initial_profiles_npz",),
                "output_dir":("output_dir",),
                "state_source":("initial_state","source"),"state_npz":("initial_state","npz"),
                "gb_source":("initial_GB","source"),"gb_npz":("initial_GB","npz"),
                "mvi_proj":("inner_mvi","projected_tolerance"),
                "mvi_inc":("inner_mvi","increment_tolerance"),
                "mvi_fd":("inner_mvi","tangent_fd_step"),
                "mvi_half":("inner_mvi","max_halfstep_iterations"),
                "mvi_cycles":("inner_mvi","max_xy_cycles"),
                "mvi_formulation":("inner_mvi","formulation"),
                "step_predictor":("step_predictor","mode"),
                "linear_res":("linear","residual_tolerance_relative"),
                "newton_res":("newton","residual_tolerance_relative"),
                "newton_inc":("newton","increment_tolerance"),
                "newton_fd":("newton","fd_step"),
                "newton_max":("newton","max_iterations"),
                "epsilon_tol":("birger","epsilon_tol"),"omega":("birger","omega"),
                "G_ref":("birger","G_ref"),"sigma_ref":("birger","sigma_ref"),
                "tol_G":("birger","tol_G"),"tol_sigma":("birger","tol_sigma"),"tol_u":("birger","tol_u"),
                "max_outer":("birger","max_outer_iterations"),
                "eps_limit":("birger","epsilon_i_limit"),"dJ_limit":("birger","delta_J_limit"),
                "phi_module":("birger","material_diagram","module"),
                "phi_callable":("birger","material_diagram","callable"),
                "phi_path":("birger","material_diagram","python_path"),
            }
            for key,path in scalar_paths.items():
                cur=cfg
                try:
                    for x in path:
                        cur=cur[x]
                except Exception:
                    cur=""
                self.vars[key].set("" if cur is None else str(cur))

            bcj=cfg.get("boundary_conditions",{})
            if bcj.get("advanced") is not None:
                self.vars["bc_mode"].set("INDIVIDUAL")
                self.advanced_bc=bcj.get("advanced")
            else:
                self.vars["bc_mode"].set("SIMPLE")
                self.vars["bc_left"].set(str(bcj.get("left") or ""))
                self.vars["bc_right"].set(str(bcj.get("right") or ""))
                self.advanced_bc=None
            self._refresh_visibility()
            self._log(f"CASE загружен: {p}\n")
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки",str(exc))

    def validate_case(self):
        try:
            cfg=self._case_dict()
            p=self._case_path_for_save()
            if p is None:
                return
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8")
            result=validate_final_case(p)
            messagebox.showinfo("Проверка входа",json.dumps(result,indent=2,ensure_ascii=False))
            self._log("VALIDATION PASS\n")
        except Exception as exc:
            messagebox.showerror("Проверка входа — ошибка",str(exc))
            self._log(f"VALIDATION FAIL: {type(exc).__name__}: {exc}\n")

    def _create_simple_radial_load(self):
        try:
            theory=self.vars["theory"].get()
            L=_float(self.vars["L"].get(),"L",positive=True)
            R=_float(self.vars["R"].get(),"R",positive=True)
            h=_float(self.vars["h"].get(),"h",positive=True)
            nx=_int(self.vars["nx"].get(),"nx",7)
            ns=_int(self.vars["ns"].get(),"ns",7)

            win=tk.Toplevel(self)
            win.title("Простая радиальная нагрузка")
            qv=tk.StringVar()
            factors=tk.StringVar(value="1")
            face=tk.StringVar(value="OUTER")

            ttk.Label(win,text="q (signed)").grid(row=0,column=0,sticky="w",padx=8,pady=4)
            ttk.Entry(win,textvariable=qv).grid(row=0,column=1,padx=8,pady=4)
            ttk.Label(win,text="Коэффициенты шагов через запятую").grid(row=1,column=0,sticky="w",padx=8,pady=4)
            ttk.Entry(win,textvariable=factors,width=32).grid(row=1,column=1,padx=8,pady=4)
            if theory=="CUF":
                ttk.Label(win,text="Поверхность").grid(row=2,column=0,sticky="w",padx=8,pady=4)
                ttk.Combobox(win,textvariable=face,values=("OUTER","INNER"),
                             state="readonly").grid(row=2,column=1,padx=8,pady=4)

            def create():
                q=_float(qv.get(),"q")
                fs=[float(x.strip()) for x in factors.get().split(",") if x.strip()]
                if not fs:
                    raise ValueError("Нужен хотя бы один коэффициент нагрузки")
                _grid=mvi.CylinderGrid(L,R,nx,ns)
                qr=np.full((ns,nx),q,float)

                if theory=="CUF":
                    basis=tebasis.get_basis(self.vars["cuf_model"].get())
                    zq=+0.5*h if face.get()=="OUTER" else -0.5*h
                    p0=np.asarray(g7.radial_nominal_face_load(qr,zq=zq,R=R,basis=basis),float)
                else:
                    p0=np.zeros((3,ns,nx),float)
                    p0[2]=qr
                loads=np.stack([f*p0 for f in fs],axis=0)

                path=filedialog.asksaveasfilename(
                    parent=win,defaultextension=".npz",
                    filetypes=[("NumPy NPZ","*.npz")])
                if not path:
                    return
                np.savez_compressed(path,loads=loads,load_factors=np.asarray(fs,float))
                self.vars["loads_npz"].set(path)
                self._log(f"Создан loads NPZ: {path}; steps={len(fs)}\n")
                win.destroy()

            ttk.Button(win,text="Создать NPZ",command=lambda:self._safe_dialog_call(create)).grid(
                row=3,column=0,columnspan=2,pady=10)

        except Exception as exc:
            messagebox.showerror("Нагрузка",str(exc))

    def _safe_dialog_call(self,fn):
        try: fn()
        except Exception as exc: messagebox.showerror("Ошибка",str(exc))

    def run_case(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Расчёт","Расчёт уже выполняется")
            return
        try:
            cfg=self._case_dict()
            p=self._case_path_for_save()
            if p is None:
                return
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8")
            validate_final_case(p)

            out=Path(cfg["output_dir"])
            if not out.is_absolute():
                out=p.parent/out
            self.run_output_dir=out.resolve()

            # Determine expected load-step count directly from input NPZ.
            lp=Path(cfg["loads_npz"])
            if not lp.is_absolute():
                lp=p.parent/lp
            with np.load(lp,allow_pickle=False) as z:
                self.expected_steps=int(np.asarray(z["loads"]).shape[0])

            self.run_kind="CASE"
            self._paused=False;self._paused_pids=[];self._pause_started_at=None;self._paused_total=0.0
            cmd=[sys.executable,str(APP_DIR/"RUN_FINAL_CASE.py"),str(p)]
            self._log("\n=== START ===\n"+" ".join(cmd)+"\n")
            self.run_btn.configure(state="disabled")
            self._progress_reset(self.expected_steps)

            self.proc=subprocess.Popen(
                cmd,cwd=str(APP_DIR),
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,encoding="utf-8",errors="replace"
            )
            threading.Thread(target=self._read_process_output,daemon=True).start()
        except Exception as exc:
            messagebox.showerror("Запуск — ошибка",str(exc))
            self._log(f"START FAIL: {type(exc).__name__}: {exc}\n")

    def run_mvi_series(self):
        if self.vars["method"].get()!="MVI":
            messagebox.showwarning("МВИ-1…8","Серия доступна только при method=MVI")
            return
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Расчёт","Расчёт уже выполняется")
            return
        try:
            cfg=self._case_dict()
            p=self._case_path_for_save()
            if p is None:
                return
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8")
            validate_final_case(p)

            out=Path(cfg["output_dir"])
            if not out.is_absolute():
                out=p.parent/out
            self.run_output_dir=out.resolve()/"MVI_1_8_SERIES"
            self.expected_steps=8
            self.run_kind="MVI_SERIES"

            cmd=[sys.executable,str(APP_DIR/"RUN_MVI_1_8.py"),str(p)]
            self._log("\n=== START MVI-1…8 ===\n"+" ".join(cmd)+"\n")
            self.run_btn.configure(state="disabled")
            self.series_btn.configure(state="disabled")
            self._progress_reset(8)

            self.proc=subprocess.Popen(
                cmd,cwd=str(APP_DIR),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,encoding="utf-8",errors="replace"
            )
            threading.Thread(target=self._read_process_output,daemon=True).start()
        except Exception as exc:
            messagebox.showerror("МВИ-1…8 — ошибка",str(exc))
            self._log(f"MVI SERIES START FAIL: {type(exc).__name__}: {exc}\n")

    def _read_process_output(self):
        assert self.proc is not None
        for line in self.proc.stdout:
            self.log_queue.put(line)
        rc=self.proc.wait()
        self.log_queue.put(("__DONE__",rc))

    def _poll_log_queue(self):
        try:
            while True:
                item=self.log_queue.get_nowait()
                if isinstance(item,tuple) and item and item[0]=="__DONE__":
                    self._on_process_done(item[1])
                elif isinstance(item,tuple) and item and item[0]=="__STOP_FAIL__":
                    self._log(f"STOP FAIL: {item[1]}\n")
                elif isinstance(item,tuple) and item and item[0]=="__POST_DONE__":
                    self._on_postprocess_done(item[1])
                elif isinstance(item,tuple) and item and item[0]=="__SUMMARY_DONE__":
                    self._on_quick_summary_done(item[1])
                else:
                    self._log(str(item))
        except queue.Empty:
            pass
        self.after(150,self._poll_log_queue)

    def _poll_progress_files(self):
        try:
            if self.proc is not None and self.proc.poll() is None and self.run_output_dir:
                out=self.run_output_dir
                if self.run_kind=="MVI_SERIES":
                    completed=len(list(out.glob("MVI_*_SUMMARY.json"))) if out.exists() else 0
                else:
                    completed=len(list(out.glob("STEP_*.json"))) if out.exists() else 0
                if self.expected_steps:
                    self._progress_update(completed,self.expected_steps)
        except Exception:
            pass
        self.after(500,self._poll_progress_files)

    def _on_process_done(self,rc):
        self.progress.stop()
        if rc==0 and self.expected_steps:
            self._progress_update(self.expected_steps,self.expected_steps)
        self.run_btn.configure(state="normal")
        try:
            self.series_btn.configure(state="normal" if self.vars["method"].get()=="MVI" else "disabled")
        except Exception:
            pass
        (self.progress_status_label.configure(text="Завершено" if rc==0 else f"Ошибка rc={rc}") if hasattr(self,"progress_status_label") else self.progress_label.configure(text="Завершено" if rc==0 else f"Ошибка rc={rc}"))
        self._log(f"=== DONE rc={rc} ===\n")
        if rc==0 and self.run_output_dir and self.run_kind=="CASE":
            self._log("Расчёт сохранён. Постобработка не запускается автоматически. Выберите FAST / STANDARD / FULL вручную.\n")
            self.progress_label.configure(text="Расчёт завершён · выберите постобработку")
        elif rc==0 and self.run_kind=="MVI_SERIES":
            self._log(f"Серия МВИ-1…8 сохранена: {self.run_output_dir}. Постобработка автоматически не запускается.\n")
        elif rc!=0:
            messagebox.showerror("Расчёт","Расчёт завершился с ошибкой. См. лог.")

    def _run_postprocess(self,mode="FAST",full=None):
        """Run postprocessing only on explicit user request.

        ``full`` is retained only for compatibility with older internal calls;
        the GUI uses the explicit FAST / STANDARD / FULL mode names.
        """
        try:
            if full is not None:
                mode="FULL" if bool(full) else "FAST"
            mode=str(mode).upper().strip()
            if mode not in ("FAST","STANDARD","FULL"):
                raise ValueError(f"Unknown postprocess mode: {mode}")
            if self.post_proc is not None and self.post_proc.poll() is None:
                messagebox.showinfo("Постобработка / Post-processing","Постобработка уже выполняется.\nPost-processing is already running.")
                return
            if self.run_output_dir is None:
                messagebox.showwarning("Постобработка / Post-processing","Сначала выберите/выполните расчёт.\nSelect or run a calculation first.")
                return
            arg={"FAST":"--fast","STANDARD":"--standard","FULL":"--full"}[mode]
            self._postprocess_mode_requested=mode
            cmd=[sys.executable,str(APP_DIR/"FINAL_POSTPROCESS.py"),str(self.run_output_dir),arg]
            self._log(f"=== POSTPROCESS {mode} START (manual) ===\n"+" ".join(cmd)+"\n")
            labels={
                "FAST":"FAST / БЫСТРАЯ постобработка…",
                "STANDARD":"STANDARD / СТАНДАРТНАЯ постобработка…",
                "FULL":"FULL / ПОЛНАЯ постобработка…",
            }
            self.progress_label.configure(text=labels[mode])
            self.post_proc=subprocess.Popen(
                cmd,cwd=str(APP_DIR),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,encoding="utf-8",errors="replace"
            )
            threading.Thread(target=self._read_postprocess_output,daemon=True).start()
        except Exception as exc:
            self._log(f"POSTPROCESS FAIL: {type(exc).__name__}: {exc}\n")

    def _read_postprocess_output(self):
        proc=self.post_proc
        if proc is None:return
        try:
            for line in proc.stdout:self.log_queue.put(line)
            rc=proc.wait()
        except Exception:
            rc=-999
        self.log_queue.put(("__POST_DONE__",rc))

    def _on_postprocess_done(self,rc):
        mode=str(getattr(self,"_postprocess_mode_requested","FAST")).upper()
        if rc==0:
            self._log(f"=== POSTPROCESS {mode} DONE ===\n")
            labels={
                "FAST":"FAST завершён / FAST complete",
                "STANDARD":"STANDARD завершён / STANDARD complete",
                "FULL":"FULL завершён / FULL complete",
            }
            self.progress_label.configure(text=labels.get(mode,"Постобработка завершена"))
            try:self.refresh_results_view()
            except Exception:pass
        else:
            self._log(f"=== POSTPROCESS {mode} FAIL rc={rc} ===\n")
            self.progress_label.configure(text=f"Ошибка постобработки / Post-process error rc={rc}")

    def _run_quick_summary(self):
        """Build a compact two-page RU/EN + English PDF from existing results."""
        try:
            if self.run_output_dir is None:
                messagebox.showwarning("Экспресс-отчёт / Quick Summary","Сначала выберите или выполните расчёт.\nSelect or run a calculation first.")
                return
            if self.post_proc is not None and self.post_proc.poll() is None:
                messagebox.showinfo("Экспресс-отчёт / Quick Summary","Дождитесь завершения текущей постобработки.\nWait until the current postprocessing is finished.")
                return
            if self.summary_proc is not None and self.summary_proc.poll() is None:
                messagebox.showinfo("Экспресс-отчёт / Quick Summary","Экспресс-отчёт уже формируется.\nQuick Summary is already being generated.")
                return
            cmd=[sys.executable,str(APP_DIR/"FINAL_QUICK_SUMMARY.py"),str(self.run_output_dir)]
            self._log("=== QUICK SUMMARY START ===\n"+" ".join(cmd)+"\n")
            self.progress_label.configure(text="Экспресс-отчёт / Quick Summary…")
            self.summary_proc=subprocess.Popen(
                cmd,cwd=str(APP_DIR),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,encoding="utf-8",errors="replace"
            )
            threading.Thread(target=self._read_quick_summary_output,daemon=True).start()
        except Exception as exc:
            self._log(f"QUICK SUMMARY FAIL: {type(exc).__name__}: {exc}\n")
            messagebox.showerror("Экспресс-отчёт / Quick Summary",str(exc))

    def _read_quick_summary_output(self):
        proc=self.summary_proc
        if proc is None:return
        try:
            for line in proc.stdout:self.log_queue.put(line)
            rc=proc.wait()
        except Exception:
            rc=-999
        self.log_queue.put(("__SUMMARY_DONE__",rc))

    def _on_quick_summary_done(self,rc):
        if rc!=0:
            self.progress_label.configure(text=f"Ошибка Quick Summary rc={rc}")
            self._log(f"=== QUICK SUMMARY FAIL rc={rc} ===\n")
            messagebox.showerror("Экспресс-отчёт / Quick Summary","Не удалось сформировать PDF. См. лог.\nCould not build the PDF. See log.")
            return
        self.progress_label.configure(text="Экспресс-отчёт готов / Quick Summary ready")
        self._log("=== QUICK SUMMARY DONE ===\n")
        pdf=Path(self.run_output_dir)/"REPORT"/"QUICK_SUMMARY_RU_EN.pdf"
        try:
            if pdf.exists():
                if os.name=="nt":os.startfile(str(pdf))
                elif sys.platform=="darwin":subprocess.Popen(["open",str(pdf)])
                else:subprocess.Popen(["xdg-open",str(pdf)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except Exception as exc:
            self._log(f"QUICK SUMMARY OPEN WARNING: {type(exc).__name__}: {exc}\n")

    def _run_mvi_series_postprocess(self):
        try:
            if self.post_proc is not None and self.post_proc.poll() is None:
                return
            cmd=[sys.executable,str(APP_DIR/"FINAL_MVI_SERIES_POSTPROCESS.py"),str(self.run_output_dir)]
            self._log("=== MVI SERIES POSTPROCESS START ===\n"+" ".join(cmd)+"\n")
            self.progress_label.configure(text="Постобработка МВИ…")
            self.post_proc=subprocess.Popen(
                cmd,cwd=str(APP_DIR),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,encoding="utf-8",errors="replace"
            )
            threading.Thread(target=self._read_postprocess_output,daemon=True).start()
        except Exception as exc:
            self._log(f"MVI SERIES POSTPROCESS FAIL: {type(exc).__name__}: {exc}\n")

    def _log(self,text):
        if not hasattr(self,"log"):
            return
        self.log.insert("end",text)
        self.log.see("end")

    def open_results(self):
        p=self.run_output_dir
        if p is None:
            s=self.vars["output_dir"].get().strip()
            if not s:
                return
            p=Path(s)
        try:
            p=Path(p).resolve()
            if sys.platform.startswith("win"):
                os.startfile(p)
            elif sys.platform=="darwin":
                subprocess.Popen(["open",str(p)])
            else:
                subprocess.Popen(["xdg-open",str(p)])
            # Opening the folder is also the user's request to inspect results.
            # Refresh the in-program result index immediately.
            try:self.refresh_results_view()
            except Exception:pass
        except Exception as exc:
            messagebox.showerror("Папка результатов",str(exc))


class EnhancedSolverGUI(SolverGUI):
    """Expanded Stage12 GUI. Integration/UI only; solver mechanics are unchanged."""

    UNIT_LABELS={
        "N_MM_MPA": {"length":"mm","stress":"MPa","pressure":"N/mm²","line_load":"N/mm","disp":"mm"},
        "SI_N_M_PA": {"length":"m","stress":"Pa","pressure":"Pa","line_load":"N/m","disp":"m"},
    }

    def _build_ui(self):
        self.geometry("1280x860")
        self.minsize(1080,720)
        top=ttk.LabelFrame(self,text="Расчётная модель",padding=8)
        top.pack(fill="x",padx=8,pady=(8,6))

        ttk.Label(top,text="Теория").grid(row=0,column=0,sticky="w")
        cb=ttk.Combobox(top,textvariable=self._v("theory","CUF"),values=THEORIES,state="readonly",width=8)
        cb.grid(row=0,column=1,padx=(4,10)); cb.bind("<<ComboboxSelected>>",lambda e:self._on_theory_change())

        ttk.Label(top,text="CUF-модель").grid(row=0,column=2,sticky="w")
        self.cuf_model_combo=ttk.Combobox(top,textvariable=self._v("cuf_model","TE1"),values=CUF_MODELS,state="readonly",width=7)
        self.cuf_model_combo.grid(row=0,column=3,padx=(4,10)); self.cuf_model_combo.bind("<<ComboboxSelected>>",lambda e:self._on_model_change())

        ttk.Label(top,text="Метод").grid(row=0,column=4,sticky="w")
        cb=ttk.Combobox(top,textvariable=self._v("method","FDM"),values=METHODS,state="readonly",width=8)
        cb.grid(row=0,column=5,padx=(4,10)); cb.bind("<<ComboboxSelected>>",lambda e:self._refresh_visibility())

        ttk.Label(top,text="Режим").grid(row=0,column=6,sticky="w")
        cb=ttk.Combobox(top,textvariable=self._v("mode","LINEAR_ELASTIC"),values=MODES,state="readonly",width=22)
        cb.grid(row=0,column=7,padx=(4,10)); cb.bind("<<ComboboxSelected>>",lambda e:self._refresh_visibility())

        ttk.Label(top,text="МВИ-N").grid(row=0,column=8,sticky="w")
        self.rank_combo=ttk.Combobox(top,textvariable=self._v("rank",""),values=[str(i) for i in range(1,9)],state="readonly",width=5)
        self.rank_combo.grid(row=0,column=9,padx=(4,10))

        ttk.Label(top,text="Единицы").grid(row=0,column=10,sticky="w")
        cb=ttk.Combobox(top,textvariable=self._v("unit_system","N_MM_MPA"),values=tuple(UNIT_SYSTEMS),state="readonly",width=12)
        cb.grid(row=0,column=11,padx=(4,4)); cb.bind("<<ComboboxSelected>>",lambda e:self._update_units())

        self.model_info=ttk.Label(top,text="")
        self.model_info.grid(row=1,column=0,columnspan=12,sticky="w",pady=(7,0))

        # Always-visible task parameter toolbar.  These actions save/restore
        # the values entered in the GUI, even when the physical case is not
        # complete enough to validate/run yet.
        parameter_toolbar=ttk.Frame(top)
        parameter_toolbar.grid(row=2,column=0,columnspan=12,sticky="w",pady=(8,0))
        ttk.Label(parameter_toolbar,text="Параметры задачи:").pack(side="left",padx=(0,6))
        ttk.Button(parameter_toolbar,text="Открыть параметры",command=self.open_parameters).pack(side="left",padx=(0,5))
        ttk.Button(parameter_toolbar,text="Сохранить параметры",command=self.save_parameters).pack(side="left",padx=5)
        ttk.Button(parameter_toolbar,text="Сохранить как…",command=self.save_parameters_as).pack(side="left",padx=5)
        self.parameter_toolbar=parameter_toolbar

        notebook=ttk.Notebook(self)
        notebook.pack(fill="both",expand=True,padx=8,pady=(0,8))
        self.notebook=notebook
        self.tab_problem=ttk.Frame(notebook,padding=10)
        self.tab_load=ttk.Frame(notebook,padding=10)
        self.tab_numeric=ttk.Frame(notebook,padding=10)
        self.tab_birger=ttk.Frame(notebook,padding=10)
        self.tab_j2=ttk.Frame(notebook,padding=10)
        self.tab_results=ttk.Frame(notebook,padding=10)
        self.tab_files=ttk.Frame(notebook,padding=10)
        self.tab_run=ttk.Frame(notebook,padding=10)
        for tab,title in ((self.tab_problem,"Постановка"),(self.tab_load,"Нагрузка"),(self.tab_numeric,"МВИ / Newton"),
                          (self.tab_birger,"Биргер"),(self.tab_j2,"J2 пластичность"),(self.tab_results,"Результаты"),(self.tab_files,"Файлы"),(self.tab_run,"Расчёт / лог")):
            notebook.add(tab,text=title)
        self._build_problem_tab()
        self._build_load_tab()
        self._build_numeric_tab()
        self._build_birger_tab()
        self._build_j2_tab()
        self._build_results_tab()
        self._build_files_tab()
        self._build_run_tab()
        self._refresh_material_mode()
        self._update_units()

    def _labeled_entry(self,parent,row,label,key,column=0,width=16):
        lab=ttk.Label(parent,text=label)
        lab.grid(row=row,column=column,sticky="w",pady=3)
        ent=ttk.Entry(parent,textvariable=self._v(key),width=width)
        if key in ("load_factors","layer_bounds","gb_thresholds"):
            ent._clipboard_list_mode=True
        ent.grid(row=row,column=column+1,sticky="ew",padx=6,pady=3)
        return lab,ent

    def _build_problem_tab(self):
        f=self.tab_problem; f.columnconfigure(0,weight=1); f.columnconfigure(1,weight=1); f.rowconfigure(0,weight=1)

        # Keep all inputs reachable on Windows displays with scaling enabled.
        left_host=ttk.Frame(f)
        left_host.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        left_host.columnconfigure(0,weight=1); left_host.rowconfigure(0,weight=1)
        self.problem_left_canvas=tk.Canvas(left_host,highlightthickness=0,borderwidth=0)
        self.problem_left_canvas.grid(row=0,column=0,sticky="nsew")
        self.problem_left_scroll=ttk.Scrollbar(
            left_host,orient="vertical",command=self.problem_left_canvas.yview,
            style="Green.Vertical.TScrollbar"
        )
        self.problem_left_scroll.grid(row=0,column=1,sticky="ns")
        self.problem_left_canvas.configure(yscrollcommand=self.problem_left_scroll.set)
        left=ttk.Frame(self.problem_left_canvas)
        self.problem_left_inner=left
        self._problem_left_window=self.problem_left_canvas.create_window((0,0),window=left,anchor="nw")
        left.columnconfigure(0,weight=1)
        left.bind("<Configure>",lambda _e:self.problem_left_canvas.configure(scrollregion=self.problem_left_canvas.bbox("all")))
        self.problem_left_canvas.bind(
            "<Configure>",
            lambda e:self.problem_left_canvas.itemconfigure(self._problem_left_window,width=max(1,e.width))
        )
        def _problem_mousewheel(event):
            delta=getattr(event,"delta",0)
            if delta:self.problem_left_canvas.yview_scroll(int(-delta/120),"units")
        self.problem_left_canvas.bind("<MouseWheel>",_problem_mousewheel)
        left.bind("<MouseWheel>",_problem_mousewheel)

        # The boundary-condition block and cylinder scheme can be taller than
        # a scaled Windows display.  Keep the complete right-hand side reachable
        # through its own scrollbar rather than clipping the scheme below screen.
        right_host=ttk.Frame(f)
        right_host.grid(row=0,column=1,sticky="nsew")
        right_host.columnconfigure(0,weight=1); right_host.rowconfigure(0,weight=1)
        self.problem_right_canvas=tk.Canvas(right_host,highlightthickness=0,borderwidth=0)
        self.problem_right_canvas.grid(row=0,column=0,sticky="nsew")
        self.problem_right_scroll=ttk.Scrollbar(
            right_host,orient="vertical",command=self.problem_right_canvas.yview,
            style="Green.Vertical.TScrollbar"
        )
        self.problem_right_scroll.grid(row=0,column=1,sticky="ns")
        self.problem_right_canvas.configure(yscrollcommand=self.problem_right_scroll.set)
        right=ttk.Frame(self.problem_right_canvas)
        self.problem_right_inner=right
        self._problem_right_window=self.problem_right_canvas.create_window(
            (0,0),window=right,anchor="nw"
        )
        right.bind(
            "<Configure>",
            lambda _e:self.problem_right_canvas.configure(
                scrollregion=self.problem_right_canvas.bbox("all")
            )
        )
        self.problem_right_canvas.bind(
            "<Configure>",
            lambda e:self.problem_right_canvas.itemconfigure(
                self._problem_right_window,width=max(1,e.width)
            )
        )

        geom=ttk.LabelFrame(left,text="Геометрия замкнутого цилиндра",padding=8); geom.pack(fill="x",pady=(0,8))
        self.unit_labels={}
        for r,(txt,key) in enumerate((("Длина L","L"),("Радиус R","R"),("Толщина h","h"))):
            lab,ent=self._labeled_entry(geom,r,txt,key); self.unit_labels[key]=lab
            ent.bind("<KeyRelease>",lambda e:self._update_grid_info())

        # Scale is physical, not a numerical method; MICRO cannot use MACRO solver.
        scale_box=ttk.LabelFrame(left,text="Масштабная теория",padding=8)
        scale_box.pack(fill="x",pady=(0,8))
        self.scale_combo=ttk.Combobox(
            scale_box,textvariable=self._v("scale_model","MACRO"),
            values=flexoscale.SCALES,state="readonly",width=18
        )
        self.scale_combo.pack(anchor="w")
        self.scale_combo.bind("<<ComboboxSelected>>",lambda _e:self._refresh_visibility())
        self.scale_note=ttk.Label(scale_box,wraplength=470,justify="left",
             text="MACRO — классическая упругость; MICRO — полная ковариантная G3D-флексоэлектрическая теория (реализация связанных PDE готовится).")
        self.scale_note.pack(fill="x",pady=(5,0))
        micro=ttk.LabelFrame(left,text="MICRO · G3D, параметры СИ",padding=8)
        micro.pack(fill="x",pady=(0,8))
        for j,(caption,key) in enumerate((
            ("Градиентная длина ell, м","micro_ell_m"),
            ("Флексокоэффициент mu1, Кл/м","micro_mu1"),
            ("Флексокоэффициент mu2, Кл/м","micro_mu2"),
            ("Проницаемость kappa, Ф/м","micro_kappa"),
            ("Плотность rho, кг/м3","micro_rho"),
            ("Вязкий коэффициент eta, с","micro_eta"),
            ("Порядок потенциала P","micro_potential_order"),
        )): self._labeled_entry(micro,j,caption,key)
        ttk.Label(micro,text="Без автоматических физических значений. Gz не подменяет G3D. Микроветка не запускается до реализации связанной системы.",
            wraplength=460).grid(row=7,column=0,columnspan=2,sticky="w")
        mat=ttk.LabelFrame(left,text="Материал",padding=8); mat.pack(fill="x",pady=(0,8))
        ttk.Label(mat,text="Источник").grid(row=0,column=0,sticky="w",pady=3)
        self.material_source_combo=ttk.Combobox(mat,textvariable=self._v("material_source","USER"),values=("USER","REFERENCE"),state="readonly",width=16)
        self.material_source_combo.grid(row=0,column=1,sticky="w",padx=6,pady=3)
        self.material_source_combo.bind("<<ComboboxSelected>>",lambda e:self._refresh_material_mode())
        ttk.Label(mat,text="Справочный материал").grid(row=1,column=0,sticky="w",pady=3)
        self.material_name_combo=ttk.Combobox(mat,textvariable=self._v("material_name",""),values=(),state="disabled",width=34)
        self.material_name_combo.grid(row=1,column=1,sticky="ew",padx=6,pady=3)
        self.material_name_combo.bind("<<ComboboxSelected>>",lambda e:self._apply_material_preset())
        lab,self.material_E0_entry=self._labeled_entry(mat,2,"Модуль E0","E0"); self.unit_labels["E0"]=lab
        _,self.material_nu0_entry=self._labeled_entry(mat,3,"Коэффициент Пуассона ν0","nu0")
        self.material_note=ttk.Label(mat,text="Пользовательский материал: введите E0 и ν0 вручную.",wraplength=470)
        self.material_note.grid(row=4,column=0,columnspan=2,sticky="w",pady=(6,0))
        ttk.Label(mat,text="Справочные значения — только удобные generic isotropic пресеты; для физического расчёта проверяйте конкретную марку материала.",wraplength=470).grid(row=5,column=0,columnspan=2,sticky="w",pady=(4,0))

        grid=ttk.LabelFrame(left,text="Глобальная сетка и квадратура",padding=8); grid.pack(fill="x")
        self._labeled_entry(grid,0,"nx","nx")[1].bind("<KeyRelease>",lambda e:self._update_grid_info())
        self._labeled_entry(grid,1,"ns","ns")[1].bind("<KeyRelease>",lambda e:self._update_grid_info())
        self._labeled_entry(grid,2,"FD stencil","stencil")
        ttk.Label(grid,text="Квадратура по толщине").grid(row=3,column=0,sticky="w",pady=3)
        self.quadrature_type_combo=ttk.Combobox(
            grid,textvariable=self._v("quadrature_type","GAUSS"),
            values=("GAUSS","LAYERED"),state="readonly",width=14
        )
        self.quadrature_type_combo.grid(row=3,column=1,sticky="w",padx=6,pady=3)
        self._labeled_entry(grid,4,"nz — точек Гаусса по толщине","n_gauss")
        self._labeled_entry(grid,5,"Границы слоёв z","layer_bounds")
        ttk.Label(grid,text="Для LAYERED: z-границы через запятую от -h/2 до +h/2; n_gauss применяется на каждый слой.",wraplength=455).grid(row=6,column=0,columnspan=2,sticky="w",pady=(2,5))
        ttk.Label(grid,text="Быстрый ввод сетки").grid(row=7,column=0,sticky="w",pady=3)
        preset=ttk.Combobox(grid,textvariable=self._v("grid_preset",""),values=("7×8","9×12","11×16","15×24","21×32","31×48"),state="readonly",width=14)
        preset.grid(row=7,column=1,sticky="w",padx=6); preset.bind("<<ComboboxSelected>>",lambda e:self._apply_grid_preset())
        self.grid_info=ttk.Label(grid,text="Δx = —; Δs = —; число неизвестных = —",wraplength=470)
        self.grid_info.grid(row=8,column=0,columnspan=2,sticky="w",pady=(7,0))

        bc=ttk.LabelFrame(right,text="Краевые условия на торцах x=0 и x=L",padding=8); bc.pack(fill="x",pady=(0,8))
        ttk.Label(bc,text="Режим КУ").grid(row=0,column=0,sticky="w")
        bcm=ttk.Combobox(bc,textvariable=self._v("bc_mode","SIMPLE"),values=("SIMPLE","INDIVIDUAL"),state="readonly",width=24)
        bcm.grid(row=0,column=1,padx=6,sticky="w"); bcm.bind("<<ComboboxSelected>>",lambda e:self._refresh_visibility())
        ttk.Label(bc,text="Левый торец").grid(row=1,column=0,sticky="w")
        self.bc_left_combo=ttk.Combobox(bc,textvariable=self._v("bc_left",""),values=("","CLAMPED","FREE"),state="readonly",width=52)
        self.bc_left_combo.grid(row=1,column=1,padx=6,sticky="ew")
        ttk.Label(bc,text="Правый торец").grid(row=2,column=0,sticky="w")
        self.bc_right_combo=ttk.Combobox(bc,textvariable=self._v("bc_right",""),values=("","CLAMPED","FREE"),state="readonly",width=52)
        self.bc_right_combo.grid(row=2,column=1,padx=6,sticky="ew")
        bc.columnconfigure(1,weight=1)
        buttons=ttk.Frame(bc); buttons.grid(row=3,column=0,columnspan=2,sticky="w",pady=(8,4))
        self.bc_edit_btn=ttk.Button(buttons,text="Ручные / Custom КУ…",command=self._edit_individual_bc)
        self.bc_edit_btn.pack(side="left",padx=(0,6))
        self.bc_engineering_btn=ttk.Button(buttons,text="Инженерные жёсткости k…",command=self._edit_engineering_bc)
        self.bc_engineering_btn.pack(side="left")
        self.bc_summary=ttk.Label(bc,text="PRESET: требуется явный выбор",justify="left",wraplength=520)
        self.bc_summary.grid(row=4,column=0,columnspan=2,sticky="w")
        self.bc_active_equations=ttk.Label(bc,text="Активные уравнения ГУ: —",justify="left",wraplength=520,foreground="#17324d")
        self.bc_active_equations.grid(row=5,column=0,columnspan=2,sticky="w",pady=(5,0))
        ttk.Label(bc,text="Preset выбирает физический тип опоры, а решатель автоматически разворачивает его в формулы текущей модели KL/TE1–TE4. Custom позволяет задавать каждую DOF вручную. FREE–FREE без обработки жёстких мод отклоняется.",wraplength=520).grid(row=6,column=0,columnspan=2,sticky="w",pady=(8,0))

        scheme=ttk.LabelFrame(right,text="Схема задачи",padding=8); scheme.pack(fill="both",expand=True)
        self.scheme_canvas=tk.Canvas(scheme,height=255,background="white",highlightthickness=1,highlightbackground="#aaa")
        self.scheme_canvas.pack(fill="both",expand=True)
        ttk.Button(scheme,text="Обновить схему",command=self._draw_problem_scheme).pack(anchor="w",pady=(6,0))

    def _build_load_tab(self):
        f=self.tab_load
        f.columnconfigure(0,weight=1); f.columnconfigure(1,weight=1); f.rowconfigure(0,weight=1)

        # Left side is intentionally scrollable. On common Windows display scaling
        # the load controls are taller than the notebook page; without a scrollbar
        # the load-step fields (lambda start / dlambda / lambda end) were clipped.
        left_host=ttk.Frame(f)
        left_host.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        left_host.columnconfigure(0,weight=1); left_host.rowconfigure(0,weight=1)
        self.load_left_canvas=tk.Canvas(left_host,highlightthickness=0,borderwidth=0)
        self.load_left_canvas.grid(row=0,column=0,sticky="nsew")
        self.load_left_scroll=ttk.Scrollbar(left_host,orient="vertical",command=self.load_left_canvas.yview)
        self.load_left_scroll.grid(row=0,column=1,sticky="ns")
        self.load_left_canvas.configure(yscrollcommand=self.load_left_scroll.set)
        left=ttk.Frame(self.load_left_canvas)
        self.load_left_inner=left
        self._load_left_window=self.load_left_canvas.create_window((0,0),window=left,anchor="nw")
        left.columnconfigure(0,weight=1)
        def _load_scrollregion(_event=None):
            self.load_left_canvas.configure(scrollregion=self.load_left_canvas.bbox("all"))
        def _load_canvas_width(event):
            self.load_left_canvas.itemconfigure(self._load_left_window,width=max(1,event.width))
        left.bind("<Configure>",_load_scrollregion)
        self.load_left_canvas.bind("<Configure>",_load_canvas_width)
        # Mouse wheel works when pointer is over the load controls, while the
        # explicit scrollbar remains available on every platform.
        def _load_mousewheel(event):
            delta=getattr(event,"delta",0)
            if delta:
                self.load_left_canvas.yview_scroll(int(-delta/120),"units")
        self.load_left_canvas.bind("<MouseWheel>",_load_mousewheel)
        left.bind("<MouseWheel>",_load_mousewheel)

        lf=ttk.LabelFrame(left,text="Тип и величина нагрузки",padding=8); lf.pack(fill="x",pady=(0,8))
        self.load_type_frame=lf
        ttk.Label(lf,text="Тип").grid(row=0,column=0,sticky="w")
        cb=ttk.Combobox(lf,textvariable=self._v("load_type","UNIFORM_RADIAL_FACE"),values=loadbuilder.LOAD_TYPES,state="readonly",width=28)
        cb.grid(row=0,column=1,sticky="w",padx=6); cb.bind("<<ComboboxSelected>>",lambda e:self._refresh_load_visibility())
        ttk.Label(lf,text="Встроенные radial face traction: UNIFORM, HARMONIC, PATCH, LINE. Произвольная нагрузка — EXTERNAL_NPZ.",wraplength=510).grid(row=1,column=0,columnspan=3,sticky="w",pady=(5,3))
        ttk.Label(lf,text="V8: кинематика нагрузки = DEAD_REFERENCE (номинальная/мёртвая): направление и величина фиксированы в отсчётной конфигурации; follower load не реализована.",wraplength=510).grid(row=2,column=0,columnspan=3,sticky="w",pady=(0,8))
        self.load_q_label,self.load_q_entry=self._labeled_entry(lf,3,"q (величина, >= 0)","load_q")
        self.load_pressure_direction_label=ttk.Label(lf,text="Направление давления")
        self.load_pressure_direction_label.grid(row=4,column=0,sticky="w",pady=3)
        self.load_pressure_direction_combo=ttk.Combobox(
            lf,textvariable=self._v("load_pressure_direction","INTERNAL"),
            values=("INTERNAL","EXTERNAL"),state="readonly",width=16)
        self.load_pressure_direction_combo.grid(row=4,column=1,sticky="w",padx=6,pady=3)
        self.load_pressure_direction_combo.bind("<<ComboboxSelected>>",lambda e:self._schedule_problem_scheme_redraw(20))
        ttk.Label(lf,text="INTERNAL: радиально наружу; EXTERNAL: радиально внутрь. В solver знак подставляется автоматически.",wraplength=510).grid(row=5,column=0,columnspan=3,sticky="w",pady=(2,5))
        self.load_line_label,self.load_line_entry=self._labeled_entry(lf,6,"p_line (величина, >= 0)","load_p_line")
        self.load_point_label,self.load_point_entry=self._labeled_entry(lf,7,"P точечной силы [N]","load_point_P")
        self.load_point_x_label,self.load_point_x_entry=self._labeled_entry(lf,8,"x0 точки [длина]","load_point_x0")
        self.load_point_t_label,self.load_point_t_entry=self._labeled_entry(lf,9,"theta0 точки [rad]","load_point_theta0")
        self.load_face_label=ttk.Label(lf,text="Поверхность нагрузки")
        self.load_face_label.grid(row=10,column=0,sticky="w",pady=3)
        self.load_face_combo=ttk.Combobox(lf,textvariable=self._v("load_face","OUTER"),values=loadbuilder.FACES,state="readonly",width=12)
        self.load_face_combo.grid(row=10,column=1,sticky="w",padx=6,pady=3)

        self.harmonic_frame=ttk.LabelFrame(left,text="Гармоническая радиальная нагрузка",padding=8); self.harmonic_frame.pack(fill="x",pady=(0,8))
        self._v("load_harmonic_m","1"); self._v("load_harmonic_n","4")
        self._labeled_entry(self.harmonic_frame,0,"m (вдоль x)","load_harmonic_m")
        self._labeled_entry(self.harmonic_frame,1,"n (по окружности)","load_harmonic_n")
        ttk.Label(self.harmonic_frame,text="q(x,s)=q0·cos(n·s/R)·sin(m·π·x/L). Для Nwoji Case 2: m=1, n=4.",wraplength=500).grid(row=2,column=0,columnspan=2,sticky="w",pady=(4,0))

        self.patch_frame=ttk.LabelFrame(left,text="Область действия нагрузки",padding=8); self.patch_frame.pack(fill="x",pady=(0,8))
        self._labeled_entry(self.patch_frame,0,"x1 (начало по длине)","load_x1"); self._labeled_entry(self.patch_frame,1,"x2 (конец по длине)","load_x2")
        ttk.Label(self.patch_frame,text="Окружная координата").grid(row=2,column=0,sticky="w",pady=3)
        self.load_coord_combo=ttk.Combobox(self.patch_frame,textvariable=self._v("load_coord","THETA"),values=loadbuilder.COORD_MODES,state="readonly",width=12)
        self.load_coord_combo.grid(row=2,column=1,sticky="w",padx=6,pady=3); self.load_coord_combo.bind("<<ComboboxSelected>>",lambda e:self._update_load_units())
        self.load_c1_label,self.load_c1_entry=self._labeled_entry(self.patch_frame,3,"θ1 / s1 (PATCH/HARMONIC)","load_c1")
        self.load_c2_label,self.load_c2_entry=self._labeled_entry(self.patch_frame,4,"θ2 / s2 (PATCH/HARMONIC)","load_c2")
        self.load_c0_label,self.load_c0_entry=self._labeled_entry(self.patch_frame,5,"θ0 / s0 (центр LINE)","load_c0")
        self.load_band_label,self.load_band_entry=self._labeled_entry(self.patch_frame,6,"Физическая ширина полосы b (LINE)","load_band_width")
        self.load_full_var=tk.BooleanVar(value=False)
        self.load_full_check=ttk.Checkbutton(self.patch_frame,text="Вся окружность (PATCH/HARMONIC)",variable=self.load_full_var,command=lambda:self._schedule_problem_scheme_redraw(20)); self.load_full_check.grid(row=7,column=0,columnspan=2,sticky="w",pady=4)
        ttk.Label(self.patch_frame,text="PATCH/HARMONIC: x1–x2 и θ1–θ2 (или s1–s2) задают область. LINE: x1–x2, центр θ0/s0 и физическая ширина b. Фронты ограниченной области сглаживаются автоматически по КР-шаблону.",wraplength=500).grid(row=8,column=0,columnspan=2,sticky="w",pady=(4,0))

        steps=ttk.LabelFrame(left,text="Шаги нагружения",padding=8); steps.pack(fill="x",pady=(0,8))
        self.load_steps_frame=steps
        ttk.Label(steps,text="Задание").grid(row=0,column=0,sticky="w",pady=3)
        self.load_step_mode_combo=ttk.Combobox(steps,textvariable=self._v("load_step_mode","COUNT"),values=("RANGE","COUNT","MANUAL_LIST"),state="readonly",width=16)
        self.load_step_mode_combo.grid(row=0,column=1,sticky="w",padx=6,pady=3)
        self.load_step_mode_combo.bind("<<ComboboxSelected>>",lambda e:self._refresh_load_step_mode())
        self.load_range_frame=ttk.Frame(steps); self.load_range_frame.grid(row=1,column=0,columnspan=2,sticky="ew")
        self._labeled_entry(self.load_range_frame,0,"λ start","load_lambda_start")
        self._labeled_entry(self.load_range_frame,1,"Δλ","load_lambda_step")
        self._labeled_entry(self.load_range_frame,2,"λ end","load_lambda_end")
        self.load_count_frame=ttk.Frame(steps); self.load_count_frame.grid(row=2,column=0,columnspan=2,sticky="ew")
        self._labeled_entry(self.load_count_frame,0,"λ end","load_lambda_end")
        self._labeled_entry(self.load_count_frame,1,"Число шагов","load_step_count")
        self.load_manual_frame=ttk.Frame(steps); self.load_manual_frame.grid(row=3,column=0,columnspan=2,sticky="ew")
        ttk.Label(self.load_manual_frame,text="λ через запятую").grid(row=0,column=0,sticky="w",pady=3)
        self.load_factors_entry=ttk.Entry(self.load_manual_frame,textvariable=self._v("load_factors","1"),width=28)
        self.load_factors_entry._clipboard_list_mode=True
        self.load_factors_entry.grid(row=0,column=1,sticky="ew",padx=6,pady=3)
        self.load_steps_info=ttk.Label(steps,text="",wraplength=500); self.load_steps_info.grid(row=4,column=0,columnspan=2,sticky="w",pady=(5,0))
        for key in ("load_lambda_start","load_lambda_step","load_lambda_end","load_step_count","load_factors"):
            self._v(key).trace_add("write",lambda *_:self._update_load_steps_info())

        self.load_file_frame=ttk.LabelFrame(left,text="Внешняя нагрузка NPZ",padding=8); self.load_file_frame.pack(fill="x",pady=(0,8))
        self.load_file_frame.columnconfigure(1,weight=1)
        ttk.Label(self.load_file_frame,text="Loads NPZ").grid(row=0,column=0,sticky="w")
        ttk.Entry(self.load_file_frame,textvariable=self._v("loads_npz"),width=48).grid(row=0,column=1,sticky="ew",padx=6)
        ttk.Button(self.load_file_frame,text="…",command=lambda:self._choose_file("loads_npz")).grid(row=0,column=2)

        prev=ttk.LabelFrame(f,text="Предпросмотр нагрузки",padding=8); prev.grid(row=0,column=1,sticky="nsew")
        self.load_preview_canvas=tk.Canvas(prev,height=510,background="white",highlightthickness=1,highlightbackground="#aaa")
        self.load_preview_canvas.pack(fill="both",expand=True)
        btns=ttk.Frame(prev); btns.pack(fill="x",pady=(7,0))
        ttk.Button(btns,text="Показать на цилиндре",command=self._draw_problem_scheme).pack(side="left")
        ttk.Button(btns,text="Карта q(x,θ)",command=self._preview_load_map).pack(side="left",padx=6)
        self.load_meta_label=ttk.Label(prev,text="Нагрузка ещё не проверена.",wraplength=520)
        self.load_meta_label.pack(anchor="w",pady=(7,0))
        self._refresh_load_visibility(); self._refresh_load_step_mode()
        if not getattr(self,"_load_preview_traces_installed",False):
            for _key in ("load_x1","load_x2","load_c1","load_c2","load_c0","load_band_width",
                         "load_harmonic_m","load_harmonic_n","load_q","load_p_line",
                         "load_pressure_direction","load_coord"):
                self._v(_key).trace_add("write",lambda *_args:self._schedule_problem_scheme_redraw())
            self._load_preview_traces_installed=True

    def _build_numeric_tab(self):
        f=self.tab_numeric; f.columnconfigure(0,weight=1); f.columnconfigure(1,weight=1)
        self.mvi_frame=ttk.LabelFrame(f,text="МВИ",padding=8); self.mvi_frame.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        for r,(lab,key) in enumerate((("Projected tolerance (relative)","mvi_proj"),("Increment tolerance","mvi_inc"),("Tangent FD step","mvi_fd"),("Max halfstep iterations","mvi_half"),("Max X↔Y cycles","mvi_cycles"))):
            self._labeled_entry(self.mvi_frame,r,lab,key)
        ttk.Label(self.mvi_frame,text="Предиктор шага").grid(row=5,column=0,sticky="w")
        ttk.Combobox(
            self.mvi_frame,textvariable=self._v("step_predictor","ERROR"),
            values=("ERROR","LINEAR"),state="readonly",width=12
        ).grid(row=5,column=1,sticky="w",padx=6)
        ttk.Label(self.mvi_frame,text="ERROR: старт с последнего принятого шага. LINEAR: экстраполяция по двум последним принятым шагам.",wraplength=500).grid(row=6,column=0,columnspan=2,sticky="w",pady=(8,0))
        self.newton_frame=ttk.LabelFrame(f,text="Newton для nonlinear FDM",padding=8); self.newton_frame.grid(row=0,column=1,sticky="nsew")
        if not self._v("newton_res","1e-9").get().strip(): self._v("newton_res").set("1e-9")
        for r,(lab,key) in enumerate((("Relative residual tolerance (V2)","newton_res"),("Increment tolerance","newton_inc"),("Relative FD Jacobian step (V3)","newton_fd"),("Max iterations","newton_max"))):
            self._labeled_entry(self.newton_frame,r,lab,key)
        ttk.Label(self.newton_frame,text="Без line-search, релаксации, автоматического дробления шага и fallback-решателей.",wraplength=500).grid(row=4,column=0,columnspan=2,sticky="w",pady=(8,0))

        self.linear_frame=ttk.LabelFrame(f,text="Линейный FDM — проверка К-2",padding=8)
        self.linear_frame.grid(row=1,column=1,sticky="new",pady=(10,0))
        if not self._v("linear_res","1e-10").get().strip(): self._v("linear_res").set("1e-10")
        self._labeled_entry(self.linear_frame,0,"Relative residual tolerance","linear_res")
        ttk.Label(
            self.linear_frame,
            text="После прямого solve невязка проверяется обязательно; рекомендуемо 1e-10.",
            wraplength=500,
        ).grid(row=1,column=0,columnspan=2,sticky="w",pady=(8,0))

    def _build_birger_tab(self):
        f=self.tab_birger; f.columnconfigure(0,weight=1); f.columnconfigure(1,weight=1)
        self.birger_frame=ttk.LabelFrame(f,text="Внешний цикл Биргера",padding=8); self.birger_frame.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        self._v("initial_slope_rtol","0.05")
        for r,(lab,key) in enumerate((("ε_i tolerance","epsilon_tol"),("ω","omega"),("G_ref","G_ref"),("σ_ref","sigma_ref"),("tol_G","tol_G"),("tol_σ","tol_sigma"),("tol_u (изменение поля, В-6)","tol_u"),("Max outer iterations","max_outer"),("ε_i limit","eps_limit"),("|J−1| limit","dJ_limit"),("K5 допуск начального наклона","initial_slope_rtol"))):
            self._labeled_entry(self.birger_frame,r,lab,key)

        curve=ttk.LabelFrame(f,text="Материальная кривая σ_i = Φ(ε_i) — не обычная σ–ε",padding=8); curve.grid(row=0,column=1,sticky="nsew")
        ttk.Label(curve,text="Способ задания").grid(row=0,column=0,sticky="w")
        cb=ttk.Combobox(curve,textvariable=self._v("curve_mode","TABLE"),values=birgercurve.CURVE_MODES,state="readonly",width=20)
        cb.grid(row=0,column=1,sticky="w",padx=6); cb.bind("<<ComboboxSelected>>",lambda e:self._refresh_curve_visibility())
        ttk.Label(curve,text="Экстраполяция").grid(row=1,column=0,sticky="w")
        self.curve_ext_combo=ttk.Combobox(curve,textvariable=self._v("curve_extrap","ERROR"),values=birgercurve.EXTRAPOLATIONS,state="readonly",width=12)
        self.curve_ext_combo.grid(row=1,column=1,sticky="w",padx=6)

        self.curve_table_frame=ttk.LabelFrame(curve,text="Точки ε_i, σ_i; кривая должна проходить через (0, 0)",padding=5); self.curve_table_frame.grid(row=2,column=0,columnspan=3,sticky="nsew",pady=5)
        self.curve_points=tk.Text(self.curve_table_frame,height=8,width=48); self.curve_points.pack(fill="both",expand=True)

        self.curve_csv_frame=ttk.Frame(curve); self.curve_csv_frame.grid(row=3,column=0,columnspan=3,sticky="ew",pady=3); self.curve_csv_frame.columnconfigure(1,weight=1)
        ttk.Label(self.curve_csv_frame,text="CSV/TXT").grid(row=0,column=0,sticky="w"); ttk.Entry(self.curve_csv_frame,textvariable=self._v("curve_csv"),width=42).grid(row=0,column=1,sticky="ew",padx=6); ttk.Button(self.curve_csv_frame,text="…",command=lambda:self._choose_file("curve_csv")).grid(row=0,column=2)

        self.curve_formula_frame=ttk.Frame(curve); self.curve_formula_frame.grid(row=4,column=0,columnspan=3,sticky="ew",pady=3); self.curve_formula_frame.columnconfigure(1,weight=1)
        ttk.Label(self.curve_formula_frame,text="Формула Φ(eps)").grid(row=0,column=0,sticky="w"); ttk.Entry(self.curve_formula_frame,textvariable=self._v("curve_formula"),width=42).grid(row=0,column=1,sticky="ew",padx=6)

        self.curve_piece_frame=ttk.LabelFrame(curve,text="Кусочная функция: epsilon_max | formula; последняя граница None",padding=5); self.curve_piece_frame.grid(row=5,column=0,columnspan=3,sticky="nsew",pady=5)
        self.curve_piecewise=tk.Text(self.curve_piece_frame,height=6,width=48); self.curve_piecewise.pack(fill="both",expand=True)

        self.curve_external_frame=ttk.LabelFrame(curve,text="Внешний Python callable",padding=5); self.curve_external_frame.grid(row=6,column=0,columnspan=3,sticky="ew",pady=4)
        self._labeled_entry(self.curve_external_frame,0,"Module","phi_module")
        self._labeled_entry(self.curve_external_frame,1,"Callable","phi_callable")
        ttk.Label(self.curve_external_frame,text="Python path").grid(row=2,column=0,sticky="w"); ttk.Entry(self.curve_external_frame,textvariable=self._v("phi_path"),width=38).grid(row=2,column=1,sticky="ew",padx=6); ttk.Button(self.curve_external_frame,text="…",command=lambda:self._choose_dir("phi_path")).grid(row=2,column=2)

        crit=ttk.LabelFrame(curve,text="Критерий для визуализации пластической зоны (необязательно)",padding=5); crit.grid(row=7,column=0,columnspan=3,sticky="ew",pady=4)
        self._labeled_entry(crit,0,"σ_T","yield_stress"); self._labeled_entry(crit,1,"ε_T","yield_strain")
        self._labeled_entry(curve,8,"ε max для предпросмотра","curve_preview_max")
        ttk.Button(curve,text="Проверить и построить кривую",command=self._preview_birger_curve).grid(row=9,column=0,columnspan=2,sticky="w",pady=(8,0))
        self.curve_status=ttk.Label(curve,text="",wraplength=520); self.curve_status.grid(row=10,column=0,columnspan=3,sticky="w",pady=(5,0))
        self._refresh_curve_visibility()


    def _build_j2_tab(self):
        """Input only for the independent incremental J2 material model."""
        f=self.tab_j2
        f.columnconfigure(0,weight=1); f.columnconfigure(1,weight=1)
        law=ttk.LabelFrame(f,text="J2 / von Mises: ассоциативная пластичность",padding=8)
        law.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        ttk.Label(law,text="Упрочнение").grid(row=0,column=0,sticky="w",pady=3)
        ttk.Combobox(
            law,textvariable=self._v("j2_hardening_kind","LINEAR"),
            values=("PERFECT","LINEAR","TABLE"),state="readonly",width=16
        ).grid(row=0,column=1,sticky="w",padx=6,pady=3)
        self._labeled_entry(law,1,"σy0","j2_sigma_y0")
        self._labeled_entry(law,2,"H (изотропное упрочнение)","j2_H")
        self._labeled_entry(law,3,"Допуск поверхности текучести","j2_yield_tolerance")
        self._labeled_entry(law,4,"Локальный допуск return-mapping","j2_local_tolerance")
        self._labeled_entry(law,5,"Макс. локальных итераций","j2_max_local_iterations")
        ttk.Label(
            law,
            text="PERFECT: H не используется. LINEAR: σy=σy0+Hκ. "
                 "TABLE: используется строго монотонная таблица справа.",
            wraplength=500
        ).grid(row=6,column=0,columnspan=2,sticky="w",pady=(8,0))

        table=ttk.LabelFrame(f,text="Табличное изотропное упрочнение",padding=8)
        table.grid(row=0,column=1,sticky="nsew")
        ttk.Label(
            table,text="Строки: κ, σy. Нужны минимум две точки; κ возрастает, σy не убывает.",
            wraplength=500
        ).pack(anchor="w")
        self.j2_hardening_table=tk.Text(table,height=14,width=48)
        self.j2_hardening_table.pack(fill="both",expand=True,pady=(6,0))

        hist=ttk.LabelFrame(f,text="Начальная принятая история J2",padding=8)
        hist.grid(row=1,column=0,columnspan=2,sticky="ew",pady=(8,0))
        ttk.Label(hist,text="Источник").grid(row=0,column=0,sticky="w")
        ttk.Combobox(
            hist,textvariable=self._v("j2_history_source","ELASTIC_ZERO"),
            values=("ELASTIC_ZERO","NPZ"),state="readonly",width=16
        ).grid(row=0,column=1,sticky="w",padx=6)
        self._file_row(hist,1,"NPZ истории J2","j2_history_npz")
        ttk.Label(
            hist,text="NPZ должен быть записан этим решателем при тех же точках и весах квадратуры. "
                 "Разгрузка и повторное нагружение используют только принятую историю.",
            wraplength=900
        ).grid(row=2,column=0,columnspan=3,sticky="w",pady=(6,0))

    def _build_results_tab(self):
        f=self.tab_results; f.columnconfigure(0,weight=0); f.columnconfigure(1,weight=1); f.rowconfigure(0,weight=1)
        opts=ttk.LabelFrame(f,text="Автоматическая постобработка",padding=8); opts.grid(row=0,column=0,sticky="nsw",padx=(0,8))
        self._labeled_entry(opts,0,"Масштаб деформации 3D","deformation_scale")
        self._labeled_entry(opts,1,"Пороги GB/G0 через запятую","gb_thresholds")
        ttk.Label(opts,text="Контрольные точки: label, x, theta_deg").grid(row=2,column=0,columnspan=2,sticky="w",pady=(7,2))
        self.control_points_text=tk.Text(opts,height=8,width=36); self.control_points_text.grid(row=3,column=0,columnspan=2,sticky="nsew")
        ttk.Label(opts,text="После расчёта сохраняются NUMERIC и принятые STEP. GRAPHS/*, 3D/*, REPORT, Excel и ZIP создаются после ручного запуска FAST / STANDARD / FULL.",wraplength=350).grid(row=4,column=0,columnspan=2,sticky="w",pady=(8,0))

        viewer=ttk.LabelFrame(f,text="Просмотр результатов внутри программы",padding=8); viewer.grid(row=0,column=1,sticky="nsew"); viewer.columnconfigure(0,weight=1); viewer.rowconfigure(3,weight=1)
        self.result_viewer=viewer
        bar=ttk.Frame(viewer); bar.grid(row=0,column=0,sticky="ew"); self.result_bar=bar
        ttk.Button(bar,text="Обновить",command=self.refresh_results_view).pack(side="left")
        ttk.Button(bar,text="Открыть папку",command=self.open_results).pack(side="left",padx=6)
        self.result_summary=ttk.Label(viewer,text="Результатов пока нет.",wraplength=650,justify="left"); self.result_summary.grid(row=1,column=0,sticky="ew",pady=6)
        pan=ttk.Panedwindow(viewer,orient="horizontal"); pan.grid(row=3,column=0,sticky="nsew"); self.result_pan=pan
        lframe=ttk.Frame(pan); rframe=ttk.Frame(pan); pan.add(lframe,weight=1); pan.add(rframe,weight=3)
        self.result_list=tk.Listbox(lframe,width=42); self.result_list.pack(fill="both",expand=True); self.result_list.bind("<<ListboxSelect>>",lambda e:self._preview_selected_result())
        self.result_preview_frame=rframe
        self.result_preview_label=ttk.Label(rframe,text="Выберите PNG-график слева.",anchor="center"); self.result_preview_label.pack(fill="both",expand=True)
        self._result_figure_canvas=None
        self._result_png_paths=[]

    def _build_files_tab(self):
        f=self.tab_files; f.columnconfigure(1,weight=1)
        self.mvi_profile_row=ttk.LabelFrame(f,text="Начальные профили МВИ",padding=6); self.mvi_profile_row.grid(row=0,column=0,columnspan=3,sticky="ew",pady=4); self.mvi_profile_row.columnconfigure(1,weight=1)
        self._file_row(self.mvi_profile_row,0,"Initial X/Y NPZ","profiles_npz")
        ttk.Label(self.mvi_profile_row,text="CUF: X(3,m,rank,nx), Y(3,m,rank,ns), где m=2/3/4/5 для TE1/2/3/4.",wraplength=820).grid(row=1,column=0,columnspan=3,sticky="w")

        self.initial_state_frame=ttk.LabelFrame(f,text="Начальное поле nonlinear FDM",padding=6); self.initial_state_frame.grid(row=1,column=0,columnspan=3,sticky="ew",pady=4); self.initial_state_frame.columnconfigure(1,weight=1)
        ttk.Label(self.initial_state_frame,text="Source").grid(row=0,column=0,sticky="w"); ttk.Combobox(self.initial_state_frame,textvariable=self._v("state_source","ZERO"),values=("ZERO","NPZ"),state="readonly",width=12).grid(row=0,column=1,sticky="w",padx=6); self._file_row(self.initial_state_frame,1,"State NPZ","state_npz")

        self.gb_frame=ttk.LabelFrame(f,text="Начальное GB для Birger",padding=6); self.gb_frame.grid(row=2,column=0,columnspan=3,sticky="ew",pady=4); self.gb_frame.columnconfigure(1,weight=1)
        ttk.Label(self.gb_frame,text="Source").grid(row=0,column=0,sticky="w"); ttk.Combobox(self.gb_frame,textvariable=self._v("gb_source","ELASTIC_G0"),values=("ELASTIC_G0","NPZ"),state="readonly",width=14).grid(row=0,column=1,sticky="w",padx=6); self._file_row(self.gb_frame,1,"GB NPZ","gb_npz")
        self._file_row(f,3,"Папка результатов","output_dir",kind="dir")
        self._file_row(f,4,"FINAL_CASE.json","case_path")
        buttons=ttk.Frame(f); buttons.grid(row=5,column=0,columnspan=3,sticky="w",pady=10)
        ttk.Button(buttons,text="Сохранить CASE",command=self.save_case).pack(side="left",padx=(0,6)); ttk.Button(buttons,text="Загрузить CASE",command=self.load_case).pack(side="left")

    def _on_theory_change(self):
        self.advanced_bc=None
        self.engineering_bc=None
        self.vars["bc_mode"].set("SIMPLE")
        self._refresh_visibility()

    def _on_model_change(self):
        self.advanced_bc=None
        self.engineering_bc=None
        self.vars["bc_mode"].set("SIMPLE")
        self._update_kinematics_labels()
        self._update_grid_info(); self._refresh_visibility()

    def _te_m(self):
        if self.vars.get("theory") and self.vars["theory"].get()=="CUF":
            return int(self.vars["cuf_model"].get()[2:])+1
        return None

    def _load_material_library(self):
        if hasattr(self,"_material_library_cache"):
            return self._material_library_cache
        path=APP_DIR/"MATERIAL_LIBRARY.json"
        data={"materials":[]}
        try:
            data=json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"MATERIAL_LIBRARY warning: {exc}\n")
        mats={str(x.get("name")):x for x in data.get("materials",[]) if isinstance(x,dict) and x.get("name")}
        self._material_library_cache=mats
        return mats

    def _refresh_material_mode(self):
        if not hasattr(self,"material_name_combo"): return
        src=self.vars.get("material_source",tk.StringVar(value="USER")).get()
        mats=self._load_material_library()
        if src=="REFERENCE":
            self.material_name_combo.configure(state="readonly",values=tuple(mats.keys()))
            if self.vars["material_name"].get() not in mats:
                self.vars["material_name"].set(next(iter(mats),""))
            self._apply_material_preset()
        else:
            # The reference selector has no meaning for a manual material.
            # Do not leave a stale steel name visible in a disabled control.
            self.vars["material_name"].set("— не используется —")
            self.material_name_combo.configure(state="disabled",values=("— не используется —",))
            self.material_note.configure(text="Пользовательский материал: введите E0 и ν0 вручную.")
        # A reference preset must remain an indivisible pair (E0, ν0).
        # Disabling the fields prevents metadata such as “Steel” from being
        # combined with manually changed numerical constants.
        readonly=(src=="REFERENCE")
        for entry in (getattr(self,"material_E0_entry",None),getattr(self,"material_nu0_entry",None)):
            if entry is not None:
                entry.configure(state="disabled" if readonly else "normal")

    def _apply_material_preset(self):
        if self.vars.get("material_source",tk.StringVar(value="USER")).get()!="REFERENCE": return
        mats=self._load_material_library(); name=self.vars.get("material_name",tk.StringVar()).get(); m=mats.get(name)
        if not m: return
        Epa=float(m["E_Pa"]); nu=float(m["nu"])
        if self.vars.get("unit_system",tk.StringVar(value="N_MM_MPA")).get()=="N_MM_MPA": E=Epa/1.0e6
        else: E=Epa
        self.vars["E0"].set(f"{E:.12g}"); self.vars["nu0"].set(f"{nu:.12g}")
        self.material_note.configure(text=f"{name}: E0={E:.6g}, ν0={nu:.6g}. {m.get('note','')}")

    def _refresh_load_step_mode(self):
        if not hasattr(self,"load_range_frame"): return
        mode=self.vars.get("load_step_mode",tk.StringVar(value="RANGE")).get()
        if hasattr(self,"load_count_frame"):
            self.load_count_frame.grid_remove()
        self.load_range_frame.grid_remove()
        self.load_manual_frame.grid_remove()
        if mode=="RANGE":
            self.load_range_frame.grid()
        elif mode=="COUNT" and hasattr(self,"load_count_frame"):
            self.load_count_frame.grid()
        else:
            self.load_manual_frame.grid()
        self._update_load_steps_info()

    def _range_factors(self):
        a=_float(self.vars["load_lambda_start"].get(),"λ start")
        d=_float(self.vars["load_lambda_step"].get(),"Δλ",positive=True)
        b=_float(self.vars["load_lambda_end"].get(),"λ end")
        if b<a: raise ValueError("λ end должен быть не меньше λ start")
        n=int(np.floor((b-a)/d+1.0e-12))+1
        vals=[a+i*d for i in range(n)]
        tol=1.0e-10*max(1.0,abs(a),abs(b))
        if vals[-1] < b-tol: vals.append(b)
        elif abs(vals[-1]-b)<=tol: vals[-1]=b
        if not vals or not np.all(np.isfinite(vals)): raise ValueError("Некорректные шаги нагрузки")
        return [float(f"{x:.15g}") for x in vals]

    def _count_factors(self):
        b=_float(self.vars["load_lambda_end"].get(),"λ end",positive=True)
        n=_int(self.vars["load_step_count"].get(),"число шагов",1)
        d=b/float(n)
        vals=[(i+1)*d for i in range(n)]
        if not vals or not np.all(np.isfinite(vals)): raise ValueError("Некорректное число шагов нагрузки")
        vals[-1]=b
        return [float(f"{x:.15g}") for x in vals]

    def _update_load_steps_info(self):
        if not hasattr(self,"load_steps_info"): return
        try:
            vals=self._parse_factors()
            self.load_steps_info.configure(text=f"Шагов: {len(vals)}; λ = "+", ".join(f"{x:.6g}" for x in vals[:8])+(" …" if len(vals)>8 else ""))
        except Exception as exc:
            self.load_steps_info.configure(text=f"Шаги не заданы корректно: {exc}")

    def _unit_switch_unsafe_reason(self):
        """Return why an automatic GUI unit conversion would be unsafe.

        Scalars owned by this GUI can be converted exactly.  External arrays,
        arbitrary formulas and advanced BC dictionaries may contain dimensional
        quantities whose powers cannot be inferred generically, so a unit-system
        switch is blocked rather than silently changing the physical problem.
        """
        if self.advanced_bc is not None:
            return "заданы индивидуальные/расширенные граничные условия"
        if self.vars.get("load_type") and self.vars["load_type"].get()=="EXTERNAL_NPZ" and self.vars.get("loads_npz") and self.vars["loads_npz"].get().strip():
            return "подключено внешнее поле нагрузки NPZ"
        if self.vars.get("state_source") and self.vars["state_source"].get()=="NPZ" and self.vars.get("state_npz") and self.vars["state_npz"].get().strip():
            return "подключено начальное состояние NPZ"
        if self.vars.get("gb_source") and self.vars["gb_source"].get()=="NPZ" and self.vars.get("gb_npz") and self.vars["gb_npz"].get().strip():
            return "подключено начальное поле GB из NPZ"
        if self.vars.get("method") and self.vars["method"].get()=="MVI" and self.vars.get("profiles_npz") and self.vars["profiles_npz"].get().strip():
            return "подключены внешние профили МВИ NPZ"
        if self.vars.get("curve_mode"):
            cm=self.vars["curve_mode"].get()
            if cm=="CSV" and self.vars.get("curve_csv") and self.vars["curve_csv"].get().strip():
                return "кривая Биргера читается из внешнего CSV/TXT"
            if cm=="FORMULA" and self.vars.get("curve_formula") and self.vars["curve_formula"].get().strip():
                return "кривая Биргера задана произвольной формулой"
            if cm=="PIECEWISE" and hasattr(self,"curve_piecewise") and self.curve_piecewise.get("1.0","end-1c").strip():
                return "кривая Биргера задана кусочными формулами"
            if cm=="EXTERNAL_PYTHON" and any(self.vars.get(k) and self.vars[k].get().strip() for k in ("phi_module","phi_callable","phi_path")):
                return "кривая Биргера задана внешним Python-кодом"
        return None

    @staticmethod
    def _unit_fmt(value):
        return f"{float(value):.12g}"

    def _convert_gui_units(self,old_system,new_system):
        """Convert GUI-owned dimensional inputs transactionally, old -> new."""
        affected=(
            "L","R","h","load_x1","load_x2","load_band_width",
            "E0","G_ref","sigma_ref","yield_stress","j2_sigma_y0","j2_H",
            "load_q","load_p_line","load_c1","load_c2","load_c0",
        )
        snapshots={k:self.vars[k].get() for k in affected if k in self.vars}
        curve_snapshot=self.curve_points.get("1.0","end-1c") if hasattr(self,"curve_points") else None
        j2_table_snapshot=self.j2_hardening_table.get("1.0","end-1c") if hasattr(self,"j2_hardening_table") else None
        cp_snapshot=self.control_points_text.get("1.0","end-1c") if hasattr(self,"control_points_text") else None
        try:
            kinds={
                "L":"length","R":"length","h":"length",
                "load_x1":"length","load_x2":"length","load_band_width":"length",
                "E0":"stress","G_ref":"stress","sigma_ref":"stress","yield_stress":"stress",
                "j2_sigma_y0":"stress","j2_H":"stress",
                "load_q":"pressure","load_p_line":"line_load",
            }
            if self.vars.get("load_coord") and self.vars["load_coord"].get()=="S":
                kinds.update({"load_c1":"length","load_c2":"length","load_c0":"length"})
            for key,kind in kinds.items():
                if key not in self.vars: continue
                txt=self.vars[key].get().strip()
                if not txt: continue
                val=funits.convert_value(float(txt),kind,old_system,new_system)
                self.vars[key].set(self._unit_fmt(val))

            # TABLE: epsilon is dimensionless; Phi/sigma column has stress units.
            if hasattr(self,"curve_points") and self.vars.get("curve_mode") and self.vars["curve_mode"].get()=="TABLE":
                out=[]
                for raw in curve_snapshot.splitlines():
                    st=raw.strip()
                    if not st or st.startswith("#"):
                        out.append(raw); continue
                    parts=[x.strip() for x in st.replace(";",",").split(",")]
                    if len(parts)<2:
                        raise ValueError("TABLE Биргера: ожидались epsilon, Phi")
                    eps=float(parts[0]); sig=float(parts[1])
                    sig2=funits.convert_value(sig,"stress",old_system,new_system)
                    out.append(f"{eps:.12g}, {sig2:.12g}")
                self.curve_points.delete("1.0","end"); self.curve_points.insert("1.0","\n".join(out))

            # J2 TABLE: kappa is dimensionless; sigma_y is a stress.
            if hasattr(self,"j2_hardening_table") and j2_table_snapshot is not None:
                out=[]
                for raw in j2_table_snapshot.splitlines():
                    st=raw.strip()
                    if not st or st.startswith("#"):
                        out.append(raw); continue
                    parts=[x.strip() for x in st.replace(";",",").split(",")]
                    if len(parts)<2:
                        raise ValueError("J2 TABLE: expected kappa, sigma_y")
                    kappa=float(parts[0]); sigma_y=float(parts[1])
                    sigma_y2=funits.convert_value(sigma_y,"stress",old_system,new_system)
                    out.append(f"{kappa:.12g}, {sigma_y2:.12g}")
                self.j2_hardening_table.delete("1.0","end")
                self.j2_hardening_table.insert("1.0","\n".join(out))

            # Postprocess control points: x is dimensional, theta is stored/displayed in degrees.
            if hasattr(self,"control_points_text") and cp_snapshot is not None:
                out=[]
                for raw in cp_snapshot.splitlines():
                    st=raw.strip()
                    if not st or st.startswith("#"):
                        out.append(raw); continue
                    parts=[x.strip() for x in st.replace(";",",").split(",")]
                    if len(parts)==2:
                        x=float(parts[0]); th=parts[1]
                        x2=funits.convert_value(x,"length",old_system,new_system)
                        out.append(f"{x2:.12g}, {th}")
                    elif len(parts)>=3:
                        label=parts[0]; x=float(parts[1]); th=parts[2]
                        x2=funits.convert_value(x,"length",old_system,new_system)
                        out.append(f"{label}, {x2:.12g}, {th}")
                    else:
                        raise ValueError("Контрольная точка: label, x, theta_deg")
                self.control_points_text.delete("1.0","end"); self.control_points_text.insert("1.0","\n".join(out))
        except Exception:
            for key,val in snapshots.items(): self.vars[key].set(val)
            if hasattr(self,"curve_points") and curve_snapshot is not None:
                self.curve_points.delete("1.0","end"); self.curve_points.insert("1.0",curve_snapshot)
            if hasattr(self,"j2_hardening_table") and j2_table_snapshot is not None:
                self.j2_hardening_table.delete("1.0","end"); self.j2_hardening_table.insert("1.0",j2_table_snapshot)
            if hasattr(self,"control_points_text") and cp_snapshot is not None:
                self.control_points_text.delete("1.0","end"); self.control_points_text.insert("1.0",cp_snapshot)
            raise

    def _update_units(self):
        new_system=self.vars.get("unit_system",tk.StringVar(value="N_MM_MPA")).get()
        try:new_system=funits.normalize_system(new_system)
        except Exception:
            new_system="N_MM_MPA"; self.vars["unit_system"].set(new_system)
        old_system=getattr(self,"_last_unit_system",None)
        if old_system and old_system!=new_system:
            reason=self._unit_switch_unsafe_reason()
            if reason:
                self.vars["unit_system"].set(old_system)
                new_system=old_system
                messagebox.showwarning(
                    "Единицы",
                    "Автоматическое переключение единиц отменено: "+reason+".\n\n"
                    "Сначала отключите/пересохраните этот внешний источник в требуемой системе единиц. "
                    "Решатель не угадывает размерность внешних данных."
                )
            else:
                try:
                    self._convert_gui_units(old_system,new_system)
                    self._log(f"V5: числовые поля GUI пересчитаны {old_system} -> {new_system}.\n")
                except Exception as exc:
                    self.vars["unit_system"].set(old_system); new_system=old_system
                    messagebox.showerror("Единицы",f"Не удалось безопасно пересчитать поля: {exc}")
        self._last_unit_system=new_system
        u=self.UNIT_LABELS.get(new_system,self.UNIT_LABELS["N_MM_MPA"])
        if hasattr(self,"unit_labels"):
            self.unit_labels["L"].configure(text=f"Длина L [{u['length']}]")
            self.unit_labels["R"].configure(text=f"Радиус R [{u['length']}]")
            self.unit_labels["h"].configure(text=f"Толщина h [{u['length']}]")
            self.unit_labels["E0"].configure(text=f"Модуль E0 [{u['stress']}]")
        if hasattr(self,"load_q_label"):
            self.load_q_label.configure(text=f"q [{u['pressure']}]")
            self.load_line_label.configure(text=f"p_line [{u['line_load']}]")
            if hasattr(self,"load_band_label"):
                self.load_band_label.configure(text=f"Ширина полосы b [{u['length']}]")
        if self.vars.get("material_source",tk.StringVar(value="USER")).get()=="REFERENCE": self._apply_material_preset()
        self._update_load_units(); self._update_grid_info()

    def _update_load_units(self):
        if not hasattr(self,"load_c1_label"): return
        cm=self.vars.get("load_coord",tk.StringVar(value="THETA")).get()
        lu=self.UNIT_LABELS.get(self.vars.get("unit_system",tk.StringVar(value="N_MM_MPA")).get(),self.UNIT_LABELS["N_MM_MPA"])["length"]
        if cm=="THETA":
            self.load_c1_label.configure(text="θ1 [rad]"); self.load_c2_label.configure(text="θ2 [rad]"); self.load_c0_label.configure(text="θ0 [rad]")
        else:
            self.load_c1_label.configure(text=f"s1 [{lu}]"); self.load_c2_label.configure(text=f"s2 [{lu}]"); self.load_c0_label.configure(text=f"s0 [{lu}]")

    def _apply_grid_preset(self):
        s=self.vars["grid_preset"].get().replace("×","x")
        try:a,b=s.lower().split("x"); self.vars["nx"].set(a); self.vars["ns"].set(b)
        except Exception:return
        self._update_grid_info()

    def _update_grid_info(self):
        try:
            L=float(self.vars["L"].get()); R=float(self.vars["R"].get()); nx=int(self.vars["nx"].get()); ns=int(self.vars["ns"].get())
            dx=L/(nx-1); ds=2*np.pi*R/ns
            if self.vars["theory"].get()=="CUF": ndof=3*self._te_m()*nx*ns
            else: ndof=3*nx*ns
            self.grid_info.configure(text=f"Δx={dx:.6g}; Δs={ds:.6g}; узлов={nx*ns}; неизвестных полного FDM-поля={ndof}")
        except Exception:
            if hasattr(self,"grid_info"): self.grid_info.configure(text="Δx = —; Δs = —; число неизвестных = —")
        try:
            if self.vars["theory"].get()=="CUF":
                m=self._te_m(); self.model_info.configure(text=f"CUF {self.vars['cuf_model'].get()}: Fτ=z^τ, τ=0…{m-1}; обобщённых полей={3*m}.  Единицы: {self.vars['unit_system'].get()}")
            else:self.model_info.configure(text=f"Kirchhoff–Love displacement formulation.  Единицы: {self.vars['unit_system'].get()}")
        except Exception: pass

    def _set_children_state(self,w,state):
        try:
            for ch in w.winfo_children():
                try: ch.configure(state=state)
                except Exception: pass
                self._set_children_state(ch,state)
        except Exception: pass

    def _refresh_visibility(self):
        theory=self.vars.get("theory",tk.StringVar(value="CUF")).get(); method=self.vars.get("method",tk.StringVar(value="FDM")).get(); mode=self.vars.get("mode",tk.StringVar(value="LINEAR_ELASTIC")).get(); nonlinear=mode.startswith("NONLINEAR"); birger=mode.endswith("BIRGER")
        try:self.cuf_model_combo.configure(state="readonly" if theory=="CUF" else "disabled")
        except Exception:pass
        try:self.rank_combo.configure(state="readonly" if method=="MVI" else "disabled")
        except Exception:pass
        if hasattr(self,"mvi_frame"):self._set_children_state(self.mvi_frame,"normal" if method=="MVI" else "disabled")
        if hasattr(self,"newton_frame"):self._set_children_state(self.newton_frame,"normal" if (method=="FDM" and nonlinear) else "disabled")
        if hasattr(self,"linear_frame"):self._set_children_state(self.linear_frame,"normal" if (method=="FDM" and mode=="LINEAR_ELASTIC") else "disabled")
        if hasattr(self,"birger_frame"):self._set_children_state(self.birger_frame,"normal" if birger else "disabled")
        if hasattr(self,"initial_state_frame"):self._set_children_state(self.initial_state_frame,"normal" if (method=="FDM" and nonlinear) else "disabled")
        if hasattr(self,"gb_frame"):self._set_children_state(self.gb_frame,"normal" if birger else "disabled")
        try:
            if theory=="CUF":
                if self.vars["load_face"].get() not in loadbuilder.FACES:
                    self.vars["load_face"].set("OUTER")
                self.load_face_label.grid()
                self.load_face_combo.grid()
                self.load_face_combo.configure(
                    state="readonly" if self.vars["load_type"].get()!="EXTERNAL_NPZ" else "disabled"
                )
            else:
                self.vars["load_face"].set("")
                self.load_face_label.grid_remove()
                self.load_face_combo.grid_remove()
        except Exception:pass
        bc_mode=self.vars.get("bc_mode",tk.StringVar(value="SIMPLE")).get()
        try:
            self.bc_left_combo.configure(state="readonly" if bc_mode=="SIMPLE" else "disabled")
            self.bc_right_combo.configure(state="readonly" if bc_mode=="SIMPLE" else "disabled")
            self.bc_edit_btn.configure(state="normal")
            self._update_bc_display()
        except Exception:pass
        try:self.series_btn.configure(state="normal" if method=="MVI" else "disabled")
        except Exception:pass
        self._refresh_load_visibility(); self._refresh_curve_visibility(); self._update_grid_info(); self._schedule_problem_scheme_redraw(20)

    def _refresh_load_visibility(self):
        if not hasattr(self,"load_file_frame"): return
        typ=self.vars.get("load_type",tk.StringVar(value="UNIFORM_RADIAL_FACE")).get()

        # q belongs only to distributed face pressure.  The line generator and
        # point force have their own amplitudes, but use the same direction
        # selector below.
        pressure_active=typ in (
            "UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE","PATCH_RADIAL_FACE"
        )
        for widget in (getattr(self,"load_q_label",None),getattr(self,"load_q_entry",None)):
            if widget is not None:
                widget.grid() if pressure_active else widget.grid_remove()
        line_active=typ=="LINE_RADIAL_GENERATOR"
        point_active=typ=="POINT_RADIAL_FORCE"
        direction_active=pressure_active or line_active or point_active
        for widget in (getattr(self,"load_pressure_direction_label",None),
                       getattr(self,"load_pressure_direction_combo",None)):
            if widget is not None:
                widget.grid() if direction_active else widget.grid_remove()
        for widget in (getattr(self,"load_line_label",None),getattr(self,"load_line_entry",None)):
            if widget is not None:
                widget.grid() if line_active else widget.grid_remove()
        for widget in (getattr(self,"load_point_label",None),getattr(self,"load_point_entry",None),
                       getattr(self,"load_point_x_label",None),getattr(self,"load_point_x_entry",None),
                       getattr(self,"load_point_t_label",None),getattr(self,"load_point_t_entry",None)):
            if widget is not None:
                widget.grid() if point_active else widget.grid_remove()

        # Show only controls that belong to the selected load type. This keeps
        # lambda stepping immediately reachable and avoids a tall disabled PATCH
        # block hiding the controls below it.
        if hasattr(self,"harmonic_frame"):
            if typ=="HARMONIC_RADIAL_FACE":
                if not self.harmonic_frame.winfo_manager():
                    self.harmonic_frame.pack(fill="x",pady=(0,8),before=self.load_steps_frame)
                self._set_children_state(self.harmonic_frame,"normal")
            else:
                self.harmonic_frame.pack_forget()

        patch_active=typ in ("PATCH_RADIAL_FACE","HARMONIC_RADIAL_FACE","LINE_RADIAL_GENERATOR")
        if hasattr(self,"patch_frame"):
            if patch_active:
                if not self.patch_frame.winfo_manager():
                    self.patch_frame.pack(fill="x",pady=(0,8),before=self.load_steps_frame)
                self._set_children_state(self.patch_frame,"normal")
                try:self.load_coord_combo.configure(state="readonly")
                except Exception:pass
                area_type=typ in ("PATCH_RADIAL_FACE","HARMONIC_RADIAL_FACE")
                for widget in (getattr(self,"load_c1_label",None),getattr(self,"load_c1_entry",None),
                               getattr(self,"load_c2_label",None),getattr(self,"load_c2_entry",None),
                               getattr(self,"load_full_check",None)):
                    if widget is not None:
                        widget.grid() if area_type else widget.grid_remove()
                for widget in (getattr(self,"load_c0_label",None),getattr(self,"load_c0_entry",None),
                               getattr(self,"load_band_label",None),getattr(self,"load_band_entry",None)):
                    if widget is not None:
                        widget.grid() if typ=="LINE_RADIAL_GENERATOR" else widget.grid_remove()
                if typ=="HARMONIC_RADIAL_FACE" and not self.vars.get("load_c1",tk.StringVar()).get().strip() and not self.vars.get("load_c2",tk.StringVar()).get().strip():
                    self.load_full_var.set(True)
            else:
                self.patch_frame.pack_forget()

        if typ=="EXTERNAL_NPZ":
            if not self.load_file_frame.winfo_manager():
                self.load_file_frame.pack(fill="x",pady=(0,8),after=self.load_steps_frame)
            self._set_children_state(self.load_file_frame,"normal")
        else:
            self.load_file_frame.pack_forget()

        if hasattr(self,"load_step_mode_combo"):
            # EXTERNAL_NPZ carries its own load-factor sequence; retain the step
            # block as a visible diagnostic but disable editing for that mode.
            self._set_children_state(self.load_range_frame,"normal" if typ!="EXTERNAL_NPZ" else "disabled")
            self._set_children_state(self.load_manual_frame,"normal" if typ!="EXTERNAL_NPZ" else "disabled")
            try:self.load_step_mode_combo.configure(state="readonly" if typ!="EXTERNAL_NPZ" else "disabled")
            except Exception:pass
        self._refresh_load_step_mode(); self._update_load_units()
        if hasattr(self,"load_left_canvas"):
            self.after_idle(lambda:self.load_left_canvas.configure(scrollregion=self.load_left_canvas.bbox("all")))

    def _refresh_curve_visibility(self):
        if not hasattr(self,"curve_table_frame"):return
        mode=self.vars.get("curve_mode",tk.StringVar(value="TABLE")).get()
        frames=((self.curve_table_frame,"TABLE"),(self.curve_csv_frame,"CSV"),(self.curve_formula_frame,"FORMULA"),(self.curve_piece_frame,"PIECEWISE"),(self.curve_external_frame,"EXTERNAL_PYTHON"))
        for fr,m in frames:
            if mode==m:fr.grid()
            else:fr.grid_remove()
        try:self.curve_ext_combo.configure(state="readonly" if mode in ("TABLE","CSV") else "disabled")
        except Exception:pass

    def _bc_model_tag(self):
        return bcreg.model_tag(
            self._v("theory","CUF").get(),
            self._v("cuf_model","TE1").get(),
        )

    def _engineering_selected_sides(self):
        if self._v("bc_mode","SIMPLE").get()!="SIMPLE":
            return ()
        return tuple(
            side for side,key in (("left","bc_left"),("right","bc_right"))
            if self._v(key).get().upper()=="ELASTIC_ENGINEERING"
        )

    def _update_bc_display(self):
        if not hasattr(self,"bc_summary"):
            return
        mode=self._v("bc_mode","SIMPLE").get()
        theory=self._v("theory","CUF").get()
        model=self._v("cuf_model","TE1").get()
        if theory in capreg.PLANNED_THEORIES:
            self.bc_summary.configure(text=f"{theory}: граничные условия разрабатываются")
            if hasattr(self,"bc_active_equations"):
                self.bc_active_equations.configure(text="Нельзя использовать KL/CUF условия для FSDT/REDDY")
            return
        if mode=="SIMPLE":
            left=self._v("bc_left","").get().upper().strip()
            right=self._v("bc_right","").get().upper().strip()
            try:
                left_exact=bcreg.resolved_preset_code(left,theory) if left else "—"
                right_exact=bcreg.resolved_preset_code(right,theory) if right else "—"
            except Exception:
                left_exact=left or "—"; right_exact=right or "—"
            self.bc_summary.configure(
                text=f"PRESET · {self._bc_model_tag()} · x=0: {left_exact} · x=L: {right_exact}"
            )
            lf=bcreg.formula_text(theory,model,left) if left else "не выбрано"
            rf=bcreg.formula_text(theory,model,right) if right else "не выбрано"
            extra=""
            sides=self._engineering_selected_sides()
            if sides:
                ready=isinstance(getattr(self,"engineering_bc",None),dict)
                extra=(" · k заданы" if ready else " · ТРЕБУЕТСЯ задать k")
            if hasattr(self,"bc_active_equations"):
                self.bc_active_equations.configure(
                    text=f"Активные уравнения ГУ:\n  x=0: {lf}\n  x=L: {rf}{extra}"
                )
            if hasattr(self,"bc_engineering_btn"):
                self.bc_engineering_btn.configure(state="normal" if sides else "disabled")
        else:
            self.bc_summary.configure(text=self._advanced_bc_summary())
            if hasattr(self,"bc_active_equations"):
                self.bc_active_equations.configure(
                    text=f"Активные уравнения ГУ: CUSTOM · {self._bc_model_tag()} · по отдельным DOF"
                )
            if hasattr(self,"bc_engineering_btn"):
                self.bc_engineering_btn.configure(state="disabled")

    def _edit_engineering_bc(self):
        sides=self._engineering_selected_sides()
        if not sides:
            messagebox.showinfo(
                "Инженерные КУ",
                "Сначала выберите «Упругая / инженерная» для левого и/или правого торца."
            )
            return
        theory=self._v("theory","CUF").get()
        current=self.engineering_bc if isinstance(getattr(self,"engineering_bc",None),dict) else {}
        win=tk.Toplevel(self)
        win.title(f"Инженерные краевые условия — {self._bc_model_tag()}")
        win.transient(self)
        host=ttk.Frame(win,padding=10); host.pack(fill="both",expand=True)
        for c,txt in enumerate(("Торец","DOF","k","q0","T̄")):
            ttk.Label(host,text=txt,font=("Segoe UI Semibold",9)).grid(row=0,column=c,padx=4,pady=3,sticky="w")
        rows=[]
        if theory=="KL":
            dofs=("u","v","w","slope")
            for side in sides:
                for field in dofs:
                    item=((current.get(side) or {}).get(field) or {})
                    rows.append((side,field,item))
        else:
            m=self._te_m()
            for side in sides:
                for comp in ("u","v","w"):
                    vals=(current.get(side) or {}).get(comp) or []
                    for tau in range(m):
                        item=vals[tau] if tau<len(vals) and isinstance(vals[tau],dict) else {}
                        rows.append((side,f"{comp}{tau}",item))
        varmap={}
        for r,(side,dof,item) in enumerate(rows,1):
            ttk.Label(host,text="x=0" if side=="left" else "x=L").grid(row=r,column=0,padx=4,pady=2,sticky="w")
            ttk.Label(host,text=dof).grid(row=r,column=1,padx=4,pady=2,sticky="w")
            kv=tk.StringVar(value="" if item.get("stiffness") is None else str(item.get("stiffness")))
            qv=tk.StringVar(value=str(item.get("reference",0.0)))
            tv=tk.StringVar(value=str(item.get("traction",item.get("value",0.0))))
            ttk.Entry(host,textvariable=kv,width=16).grid(row=r,column=2,padx=4,pady=2)
            ttk.Entry(host,textvariable=qv,width=16).grid(row=r,column=3,padx=4,pady=2)
            ttk.Entry(host,textvariable=tv,width=16).grid(row=r,column=4,padx=4,pady=2)
            varmap[(side,dof)]=(kv,qv,tv)
        note=(
            "Уравнение каждой строки: T + k(q-q0) = T̄.  "
            "k вводится в согласованных единицах текущей модели.  k=0 допустимо для отдельной DOF, "
            "но на каждом инженерном торце хотя бы одна жёсткость должна быть > 0."
        )
        ttk.Label(host,text=note,wraplength=700).grid(row=len(rows)+1,column=0,columnspan=5,sticky="w",pady=(8,4))
        def save():
            eng={}
            positive={side:False for side in sides}
            if theory=="KL":
                for side in sides:
                    eng[side]={}
                    for field in ("u","v","w","slope"):
                        kv,qv,tv=varmap[(side,field)]
                        k=_float(kv.get(),f"k {side}.{field}",nonnegative=True)
                        q0=_float(qv.get(),f"q0 {side}.{field}")
                        tb=_float(tv.get(),f"Tbar {side}.{field}")
                        positive[side]|=(k>0.0)
                        eng[side][field]={"stiffness":k,"reference":q0,"traction":tb}
            else:
                m=self._te_m()
                for side in sides:
                    eng[side]={c:[None]*m for c in ("u","v","w")}
                    for comp in ("u","v","w"):
                        for tau in range(m):
                            kv,qv,tv=varmap[(side,f"{comp}{tau}")]
                            k=_float(kv.get(),f"k {side}.{comp}{tau}",nonnegative=True)
                            q0=_float(qv.get(),f"q0 {side}.{comp}{tau}")
                            tb=_float(tv.get(),f"Tbar {side}.{comp}{tau}")
                            positive[side]|=(k>0.0)
                            eng[side][comp][tau]={"stiffness":k,"reference":q0,"traction":tb}
            bad=[side for side,ok in positive.items() if not ok]
            if bad:
                raise ValueError("Для инженерного торца хотя бы одна жёсткость k должна быть > 0: "+", ".join(bad))
            # Preserve a previously configured opposite elastic side if it is not
            # currently being edited; normal preset sides do not need entries.
            self.engineering_bc=eng
            self._update_bc_display()
            win.destroy()
        ttk.Button(host,text="Сохранить инженерные КУ",command=lambda:self._safe_dialog_call(save)).grid(
            row=len(rows)+2,column=0,columnspan=5,pady=(6,2)
        )

    def _advanced_bc_summary(self):
        if not self.advanced_bc:
            return f"CUSTOM · {self._bc_model_tag()}: не задано"
        try:
            if self._v("theory").get()=="KL":
                parts=[]
                for side in ("left","right"):
                    kinds=[self.advanced_bc[side][f]["kind"] for f in ("u","v","w","slope")]
                    parts.append(f"{side}: "+"/".join(kinds))
                return f"CUSTOM · KL · "+" ; ".join(parts)
            m=self._te_m()
            return f"CUSTOM · {self._bc_model_tag()}: {3*m} generalized DOF per edge"
        except Exception:
            return f"CUSTOM · {self._bc_model_tag()}: задано"

    def _preset_end_as_custom(self,side):
        theory=self._v("theory","CUF").get()
        code=self._v("bc_left" if side=="left" else "bc_right","").get().upper().strip()
        if not code:
            return None
        eng=self.engineering_bc if isinstance(getattr(self,"engineering_bc",None),dict) else {}
        if theory=="KL":
            kinds=bcreg.kl_kinds(code)
            out={}
            for field,kind in zip(("u","v","w","slope"),kinds):
                item={"kind":kind,"value":0.0}
                if kind=="ELASTIC":
                    try:e=eng[side][field]
                    except Exception as exc:
                        raise ValueError("Сначала задайте инженерные жёсткости k для выбранного торца") from exc
                    item.update(stiffness=float(e["stiffness"]),reference=float(e.get("reference",0.0)),value=float(e.get("traction",0.0)))
                out[field]=item
            return out
        m=self._te_m(); out={}
        for comp_index,comp in enumerate(("u","v","w")):
            vals=[]
            for tau in range(m):
                kind=bcreg.cuf_kinds_tau(code,tau)[comp_index]
                item={"kind":kind,"data":0.0}
                if kind=="ELASTIC":
                    try:e=eng[side][comp][tau]
                    except Exception as exc:
                        raise ValueError("Сначала задайте инженерные жёсткости k для выбранного торца") from exc
                    item.update(stiffness=float(e["stiffness"]),reference=float(e.get("reference",0.0)),data=float(e.get("traction",0.0)))
                vals.append(item)
            out[comp]=vals
        return out

    def _default_individual_bc(self,theory):
        return {
            "left":self._preset_end_as_custom("left"),
            "right":self._preset_end_as_custom("right"),
        }

    def _edit_individual_bc(self):
        theory=self._v("theory","CUF").get()
        if self._v("bc_mode","SIMPLE").get()!="INDIVIDUAL":
            try:
                self.advanced_bc=self._default_individual_bc(theory)
            except Exception as exc:
                messagebox.showerror("Custom КУ",str(exc)); return
            if self.advanced_bc.get("left") is None or self.advanced_bc.get("right") is None:
                messagebox.showerror("Custom КУ","Сначала выберите Preset для обоих торцов или задайте Custom вручную после выбора модели.")
                return
            self._v("bc_mode").set("INDIVIDUAL")
            self._refresh_visibility()
        elif self.advanced_bc is None:
            # Start from CLAMPED only as an explicit editor template; this does
            # not affect the solver until the user saves the Custom table.
            oldl,oldr=self._v("bc_left").get(),self._v("bc_right").get()
            try:
                self._v("bc_left").set("CLAMPED"); self._v("bc_right").set("CLAMPED")
                self.advanced_bc=self._default_individual_bc(theory)
            finally:
                self._v("bc_left").set(oldl); self._v("bc_right").set(oldr)

        win=tk.Toplevel(self)
        win.title(f"Ручные / Custom КУ — {self._bc_model_tag()}")
        win.transient(self)
        outer=ttk.Frame(win,padding=10); outer.pack(fill="both",expand=True)
        canvas=tk.Canvas(outer,highlightthickness=0,height=560,width=850)
        sb=ttk.Scrollbar(outer,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left",fill="both",expand=True); sb.pack(side="right",fill="y")
        host=ttk.Frame(canvas); wid=canvas.create_window((0,0),window=host,anchor="nw")
        host.bind("<Configure>",lambda _e:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",lambda e:canvas.itemconfigure(wid,width=max(1,e.width)))
        headers=("Торец","DOF","Тип","q̄ / T̄","k","q0")
        for c,h in enumerate(headers):
            ttk.Label(host,text=h,font=("Segoe UI Semibold",9)).grid(row=0,column=c,padx=4,pady=4,sticky="w")

        rows=[]
        if theory=="KL":
            for side in ("left","right"):
                for field in ("u","v","w","slope"):
                    item=self.advanced_bc.get(side,{}).get(field,{"kind":"NATURAL","value":0.0})
                    rows.append((side,field,item))
        else:
            m=self._te_m()
            for side in ("left","right"):
                for comp in ("u","v","w"):
                    items=self.advanced_bc.get(side,{}).get(comp,[])
                    if len(items)!=m:items=[{"kind":"NATURAL","data":0.0} for _ in range(m)]
                    for tau,item in enumerate(items):rows.append((side,f"{comp}{tau}",item))
        vars_map={}
        for r,(side,field,item) in enumerate(rows,1):
            ttk.Label(host,text="x=0" if side=="left" else "x=L").grid(row=r,column=0,padx=4,pady=2,sticky="w")
            ttk.Label(host,text=field).grid(row=r,column=1,padx=4,pady=2,sticky="w")
            kv=tk.StringVar(value=str(item.get("kind","NATURAL")).upper())
            val=item.get("value",item.get("data",0.0))
            vv=tk.StringVar(value=str(val)); kk=tk.StringVar(value="" if item.get("stiffness") is None else str(item.get("stiffness")))
            q0=tk.StringVar(value=str(item.get("reference",0.0)))
            ttk.Combobox(host,textvariable=kv,values=("GEOMETRIC","NATURAL","ELASTIC"),state="readonly",width=13).grid(row=r,column=2,padx=4,pady=2)
            ttk.Entry(host,textvariable=vv,width=15).grid(row=r,column=3,padx=4,pady=2)
            ttk.Entry(host,textvariable=kk,width=15).grid(row=r,column=4,padx=4,pady=2)
            ttk.Entry(host,textvariable=q0,width=15).grid(row=r,column=5,padx=4,pady=2)
            vars_map[(side,field)]=(kv,vv,kk,q0)
        ttk.Label(
            host,
            text="GEOMETRIC: q=q̄.  NATURAL: T=T̄.  ELASTIC: T+k(q-q0)=T̄.  Для GEOMETRIC/NATURAL поля k и q0 игнорируются.",
            wraplength=800
        ).grid(row=len(rows)+1,column=0,columnspan=6,sticky="w",pady=(8,4))

        def parse_item(side,field,data_key):
            kv,vv,kk,q0=vars_map[(side,field)]
            kind=kv.get().upper()
            if kind not in ("GEOMETRIC","NATURAL","ELASTIC"):
                raise ValueError(f"Не выбран тип КУ: {side}.{field}")
            item={"kind":kind,data_key:_float(vv.get(),f"{side}.{field}")}
            if kind=="ELASTIC":
                item["stiffness"]=_float(kk.get(),f"k {side}.{field}",nonnegative=True)
                item["reference"]=_float(q0.get(),f"q0 {side}.{field}")
            return item

        def save():
            if theory=="KL":
                adv={"left":{},"right":{}}
                for side in ("left","right"):
                    for field in ("u","v","w","slope"):
                        adv[side][field]=parse_item(side,field,"value")
            else:
                m=self._te_m(); adv={"left":{c:[None]*m for c in ("u","v","w")},"right":{c:[None]*m for c in ("u","v","w")}}
                for side in ("left","right"):
                    for comp in ("u","v","w"):
                        for tau in range(m):
                            adv[side][comp][tau]=parse_item(side,f"{comp}{tau}","data")
            self.advanced_bc=adv
            self._v("bc_mode").set("INDIVIDUAL")
            self._refresh_visibility()
            win.destroy()
        ttk.Button(host,text="Сохранить Custom КУ",command=lambda:self._safe_dialog_call(save)).grid(
            row=len(rows)+2,column=0,columnspan=6,pady=10
        )

    def _parse_factors(self):
        mode=self.vars.get("load_step_mode",tk.StringVar(value="MANUAL_LIST")).get()
        if mode=="RANGE":
            return self._range_factors()
        if mode=="COUNT":
            return self._count_factors()
        import re
        vals=[]
        for x in re.split(r"[,;\s]+",self.vars["load_factors"].get().strip()):
            if x.strip():vals.append(float(x.strip()))
        if not vals or not np.all(np.isfinite(vals)):raise ValueError("Нужен непустой конечный список коэффициентов нагрузки")
        return vals

    def _load_definition(self):
        typ=self.vars["load_type"].get(); spec={"type":typ,"kinematics":"DEAD_REFERENCE"}
        if typ=="EXTERNAL_NPZ":
            p=self.vars["loads_npz"].get().strip()
            if not p:raise ValueError("Для EXTERNAL_NPZ выберите файл нагрузки")
            spec["npz"]=p; return spec
        spec["load_factors"]=self._parse_factors()
        spec["step_definition"]={"mode":self.vars.get("load_step_mode",tk.StringVar(value="MANUAL_LIST")).get()}
        if spec["step_definition"]["mode"]=="RANGE":
            spec["step_definition"].update({"lambda_start":float(self.vars["load_lambda_start"].get()),"lambda_step":float(self.vars["load_lambda_step"].get()),"lambda_end":float(self.vars["load_lambda_end"].get())})
        elif spec["step_definition"]["mode"]=="COUNT":
            n=_int(self.vars["load_step_count"].get(),"число шагов",1)
            b=_float(self.vars["load_lambda_end"].get(),"λ end",positive=True)
            spec["step_definition"].update({"lambda_end":b,"step_count":n,"lambda_step":b/float(n)})
        if self.vars["theory"].get()=="CUF":spec["face"]=self.vars["load_face"].get()
        x1=self.vars["load_x1"].get().strip(); x2=self.vars["load_x2"].get().strip()
        if x1:spec["x1"]=float(x1)
        if x2:spec["x2"]=float(x2)
        if typ in ("UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE","PATCH_RADIAL_FACE"):
            qmag=_float(self.vars["load_q"].get(),"q",nonnegative=True)
            direction=self.vars.get("load_pressure_direction",tk.StringVar(value="INTERNAL")).get().upper()
            if direction not in ("INTERNAL","EXTERNAL"): raise ValueError("Направление давления должно быть INTERNAL или EXTERNAL")
            spec["pressure_direction"]=direction
            spec["q_magnitude"]=qmag
            spec["q"]=qmag if direction=="INTERNAL" else -qmag
        if typ=="HARMONIC_RADIAL_FACE":
            spec["m"]=_int(self.vars["load_harmonic_m"].get(),"m",1); spec["n"]=_int(self.vars["load_harmonic_n"].get(),"n",0)
        if typ in ("PATCH_RADIAL_FACE","HARMONIC_RADIAL_FACE","LINE_RADIAL_GENERATOR"):
            spec["circumferential_coordinate"]=self.vars["load_coord"].get()
        if typ in ("PATCH_RADIAL_FACE","HARMONIC_RADIAL_FACE"):
            spec["full_circumference"]=bool(self.load_full_var.get())
            if not spec["full_circumference"]:
                spec["c1"]=_float(self.vars["load_c1"].get(),"c1"); spec["c2"]=_float(self.vars["load_c2"].get(),"c2")
        if typ=="POINT_RADIAL_FORCE":
            Pmag=_float(self.vars["load_point_P"].get(),"P",nonnegative=True)
            direction=self.vars.get("load_pressure_direction",tk.StringVar(value="INTERNAL")).get().upper()
            if direction not in ("INTERNAL","EXTERNAL"): raise ValueError("Направление силы должно быть INTERNAL или EXTERNAL")
            spec["pressure_direction"]=direction
            spec["P_magnitude"]=Pmag
            spec["P"]=Pmag if direction=="INTERNAL" else -Pmag
            spec["x0"]=_float(self.vars["load_point_x0"].get(),"x0")
            spec["theta0"]=_float(self.vars["load_point_theta0"].get(),"theta0")
        if typ=="LINE_RADIAL_GENERATOR":
            pmag=_float(self.vars["load_p_line"].get(),"p_line",nonnegative=True)
            direction=self.vars.get("load_pressure_direction",tk.StringVar(value="INTERNAL")).get().upper()
            if direction not in ("INTERNAL","EXTERNAL"): raise ValueError("Направление давления должно быть INTERNAL или EXTERNAL")
            spec["pressure_direction"]=direction
            spec["p_line_magnitude"]=pmag
            spec["p_line"]=pmag if direction=="INTERNAL" else -pmag
            spec["c0"]=_float(self.vars["load_c0"].get(),"c0")
            spec["band_width"]=_float(self.vars["load_band_width"].get(),"band_width",positive=True)
        return spec

    def _parse_curve_points(self):
        rows=[]
        for ln in self.curve_points.get("1.0","end").splitlines():
            s=ln.strip()
            if not s or s.startswith("#"):continue
            parts=s.replace(";"," ").replace(","," ").split()
            if len(parts)<2:raise ValueError(f"Строка кривой должна содержать ε и σ: {ln}")
            rows.append([float(parts[0]),float(parts[1])])
        return rows

    def _parse_piecewise(self):
        seg=[]
        for ln in self.curve_piecewise.get("1.0","end").splitlines():
            s=ln.strip()
            if not s or s.startswith("#"):continue
            if "|" in s:a,b=s.split("|",1)
            elif "\t" in ln:a,b=ln.split("\t",1)
            else:raise ValueError("Кусочная функция: каждая строка должна быть 'epsilon_max | formula' (можно вставить 2 столбца из Excel)")
            a=a.strip(); b=b.strip(); mx=None if a.lower() in ("none","null","inf","*") else float(a); seg.append({"epsilon_max":mx,"formula":b})
        return seg

    def _curve_spec(self):
        mode=self.vars["curve_mode"].get(); spec={"mode":mode}
        if mode=="TABLE":spec.update(points=self._parse_curve_points(),extrapolation=self.vars["curve_extrap"].get())
        elif mode=="CSV":
            p=self.vars["curve_csv"].get().strip();
            if not p:raise ValueError("Не выбран CSV/TXT кривой")
            spec.update(path=p,extrapolation=self.vars["curve_extrap"].get())
        elif mode=="FORMULA":spec["formula"]=self.vars["curve_formula"].get().strip()
        elif mode=="PIECEWISE":spec["segments"]=self._parse_piecewise()
        elif mode=="EXTERNAL_PYTHON":spec["material_diagram"]={"module":self.vars["phi_module"].get().strip() or None,"callable":self.vars["phi_callable"].get().strip() or None,"python_path":self.vars["phi_path"].get().strip() or None}
        ys=self.vars["yield_stress"].get().strip(); ye=self.vars["yield_strain"].get().strip()
        if ys:spec["yield_stress"]=float(ys)
        if ye:spec["yield_strain"]=float(ye)
        return spec

    def _parse_control_points(self):
        cps=[]
        for k,ln in enumerate(self.control_points_text.get("1.0","end").splitlines(),1):
            s=ln.strip()
            if not s or s.startswith("#"):continue
            p=[x.strip() for x in s.replace("\t",",").replace(";",",").split(",")]
            if len(p)==2:label=f"P{k}"; xs,ts=p
            elif len(p)>=3:label,xs,ts=p[:3]
            else:raise ValueError("Контрольная точка: label, x, theta_deg")
            cps.append({"label":label,"x":float(xs),"theta":float(ts)*np.pi/180.0})
        return cps

    def _postprocess_spec(self):
        d=self.vars["deformation_scale"].get().strip(); th=self.vars["gb_thresholds"].get().strip()
        return {"deformation_scale":1.0 if not d else float(d),"gb_ratio_thresholds":[float(x.strip()) for x in th.replace(";",",").split(",") if x.strip()],"control_points":self._parse_control_points()}

    def _case_dict(self):
        theory=self.vars["theory"].get(); method=self.vars["method"].get(); mode=self.vars["mode"].get(); birger=mode.endswith("BIRGER"); nonlinear=mode.startswith("NONLINEAR")
        model_tag=bcreg.model_tag(theory,self._v("cuf_model","TE1").get())
        if self.vars["bc_mode"].get()=="SIMPLE":
            bc={
                "model":model_tag,
                "registry_version":bcreg.BC_REGISTRY_VERSION,
                "left":self.vars["bc_left"].get(),
                "right":self.vars["bc_right"].get(),
                "advanced":None,
                "engineering":copy.deepcopy(self.engineering_bc) if isinstance(self.engineering_bc,dict) else None,
            }
        else:
            bc={
                "model":model_tag,
                "registry_version":bcreg.BC_REGISTRY_VERSION,
                "left":None,"right":None,
                "advanced":copy.deepcopy(self.advanced_bc),
                "engineering":None,
            }
        material_source=self.vars.get("material_source",tk.StringVar(value="USER")).get()
        material_name=(self._v("user_material_name").get().strip() if material_source=="USER" else self.vars.get("material_name",tk.StringVar()).get().strip())
        cfg={
            "theory":theory,"method":method,"mode":mode,"rank":int(self.vars["rank"].get()) if method=="MVI" else None,
            "units":{"system":self.vars["unit_system"].get()},
            "geometry":{"L":_float(self.vars["L"].get(),"L",positive=True),"R":_float(self.vars["R"].get(),"R",positive=True),"h":_float(self.vars["h"].get(),"h",positive=True)},
            "material":{"E0":_float(self.vars["E0"].get(),"E0",positive=True),"nu0":_float(self.vars["nu0"].get(),"ν0"),"source":material_source,"name":material_name or None},
            "grid":{"nx":_int(self.vars["nx"].get(),"nx",7),"ns":_int(self.vars["ns"].get(),"ns",7),"stencil":_int(self.vars["stencil"].get(),"stencil",3)},
            "quadrature":(
                {"type":"GAUSS","n_gauss":_int(self.vars["n_gauss"].get(),"n_gauss",1)}
                if self._v("quadrature_type","GAUSS").get()=="GAUSS" else
                {"type":"LAYERED",
                 "n_gauss":_int(self.vars["n_gauss"].get(),"n_gauss per layer",1),
                 "layer_bounds":[float(x.strip()) for x in self._v("layer_bounds").get().replace(";",",").split(",") if x.strip()]}
            ),
            "boundary_conditions":bc,"load_definition":self._load_definition(),"loads_npz":self.vars["loads_npz"].get().strip() or None,
            "initial_profiles_npz":self.vars["profiles_npz"].get().strip() or None,
            "initial_state":{"source":self.vars["state_source"].get() if (method=="FDM" and nonlinear) else None,"npz":self.vars["state_npz"].get().strip() or None},
            "initial_GB":{"source":self.vars["gb_source"].get() if birger else None,"npz":self.vars["gb_npz"].get().strip() or None},
            "inner_mvi":{"projected_tolerance":_float(self.vars["mvi_proj"].get(),"MVI projected tolerance",positive=True) if method=="MVI" else None,"increment_tolerance":_float(self.vars["mvi_inc"].get(),"MVI increment tolerance",positive=True) if method=="MVI" else None,"tangent_fd_step":_float(self.vars["mvi_fd"].get(),"MVI tangent FD step",positive=True) if method=="MVI" else None,"max_halfstep_iterations":_int(self.vars["mvi_half"].get(),"MVI max halfstep",1) if method=="MVI" else None,"max_xy_cycles":_int(self.vars["mvi_cycles"].get(),"MVI max cycles",1) if method=="MVI" else None,"detf_guarded_correction":True if method=="MVI" else None,"max_correction_halvings":24 if method=="MVI" else None},
            "step_predictor":{"mode":self._v("step_predictor","ERROR").get() if method=="MVI" else "ERROR"},
            "linear":{"residual_tolerance_relative":_float(self.vars["linear_res"].get(),"Linear relative residual tolerance",positive=True) if (method=="FDM" and mode=="LINEAR_ELASTIC") else None},
            "newton":{"residual_tolerance_relative":_float(self.vars["newton_res"].get(),"Newton relative residual tolerance",positive=True) if (method=="FDM" and nonlinear) else None,"increment_tolerance":_float(self.vars["newton_inc"].get(),"Newton increment tolerance",positive=True) if (method=="FDM" and nonlinear) else None,"fd_step":_float(self.vars["newton_fd"].get(),"Newton relative FD step",positive=True) if (method=="FDM" and nonlinear) else None,"max_iterations":_int(self.vars["newton_max"].get(),"Newton max iterations",1) if (method=="FDM" and nonlinear) else None},
            "birger":{"epsilon_tol":_float(self.vars["epsilon_tol"].get(),"epsilon_tol",nonnegative=True) if birger else None,"omega":_float(self.vars["omega"].get(),"omega",positive=True) if birger else None,"G_ref":_float(self.vars["G_ref"].get(),"G_ref",positive=True) if birger else None,"sigma_ref":_float(self.vars["sigma_ref"].get(),"sigma_ref",positive=True) if birger else None,"tol_G":_float(self.vars["tol_G"].get(),"tol_G",positive=True) if birger else None,"tol_sigma":_float(self.vars["tol_sigma"].get(),"tol_sigma",positive=True) if birger else None,"tol_u":_float(self.vars["tol_u"].get(),"tol_u",positive=True) if birger else None,"max_outer_iterations":_int(self.vars["max_outer"].get(),"max_outer_iterations",1) if birger else None,"epsilon_i_limit":_float(self.vars["eps_limit"].get(),"epsilon_i_limit",positive=True) if birger else None,"delta_J_limit":_float(self.vars["dJ_limit"].get(),"delta_J_limit",positive=True) if birger else None,"limit_policy":"REJECT" if birger else None,"initial_slope_rtol":_float(self.vars["initial_slope_rtol"].get(),"initial_slope_rtol",positive=True) if birger else None,"reject_local_unloading":True if birger else None,"unloading_tolerance":1.0e-12 if birger else None,"curve":self._curve_spec() if birger else None},
            "postprocess":self._postprocess_spec(),"output_dir":self.vars["output_dir"].get().strip() or None,
        }
        if theory=="CUF":cfg["cuf_model"]=self.vars["cuf_model"].get()
        return cfg

    def _preview_load_map(self):
        try:
            cfg=self._case_dict(); spec=cfg["load_definition"]
            if spec["type"]=="EXTERNAL_NPZ":raise ValueError("Для EXTERNAL_NPZ карта определяется содержимым файла; используйте проверку CASE/результаты.")
            basis=tebasis.get_basis(cfg.get("cuf_model","TE1")) if cfg["theory"]=="CUF" else None
            _,_,meta,radial=loadbuilder.build_generated_loads(spec,theory=cfg["theory"],L=cfg["geometry"]["L"],R=cfg["geometry"]["R"],h=cfg["geometry"]["h"],nx=cfg["grid"]["nx"],ns=cfg["grid"]["ns"],stencil=cfg["grid"]["stencil"],basis=basis)
            self.load_meta_label.configure(text=json.dumps(meta,ensure_ascii=False))
            win=tk.Toplevel(self); win.title("Карта radial surface load")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            labels=self._gui_units();L=cfg["geometry"]["L"];R=cfg["geometry"]["R"]
            fig=Figure(figsize=(7.5,4.8)); ax=fig.add_subplot(111); im=ax.imshow(radial,aspect="auto",origin="lower",extent=(0.0,L,0.0,2*np.pi*R)); ax.set_xlabel(f"x [{labels['length']}]"); ax.set_ylabel(f"s [{labels['length']}]"); ax.set_title(f"{spec['type']}: распределение радиальной нагрузки"); cb=fig.colorbar(im,ax=ax);cb.set_label(f"q [{labels['pressure']}]"); canv=FigureCanvasTkAgg(fig,master=win); canv.draw(); canv.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:messagebox.showerror("Предпросмотр нагрузки",str(exc))

    def _preview_birger_curve(self):
        try:
            spec=self._curve_spec(); mode=spec["mode"]
            if mode=="EXTERNAL_PYTHON":raise ValueError("Внешний Python callable проверяется dispatcher при валидации CASE; локальный график без импорта не строится.")
            mx=self.vars["curve_preview_max"].get().strip(); mx=None if not mx else float(mx)
            x,y=birgercurve.sample_curve(spec,base_dir=Path(self.vars["case_path"].get()).parent if self.vars["case_path"].get().strip() else APP_DIR,epsilon_max=mx,n=300)
            self.curve_status.configure(text=f"Кривая PASS: ε∈[{x.min():.6g},{x.max():.6g}], σ∈[{y.min():.6g},{y.max():.6g}]")
            win=tk.Toplevel(self); win.title("Кривая Биргера σ_i = Φ(ε_i)")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            fig=Figure(figsize=(7,4.8)); ax=fig.add_subplot(111); ax.plot(x,y); ax.set_xlabel("ε_i [-]"); ax.set_ylabel(f"σ_i [{self._gui_units()['stress']}]"); ax.grid(True,alpha=.3); ax.set_title("Материальная кривая Биргера"); canv=FigureCanvasTkAgg(fig,master=win); canv.draw(); canv.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:self.curve_status.configure(text=f"FAIL: {exc}"); messagebox.showerror("Кривая Биргера",str(exc))

    def _schedule_problem_scheme_redraw(self, delay_ms=180):
        """Debounced GUI-only redraw.

        Entry widgets emit a trace on every typed character.  Redrawing on every
        intermediate value made the preview flicker/disappear while editing.
        This schedules one redraw after the user pauses briefly.
        """
        try:
            old=getattr(self,"_problem_scheme_after_id",None)
            if old:
                try:self.after_cancel(old)
                except Exception:pass
            self._problem_scheme_after_id=self.after(
                max(1,int(delay_ms)), self._safe_problem_scheme_redraw
            )
        except Exception:
            pass

    def _safe_problem_scheme_redraw(self):
        self._problem_scheme_after_id=None
        try:
            self._draw_problem_scheme()
        except Exception as exc:
            # Preview errors must never break input editing or calculation.
            try:self._log(f"PREVIEW WARNING: {type(exc).__name__}: {exc}\n")
            except Exception:pass

    def _draw_problem_scheme(self):
        if not hasattr(self,"scheme_canvas"):return
        c=self.scheme_canvas; c.delete("all")
        w=max(c.winfo_width(),420); h=max(c.winfo_height(),230)
        x0=70; x1=w-70; cy=h/2; ry=min(75,h*.32)
        c.create_rectangle(x0,cy-ry,x1,cy+ry,outline="black",width=2)
        c.create_oval(x0-18,cy-ry,x0+18,cy+ry,outline="black",width=2)
        c.create_oval(x1-18,cy-ry,x1+18,cy+ry,outline="black",width=2)
        c.create_text((x0+x1)/2,25,text=f"{self.vars.get('theory',tk.StringVar(value='')).get()} {self.vars.get('cuf_model',tk.StringVar(value='')).get() if self.vars.get('theory',tk.StringVar()).get()=='CUF' else ''}")
        l=self.vars.get("bc_left",tk.StringVar()).get()
        r=self.vars.get("bc_right",tk.StringVar()).get()
        theory_now=self.vars.get("theory",tk.StringVar(value="CUF")).get()
        try:
            l_show=bcreg.resolved_preset_code(l,theory_now) if l else self.vars.get('bc_mode',tk.StringVar()).get()
            r_show=bcreg.resolved_preset_code(r,theory_now) if r else self.vars.get('bc_mode',tk.StringVar()).get()
        except Exception:
            l_show=l or self.vars.get('bc_mode',tk.StringVar()).get()
            r_show=r or self.vars.get('bc_mode',tk.StringVar()).get()
        c.create_text(x0,cy+ry+20,text=f"x=0: {l_show}")
        c.create_text(x1,cy+ry+20,text=f"x=L: {r_show}")

        typ=self.vars.get("load_type",tk.StringVar(value="")).get()
        direction=self.vars.get("load_pressure_direction",tk.StringVar(value="INTERNAL")).get().upper()
        direction_ru=("ВНЕШНЕЕ: снаружи → внутрь" if direction=="EXTERNAL"
                      else "ВНУТРЕННЕЕ: изнутри → наружу")
        c.create_text((x0+x1)/2,h-18,text=f"Нагрузка: {typ} · {direction_ru}")

        def _fvar(key,default):
            try:
                s=self.vars.get(key,tk.StringVar(value="")).get().strip()
                return float(s) if s else float(default)
            except Exception:
                return float(default)

        Lval=max(_fvar("L",1.0),1e-30)
        Rval=max(_fvar("R",1.0),1e-30)
        rx1=max(0.0,min(Lval,_fvar("load_x1",0.0)))
        rx2=max(0.0,min(Lval,_fvar("load_x2",Lval)))
        if rx2<rx1: rx1,rx2=rx2,rx1
        limited_x=typ in ("PATCH_RADIAL_FACE","HARMONIC_RADIAL_FACE","LINE_RADIAL_GENERATOR")
        ax0=(x0+45)+(x1-x0-90)*(rx1/Lval if limited_x else 0.0)
        ax1=(x0+45)+(x1-x0-90)*(rx2/Lval if limited_x else 1.0)

        if typ!="EXTERNAL_NPZ":
            for xx in np.linspace(ax0,ax1,6):
                if direction=="EXTERNAL":
                    c.create_line(xx,cy-ry-30,xx,cy-ry+1,arrow=tk.LAST,width=1.4)
                else:
                    c.create_line(xx,cy-ry+22,xx,cy-ry-18,arrow=tk.LAST,width=1.4)

        # Main right-side preview: cylinder, not an unwrapped rectangle.
        if hasattr(self,"load_preview_canvas"):
            p=self.load_preview_canvas; p.delete("all")
            pw=max(p.winfo_width(),430); ph=max(p.winfo_height(),360)

            margin=55
            lx=margin+35
            rx=pw-margin-35
            midy=ph*0.47
            rad=min(ph*0.22,72)
            cap=26

            # Cylinder body with simple perspective.
            p.create_line(lx,midy-rad,rx,midy-rad,width=2)
            p.create_line(lx,midy+rad,rx,midy+rad,width=2)
            p.create_oval(lx-cap,midy-rad,lx+cap,midy+rad,outline="black",width=2)
            p.create_oval(rx-cap,midy-rad,rx+cap,midy+rad,outline="black",width=2)
            p.create_text(pw/2,25,text="Предпросмотр нагрузки на цилиндре",font=("Segoe UI Semibold",10))

            def xpix(x):
                return lx+(rx-lx)*max(0.0,min(1.0,float(x)/Lval))

            # Approximate visible circumferential placement by vertical location
            # on the shell silhouette.  Exact field remains available through
            # the dedicated q(x,theta) map button.
            coord_mode=self.vars.get("load_coord",tk.StringVar(value="THETA")).get().upper()
            period=2*np.pi if coord_mode=="THETA" else 2*np.pi*Rval

            def coord_to_y(cc):
                z=(float(cc)%period)/period
                return midy-rad*np.cos(2*np.pi*z)

            def circumference_span():
                if typ in ("PATCH_RADIAL_FACE","HARMONIC_RADIAL_FACE"):
                    full=bool(getattr(self,"load_full_var",tk.BooleanVar(value=True)).get())
                    if full:
                        return None
                    return (_fvar("load_c1",0.0),_fvar("load_c2",period))
                if typ=="LINE_RADIAL_GENERATOR":
                    c0=_fvar("load_c0",0.0)
                    bw=max(_fvar("load_band_width",0.0),0.0)
                    half=(0.5*bw/Rval if coord_mode=="THETA" else 0.5*bw)
                    return (c0-half,c0+half)
                return None

            # Draw region on cylinder.
            if typ in ("PATCH_RADIAL_FACE","HARMONIC_RADIAL_FACE"):
                xa=xpix(rx1); xb=xpix(rx2)
                span=circumference_span()
                if span is None:
                    ytop=midy-rad+5; ybot=midy+rad-5
                else:
                    ys=[coord_to_y(span[0]),coord_to_y(span[1])]
                    ytop=max(midy-rad+5,min(ys)-12)
                    ybot=min(midy+rad-5,max(ys)+12)
                    if abs(ybot-ytop)<20:
                        ytop=max(midy-rad+5,(ytop+ybot)/2-12)
                        ybot=min(midy+rad-5,(ytop+ybot)/2+12)
                p.create_rectangle(xa,ytop,xb,ybot,outline="#444",width=2,dash=(5,3) if typ=="HARMONIC_RADIAL_FACE" else None)
                if typ=="PATCH_RADIAL_FACE":
                    p.create_text((xa+xb)/2,ytop-12,text=f"PATCH  a={rx2-rx1:.4g}",font=("Segoe UI",9))
                else:
                    p.create_text((xa+xb)/2,ytop-12,text=f"HARMONIC  m={int(round(_fvar('load_harmonic_m',1)))}  n={int(round(_fvar('load_harmonic_n',0)))}",font=("Segoe UI",9))

            elif typ=="LINE_RADIAL_GENERATOR":
                xa=xpix(rx1); xb=xpix(rx2)
                c0=_fvar("load_c0",0.0)
                y0=coord_to_y(c0)
                bw=max(_fvar("load_band_width",0.0),0.0)
                visual=max(8,min(26,8+20*bw/max(2*np.pi*Rval,1e-30)))
                p.create_rectangle(xa,y0-visual/2,xb,y0+visual/2,outline="#444",width=2)
                p.create_text((xa+xb)/2,y0-visual/2-12,text=f"LINE/BAND  b={bw:.4g}",font=("Segoe UI",9))

            # Load arrows: fewer and larger.
            if typ!="EXTERNAL_NPZ":
                xs=np.linspace(xpix(rx1 if limited_x else 0.0),xpix(rx2 if limited_x else Lval),5)
                if typ=="HARMONIC_RADIAL_FACE":
                    m=max(1,int(round(_fvar("load_harmonic_m",1))))
                    for j,xx in enumerate(xs):
                        phase=np.sin(m*np.pi*(rx1+(rx2-rx1)*j/max(len(xs)-1,1))/Lval)
                        sgn=1 if phase>=0 else -1
                        if direction=="EXTERNAL": sgn*=-1
                        if sgn>0:
                            p.create_line(xx,midy-rad+25,xx,midy-rad-18,arrow=tk.LAST,width=1.6)
                        else:
                            p.create_line(xx,midy-rad-24,xx,midy-rad+10,arrow=tk.LAST,width=1.6)
                else:
                    for xx in xs:
                        if direction=="EXTERNAL":
                            p.create_line(xx,midy-rad-28,xx,midy-rad+8,arrow=tk.LAST,width=1.6)
                        else:
                            p.create_line(xx,midy-rad+24,xx,midy-rad-18,arrow=tk.LAST,width=1.6)

            # Dimension / position summary.
            details=[]
            if limited_x:
                details.append(f"x1={rx1:.6g}, x2={rx2:.6g}, a={rx2-rx1:.6g}")
            if typ in ("PATCH_RADIAL_FACE","HARMONIC_RADIAL_FACE"):
                full=bool(getattr(self,"load_full_var",tk.BooleanVar(value=True)).get())
                if full:
                    details.append("по окружности: вся поверхность")
                else:
                    details.append(f"c1={_fvar('load_c1',0):.6g}, c2={_fvar('load_c2',period):.6g}")
            if typ=="LINE_RADIAL_GENERATOR":
                details.append(f"центр={_fvar('load_c0',0):.6g}, b={_fvar('load_band_width',0):.6g}")
            p.create_text(pw/2,ph-52,text=";  ".join(details) if details else "Равномерная нагрузка по всей оболочке",font=("Segoe UI",9))
            p.create_text(pw/2,ph-28,text=f"{direction_ru}. Точная карта распределения — кнопка «Карта q(x,θ)».",font=("Segoe UI",9))

    def run_case(self):
        if self.proc is not None and self.proc.poll() is None:messagebox.showwarning("Расчёт","Расчёт уже выполняется");return
        try:
            cfg=self._case_dict(); p=self._case_path_for_save();
            if p is None:return
            p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8"); validate_final_case(p)
            out=Path(cfg["output_dir"]); out=out if out.is_absolute() else p.parent/out; self.run_output_dir=out.resolve(); ld=cfg["load_definition"]
            if ld["type"]=="EXTERNAL_NPZ":
                lp=Path(ld["npz"]); lp=lp if lp.is_absolute() else p.parent/lp
                with np.load(lp,allow_pickle=False) as z:self.expected_steps=int(np.asarray(z["loads"]).shape[0])
            else:self.expected_steps=len(ld["load_factors"])
            self.run_kind="CASE"; cmd=[sys.executable,str(APP_DIR/"RUN_FINAL_CASE.py"),str(p)]; self._log("\n=== START ===\n"+" ".join(cmd)+"\n"); self.run_btn.configure(state="disabled"); self._progress_reset(self.expected_steps)
            self.proc=subprocess.Popen(cmd,cwd=str(APP_DIR),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding="utf-8",errors="replace"); self._set_top_running_state(True); threading.Thread(target=self._read_process_output,daemon=True).start()
        except Exception as exc:
            self._set_top_running_state(False)
            messagebox.showerror("Запуск — ошибка",str(exc)); self._log(f"START FAIL: {type(exc).__name__}: {exc}\n")

    def _on_process_done(self,rc):
        self._set_top_running_state(False)
        if bool(getattr(self,"_stop_requested",False)):
            self.progress.stop()
            try:self.run_btn.configure(state="normal")
            except Exception:pass
            try:self.series_btn.configure(state="normal" if self.vars["method"].get()=="MVI" else "disabled")
            except Exception:pass
            self._online_live_point=None
            try:self._online_refresh_steps()
            except Exception:pass
            try:
                accepted=len(getattr(self,"_online_step_cache",{}) or {});total=int(self.expected_steps or 0);self._progress_update(accepted,total,label_prefix=f"Остановлено; принято шагов: {accepted}")
            except Exception:pass
            n=max(len(getattr(self,"_online_step_cache",{}) or {}),len(self._accepted_step_numbers_on_disk()))
            if self.expected_steps:self._progress_update(n,self.expected_steps,label_prefix="Остановлено")
            else:self.progress_label.configure(text=f"Остановлено; сохранено шагов: {n}")
            try:self._online_set("stage","STOPPED BY USER")
            except Exception:pass
            self._log(f"=== STOPPED BY USER; completed steps preserved: {n} ===\n")
            self._stop_requested=False
            if n>0 and self.run_output_dir:
                if self.run_kind=="MVI_SERIES":
                    self._log("Серия остановлена. Завершённые варианты сохранены; постобработка автоматически не запускается.\n")
                else:
                    self._log("Остановлено. Принятые шаги сохранены. FAST / STANDARD / FULL запускаются только вручную.\n")
            elif self.run_kind=="CASE":
                self._log("STOP произошёл до первого принятого шага: корректных полей для постобработки ещё нет.\n")
            return
        if rc!=0 and self.run_kind=="CASE" and self.run_output_dir:
            summary={}
            for name in ("FINAL_RUN_SUMMARY.json","RUN_RESULT.json","SERIES_SUMMARY.json"):
                p=Path(self.run_output_dir)/name
                if not p.exists():continue
                try:
                    summary=json.loads(p.read_text(encoding="utf-8"))
                    break
                except Exception:
                    summary={}
            status=str(summary.get("status",""))
            try:accepted=int(summary.get("accepted_steps",0) or summary.get("completed_steps",0) or 0)
            except Exception:accepted=0
            if accepted>0 and status in ("LOAD_SERIES_STOPPED","PARTIAL_LOAD_SERIES"):
                self.progress.stop()
                try:self.run_btn.configure(state="normal")
                except Exception:pass
                try:self.series_btn.configure(state="normal" if self.vars["method"].get()=="MVI" else "disabled")
                except Exception:pass
                try:self._online_refresh_steps()
                except Exception:pass
                n=len(getattr(self,"_online_step_cache",{}) or {}) or accepted
                if self.expected_steps:self._progress_update(n,self.expected_steps,label_prefix="Остановлено")
                else:self.progress_label.configure(text=f"Остановлено; сохранено шагов: {n}")
                try:self._online_set("stage",status)
                except Exception:pass
                failed=summary.get("failed_step_index")
                reason=summary.get("failed_step_status") or status
                code=summary.get("failed_step_code") or ""
                diagnostic=summary.get("failed_step_diagnostic_ru") or ""
                recommendation=summary.get("failed_step_recommendation_ru") or ""
                extra=("; "+diagnostic) if diagnostic else ""
                self._log(f"=== LOAD SERIES STOPPED; accepted steps preserved: {n}; failed_step={failed}; code={code}; reason={reason}{extra} ===\n")
                self._log("Принятые шаги сохранены. FAST / STANDARD / FULL запускаются только вручную.\n")
                details=[]
                if code: details.append(f"Тип: {code}")
                if diagnostic: details.append(diagnostic)
                if recommendation: details.append(recommendation)
                if not details: details.append(f"Причина: {reason}")
                messagebox.showwarning(
                    "Расчёт остановлен",
                    f"Серия остановлена на шаге {failed}; принято шагов: {n}.\n"
                    + "\n".join(details)
                    + "\n\n"
                    "Принятые шаги сохранены. FAST / STANDARD / FULL запускаются только вручную."
                )
                return
        super()._on_process_done(rc)
        try:self._online_refresh_steps()
        except Exception:pass
        if rc==0 and self.run_output_dir:
            try:self.refresh_results_view()
            except Exception:pass
        if rc!=0 and self.run_kind=="CASE" and self.run_output_dir:
            n=max(len(getattr(self,"_online_step_cache",{}) or {}),len(self._accepted_step_numbers_on_disk()))
            if n>0:
                if self.expected_steps:self._progress_update(n,self.expected_steps,label_prefix="Остановлено")
                self._log(f"=== NONCONVERGED; accepted steps preserved: {n}; postprocess is manual ===\n")
                self._log("FAST / STANDARD / FULL доступны для сохранённых принятых шагов и запускаются вручную.\n")

    def refresh_results_view(self):
        try:
            p=self.run_output_dir
            if p is None:
                s=self.vars.get("output_dir",tk.StringVar()).get().strip();
                if not s:return
                p=Path(s)
            p=Path(p).resolve(); report=p/"REPORT"/"CALCULATION_SUMMARY.json"
            if report.exists():
                j=json.loads(report.read_text(encoding="utf-8")); system=str((j.get("units") or {}).get("system",self._v("unit_system","N_MM_MPA").get())); du=funits.unit_labels(system)["displacement"]; self.result_summary.configure(text=f"status={j.get('status')} | theory={j.get('theory')} {j.get('cuf_model') or ''} | method={j.get('method')} | mode={j.get('mode')} | steps={j.get('steps')} | STEP={j.get('selected_state_file')} | |w|max={j.get('w_abs_max_final')} {du} | w_ref={j.get('w_reference_final')} {du} | w_center={j.get('w_center_final')} {du} | derived_error={j.get('derived_error')}")
            else:self.result_summary.configure(text=f"Папка: {p}\nREPORT/CALCULATION_SUMMARY.json ещё не найден.")
            paths=sorted(list((p/"GRAPHS").rglob("*.png"))+list((p/"3D").rglob("*.png"))) if p.exists() else []
            self._result_png_paths=paths; self.result_list.delete(0,"end")
            for q in paths:self.result_list.insert("end",str(q.relative_to(p)))
        except Exception as exc:self.result_summary.configure(text=f"Ошибка чтения результатов: {exc}")

    def _preview_selected_result(self):
        try:
            sel=self.result_list.curselection();
            if not sel:return
            path=self._result_png_paths[sel[0]]
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.image as mpimg
            if self._result_figure_canvas is not None:
                self._result_figure_canvas.get_tk_widget().destroy(); self._result_figure_canvas=None
            try:self.result_preview_label.pack_forget()
            except Exception:pass
            fig=Figure(figsize=(7,5)); ax=fig.add_subplot(111); ax.imshow(mpimg.imread(path)); ax.axis("off"); ax.set_title(path.name); fig.tight_layout(); self._result_current_figure=fig; self._result_figure_canvas=FigureCanvasTkAgg(fig,master=self.result_preview_frame); self._result_figure_canvas.draw(); self._result_figure_canvas.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:messagebox.showerror("Просмотр графика",str(exc))

    def load_case(self):
        p=filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not p:return
        try:
            cfg=json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(cfg.get("newton"),dict) and "residual_tolerance" in cfg["newton"]:
                raise ValueError("V2: старое поле newton.residual_tolerance (абсолютное) запрещено; используйте residual_tolerance_relative, рекомендуемо 1e-9")
            self.vars["case_path"].set(p)
            if cfg.get("theory")=="CUF_TE1":cfg["theory"]="CUF";cfg["cuf_model"]="TE1"
            bc_loaded=cfg.get("boundary_conditions") or {}
            if isinstance(bc_loaded,dict):
                saved_model=bc_loaded.get("model")
                expected_model=bcreg.model_tag(cfg.get("theory","CUF"),cfg.get("cuf_model","TE1"))
                if saved_model not in (None,"") and str(saved_model).upper().strip()!=expected_model:
                    raise ValueError(
                        f"CASE содержит КУ для {saved_model}, но активная кинематическая модель {expected_model}. "
                        "Краевые условия необходимо пересоздать для выбранной модели."
                    )
            paths={"theory":("theory",),"cuf_model":("cuf_model",),"method":("method",),"mode":("mode",),"rank":("rank",),"unit_system":("units","system"),"L":("geometry","L"),"R":("geometry","R"),"h":("geometry","h"),"E0":("material","E0"),"nu0":("material","nu0"),"nx":("grid","nx"),"ns":("grid","ns"),"stencil":("grid","stencil"),"quadrature_type":("quadrature","type"),"n_gauss":("quadrature","n_gauss"),"layer_bounds":("quadrature","layer_bounds"),"profiles_npz":("initial_profiles_npz",),"output_dir":("output_dir",),"state_source":("initial_state","source"),"state_npz":("initial_state","npz"),"gb_source":("initial_GB","source"),"gb_npz":("initial_GB","npz"),"mvi_proj":("inner_mvi","projected_tolerance"),"mvi_inc":("inner_mvi","increment_tolerance"),"mvi_fd":("inner_mvi","tangent_fd_step"),"mvi_half":("inner_mvi","max_halfstep_iterations"),"mvi_cycles":("inner_mvi","max_xy_cycles"),"step_predictor":("step_predictor","mode"),"linear_res":("linear","residual_tolerance_relative"),"newton_res":("newton","residual_tolerance_relative"),"newton_inc":("newton","increment_tolerance"),"newton_fd":("newton","fd_step"),"newton_max":("newton","max_iterations"),"epsilon_tol":("birger","epsilon_tol"),"omega":("birger","omega"),"G_ref":("birger","G_ref"),"sigma_ref":("birger","sigma_ref"),"tol_G":("birger","tol_G"),"tol_sigma":("birger","tol_sigma"),"tol_u":("birger","tol_u"),"max_outer":("birger","max_outer_iterations"),"eps_limit":("birger","epsilon_i_limit"),"dJ_limit":("birger","delta_J_limit"),"initial_slope_rtol":("birger","initial_slope_rtol")}
            for key,path in paths.items():
                cur=cfg
                try:
                    for x in path:cur=cur[x]
                except Exception:cur=""
                if key in self.vars:self.vars[key].set("" if cur is None else str(cur))
            # Loading a CASE is not a user unit switch: its values are already
            # expressed in the CASE-declared system.  Reset the GUI conversion
            # anchor before refreshing labels/material presets.
            self._last_unit_system=self.vars.get("unit_system",tk.StringVar(value="N_MM_MPA")).get()
            mj=cfg.get("material",{}) or {}
            self.vars["material_source"].set(str(mj.get("source","USER")))
            self.vars["material_name"].set(str(mj.get("name") or ""))
            self._refresh_material_mode()
            bc=cfg.get("boundary_conditions",{}); self.advanced_bc=bc.get("advanced"); self.engineering_bc=bc.get("engineering")
            if self.advanced_bc is None:self.vars["bc_mode"].set("SIMPLE");self.vars["bc_left"].set(str(bc.get("left") or ""));self.vars["bc_right"].set(str(bc.get("right") or ""))
            else:self.vars["bc_mode"].set("INDIVIDUAL")
            ld=cfg.get("load_definition")
            if isinstance(ld,dict):
                typ=str(ld.get("type","EXTERNAL_NPZ"))
                self.vars["load_type"].set(typ)
                self.vars["loads_npz"].set(str(ld.get("npz") or cfg.get("loads_npz") or ""))
                self.vars["load_factors"].set(",".join(str(x) for x in ld.get("load_factors",[1])))
                sd=ld.get("step_definition",{}) or {}
                self.vars["load_step_mode"].set(str(sd.get("mode","MANUAL_LIST")))
                self.vars["load_lambda_start"].set(str(sd.get("lambda_start",0.1)))
                self.vars["load_lambda_step"].set(str(sd.get("lambda_step",0.1)))
                self.vars["load_lambda_end"].set(str(sd.get("lambda_end",1.0)))
                default_surface="OUTER" if theory=="CUF" else "MIDDLE"
                self.vars["load_face"].set(str(ld.get("surface",ld.get("face",default_surface))))
                qstored=ld.get("q")
                pstored=ld.get("p_line")
                Pstored=ld.get("P")
                self.vars["load_q"].set("" if qstored is None else str(abs(float(ld.get("q_magnitude",qstored)))))
                self.vars["load_harmonic_m"].set(str(ld.get("m",1)))
                self.vars["load_harmonic_n"].set(str(ld.get("n",4)))
                self.vars["load_p_line"].set("" if pstored is None else str(abs(float(ld.get("p_line_magnitude",pstored)))))
                self.vars["load_point_P"].set("" if Pstored is None else str(abs(float(ld.get("P_magnitude",Pstored)))))
                self.vars["load_point_x0"].set(str(ld.get("x0","") if ld.get("x0") is not None else ""))
                self.vars["load_point_theta0"].set(str(ld.get("theta0","") if ld.get("theta0") is not None else ""))
                direction=str(ld.get("pressure_direction") or ("EXTERNAL" if ((qstored is not None and float(qstored)<0) or (pstored is not None and float(pstored)<0) or (Pstored is not None and float(Pstored)<0)) else "INTERNAL")).upper()
                self.vars["load_pressure_direction"].set(direction)
                self.vars["load_x1"].set(str(ld.get("x1","") if ld.get("x1") is not None else ""))
                self.vars["load_x2"].set(str(ld.get("x2","") if ld.get("x2") is not None else ""))
                self.vars["load_coord"].set(str(ld.get("circumferential_coordinate","THETA")))
                self.vars["load_c1"].set(str(ld.get("c1","") if ld.get("c1") is not None else ""))
                self.vars["load_c2"].set(str(ld.get("c2","") if ld.get("c2") is not None else ""))
                self.vars["load_c0"].set(str(ld.get("c0","") if ld.get("c0") is not None else ""))
                self.vars["load_band_width"].set(str(ld.get("band_width","") if ld.get("band_width") is not None else ""))
                self.load_full_var.set(bool(ld.get("full_circumference",False)))
                if str(sd.get("mode","")).upper()=="COUNT":
                    sc=sd.get("step_count")
                    if sc is None:sc=len(ld.get("load_factors") or [])
                    if sc:self.vars["load_step_count"].set(str(sc))
            else:self.vars["load_type"].set("EXTERNAL_NPZ");self.vars["loads_npz"].set(str(cfg.get("loads_npz") or "")); self.vars["load_step_mode"].set("MANUAL_LIST")
            bj=cfg.get("birger",{}) or {}; curve=bj.get("curve") if isinstance(bj,dict) else None
            if isinstance(curve,dict):
                cm=str(curve.get("mode","TABLE"));self.vars["curve_mode"].set(cm);self.vars["curve_extrap"].set(str(curve.get("extrapolation","ERROR")));self.vars["yield_stress"].set(str(curve.get("yield_stress","") if curve.get("yield_stress") is not None else ""));self.vars["yield_strain"].set(str(curve.get("yield_strain","") if curve.get("yield_strain") is not None else ""))
                self.curve_points.delete("1.0","end");
                if cm=="TABLE":self.curve_points.insert("1.0","\n".join(f"{a[0]}, {a[1]}" for a in curve.get("points",[])))
                self.vars["curve_csv"].set(str(curve.get("path","") or ""));self.vars["curve_formula"].set(str(curve.get("formula","") or ""));self.curve_piecewise.delete("1.0","end")
                if cm=="PIECEWISE":self.curve_piecewise.insert("1.0","\n".join(f"{('None' if s.get('epsilon_max') is None else s.get('epsilon_max'))} | {s.get('formula','')}" for s in curve.get("segments",[])))
                md=curve.get("material_diagram",{}) if cm=="EXTERNAL_PYTHON" else {}; self.vars["phi_module"].set(str(md.get("module","") or ""));self.vars["phi_callable"].set(str(md.get("callable","") or ""));self.vars["phi_path"].set(str(md.get("python_path","") or ""))
            pp=cfg.get("postprocess",{}) or {};self.vars["deformation_scale"].set(str(pp.get("deformation_scale","") if pp.get("deformation_scale") is not None else ""));self.vars["gb_thresholds"].set(",".join(str(x) for x in pp.get("gb_ratio_thresholds",[])));self.control_points_text.delete("1.0","end");self.control_points_text.insert("1.0","\n".join(f"{x.get('label','')}, {x.get('x')}, {float(x.get('theta',0))*180/np.pi}" for x in pp.get("control_points",[])))
            self._refresh_visibility(); self._update_units(); self._log(f"CASE загружен: {p}\n")
        except Exception as exc:messagebox.showerror("Ошибка загрузки",str(exc))



class FinalSolverGUI(EnhancedSolverGUI):
    """Stage12 final UI fixes. Solver mechanics remain unchanged."""

    def _build_ui(self):
        super()._build_ui()
        # Dedicated live calculation page.  It is a GUI/output layer only; the
        # mechanics are untouched.
        self.tab_online=ttk.Frame(self.notebook,padding=8)
        try: online_index=self.notebook.index(self.tab_results)
        except Exception: online_index="end"
        self.notebook.insert(online_index,self.tab_online,text="Онлайн")
        self._build_online_tab()
        # Always-visible calculation controls. Reuse the existing run/open logic;
        # only STOP is new and terminates the current calculation subprocess.
        bar=self.parameter_toolbar
        ttk.Separator(bar,orient="vertical").pack(side="left",fill="y",padx=10)
        ttk.Label(bar,text="Расчёт:").pack(side="left",padx=(0,6))
        self.top_run_btn=ttk.Button(bar,text="Пуск",command=self.run_case)
        self.top_run_btn.pack(side="left",padx=(0,5))
        self.top_pause_btn=ttk.Button(bar,text="Пауза",command=self.toggle_pause_calculation,state="disabled")
        self.top_pause_btn.pack(side="left",padx=5)
        self.top_stop_btn=ttk.Button(bar,text="Стоп",command=self.stop_calculation,state="disabled")
        self.top_stop_btn.pack(side="left",padx=5)
        self.top_results_btn=ttk.Button(bar,text="Открыть папку результатов",command=self.open_results)
        self.top_results_btn.pack(side="left",padx=5)


    def _build_results_tab(self):
        super()._build_results_tab()
        self._result_current_figure=None
        ttk.Button(self.result_bar,text="Открыть график отдельно",command=self._open_current_result_figure).pack(side="left",padx=6)
        ttk.Button(self.result_bar,text="Открыть Excel",command=self._open_excel_results).pack(side="left",padx=6)
        ttk.Button(self.result_bar,text="FAST / БЫСТРЫЙ",command=lambda:self._run_postprocess("FAST")).pack(side="left",padx=4)
        ttk.Button(self.result_bar,text="STANDARD / СТАНДАРТНЫЙ",command=lambda:self._run_postprocess("STANDARD")).pack(side="left",padx=4)
        ttk.Button(self.result_bar,text="FULL / ПОЛНЫЙ",command=lambda:self._run_postprocess("FULL")).pack(side="left",padx=4)
        ttk.Button(self.result_bar,text="Экспресс-отчёт / Quick Summary",command=self._run_quick_summary).pack(side="left",padx=6)
        ttk.Label(self.result_bar,text="Группа").pack(side="left",padx=(12,3))
        self.result_group_combo=ttk.Combobox(
            self.result_bar,textvariable=self._v("result_group","ВСЕ"),state="readonly",width=18,
            values=("ВСЕ","Общие","Перемещения","Деформации","Напряжения","Результанты","По толщине","Пластичность","Сходимость","CUF diagnostics","Birger","3D"),
        )
        self.result_group_combo.pack(side="left")
        self.result_group_combo.bind("<<ComboboxSelected>>",lambda e:self._filter_result_pngs())
        # Direct numeric viewer: selected load step / field / thickness level.
        bar2=ttk.Frame(self.result_viewer); bar2.grid(row=2,column=0,sticky="ew",pady=(2,5))
        ttk.Label(bar2,text="Шаг").pack(side="left")
        self.result_step_combo=ttk.Combobox(bar2,textvariable=self._v("result_step",""),state="readonly",width=9)
        self.result_step_combo.pack(side="left",padx=(4,8)); self.result_step_combo.bind("<<ComboboxSelected>>",lambda e:self._result_step_changed())
        ttk.Label(bar2,text="Поле").pack(side="left")
        self.result_field_combo=ttk.Combobox(bar2,textvariable=self._v("result_field","w"),state="readonly",width=26)
        self.result_field_combo.pack(side="left",padx=(4,8))
        ttk.Label(bar2,text="z / слой").pack(side="left")
        # Editable: CUF displacement can be reconstructed at an arbitrary z.
        # Stress/strain/material fields remain tied to actual integration
        # levels and use the nearest level when a custom z is entered.
        self.result_z_combo=ttk.Combobox(bar2,textvariable=self._v("result_z","0"),state="normal",width=21)
        self.result_z_combo.pack(side="left",padx=(4,8))
        ttk.Label(bar2,text="τ").pack(side="left")
        self.result_tau_combo=ttk.Combobox(bar2,textvariable=self._v("result_tau","0"),state="readonly",width=5)
        self.result_tau_combo.pack(side="left",padx=(4,8))
        ttk.Button(bar2,text="Показать поле",command=self._preview_numeric_result).pack(side="left",padx=3)

        # Keep thickness/section inspection on its own row.  These controls
        # must never be pushed off-screen by the 3-D/VIDEO buttons.
        self.result_pan.grid_configure(row=6)
        self.result_viewer.rowconfigure(3,weight=0)
        self.result_viewer.rowconfigure(4,weight=0)
        self.result_viewer.rowconfigure(5,weight=0)
        self.result_viewer.rowconfigure(6,weight=1)
        bar3=ttk.Frame(self.result_viewer); bar3.grid(row=3,column=0,sticky="ew",pady=(0,4))
        ttk.Label(bar3,text="По толщине / сечения: x-index").pack(side="left")
        self.result_ix_spin=ttk.Spinbox(bar3,textvariable=self._v("result_ix","0"),from_=0,to=0,width=6)
        self.result_ix_spin.pack(side="left",padx=(4,8))
        ttk.Label(bar3,text="s-index").pack(side="left")
        self.result_is_spin=ttk.Spinbox(bar3,textvariable=self._v("result_is","0"),from_=0,to=0,width=6)
        self.result_is_spin.pack(side="left",padx=(4,10))
        ttk.Button(bar3,text="Профиль по толщине",command=self._preview_thickness_profile).pack(side="left",padx=3)
        ttk.Button(bar3,text="Разрез x-z",command=lambda:self._preview_numeric_cut("x-z")).pack(side="left",padx=3)
        ttk.Button(bar3,text="Разрез s-z",command=lambda:self._preview_numeric_cut("s-z")).pack(side="left",padx=3)
        ttk.Button(bar3,text="Открыть THROUGH_THICKNESS",command=lambda:self._select_result_group("По толщине")).pack(side="left",padx=(10,3))

        # Engineering 3-D modes.  Keep them on two rows so none of the
        # controls can disappear on a normal Windows desktop width.
        bar4=ttk.Frame(self.result_viewer); bar4.grid(row=4,column=0,sticky="ew",pady=(0,3))
        ttk.Label(bar4,text="3D геометрия:").pack(side="left")
        ttk.Button(bar4,text="Цельная оболочка",command=self._preview_numeric_3d).pack(side="left",padx=(8,3))
        ttk.Button(bar4,text="Продольный разрез",command=lambda:self._preview_engineering_3d_cut("longitudinal")).pack(side="left",padx=3)
        ttk.Button(bar4,text="Поперечный разрез",command=lambda:self._preview_engineering_3d_cut("transverse")).pack(side="left",padx=3)
        ttk.Button(bar4,text="Комбинированный разрез",command=lambda:self._preview_engineering_3d_cut("combined")).pack(side="left",padx=3)
        bar5=ttk.Frame(self.result_viewer); bar5.grid(row=5,column=0,sticky="ew",pady=(0,5))
        ttk.Label(bar5,text="Просмотр / экспорт:").pack(side="left")
        ttk.Button(bar5,text="3D объёмное поле",command=self._preview_numeric_volume_3d).pack(side="left",padx=(8,3))
        ttk.Button(bar5,text="Сохранить PNG",command=self._save_current_result_png).pack(side="left",padx=3)
        ttk.Label(bar5,text="Толщина среза ×").pack(side="left",padx=(12,3))
        self._result_visual_thickness_var=tk.StringVar(value="5")
        ttk.Combobox(bar5,textvariable=self._result_visual_thickness_var,state="readonly",width=4,values=("1","2","5","10")).pack(side="left",padx=(0,6))
        ttk.Button(bar5,text="VIDEO",command=self._open_video_export_dialog).pack(side="left",padx=(6,3))
        self._numeric_result_index=None
        self._all_result_png_paths=[]
        self.result_list.bind("<Double-Button-1>",lambda _e:self._open_current_result_figure(),add="+")

    def _open_figure_in_window(self,source_figure,title):
        """Open an independent, resizable copy of any GUI graph."""
        if source_figure is None:
            raise ValueError("Сначала постройте или выберите график")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        win=tk.Toplevel(self)
        win.title(title)
        win.geometry("1100x760")
        win.minsize(650,420)
        win.resizable(True,True)
        try:
            fig=copy.deepcopy(source_figure)
        except Exception:
            source_figure.canvas.draw()
            rgba=np.asarray(source_figure.canvas.buffer_rgba()).copy()
            fig=Figure(figsize=(10,7)); ax=fig.add_subplot(111); ax.imshow(rgba); ax.axis("off")
        canvas=FigureCanvasTkAgg(fig,master=win)
        toolbar=NavigationToolbar2Tk(canvas,win,pack_toolbar=False)
        toolbar.update(); toolbar.pack(side="top",fill="x")
        canvas.draw(); canvas.get_tk_widget().pack(side="top",fill="both",expand=True)
        win._cyl_figure=fig; win._cyl_canvas=canvas; win._cyl_toolbar=toolbar
        win.focus_force()
        return win

    def _open_current_result_figure(self):
        try:
            fig=getattr(self,"_result_current_figure",None)
            if fig is None:
                self._preview_selected_result()
                fig=getattr(self,"_result_current_figure",None)
            self._open_figure_in_window(fig,"График результата — отдельное окно")
        except Exception as exc:
            messagebox.showerror("Отдельный график",str(exc))

    def _open_online_figure(self,key):
        figures={
            "load":("online_fig_load","Нагрузка–прогиб"),
            "shell":("online_fig_shell","3D оболочка"),
            "field":("online_fig_field","Поле / пластичность"),
            "conv":("online_fig_conv","Сходимость"),
        }
        try:
            attr,title=figures[key]
            self._open_figure_in_window(getattr(self,attr,None),title+" — отдельное окно")
        except Exception as exc:
            messagebox.showerror("Отдельный график",str(exc))

    def _open_excel_results(self):
        try:
            out=self._result_output_dir()
            if out is None: raise ValueError("Папка текущего расчёта не определена")
            files=sorted((Path(out)/"EXCEL").glob("*__FULL_RESULTS.xlsx"),key=lambda p:p.stat().st_mtime,reverse=True)
            if not files:files=sorted(Path(out).rglob("*.xlsx"),key=lambda p:p.stat().st_mtime,reverse=True)
            if not files: raise ValueError("Excel ещё не создан. Дождитесь завершения автоматической постобработки.")
            p=files[0]
            if sys.platform.startswith("win"): os.startfile(p)
            elif sys.platform=="darwin": subprocess.Popen(["open",str(p)])
            else: subprocess.Popen(["xdg-open",str(p)])
        except Exception as exc: messagebox.showerror("Excel результатов",str(exc))

    def _select_result_group(self,group):
        """Select a result group and refresh the list without touching data."""
        try:
            self._v("result_group","ВСЕ").set(str(group))
            self._filter_result_pngs()
        except Exception as exc:
            messagebox.showerror("Результаты",str(exc))

    def _filter_result_pngs(self):
        paths=list(getattr(self,"_all_result_png_paths",[]) or [])
        group=self._v("result_group","ВСЕ").get()
        if group!="ВСЕ":
            def belongs(p):
                out=self._result_output_dir()
                try:r=str(Path(p).relative_to(out)).replace("\\","/").upper()
                except Exception:r=str(p).replace("\\","/").upper()
                checks={
                    "Общие":("GRAPHS/LOAD_DEFLECTION/",),
                    "Перемещения":("GRAPHS/DISPLACEMENTS/",),
                    "Деформации":("GRAPHS/STRAINS/",),
                    "Напряжения":("GRAPHS/STRESSES/","3D/STRESS_FIELDS/"),
                    "Результанты":("GRAPHS/RESULTANTS/",),
                    # FULL publishes each physical z-profile once in the
                    # public GRAPHS/THROUGH_THICKNESS folder; XZ/SZ cuts live
                    # under GRAPHS/SECTIONS.  Both belong to this result group.
                    "По толщине":("GRAPHS/THROUGH_THICKNESS/","GRAPHS/SECTIONS/"),
                    "Пластичность":("GRAPHS/PLASTICITY/",),
                    "Сходимость":("GRAPHS/CONVERGENCE/",),
                    "CUF diagnostics":("GRAPHS/CUF_DIAGNOSTICS/",),
                    "Birger":("GRAPHS/BIRGER/",),
                    "3D":("3D/",),
                }
                return any(r.startswith(x) for x in checks.get(group,()))
            paths=[p for p in paths if belongs(p)]
            if group=="По толщине":
                out=self._result_output_dir()
                def thickness_priority(p):
                    try:r=str(Path(p).relative_to(out)).replace("\\","/").upper()
                    except Exception:r=str(p).replace("\\","/").upper()
                    if r.startswith("GRAPHS/THROUGH_THICKNESS/"):return (0,r)
                    if "/Z_PROFILES/" in r:return (1,r)
                    if "/XZ/" in r:return (2,r)
                    if "/SZ/" in r:return (3,r)
                    return (4,r)
                paths=sorted(paths,key=thickness_priority)
        self._result_png_paths=paths
        self.result_list.delete(0,"end")
        out=self._result_output_dir()
        for q in paths:
            try:txt=str(q.relative_to(out))
            except Exception:txt=str(q)
            self.result_list.insert("end",txt)

    def _build_online_tab(self):
        # The live dashboard contains a status block, a steps table and
        # Matplotlib canvases.  On a scaled Windows display they must retain
        # their natural height instead of compressing the plot to a thin strip.
        # Keep the complete page reachable through one explicit vertical
        # scrollbar; this is UI-only and does not affect calculation data.
        host=self.tab_online
        host.columnconfigure(0,weight=1); host.rowconfigure(0,weight=1)
        self.online_scroll_canvas=tk.Canvas(host,highlightthickness=0,borderwidth=0)
        self.online_scroll_canvas.grid(row=0,column=0,sticky="nsew")
        self.online_scrollbar=ttk.Scrollbar(
            host,orient="vertical",command=self.online_scroll_canvas.yview,
            style="Green.Vertical.TScrollbar",
        )
        self.online_scrollbar.grid(row=0,column=1,sticky="ns")
        self.online_scroll_canvas.configure(yscrollcommand=self.online_scrollbar.set)
        f=ttk.Frame(self.online_scroll_canvas)
        self.online_scroll_inner=f
        self._online_scroll_window=self.online_scroll_canvas.create_window(
            (0,0),window=f,anchor="nw"
        )
        f.bind(
            "<Configure>",
            lambda _e:self.online_scroll_canvas.configure(
                scrollregion=self.online_scroll_canvas.bbox("all")
            )
        )
        self.online_scroll_canvas.bind(
            "<Configure>",
            lambda e:self.online_scroll_canvas.itemconfigure(
                self._online_scroll_window,width=max(1,e.width)
            )
        )
        def _online_mousewheel(event):
            delta=getattr(event,"delta",0)
            if delta:
                self.online_scroll_canvas.yview_scroll(int(-delta/120),"units")
        self.online_scroll_canvas.bind("<MouseWheel>",_online_mousewheel)
        f.bind("<MouseWheel>",_online_mousewheel)

        f.columnconfigure(0,weight=1); f.rowconfigure(2,weight=1)
        stat=ttk.LabelFrame(f,text="Текущее состояние расчёта",padding=7); stat.grid(row=0,column=0,sticky="ew")
        labels=[
            ("elapsed","Время"),("stage","Этап"),("step","Шаг"),("load","λ / нагрузка"),
            ("newton","Newton"),("residual","Residual"),("increment","Increment"),("wmax","wmax / координаты"),
            ("mvi","МВИ rank/cycle/half-step"),("rproj","Rproj / Δfield"),("birger","Birger outer"),("etas","ηG / ησ"),
            ("gb","GB/G0 min/max"),("plastic","Vp/V | Vnl/V"),("detf","min detF / max|J−1|"),
        ]
        self.online_value_labels={}
        for i,(key,title) in enumerate(labels):
            r=i//4; c=(i%4)*2
            ttk.Label(stat,text=title+":").grid(row=r,column=c,sticky="w",padx=(2,4),pady=2)
            lab=ttk.Label(stat,text="—",width=24); lab.grid(row=r,column=c+1,sticky="w",padx=(0,10),pady=2)
            self.online_value_labels[key]=lab

        stepsf=ttk.LabelFrame(f,text="Текущий и завершённые шаги",padding=5); stepsf.grid(row=1,column=0,sticky="ew",pady=(6,6))
        cols=("step","load","status","wmax","iters","residual","gbmin","plastic","time")
        steps_body=ttk.Frame(stepsf)
        steps_body.pack(fill="x",expand=True)
        self.online_tree=ttk.Treeview(steps_body,columns=cols,show="headings",height=4)
        self.online_steps_scroll=ttk.Scrollbar(
            steps_body,orient="vertical",command=self.online_tree.yview,
            style="Green.Vertical.TScrollbar",
        )
        self.online_tree.configure(yscrollcommand=self.online_steps_scroll.set)
        heads={"step":"Шаг","load":"λ","status":"Статус","wmax":"|w|max","iters":"Итерации","residual":"Residual","gbmin":"min GB/G0","plastic":"Vp/V | Vnl/V","time":"t, s"}
        widths={"step":50,"load":65,"status":165,"wmax":95,"iters":65,"residual":95,"gbmin":85,"plastic":105,"time":75}
        for c in cols:self.online_tree.heading(c,text=heads[c]);self.online_tree.column(c,width=widths[c],anchor="center")
        self.online_tree.pack(side="left",fill="x",expand=True)
        self.online_steps_scroll.pack(side="right",fill="y")
        self.online_tree.bind("<<TreeviewSelect>>",lambda e:self._online_tree_select())

        nb=ttk.Notebook(f); nb.grid(row=2,column=0,sticky="nsew")
        self.online_plot_tabs={}
        for key,title in (("load","Нагрузка–прогиб"),("shell","3D оболочка"),("field","Поле / пластичность"),("conv","Сходимость"),("log","Журнал онлайн")):
            fr=ttk.Frame(nb); nb.add(fr,text=title); self.online_plot_tabs[key]=fr
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        for key in ("load","shell","conv"):
            pbar=ttk.Frame(self.online_plot_tabs[key]);pbar.pack(fill="x",padx=4,pady=3)
            ttk.Button(pbar,text="Открыть график в отдельном окне",command=lambda k=key:self._open_online_figure(k)).pack(side="right")
        self.online_fig_load=Figure(figsize=(6.5,4.5)); self.online_ax_load=self.online_fig_load.add_subplot(111)
        self.online_canvas_load=FigureCanvasTkAgg(self.online_fig_load,master=self.online_plot_tabs["load"]); self.online_canvas_load.get_tk_widget().pack(fill="both",expand=True)
        self.online_canvas_load.mpl_connect("button_press_event",self._online_load_click)
        self.online_fig_shell=Figure(figsize=(7,5)); self.online_ax_shell=self.online_fig_shell.add_subplot(111,projection="3d")
        self.online_canvas_shell=FigureCanvasTkAgg(self.online_fig_shell,master=self.online_plot_tabs["shell"]); self.online_canvas_shell.get_tk_widget().pack(fill="both",expand=True)
        fbar=ttk.Frame(self.online_plot_tabs["field"]); fbar.pack(fill="x",padx=4,pady=3)
        ttk.Label(fbar,text="Поле последнего/выбранного завершённого шага:").pack(side="left")
        self.online_field_combo=ttk.Combobox(
            fbar,textvariable=self._v("online_field","AUTO"),state="readonly",width=28,
            values=("AUTO","w","GB/G0 min","epsilon_i max","sigma_i max","Birger nonlinear zone","Plastic zone","Plastic thickness fraction","unloading mask"),
        )
        self.online_field_combo.pack(side="left",padx=6)
        ttk.Button(fbar,text="Открыть график в отдельном окне",command=lambda:self._open_online_figure("field")).pack(side="right")
        self.online_field_combo.bind("<<ComboboxSelected>>",lambda e:self._render_online_step(self._online_current_step) if self._online_current_step else None)
        self.online_fig_field=Figure(figsize=(6.5,4.5)); self.online_ax_field=self.online_fig_field.add_subplot(111)
        self.online_canvas_field=FigureCanvasTkAgg(self.online_fig_field,master=self.online_plot_tabs["field"]); self.online_canvas_field.get_tk_widget().pack(fill="both",expand=True)
        self.online_fig_conv=Figure(figsize=(6.5,4.5)); self.online_ax_conv=self.online_fig_conv.add_subplot(111)
        self.online_canvas_conv=FigureCanvasTkAgg(self.online_fig_conv,master=self.online_plot_tabs["conv"]); self.online_canvas_conv.get_tk_widget().pack(fill="both",expand=True)
        logbar=ttk.Frame(self.online_plot_tabs["log"]);logbar.pack(fill="x",padx=5,pady=(5,3))
        ttk.Label(logbar,text="Прокрутка вверх приостанавливает автопрокрутку; возврат вниз включает её снова.").pack(side="left")
        ttk.Button(logbar,text="Очистить",command=self._clear_logs).pack(side="right")
        ttk.Button(logbar,text="Открыть большой журнал",command=self._open_detached_online_log).pack(side="right",padx=(0,6))
        self.online_autoscroll_btn=ttk.Button(logbar,text="Автопрокрутка: вкл",command=lambda:self._toggle_log_autoscroll("online_log"))
        self.online_autoscroll_btn.pack(side="right",padx=(0,6))
        logbody=ttk.Frame(self.online_plot_tabs["log"]);logbody.pack(fill="both",expand=True,padx=5,pady=(0,5))
        self.online_log=tk.Text(logbody,wrap="word",height=24,font=("Consolas",9),background="#fbfcfd",foreground="#21313f")
        self.online_log.pack(side="left",fill="both",expand=True)
        self.online_log_scroll=ttk.Scrollbar(logbody,orient="vertical",style="Green.Vertical.TScrollbar")
        self.online_log_scroll.pack(side="right",fill="y")
        self.online_log.configure(yscrollcommand=self.online_log_scroll.set)
        self._register_log_scrolling("online_log",self.online_log,self.online_log_scroll,self.online_autoscroll_btn)
        self._reset_online_view()

    def _open_detached_online_log(self):
        """Open a large resizable mirror of the live calculation journal."""
        win=getattr(self,"_online_log_window",None)
        try:
            if win is not None and win.winfo_exists():
                win.deiconify()
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

        win=tk.Toplevel(self)
        self._online_log_window=win
        win.title("Журнал онлайн — CYL_SOLVER")
        win.geometry("1100x650")
        win.minsize(700,400)
        win.resizable(True,True)

        def close_window():
            self.online_log_detached=None
            self._online_log_window=None
            try:self._log_autoscroll_buttons.pop("online_log_detached",None)
            except Exception:pass
            try:self._log_autoscroll_states.pop("online_log_detached",None)
            except Exception:pass
            try:win.destroy()
            except Exception:pass

        def maximize_window():
            try:
                win.state("zoomed")
            except Exception:
                width=max(700,win.winfo_screenwidth()-80)
                height=max(400,win.winfo_screenheight()-120)
                win.geometry(f"{width}x{height}+20+20")

        toolbar=ttk.Frame(win,padding=(8,7,8,5));toolbar.pack(fill="x")
        ttk.Label(
            toolbar,
            text="Журнал расчёта обновляется автоматически и прокручивается к новой строке.",
        ).pack(side="left")
        ttk.Button(toolbar,text="Закрыть",command=close_window).pack(side="right")
        ttk.Button(toolbar,text="Развернуть",command=maximize_window).pack(side="right",padx=(0,6))
        ttk.Button(toolbar,text="Очистить",command=self._clear_logs).pack(side="right",padx=(0,6))
        detached_autoscroll_btn=ttk.Button(toolbar,text="Автопрокрутка: вкл",command=lambda:self._toggle_log_autoscroll("online_log_detached"))
        detached_autoscroll_btn.pack(side="right",padx=(0,6))

        body=ttk.Frame(win,padding=(8,0,8,8));body.pack(fill="both",expand=True)
        text=tk.Text(body,wrap="word",font=("Consolas",10),background="#fbfcfd",foreground="#21313f")
        text.pack(side="left",fill="both",expand=True)
        scroll=ttk.Scrollbar(body,orient="vertical",style="Green.Vertical.TScrollbar")
        scroll.pack(side="right",fill="y")
        text.configure(yscrollcommand=scroll.set)
        self.online_log_detached=text
        self._register_log_scrolling("online_log_detached",text,scroll,detached_autoscroll_btn)

        try:
            current=self.online_log.get("1.0","end-1c")
            if current:text.insert("1.0",current)
            text.see("end")
        except Exception:
            pass

        win.protocol("WM_DELETE_WINDOW",close_window)
        win.focus_force()

    def _register_log_scrolling(self,name,text_widget,scrollbar,button):
        if not hasattr(self,"_log_autoscroll_states"):self._log_autoscroll_states={}
        if not hasattr(self,"_log_autoscroll_buttons"):self._log_autoscroll_buttons={}
        self._log_autoscroll_states[name]=True
        self._log_autoscroll_buttons[name]=button
        scrollbar.configure(command=lambda *args,n=name:self._scroll_log_widget(n,*args))
        for sequence in ("<MouseWheel>","<Button-4>","<Button-5>","<Prior>","<Next>","<Home>","<End>","<Up>","<Down>"):
            text_widget.bind(sequence,lambda _event,n=name:self.after_idle(lambda:self._sync_log_autoscroll(n)),add="+")
        self._update_log_autoscroll_button(name)

    def _scroll_log_widget(self,name,*args):
        widget=getattr(self,name,None)
        if widget is None:return
        try:widget.yview(*args)
        except Exception:return
        self._sync_log_autoscroll(name)

    def _sync_log_autoscroll(self,name):
        widget=getattr(self,name,None)
        if widget is None:return
        try:at_bottom=float(widget.yview()[1])>=0.999
        except Exception:return
        self._set_log_autoscroll(name,at_bottom)

    def _set_log_autoscroll(self,name,enabled):
        if not hasattr(self,"_log_autoscroll_states"):self._log_autoscroll_states={}
        self._log_autoscroll_states[name]=bool(enabled)
        self._update_log_autoscroll_button(name)

    def _toggle_log_autoscroll(self,name):
        states=getattr(self,"_log_autoscroll_states",{})
        enabled=not bool(states.get(name,True))
        self._set_log_autoscroll(name,enabled)
        if enabled:
            widget=getattr(self,name,None)
            if widget is not None:
                try:widget.see("end")
                except Exception:pass

    def _update_log_autoscroll_button(self,name):
        button=getattr(self,"_log_autoscroll_buttons",{}).get(name)
        if button is None:return
        enabled=bool(getattr(self,"_log_autoscroll_states",{}).get(name,True))
        try:button.configure(text="Автопрокрутка: вкл" if enabled else "Автопрокрутка: выкл")
        except Exception:pass

    def _online_load_click(self,event):
        if event.xdata is None or event.ydata is None or not self._online_step_cache:return
        pts=sorted(self._online_step_cache.values(),key=lambda x:x["step"])
        if not pts:return
        yc=np.asarray([p["load"] for p in pts],float)
        xc=np.asarray([p.get("wcenter_abs",p["wmax"]) for p in pts],float)
        xm=np.asarray([p["wmax"] for p in pts],float)
        # Both curves are selectable.  This is important for antisymmetric or
        # local loading, where the geometric centre may legitimately stay near zero.
        sx=max(float(np.ptp(np.r_[xc,xm])),float(np.max(np.abs(np.r_[xc,xm]))),1e-30)
        sy=max(float(np.ptp(yc)),float(np.max(np.abs(yc))),1e-30)
        dc=((xc-float(event.xdata))/sx)**2+((yc-float(event.ydata))/sy)**2
        dm=((xm-float(event.xdata))/sx)**2+((yc-float(event.ydata))/sy)**2
        d=np.minimum(dc,dm)
        p=pts[int(np.argmin(d))]; step=int(p["step"])
        try:
            iid=f"step_{step:04d}"; self.online_tree.selection_set(iid); self.online_tree.see(iid)
        except Exception:pass
        self._render_online_step(step)

    def _saved_run_metadata(self):
        """Return immutable metadata for the active calculation/result folder.

        Online/result rendering must never depend on GUI fields that can be
        edited after Start. Prefer the solver summary, then postprocess report,
        then the exact INPUT/FINAL_CASE.json snapshot.
        """
        out=getattr(self,"run_output_dir",None)
        if out is None:
            try:out=self._result_output_dir()
            except Exception:out=None
        if out is None:return {}
        out=Path(out)
        for fp in (out/"FINAL_RUN_SUMMARY.json",out/"REPORT"/"CALCULATION_SUMMARY.json",out/"INPUT"/"FINAL_CASE.json"):
            if not fp.exists():continue
            try:
                j=json.loads(fp.read_text(encoding="utf-8"))
                if isinstance(j,dict):return j
            except Exception:pass
        return {}

    def _load_context_from_saved_run(self):
        """Read display-only load/unit metadata from the current run folder.

        The online graph must describe the calculation that is actually running,
        not whatever values the user may edit in the GUI after Start.
        """
        spec=getattr(self,"_online_run_load_definition",None)
        system=getattr(self,"_online_run_unit_system",None)
        if isinstance(spec,dict) and spec:
            return spec, (system or self._v("unit_system","N_MM_MPA").get())
        out=getattr(self,"run_output_dir",None)
        if out is not None:
            out=Path(out)
            for fp in (out/"FINAL_RUN_SUMMARY.json",out/"REPORT"/"CALCULATION_SUMMARY.json",out/"INPUT"/"FINAL_CASE.json"):
                if not fp.exists():continue
                try:
                    j=json.loads(fp.read_text(encoding="utf-8"))
                    ld=j.get("load_definition") or {}
                    units=j.get("units") or {}
                    if isinstance(ld,dict) and ld:
                        return ld,str(units.get("system") or self._v("unit_system","N_MM_MPA").get())
                except Exception:
                    pass
        return None,None

    def _current_load_axis(self,factors):
        saved_spec,saved_system=self._load_context_from_saved_run()
        if saved_spec is not None:
            try:return funits.load_axis(factors,saved_spec,saved_system)
            except Exception:pass
        typ=self._v("load_type","EXTERNAL_NPZ").get()
        spec={"type":typ}
        try:
            if typ in ("UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE","PATCH_RADIAL_FACE"):
                spec["q"]=float(self._v("load_q").get())
            elif typ=="POINT_RADIAL_FORCE":
                pmag=abs(float(self._v("load_point_P").get()))
                direction=self._v("load_pressure_direction","INTERNAL").get().upper()
                spec["P"]=pmag if direction=="INTERNAL" else -pmag
            elif typ=="LINE_RADIAL_GENERATOR":
                pmag=abs(float(self._v("load_p_line").get()))
                direction=self._v("load_pressure_direction","INTERNAL").get().upper()
                spec["p_line"]=pmag if direction=="INTERNAL" else -pmag
        except Exception:
            pass
        system=self._v("unit_system","N_MM_MPA").get()
        return funits.load_axis(factors,spec,system)

    def _gui_units(self):
        # While a run/result folder is active, labels must stay tied to the
        # unit system of that saved calculation even if the editable GUI fields
        # are changed afterwards.
        try:
            _spec,system=self._load_context_from_saved_run()
            return funits.unit_labels(system or self._v("unit_system","N_MM_MPA").get())
        except Exception:return funits.unit_labels("N_MM_MPA")

    def _online_field_unit(self,choice):
        labels=self._gui_units(); c=str(choice).lower()
        if c.startswith(("u","v","w")):return labels["displacement"]
        if "sigma" in c or "sig_" in c or c.startswith("s_"):return labels["stress"]
        if c=="q_r" or "radial surface load" in c:return labels["pressure"]
        if c.startswith("r:n"):return labels["line_load"]
        if c.startswith("r:m"):return labels["force"]
        return "-"

    def _online_set(self,key,value):
        lab=getattr(self,"online_value_labels",{}).get(key)
        if lab is not None: lab.configure(text="—" if value is None else str(value))

    def _format_live_wmax(self,j):
        if j.get("w_abs_max") is None:return None
        lu=self._gui_units()["length"]
        txt=f"{float(j['w_abs_max']):.6g} {lu}"
        if j.get("wmax_ix") is not None and j.get("wmax_is") is not None:
            try:
                ix=int(j["wmax_ix"]); js=int(j["wmax_is"])
                if self._v("method","FDM").get()=="MVI":
                    nx=int(self._v("mvi_nx").get()); ns=int(self._v("mvi_ns").get())
                else:
                    nx=int(self._v("nx").get()); ns=int(self._v("ns").get())
                L=float(self._v("L").get()); R=float(self._v("R").get())
                x=L*ix/(nx-1) if nx>1 else 0.0; th=2*np.pi*js/ns; s=R*th
                txt+=f"; x={x:.6g} {lu}, s={s:.6g} {lu}, θ={th:.4g} rad"
            except Exception:
                txt+=f"; ix={j.get('wmax_ix')}, is={j.get('wmax_is')}"
        return txt

    def _reset_online_view(self):
        self._online_last_seq=-1; self._online_progress_pos=0; self._online_progress_path=None
        self._online_run_load_definition=None; self._online_run_unit_system=None
        self._online_step_cache={}; self._online_current_step=None; self._online_step_elapsed={}; self._online_live_point=None
        self._online_conv={"newton_x":[],"newton_r":[],"newton_inc":[],"mvi_x":[],"mvi_r":[],"mvi_inc":[],"birger_x":[],"etaG":[],"etaS":[]}
        if hasattr(self,"online_tree"):
            for iid in self.online_tree.get_children(): self.online_tree.delete(iid)
        for k in getattr(self,"online_value_labels",{}): self._online_set(k,"—")
        if hasattr(self,"online_ax_load"):
            _,load_label,_=self._current_load_axis([0.0]); du=self._gui_units()["displacement"]
            self.online_ax_load.clear(); self.online_ax_load.plot([0],[0],marker="o",label="|w(center)|"); self.online_ax_load.plot([0],[0],marker="x",label="|w|max"); self.online_ax_load.set_xlabel(f"прогиб [{du}]"); self.online_ax_load.set_ylabel(load_label); self.online_ax_load.set_title("Нагрузка–прогиб: центр и максимум"); self.online_ax_load.grid(True,alpha=.3); self.online_ax_load.legend(); self.online_canvas_load.draw_idle()
            try:self.online_tree.heading("load",text=load_label);self.online_tree.heading("wmax",text=f"|w|max [{du}]")
            except Exception:pass
            self.online_ax_shell.clear(); self.online_ax_shell.set_title("3D: завершённых шагов пока нет"); self.online_canvas_shell.draw_idle()
            self.online_ax_field.clear(); self.online_ax_field.set_title("Поле: завершённых шагов пока нет"); self.online_canvas_field.draw_idle()
            self.online_ax_conv.clear(); self.online_ax_conv.set_title("Сходимость текущего шага"); self.online_ax_conv.set_xlabel("Итерация / цикл [-]"); self.online_ax_conv.set_ylabel("Относительная норма [-]"); self.online_canvas_conv.draw_idle()

    def _load_step_visual_state(self,step):
        base=self.run_output_dir
        if base is None:
            txt=self.vars.get("output_dir",tk.StringVar()).get().strip(); base=Path(txt) if txt else None
        if base is None:return None
        out=Path(base).resolve()
        candidates=[out/f"STEP_{int(step):04d}_STATE.npz",out/f"STEP_{int(step):04d}.npz"]
        p=next((x for x in candidates if x.exists()),None)
        if p is None:return None
        with np.load(p,allow_pickle=False) as z:data={k:np.asarray(z[k]) for k in z.files}
        if "q" in data:
            q=np.asarray(data["q"],float); u,v,w=q[0,0],q[1,0],q[2,0]
        elif "U" in data:
            U=np.asarray(data["U"],float); u,v,w=U[0],U[1],U[2]
        elif "X" in data and "Y" in data:
            X=np.asarray(data["X"],float);Y=np.asarray(data["Y"],float)
            if X.ndim==3:
                U=np.einsum("crx,crs->csx",X,Y);u,v,w=U[0],U[1],U[2]
            elif X.ndim==4:
                q=np.einsum("cmrx,cmrs->cmsx",X,Y);u,v,w=q[0,0],q[1,0],q[2,0]
            else:return None
        else:return None
        return {
            "path":p,"raw":data,"u":np.asarray(u,float),"v":np.asarray(v,float),"w":np.asarray(w,float),
            "GB":data.get("GB"),"epsilon_i":data.get("epsilon_i"),"sigma_i":data.get("sigma_i"),
            "detF":data.get("detF"),"unloading_mask":data.get("unloading_mask"),
        }

    def _online_factor_for_step(self,num,j=None):
        """Load factor for an accepted state, including STOP race recovery."""
        if isinstance(j,dict) and j.get("load_factor") is not None:
            try:return float(j.get("load_factor"))
            except Exception:pass
        spec,_system=self._load_context_from_saved_run()
        raw=spec.get("definition") if isinstance(spec,dict) and isinstance(spec.get("definition"),dict) else spec
        if isinstance(raw,dict):
            vals=raw.get("load_factors")
            if isinstance(vals,(list,tuple)) and 1 <= int(num) <= len(vals):
                try:return float(vals[int(num)-1])
                except Exception:pass
            meta=spec.get("generated_meta") if isinstance(spec,dict) else None
            vals=(meta or {}).get("load_factors") if isinstance(meta,dict) else None
            if isinstance(vals,(list,tuple)) and 1 <= int(num) <= len(vals):
                try:return float(vals[int(num)-1])
                except Exception:pass
            npz_value=raw.get("npz")
            if npz_value:
                try:
                    lp=Path(str(npz_value))
                    if lp.exists():
                        with np.load(lp,allow_pickle=False) as z:
                            if "load_factors" in z.files:
                                vals=np.asarray(z["load_factors"],float).reshape(-1)
                                if 1 <= int(num) <= vals.size:return float(vals[int(num)-1])
                except Exception:pass
        out=getattr(self,"run_output_dir",None)
        if out is not None:
            gp=Path(out)/"GENERATED_LOADS.npz"
            if gp.exists():
                try:
                    with np.load(gp,allow_pickle=False) as z:
                        if "load_factors" in z.files:
                            vals=np.asarray(z["load_factors"],float).reshape(-1)
                            if 1 <= int(num) <= vals.size:return float(vals[int(num)-1])
                except Exception:pass
        return float(num)

    def _accepted_step_numbers_on_disk(self):
        out=getattr(self,"run_output_dir",None)
        if out is None:return []
        out=Path(out)
        nums=set()
        for pat in ("STEP_[0-9][0-9][0-9][0-9]_STATE.npz","STEP_[0-9][0-9][0-9][0-9].npz"):
            for p in out.glob(pat):
                if "NOT_CONVERGED" in p.name:continue
                try:nums.add(int(p.name.split("_")[1].split(".")[0]))
                except Exception:pass
        return sorted(nums)

    def _online_refresh_steps(self):
        if self.run_output_dir is None:return
        out=Path(self.run_output_dir)
        if not out.exists():return
        # Accepted STATE files are authoritative. STEP JSON may legitimately lag
        # by one atomic write when the user presses STOP at exactly that moment.
        nums=set(self._accepted_step_numbers_on_disk())
        for jp in out.glob("STEP_[0-9][0-9][0-9][0-9].json"):
            try:nums.add(int(jp.stem.split("_")[1]))
            except Exception:pass
        changed=False
        running=(self.proc is not None and self.proc.poll() is None)
        for num in sorted(nums):
            jp=out/f"STEP_{int(num):04d}.json"
            # STEP state is written atomically before STEP metadata.  While the
            # solver is running, never guess nominal/substep identity from the
            # bare NPZ: doing so transiently counts adaptive substeps as normal
            # load steps and makes the table/progress bar drift out of sync.
            if running and not jp.exists():
                continue
            try:j=json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else {}
            except Exception:
                if running:
                    continue
                j={}
            vis=self._load_step_visual_state(num)
            if vis is None:continue
            w=vis["w"]; js,ix=np.unravel_index(int(np.argmax(np.abs(w))),w.shape); wmax=float(np.max(np.abs(w)))
            center_is=int(w.shape[0]//2); center_ix=int(w.shape[1]//2); wcenter=float(w[center_is,center_ix]); wcenter_abs=abs(wcenter)
            factor=self._online_factor_for_step(num,j); actual_load=float(self._current_load_axis([factor])[0][0]); status=str(j.get("status",j.get("solver_status","ACCEPTED_STATE")))
            it=j.get("iterations",j.get("solver_iterations",j.get("outer_iterations","")))
            residual=j.get("residual_convergence_relative",j.get("residual_relative",j.get("residual_inf","")))
            gbmin=""
            if vis.get("GB") is not None:
                try:
                    meta=self._saved_run_metadata(); mat=(meta.get("material") or {}) if isinstance(meta,dict) else {}
                    E=float(mat.get("E0",self._v("E0").get()));nu=float(mat.get("nu0",self._v("nu0").get()));G0=E/(2*(1+nu));gbmin=float(np.min(np.asarray(vis["GB"],float))/G0)
                except Exception:pass
            pv=j.get("plastic_volume_fraction",""); nv=j.get("birger_nonlinear_volume_fraction","")
            elapsed=getattr(self,"_online_step_elapsed",{}).get(num,"")
            requested_index=int(j.get("requested_step_index",num) or num)
            is_substep=bool(j.get("is_substep",False))
            row={"step":num,"requested_step_index":requested_index,"is_substep":is_substep,"factor":factor,"load":actual_load,"status":status,"wmax":wmax,"wcenter":wcenter,"wcenter_abs":wcenter_abs,"center_is":center_is,"center_ix":center_ix,"iters":it,"residual":residual,"gbmin":gbmin,"plastic":pv,"nonlinear":nv,"elapsed":elapsed,"vis":vis,"wmax_is":int(js),"wmax_ix":int(ix)}
            self._online_step_cache[num]=row
            ztxt=""
            if pv not in (None,""):ztxt=f"{float(pv):.3g}"
            if nv not in (None,""):ztxt+=(" | " if ztxt else "")+f"{float(nv):.3g}"
            # The main completed-step table represents the user-requested
            # nominal load series only. Adaptive accepted substeps are internal
            # continuation states; showing them as ordinary rows makes the table
            # drift ahead of the 1..N nominal progress counter.
            iid=f"step_{num:04d}"
            if is_substep:
                if self.online_tree.exists(iid):
                    self.online_tree.delete(iid)
            else:
                vals=(requested_index,f"{actual_load:.6g}",status,f"{wmax:.6g}",it,(f"{float(residual):.3e}" if residual not in (None,"") else ""),(f"{float(gbmin):.6g}" if gbmin not in (None,"") else ""),ztxt,(f"{float(elapsed):.2f}" if elapsed not in (None,"") else ""))
                if self.online_tree.exists(iid): self.online_tree.item(iid,values=vals)
                else: self.online_tree.insert("","end",iid=iid,values=vals); changed=True
        if self._online_step_cache:
            pts=sorted(
                (x for x in self._online_step_cache.values() if not bool(x.get("is_substep",False))),
                key=lambda x:int(x.get("requested_step_index",x.get("step",0)) or 0)
            )
            self.online_ax_load.clear()
            if not pts:
                return
            yy=[0]+[x["load"] for x in pts]
            self.online_ax_load.plot([0]+[x.get("wcenter_abs",x["wmax"]) for x in pts],yy,marker="o",label="|w(center)|")
            self.online_ax_load.plot([0]+[x["wmax"] for x in pts],yy,marker="x",label="|w|max")
            _,load_label,_=self._current_load_axis([0.0]);du=self._gui_units()["displacement"]
            self.online_ax_load.set_xlabel(f"прогиб [{du}]");self.online_ax_load.set_ylabel(load_label);self.online_ax_load.set_title("Нагрузка–прогиб: центр и максимум");self.online_ax_load.grid(True,alpha=.3);self.online_ax_load.legend();self.online_canvas_load.draw_idle()
            try:self.online_tree.heading("load",text=load_label);self.online_tree.heading("wmax",text=f"|w|max [{du}]")
            except Exception:pass
            latest=int(pts[-1]["step"])
            if changed and (self.proc is not None and self.proc.poll() is None): self._render_online_step(latest)

    def _online_completed_requested_count(self):
        """Count accepted original load targets; adaptive substeps are not nominal progress."""
        rows=(getattr(self,"_online_step_cache",{}) or {}).values()
        return len({
            int(row.get("requested_step_index",row.get("step",0)) or 0)
            for row in rows
            if not bool(row.get("is_substep",False))
            and int(row.get("requested_step_index",row.get("step",0)) or 0)>0
        })

    def _online_nominal_step_text(self,j):
        total=int(j.get("total_steps") or self.expected_steps or 0)
        requested=int(j.get("requested_step_index") or j.get("step_index") or 0)
        accepted=int(j.get("accepted_step_index") or j.get("step_index") or 0)
        if bool(j.get("is_substep",False)):
            return f"{requested}/{total} · подшаг {accepted}" if total else f"{requested} · подшаг {accepted}"
        return f"{requested}/{total}" if total else str(requested)

    def _online_update_live_tree_row(self,j,actual_load=None,status=None):
        """Show the current unconverged step in the online table without marking it accepted."""
        if not hasattr(self,"online_tree") or not isinstance(j,dict):return
        accepted_step=int(j.get("accepted_step_index",j.get("step_index",0)) or 0)
        requested_step=int(j.get("requested_step_index",accepted_step) or accepted_step)
        step=(f"{requested_step} [подшаг {accepted_step}]" if bool(j.get("is_substep",False)) else str(requested_step))
        elapsed=float(j.get("elapsed_s",0.0) or 0.0)
        iid="live_current"
        old=list(self.online_tree.item(iid,"values")) if self.online_tree.exists(iid) else ["","","","","","","","",""]
        while len(old)<9:old.append("")
        old[0]=step or old[0]
        if actual_load is not None:old[1]=f"{float(actual_load):.6g}"
        old[2]=str(status or j.get("stage") or j.get("event") or "RUNNING")
        if j.get("w_abs_max") is not None:old[3]=f"{float(j.get('w_abs_max')):.6g}"
        it=j.get("iteration",j.get("cycle",j.get("outer_iteration",j.get("halfstep_iteration"))))
        if it is not None:old[4]=str(it)
        rr=j.get("residual_convergence_relative",j.get("residual_relative",j.get("projected_relative",j.get("projected_inf"))))
        if rr is not None:
            try:old[5]=f"{float(rr):.3e}"
            except Exception:old[5]=str(rr)
        if j.get("GB_ratio_min") is not None:
            try:old[6]=f"{float(j.get('GB_ratio_min')):.6g}"
            except Exception:pass
        pv=j.get("plastic_volume_fraction");nv=j.get("birger_nonlinear_volume_fraction")
        if pv is not None or nv is not None:
            a="—" if pv is None else f"{float(pv):.3g}";b="—" if nv is None else f"{float(nv):.3g}";old[7]=a+" | "+b
        old[8]=f"{elapsed:.1f}"
        if self.online_tree.exists(iid):self.online_tree.item(iid,values=tuple(old))
        else:self.online_tree.insert("",0,iid=iid,values=tuple(old),tags=("running",))
        try:self.online_tree.tag_configure("running",background="#fff5da",foreground="#684b12")
        except Exception:pass

    def _online_redraw_load_with_live(self):
        """Redraw accepted load-deflection history plus current unconverged point."""
        if not hasattr(self,"online_ax_load"): return
        pts=sorted(
            (x for x in (getattr(self,"_online_step_cache",{}) or {}).values()
             if not bool(x.get("is_substep",False))),
            key=lambda x:int(x.get("requested_step_index",x.get("step",0)) or 0)
        )
        ax=self.online_ax_load;ax.clear();du=self._gui_units()["displacement"]
        try:load_label=self._current_load_axis([1.0])[1]
        except Exception:load_label="Нагрузка / Load"
        yy=[0.0]+[x["load"] for x in pts]
        ax.plot([0.0]+[x.get("wcenter_abs",x["wmax"]) for x in pts],yy,marker="o",markersize=4,linewidth=1.35,label="|w(center)| — принято")
        ax.plot([0.0]+[x["wmax"] for x in pts],yy,marker="x",markersize=5,linewidth=1.35,label="|w|max — принято")
        live=getattr(self,"_online_live_point",None)
        if isinstance(live,dict) and live.get("wmax") is not None and live.get("load") is not None:
            x=float(live["wmax"]);y=float(live["load"])
            ax.scatter([x],[y],s=88,zorder=7,color="#d96d3e",edgecolors="white",linewidths=.9,label="текущий / live")
            ax.axhline(y,color="#7f93a6",ls="--",lw=.85,alpha=.45);ax.axvline(x,color="#c8a272",ls=":",lw=.8,alpha=.35)
            ax.text(x,y,f"  STEP {live.get('step','?')} · Newton {live.get('iter','?')}",fontsize=7.5,color="#334e68",va="bottom",ha="left")
        ax.set_xlabel(f"прогиб [{du}]");ax.set_ylabel(load_label);ax.set_title("Нагрузка–прогиб: принятые шаги + текущий Newton");ax.grid(True,alpha=.25);ax.legend(fontsize=8)
        self.online_canvas_load.draw_idle()

    def _render_online_step(self,step):
        row=self._online_step_cache.get(int(step))
        if not row:return
        vis=row["vis"];u,v,w=vis["u"],vis["v"],vis["w"]
        try:
            meta=self._saved_run_metadata(); geom=(meta.get("geometry") or {}) if isinstance(meta,dict) else {}
            L=float(geom.get("L",self._v("L").get()));R=float(geom.get("R",self._v("R").get()))
        except Exception:return
        ns,nx=w.shape;xx=np.linspace(0,L,nx);th=np.linspace(0,2*np.pi,ns,endpoint=False);X0,T=np.meshgrid(xx,th);c=np.cos(T);ss=np.sin(T)
        scale=1.0
        try:
            txt=self._v("deformation_scale","1").get().strip(); scale=float(txt) if txt else 1.0
        except Exception:pass
        X=X0+scale*u;Y=(R+scale*w)*c-scale*v*ss;Z=(R+scale*w)*ss+scale*v*c
        # Matplotlib does not connect the last theta row to the first one.
        # Duplicate the first circumferential row at theta=2*pi only for
        # visualization, so a physically closed cylinder has no artificial seam.
        Xp=np.concatenate((X,X[:1,:]),axis=0);Yp=np.concatenate((Y,Y[:1,:]),axis=0);Zp=np.concatenate((Z,Z[:1,:]),axis=0);wp=np.concatenate((w,w[:1,:]),axis=0)
        self.online_fig_shell.clear(); ax=self.online_fig_shell.add_subplot(111,projection="3d"); self.online_ax_shell=ax
        lu=self._gui_units()["length"]
        mpl=__import__('matplotlib');import FINAL_VIDEO_EXPORT as fvideo;norm,cmap=fvideo._engineering_norm_cmap(float(np.min(w)),float(np.max(w)) if float(np.max(w))!=float(np.min(w)) else float(np.min(w))+1e-30);ax.plot_surface(Xp,Yp,Zp,facecolors=cmap(norm(wp)),linewidth=0,alpha=.92);sm=mpl.cm.ScalarMappable(norm=norm,cmap=cmap);sm.set_array([]);self.online_fig_shell.colorbar(sm,ax=ax,shrink=.65,label=f"w [{lu}]");ax.set_title(f"Шаг {step}: деформированная оболочка, масштаб={scale:g}");ax.set_xlabel(f"x [{lu}]");ax.set_ylabel(f"Y [{lu}]");ax.set_zlabel(f"Z [{lu}]");self.online_canvas_shell.draw_idle()
        self.online_fig_field.clear(); af=self.online_fig_field.add_subplot(111); self.online_ax_field=af
        choice=self._v("online_field","AUTO").get()
        if choice=="AUTO": choice="GB/G0 min" if vis.get("GB") is not None else "w"
        fld=w;title=f"Step {step}: w(x,s)"
        try:
            if choice=="GB/G0 min" and vis.get("GB") is not None:
                meta=self._saved_run_metadata();mat=(meta.get("material") or {}) if isinstance(meta,dict) else {}
                E=float(mat.get("E0",self._v("E0").get()));nu=float(mat.get("nu0",self._v("nu0").get()));G0=E/(2*(1+nu));fld=np.min(np.asarray(vis["GB"],float)/G0,axis=0);title=f"Step {step}: min GB/G0 through thickness"
            elif choice=="epsilon_i max" and vis.get("epsilon_i") is not None:
                fld=np.max(np.asarray(vis["epsilon_i"],float),axis=0);title=f"Step {step}: max epsilon_i through thickness"
            elif choice=="sigma_i max" and vis.get("sigma_i") is not None:
                fld=np.max(np.asarray(vis["sigma_i"],float),axis=0);title=f"Step {step}: max sigma_i through thickness"
            elif choice=="Birger nonlinear zone" and vis.get("GB") is not None:
                meta=self._saved_run_metadata();mat=(meta.get("material") or {}) if isinstance(meta,dict) else {}
                E=float(mat.get("E0",self._v("E0").get()));nu=float(mat.get("nu0",self._v("nu0").get()));G0=E/(2*(1+nu));fld=np.any(np.asarray(vis["GB"],float)/G0<(1-1e-10),axis=0).astype(float);title=f"Step {step}: Birger nonlinear zone anywhere through thickness"
            elif choice in ("Plastic zone","Plastic thickness fraction"):
                mask=None
                meta=self._saved_run_metadata();curve=((meta.get("birger_definition") or {}).get("curve") or {}) if isinstance(meta,dict) else {}
                ys=curve.get("yield_stress"); ye=curve.get("yield_strain")
                if ys is None:
                    txt=self._v("yield_stress","").get().strip(); ys=(float(txt) if txt else None)
                if ye is None:
                    txt=self._v("yield_strain","").get().strip(); ye=(float(txt) if txt else None)
                if ys is not None and vis.get("sigma_i") is not None: mask=np.asarray(vis["sigma_i"],float)>=float(ys)
                elif ye is not None and vis.get("epsilon_i") is not None: mask=np.asarray(vis["epsilon_i"],float)>=float(ye)
                if mask is None: raise ValueError("Plastic zone requires σ_T or ε_T and Birger material-point output")
                if choice=="Plastic zone":
                    fld=np.any(mask,axis=0).astype(float);title=f"Step {step}: plastic zone anywhere through thickness"
                else:
                    raw=vis.get("raw") or {}; zz=raw.get("quadrature_z"); ww=raw.get("quadrature_w")
                    if zz is None or ww is None: raise ValueError("Weighted plastic thickness fraction is unavailable before quadrature provenance is saved")
                    zz=np.asarray(zz,float);ww=np.asarray(ww,float);wz=ww*(1.0+zz/float(R));fld=np.sum(mask.astype(float)*wz[:,None,None],axis=0)/float(np.sum(wz));title=f"Step {step}: plastic thickness fraction hp/h (weighted)"
            elif choice=="unloading mask" and vis.get("unloading_mask") is not None:
                fld=np.any(np.asarray(vis["unloading_mask"],bool),axis=0).astype(float);title=f"Step {step}: unloading anywhere through thickness"
        except Exception:
            fld=w;title=f"Step {step}: w(x,s)"
        im=af.imshow(fld,aspect="auto",origin="lower",extent=(0.0,L,0.0,2*np.pi*R));af.set_title(title);af.set_xlabel(f"x [{lu}]");af.set_ylabel(f"s [{lu}]");cb=self.online_fig_field.colorbar(im,ax=af);cb.set_label(f"[{self._online_field_unit(choice)}]");self.online_canvas_field.draw_idle()
        self._online_current_step=int(step)

    def _online_tree_select(self):
        sel=self.online_tree.selection()
        if not sel:return
        try:self._render_online_step(int(sel[0].split("_")[1]))
        except Exception:pass

    def _online_update_event(self,j):
        if not isinstance(j,dict):return
        seq=int(j.get("seq",-1))
        if seq<=getattr(self,"_online_last_seq",-1):return
        self._online_last_seq=seq
        self._online_last_event_wall=time.monotonic()
        self._online_last_heartbeat_wall=self._online_last_event_wall
        elapsed=float(j.get("elapsed_s",0.0) or 0.0)
        self._online_set("elapsed",f"{elapsed:.1f} s")
        self._online_set("stage",j.get("stage",j.get("event","—")))
        if j.get("step_index") is not None:self._online_set("step",f"{j.get('step_index')} / {j.get('total_steps','?')}")
        ltxt=None; actual_load=None
        if j.get("load_factor") is not None:
            lf=float(j.get("load_factor")); ltxt=f"λ={lf:.6g}"
            try:
                values,label,key=self._current_load_axis([lf])
                value=float(values[0]); actual_load=value
                saved_spec,_saved_system=self._load_context_from_saved_run()
                spec=saved_spec if isinstance(saved_spec,dict) else {}
                raw=spec.get("definition") if isinstance(spec.get("definition"),dict) else spec
                direction=str((raw or {}).get("pressure_direction") or "").upper()
                suffix=("; внешн." if direction=="EXTERNAL" else "; внутр." if direction=="INTERNAL" else "")
                if key!="load_factor":ltxt+=f"; {key}={value:.6g} {label.split('[',1)[-1].rstrip(']')}{suffix}"
            except Exception:pass
            self._online_set("load",ltxt)
        ev=j.get("event")
        redraw=False
        if ev=="step_start":
            self._online_conv={"newton_x":[],"newton_r":[],"newton_inc":[],"mvi_x":[],"mvi_r":[],"mvi_inc":[],"birger_x":[],"etaG":[],"etaS":[]}
            redraw=True
            step=self._online_nominal_step_text(j)
            if hasattr(self,"progress_status_label"):
                self.progress_status_label.configure(text=f"Сейчас: шаг {step}")
            try:
                accepted=self._online_completed_requested_count(); total=int(j.get("total_steps") or self.expected_steps or 0)
                self.progress_label.configure(text=f"Принято {accepted}/{total} · Сейчас {step}")
                self.progress.set_text(f"Принято {accepted}/{total}")
            except Exception:pass
            self._online_update_live_tree_row(j,actual_load,status="RUNNING / идёт")
            load_part=f" · {ltxt}" if ltxt else ""
            subdepth=int(j.get("subdivision_depth",0) or 0)
            is_substep=bool(j.get("is_substep",False))
            subtext=(f"subdivision depth={subdepth}" if is_substep or subdepth>0
                     else "subdivision: not used")
            self._log(f"[{elapsed:8.1f} s] Load step {step}{load_part} · {subtext}\n")
        elif ev=="newton_iteration":
            it=int(j.get("iteration",0)); rr=float(j.get("residual_convergence_relative",j.get("residual_relative",0)) or 0); inc=float(j.get("increment_inf",0) or 0)
            self._online_conv["newton_x"].append(it);self._online_conv["newton_r"].append(max(rr,1e-300));self._online_conv["newton_inc"].append(max(inc,1e-300))
            self._online_set("newton",f"iter {it}");self._online_set("residual",f"{rr:.3e}");self._online_set("increment",f"{inc:.3e}")
            if hasattr(self,"progress_status_label"):
                self.progress_status_label.configure(text=f"Шаг {self._online_nominal_step_text(j)} · Newton {it} · Rrel={rr:.2e}")
            wtxt=""
            if j.get("w_abs_max") is not None:
                self._online_set("wmax",self._format_live_wmax(j)); wtxt=f" · |w|max={float(j.get('w_abs_max')):.6g}"
            self._log(f"[{elapsed:8.1f} s]   Newton {it} · Rrel={rr:.3e} · Δ={inc:.3e}{wtxt}\n")
            if j.get("w_abs_max") is not None and actual_load is not None:
                self._online_live_point={"step":int(j.get("step_index",0) or 0),"iter":it,"wmax":float(j.get("w_abs_max")),"load":float(actual_load)}
                try:self._online_redraw_load_with_live()
                except Exception:pass
            try:
                accepted=self._online_completed_requested_count();total=int(j.get("total_steps") or self.expected_steps or 0)
                self.progress_label.configure(text=f"Принято {accepted}/{total} · Сейчас {self._online_nominal_step_text(j)} · Newton {it}")
                self.progress.set_text(f"Принято {accepted}/{total}")
            except Exception:pass
            self._online_update_live_tree_row(j,actual_load,status=f"NEWTON {it}")
            redraw=True
        elif ev=="mvi_halfstep_iteration":
            half=str(j.get("halfstep","?"));cyc=j.get("cycle","?");maxcyc=j.get("max_xy_cycles","?")
            it=j.get("halfstep_iteration","?");maxit=j.get("max_halfstep_iterations","?")
            rp=float(j.get("projected_inf",0) or 0);rpr=j.get("projected_relative");nd=j.get("active_directions","?")
            self._online_set("stage",f"МВИ: полушаг {half}")
            self._online_set("mvi",f"rank {j.get('rank','?')}; цикл {cyc}/{maxcyc}; {half}; ит. {it}/{maxit}")
            if rpr is None:
                rtxt=f"Rproj={rp:.3e}"
            else:
                rtxt=f"Rproj_rel={float(rpr):.3e}; abs={rp:.3e}"
            self._online_set("rproj",f"{rtxt}; направлений={nd}")
            self._log(f"[{elapsed:8.1f} s] MVI-{j.get('rank','?')} · cycle {cyc}/{maxcyc} · {half}-halfstep · inner iteration {it}/{maxit} · {rtxt} · directions={nd}\n")
        elif ev=="mvi_tangent_progress":
            half=str(j.get("halfstep","?"));cyc=j.get("cycle","?")
            it=j.get("halfstep_iteration","?");maxit=j.get("max_halfstep_iterations","?")
            col=int(j.get("tangent_column",0) or 0);total=int(j.get("tangent_columns",0) or 0)
            pct=float(j.get("tangent_percent",0) or 0)
            self._online_set("stage",f"МВИ: касательная {half}")
            self._online_set("mvi",f"rank {j.get('rank','?')}; цикл {cyc}; {half}; ит. {it}/{maxit}")
            self._online_set("rproj",f"касательная {col}/{total} ({pct:.0f}%)")
            self._log(f"[{elapsed:8.1f} s]   {half}-tangent · iteration {it}/{maxit} · column {col}/{total} ({pct:.0f}%)\n")
        elif ev=="mvi_correction_damped":
            half=str(j.get("halfstep","?"));cyc=j.get("cycle","?")
            it=j.get("halfstep_iteration","?")
            factor=float(j.get("damping_factor",1.0) or 1.0)
            halvings=int(j.get("correction_halvings",0) or 0)
            self._online_set("stage",f"МВИ: detF-guard {half}")
            self._online_set("mvi",f"rank {j.get('rank','?')}; цикл {cyc}; {half}; damping={factor:.3g}")
            self._log(f"[{elapsed:8.1f} s]   {half}-correction reduced: factor={factor:.6g}, halves={halvings}, reason detF<0\n")
        elif ev=="mvi_tangent_damped":
            half=str(j.get("halfstep","?"));cyc=j.get("cycle","?")
            it=j.get("halfstep_iteration","?")
            col=j.get("tangent_column","?")
            factor=float(j.get("damping_factor",1.0) or 1.0)
            halvings=int(j.get("tangent_halvings",0) or 0)
            self._online_set("stage",f"МВИ: detF-guard касательной {half}")
            self._online_set("mvi",f"rank {j.get('rank','?')}; цикл {cyc}; {half}; FD factor={factor:.3g}")
            self._log(f"[{elapsed:8.1f} s]   {half}-tangent: column {col}, FD-step factor={factor:.6g}, halves={halvings}, reason detF<0\n")
        elif ev=="mvi_halfstep":
            self._online_set("mvi",f"rank {j.get('rank','?')}; {j.get('halfstep')} half; it={j.get('half_iterations')}")
            rp=float(j.get("projected_inf",0) or 0);rpr=j.get("projected_relative")
            rtxt=f"Rproj={rp:.3e}" if rpr is None else f"Rproj_rel={float(rpr):.3e}; abs={rp:.3e}"
            self._online_set("rproj",f"{rtxt}; corr={float(j.get('correction_norm',0)):.3e}")
            self._log(f"[{elapsed:8.1f} s] Halfstep {j.get('halfstep','?')} completed · iterations={j.get('half_iterations','?')} · {rtxt} · correction={float(j.get('correction_norm',0) or 0):.3e}\n")
        elif ev=="mvi_cycle":
            cyc=int(j.get("cycle",0));rp=float(j.get("projected_inf",0) or 0);rpr=j.get("projected_relative");di=float(j.get("field_increment_rel",0) or 0)
            rplot=rp if rpr is None else float(rpr)
            self._online_conv["mvi_x"].append(cyc);self._online_conv["mvi_r"].append(max(rplot,1e-300));self._online_conv["mvi_inc"].append(max(di,1e-300))
            rtxt=f"Rproj={rp:.3e}" if rpr is None else f"Rproj_rel={float(rpr):.3e}; abs={rp:.3e}"
            self._online_set("mvi",f"rank {j.get('rank','?')}; cycle {cyc}");self._online_set("rproj",f"{rtxt} / Δ={di:.3e}")
            wtxt=""
            if j.get("w_abs_max") is not None:
                self._online_set("wmax",self._format_live_wmax(j))
                wtxt=f" · |w|max={self._format_live_wmax(j)}"
            def _mvi_rel_text(key,label):
                value=j.get(key)
                if value is None:
                    return f"{label}=n/a"
                try:
                    return f"{label}={float(value):.3e}"
                except Exception:
                    return f"{label}={value}"
            fulltxt=" · ".join((
                _mvi_rel_text("full_residual_relative_to_load","Rfull_rel"),
                _mvi_rel_text("interior_residual_relative_to_load","Rint_rel"),
                _mvi_rel_text("boundary_residual_relative_to_load","Rbc_rel"),
            ))
            self._log(f"[{elapsed:8.1f} s] MVI cycle {cyc} completed · {rtxt} · Δfield={di:.3e}{wtxt}\n")
            self._log(f"[{elapsed:8.1f} s]   MVI diagnostics · {fulltxt}{wtxt}\n")
            self._online_update_live_tree_row(j,actual_load,status=f"MVI cycle {cyc}")
            redraw=True
        elif ev=="birger_outer":
            k=int(j.get("outer_iteration",0));eg=float(j.get("eta_G",0) or 0);es=float(j.get("eta_sigma",0) or 0)
            self._online_conv["birger_x"].append(k);self._online_conv["etaG"].append(max(eg,1e-300));self._online_conv["etaS"].append(max(es,1e-300))
            gbmin=float(j.get('GB_ratio_min',0) or 0); gbmax=float(j.get('GB_ratio_max',0) or 0)
            self._online_set("birger",f"outer {k}");self._online_set("etas",f"{eg:.3e} / {es:.3e}");self._online_set("gb",f"{gbmin:.6g} / {gbmax:.6g}");self._online_set("detf",f"max|J−1|={float(j.get('max_abs_J_minus_1',0)):.3e}")
            if hasattr(self,"progress_status_label"):
                self.progress_status_label.configure(text=f"Шаг {self._online_nominal_step_text(j)} · Birger outer {k} · ηG={eg:.2e}")
            self._log(f"[{elapsed:8.1f} s]   Birger outer {k} · ηG={eg:.3e} · ησ={es:.3e} · GB/G0={gbmin:.6g}…{gbmax:.6g}\n")
            self._online_update_live_tree_row(j,actual_load,status=f"BIRGER {k}")
            redraw=True
        elif ev=="birger_forecast":
            rho=float(j.get("contraction_rate",0) or 0)
            need=int(j.get("estimated_remaining_outer_iterations",0) or 0)
            self._online_set("birger",f"прогноз: ρ={rho:.4g}; осталось ≈{need} outer")
        elif ev=="step_complete":
            try:
                if hasattr(self,"online_tree") and self.online_tree.exists("live_current"):self.online_tree.delete("live_current")
            except Exception:pass
            self._online_live_point=None
            if j.get("step_index") is not None:
                try:self._online_step_elapsed[int(j.get("step_index"))]=float(j.get("elapsed_s",0.0))
                except Exception:pass
            # The solver publishes an accepted STEP atomically before emitting
            # step_complete.  Refresh immediately so the middle completed-step
            # table and load-deflection graph never look frozen until the next
            # background poll.
            try:self._online_refresh_steps()
            except Exception:pass
            if j.get("w_abs_max") is not None:self._online_set("wmax",self._format_live_wmax(j))
            if j.get("GB_ratio_min") is not None:
                self._online_set("gb",f"{float(j.get('GB_ratio_min')):.6g} / {float(j.get('GB_ratio_max')):.6g}")
            if j.get("max_abs_J_minus_1") is not None:
                self._online_set("detf",f"max|J−1|={float(j.get('max_abs_J_minus_1')):.3e}")
            pv=j.get("plastic_volume_fraction");nv=j.get("birger_nonlinear_volume_fraction")
            if pv is not None or nv is not None:
                a="—" if pv is None else f"{float(pv):.4g}"
                b="—" if nv is None else f"{float(nv):.4g}"
                self._online_set("plastic",f"{a} | {b}")
            if hasattr(self,"progress_status_label"):
                self.progress_status_label.configure(text=f"Принят шаг {self._online_nominal_step_text(j)}")
            self._log(f"[{elapsed:8.1f} s] Load step {self._online_nominal_step_text(j)} completed and saved\n")
        elif ev=="run_done":
            self._online_set("stage",j.get("status","DONE"))
            self._log(f"[{elapsed:8.1f} s] Calculation completed: {j.get('status','DONE')} · saved steps={j.get('completed_steps','?')}\n")
        # Any solver event belonging to the current load step must be visible in
        # the middle table.  This includes long MVI/FDM tangent-build events,
        # which otherwise made the table look frozen while the solver was alive.
        if ev not in ("step_complete","run_done") and j.get("step_index") is not None:
            try:
                live_status=str(j.get("stage") or ev or "RUNNING")
                if ev=="mvi_tangent_progress":
                    live_status=f"касательная {j.get('tangent_column','?')}/{j.get('tangent_columns','?')}"
                self._online_update_live_tree_row(j,actual_load,status=live_status)
            except Exception:pass
        if redraw:self._redraw_online_convergence()

    def _redraw_online_convergence(self):
        if not hasattr(self,"online_ax_conv"):return
        d=self._online_conv; ax=self.online_ax_conv;ax.clear(); plotted=False
        if d["newton_x"]:
            ax.semilogy(d["newton_x"],d["newton_r"],marker="o",label="Newton residual");ax.semilogy(d["newton_x"],d["newton_inc"],marker="x",label="Newton increment");plotted=True
        if d["mvi_x"]:
            ax.semilogy(d["mvi_x"],d["mvi_r"],marker="o",label="MVI Rproj");ax.semilogy(d["mvi_x"],d["mvi_inc"],marker="x",label="MVI Δfield");plotted=True
        if d["birger_x"]:
            ax.semilogy(d["birger_x"],d["etaG"],marker="o",label="Birger ηG");ax.semilogy(d["birger_x"],d["etaS"],marker="x",label="Birger ησ");plotted=True
        if d["mvi_x"] and not d["newton_x"] and not d["birger_x"]:
            ax.set_title("Сходимость МВИ: проекционные уравнения")
            ax.set_ylabel("Относительная проекционная невязка Rproj [-]")
        else:
            ax.set_title("Сходимость текущего шага")
            ax.set_ylabel("Относительная норма [-]")
        ax.set_xlabel("Итерация / цикл [-]");ax.grid(True,alpha=.3)
        if plotted:ax.legend()
        self.online_canvas_conv.draw_idle()

    def _poll_progress_files(self):
        try:
            if self.run_output_dir:
                if self.proc is not None and self.proc.poll() is None and getattr(self,"_online_wall_start",None) is not None:
                    now=time.monotonic()
                    paused_total=float(getattr(self,"_paused_total",0.0) or 0.0)
                    if bool(getattr(self,"_paused",False)) and getattr(self,"_pause_started_at",None) is not None:
                        paused_total+=max(0.0,now-float(self._pause_started_at))
                    live_elapsed=max(0.0,now-self._online_wall_start-paused_total)
                    self._online_set("elapsed",f"{live_elapsed:.1f} s")
                    last_evt=float(getattr(self,"_online_last_event_wall",self._online_wall_start) or self._online_wall_start)
                    last_hb=float(getattr(self,"_online_last_heartbeat_wall",last_evt) or last_evt)
                    quiet=max(0.0,now-last_evt)
                    if quiet>=12.0 and now-last_hb>=12.0:
                        stage="расчёт"
                        try:
                            stage=str(self.online_vars.get("stage").get() or "расчёт")
                        except Exception:pass
                        if hasattr(self,"progress_status_label"):
                            self.progress_status_label.configure(text=f"{stage} · расчёт продолжается · {live_elapsed:.0f} s")
                        self._log(f"[{live_elapsed:8.1f} s]   {stage}: calculation continues; waiting for the next solver event\n")
                        self._online_last_heartbeat_wall=now
                out=Path(self.run_output_dir); logp=out/"LIVE_PROGRESS.jsonl"
                if logp.exists():
                    if self._online_progress_path!=logp:
                        self._online_progress_path=logp;self._online_progress_pos=0
                    with logp.open("r",encoding="utf-8",errors="replace") as f:
                        f.seek(self._online_progress_pos)
                        for line in f:
                            try:self._online_update_event(json.loads(line))
                            except Exception:pass
                        self._online_progress_pos=f.tell()
                self._online_refresh_steps()
                if self.proc is not None and self.proc.poll() is None:
                    if self.run_kind=="MVI_SERIES":
                        completed=len(list(out.glob("MVI_*_SUMMARY.json"))) if out.exists() else 0
                    else:
                        completed=self._online_completed_requested_count()
                    if self.expected_steps:self._progress_update(completed,self.expected_steps)
        except Exception:
            pass
        self.after(350,self._poll_progress_files)

    def _result_output_dir(self):
        p=self.run_output_dir
        if p is None:
            txt=self.vars.get("output_dir",tk.StringVar()).get().strip(); p=Path(txt) if txt else None
        return None if p is None else Path(p).resolve()

    def refresh_results_view(self):
        super().refresh_results_view()
        try:
            self._all_result_png_paths=list(getattr(self,"_result_png_paths",[]) or [])
            self._filter_result_pngs()
            out=self._result_output_dir(); idxp=out/"NUMERIC"/"RESULTS_INDEX.json" if out else None
            if idxp is None or not idxp.exists():return
            self._numeric_result_index=json.loads(idxp.read_text(encoding="utf-8"))
            vals=[str(x["step"]) for x in self._numeric_result_index.get("steps",[])]
            self.result_step_combo.configure(values=vals)
            if vals:
                if self._v("result_step").get() not in vals:self._v("result_step").set(vals[-1])
                self._result_step_changed()
        except Exception as exc:self._log(f"RESULT INDEX warning: {exc}\n")

    def _result_step_changed(self):
        """Refresh selectable result fields for any accepted STEP.

        FULL stores DERIVED only for the last accepted step by design.  Earlier
        accepted steps are reconstructed in memory through _result_derived_data
        so the GUI field/thickness/3-D controls remain usable without generating
        heavy intermediate postprocessing files.
        """
        try:
            step=int(self._v("result_step").get())
            d=self._result_derived_data(step)
            keys=set(d)
            zvals=np.asarray(d.get("z",[0.0]),float)
            qshape=np.asarray(d["q_coefficients"]).shape if "q_coefficients" in keys else None
            zstd=np.asarray(d.get("standard_z",[]),float)
            fields=["w0","u0","v0"]
            if "q_coefficients" in keys:fields += ["u(z)","v(z)","w(z)"]
            if "epsilon_i" in keys:fields += ["epsilon_i","sigma_i"]
            if "detF" in keys:fields += ["detF"]
            if "radial_surface_load" in keys:fields += ["q_r"]
            # Birger fields are exposed only when they are actually reconstructed
            # for a Birger run; service GB=G0 arrays in Hooke runs stay hidden.
            if "GB_ratio" in keys:
                fields += ["GB_ratio","birger_nonlinear_thickness_fraction","birger_nonlinear_mask"]
            if "unloading_mask" in keys:fields += ["unloading_mask"]
            if "E" in keys:fields += ["E_xx","E_ss","E_rr","E_xs","E_xr","E_sr","S_xx","S_ss","S_rr","S_xs","S_xr","S_sr","u_tau","v_tau","w_tau"]
            if "strain" in keys:fields += ["KL_eps_xx","KL_eps_ss","KL_eps_zz","KL_gamma_xs","KL_sig_xx","KL_sig_ss","KL_tau_xs"]
            fields += ["R:"+k[len("resultant_"):] for k in sorted(keys) if k.startswith("resultant_")]
            fields=list(dict.fromkeys(fields))
            self.result_field_combo.configure(values=fields)
            if self._v("result_field").get() not in fields:self._v("result_field").set(fields[0])
            zv=[f"g{i}: z={x:.6g}" for i,x in enumerate(zvals)]
            for x in zstd:
                txt=f"z={float(x):.6g}"
                if txt not in zv:zv.append(txt)
            self.result_z_combo.configure(values=zv)
            if zv and (not self._v("result_z").get().strip() or self._v("result_z").get() not in zv):
                self._v("result_z").set(zv[min(len(zvals)//2,len(zvals)-1)])
            nt=1 if qshape is None else int(qshape[1]);tv=[str(i) for i in range(max(nt,1))]
            self.result_tau_combo.configure(values=tv)
            if self._v("result_tau").get() not in tv:self._v("result_tau").set(tv[0])
            shape2=None
            for kk in ("epsilon_i","GB_ratio","radial_surface_load","q_coefficients","strain","E","S"):
                if kk in d:
                    aa=np.asarray(d[kk])
                    if aa.ndim>=2:
                        shape2=aa.shape[-2:];break
            if shape2 is None:
                vis=self._load_step_visual_state(step)
                if vis is not None:shape2=np.asarray(vis["w"]).shape[-2:]
            if shape2 is not None:
                ns0,nx0=int(shape2[0]),int(shape2[1])
                self.result_ix_spin.configure(to=max(0,nx0-1));self.result_is_spin.configure(to=max(0,ns0-1))
                try:self._v("result_ix").set(str(max(0,min(int(float(self._v("result_ix").get() or 0)),nx0-1))))
                except Exception:self._v("result_ix").set(str(nx0//2))
                try:self._v("result_is").set(str(max(0,min(int(float(self._v("result_is").get() or 0)),ns0-1))))
                except Exception:self._v("result_is").set("0")
        except Exception as exc:self._log(f"RESULT step warning: {exc}\n")

    def _selected_result_z(self,zvals):
        txt=str(self._v("result_z","0").get()).strip()
        if ":" in txt:
            left,right=txt.split(":",1)
            if left.strip().lower().startswith("g"):
                i=int(left.strip()[1:]); i=max(0,min(i,len(zvals)-1)); return float(zvals[i]),i,True
            txt=right.strip()
        if txt.lower().startswith("z="):txt=txt.split("=",1)[1].strip()
        try:z=float(txt)
        except Exception:z=0.0
        i=int(np.argmin(np.abs(np.asarray(zvals,float)-z))) if len(zvals) else 0
        return z,i,False

    def _numeric_field_array(self,step,field):
        vis=self._load_step_visual_state(step)
        if vis is None:raise ValueError("State file not found")
        if field in ("w0","u0","v0"):
            key=field[0];return np.asarray(vis[key],float),field+" (midsurface coefficient)"
        d=self._result_derived_data(step);keys=set(d)
        zvals=np.asarray(d.get("z",[0.0]),float)
        zsel,zi,_is_actual=self._selected_result_z(zvals)
        try:tau=int(self._v("result_tau","0").get() or 0)
        except Exception:tau=0
        if field in ("u(z)","v(z)","w(z)"):
            if "q_coefficients" not in keys:raise ValueError("Arbitrary-z displacement is available for CUF results only")
            q=np.asarray(d["q_coefficients"],float);model=str((self._numeric_result_index or {}).get("cuf_model") or (self._saved_run_metadata().get("cuf_model") if isinstance(self._saved_run_metadata(),dict) else None) or "TE1");basis=tebasis.get_basis(model)
            F=np.asarray(basis.values(float(zsel)),float);ci={"u(z)":0,"v(z)":1,"w(z)":2}[field]
            arr=np.einsum("t,tsx->sx",F,q[ci],optimize=False);return arr,f"{field}, z={zsel:.6g} (exact CUF reconstruction)"
        if field in ("epsilon_i","sigma_i","GB_ratio","plastic_mask","birger_nonlinear_mask","detF","unloading_mask"):
            if field not in keys:raise ValueError(f"Field {field} is not available for this step")
            arr=np.asarray(d[field]);zi=max(0,min(zi,arr.shape[0]-1));return arr[zi],f"{field}, z-index={zi}"
        if field in ("plastic_thickness_fraction","birger_nonlinear_thickness_fraction"):
            if field not in keys:raise ValueError(f"Field {field} is not available for this step")
            return np.asarray(d[field]),field
        if field=="q_r":
            if "radial_surface_load" not in keys:raise ValueError("radial surface load is not available")
            return np.asarray(d["radial_surface_load"],float),"radial surface load q_r"
        if field in ("u_tau","v_tau","w_tau"):
            if "q_coefficients" not in keys:raise ValueError("CUF coefficients are not available")
            q=np.asarray(d["q_coefficients"]);ci={"u_tau":0,"v_tau":1,"w_tau":2}[field];tau=max(0,min(tau,q.shape[1]-1));return q[ci,tau],f"{field}, tau={tau}"
        if field.startswith("E_") or field.startswith("S_"):
            names=("xx","ss","rr","xs","xr","sr");base=field[0];comp=names.index(field[2:]);arr=np.asarray(d[base],float);zi=max(0,min(zi,arr.shape[0]-1));return arr[zi,comp],f"{field}, z-index={zi}"
        if field.startswith("KL_eps_") or field=="KL_gamma_xs":
            names=("KL_eps_xx","KL_eps_ss","KL_eps_zz","KL_gamma_xs");comp=names.index(field);arr=np.asarray(d["strain"],float);zi=max(0,min(zi,arr.shape[0]-1));return arr[zi,comp],f"{field}, z-index={zi}"
        if field.startswith("KL_sig_") or field=="KL_tau_xs":
            names=("KL_sig_xx","KL_sig_ss","KL_tau_xs");comp=names.index(field);arr=np.asarray(d["stress"],float);zi=max(0,min(zi,arr.shape[0]-1));return arr[zi,comp],f"{field}, z-index={zi}"
        if field.startswith("R:"):
            key="resultant_"+field[2:]
            if key not in d:raise ValueError(f"Resultant {field[2:]} is not available")
            arr=np.asarray(d[key],float)
            if arr.ndim==2:return arr,field[2:]
            if arr.ndim==3:
                tau=max(0,min(tau,arr.shape[0]-1));return arr[tau],f"{field[2:]}, tau={tau}"
            raise ValueError(f"Resultant {field[2:]} has unsupported shape {arr.shape}")
        raise ValueError(f"Unsupported field {field}")

    def _preview_numeric_result(self):
        try:
            step=int(self._v("result_step").get());field=self._v("result_field").get();a,title=self._numeric_field_array(step,field)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._result_figure_canvas is not None:self._result_figure_canvas.get_tk_widget().destroy();self._result_figure_canvas=None
            try:self.result_preview_label.pack_forget()
            except Exception:pass
            L,R=self._result_geometry_from_output();labels=self._gui_units();unit=self._online_field_unit(field)
            arr=np.asarray(a,float)
            finite=arr[np.isfinite(arr)]
            mn=float(np.min(finite)) if finite.size else float("nan");mx=float(np.max(finite)) if finite.size else float("nan");rng=mx-mn
            note=f"min={mn:.6g}; max={mx:.6g}; Δ={rng:.3g}"
            if arr.ndim==2 and arr.shape[0]>1 and finite.size:
                circ=float(np.nanmax(np.ptp(arr,axis=0)));scale=max(abs(mn),abs(mx),abs(rng),1e-30)
                if circ/scale<=1e-8:note+=f"; осесимметрично по θ/s (max Δθ={circ:.3g})"
            fig=Figure(figsize=(7,5));ax=fig.add_subplot(111);im=ax.imshow(arr,aspect="auto",origin="lower",extent=(0.0,L,0.0,2*np.pi*R));ax.set_title(f"Шаг {step}: {title}");ax.set_xlabel(f"x [{labels['length']}]");ax.set_ylabel(f"s [{labels['length']}]");cb=fig.colorbar(im,ax=ax);cb.set_label(f"[{unit}]");fig.text(0.5,0.01,note,ha="center",fontsize=8);fig.tight_layout(rect=(0,0.045,1,1));self._result_current_figure=fig;self._result_figure_canvas=FigureCanvasTkAgg(fig,master=self.result_preview_frame);self._result_figure_canvas.draw();self._result_figure_canvas.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:messagebox.showerror("Числовое поле",str(exc))

    def _result_selected_point(self,shape2):
        ns,nx=(int(shape2[-2]),int(shape2[-1]))
        try:ix=int(float(self._v("result_ix","0").get() or 0))
        except Exception:ix=0
        try:js=int(float(self._v("result_is","0").get() or 0))
        except Exception:js=0
        return max(0,min(ix,nx-1)),max(0,min(js,ns-1))

    def _result_derived_data(self,step):
        """Load saved DERIVED data or reconstruct it on demand from an accepted STEP.

        This is visualization-only.  It never calls the solver, changes MVI state, or
        writes a FULL package.  The on-demand path is what makes 3-D inspection of an
        intermediate accepted step possible while FULL remains limited to the last
        accepted step.
        """
        step=int(step); out=self._result_output_dir()
        if out is None: raise ValueError("Папка результатов не выбрана")
        dp=out/"NUMERIC"/f"STEP_{step:04d}_DERIVED.npz"
        if dp.exists():
            with np.load(dp,allow_pickle=False) as zf:
                return {k:np.asarray(zf[k]) for k in zf.files}
        cache=getattr(self,"_result_dynamic_derived_cache",None)
        if isinstance(cache,dict) and cache.get("step")==step and isinstance(cache.get("data"),dict):
            return cache["data"]
        candidates=(out/f"STEP_{step:04d}_STATE.npz",out/f"STEP_{step:04d}.npz")
        sp=next((q for q in candidates if q.exists()),None)
        if sp is None: raise ValueError(f"Принятый STEP {step} не найден")
        # Reuse the exact postprocessor reconstruction in memory.  No output files
        # are generated here, so the established FULL/Excel/sections output is untouched.
        import FINAL_POSTPROCESS as _fpost
        summary=_fpost._summary(out)
        st=_fpost._state_from_npz(sp)
        if "q" in st:
            d=_fpost._cuf_derived(st,summary)
            d["q_coefficients"]=np.asarray(st["q"],float)
        elif "U" in st:
            d=_fpost._kl_derived(st,summary)
        else:
            raise ValueError(f"STEP {step} не содержит реконструируемого CUF/KL состояния")
        # Add material-zone arrays only when the saved run actually uses Birger.
        mode=str(summary.get("mode") or "").upper()
        if mode.endswith("BIRGER") and "GB" in d:
            try:
                geom=summary.get("geometry") or {}; mat=summary.get("material") or {}
                arr,_info=_fpost._birger_zone_arrays(
                    d,summary,float(geom.get("R")),float(mat.get("E0")),float(mat.get("nu0"))
                )
                d.update(arr)
            except Exception:
                pass
        raw=st.get("raw") if isinstance(st,dict) else None
        if isinstance(raw,dict):
            for key in ("unloading_mask","plastic_mask","detF"):
                if key in raw and key not in d: d[key]=np.asarray(raw[key])
        d={k:np.asarray(v) if isinstance(v,(list,tuple,np.ndarray)) else v for k,v in d.items()}
        self._result_dynamic_derived_cache={"step":step,"data":d}
        return d

    def _result_volume_displacements(self,step,zvals):
        """Return u,v,w arrays shaped [z,s,x] for 3-D geometry rendering."""
        d=self._result_derived_data(step); zvals=np.asarray(zvals,float)
        if "q_coefficients" in d:
            q=np.asarray(d["q_coefficients"],float)
            model=str((self._numeric_result_index or {}).get("cuf_model") or (self._saved_run_metadata().get("cuf_model") if isinstance(self._saved_run_metadata(),dict) else None) or "TE1")
            basis=tebasis.get_basis(model)
            F=np.asarray([basis.values(float(zz)) for zz in zvals],float)
            disp=np.einsum("gt,ctsx->gcsx",F,q,optimize=False)
            return disp[:,0],disp[:,1],disp[:,2]
        vis=self._load_step_visual_state(step)
        if vis is None: raise ValueError("State file not found")
        u=np.asarray(vis["u"],float);v=np.asarray(vis["v"],float);w=np.asarray(vis["w"],float)
        return (np.repeat(u[None,:,:],len(zvals),axis=0),
                np.repeat(v[None,:,:],len(zvals),axis=0),
                np.repeat(w[None,:,:],len(zvals),axis=0))

    def _numeric_3d_field(self,step,field):
        """Return (z, array[z,s,x], label) from saved or on-demand result data."""
        zf=self._result_derived_data(step); keys=set(zf)
        zvals=np.asarray(zf["z"],float) if "z" in keys else np.asarray([0.0])
        if field in ("u(z)","v(z)","w(z)"):
            if "q_coefficients" not in keys:raise ValueError("Through-thickness displacement reconstruction is available for CUF only")
            q=np.asarray(zf["q_coefficients"],float); model=str((self._numeric_result_index or {}).get("cuf_model") or (self._saved_run_metadata().get("cuf_model") if isinstance(self._saved_run_metadata(),dict) else None) or "TE1");basis=tebasis.get_basis(model)
            F=np.asarray([basis.values(float(zz)) for zz in zvals],float);ci={"u(z)":0,"v(z)":1,"w(z)":2}[field]
            return zvals,np.einsum("gt,tsx->gsx",F,q[ci],optimize=False),field
        direct=("epsilon_i","sigma_i","GB_ratio","plastic_mask","birger_nonlinear_mask","detF","unloading_mask")
        if field in direct:
            if field not in keys:raise ValueError(f"Field {field} is not stored/reconstructable for this step")
            a=np.asarray(zf[field],float)
            if a.ndim!=3:raise ValueError(f"Field {field} is not a z-s-x field")
            return zvals,a,field
        if field.startswith("E_") or field.startswith("S_"):
            names=("xx","ss","rr","xs","xr","sr");base=field[0];comp=names.index(field[2:]);a=np.asarray(zf[base],float)
            return zvals,a[:,comp],field
        if field.startswith("KL_eps_") or field=="KL_gamma_xs":
            names=("KL_eps_xx","KL_eps_ss","KL_eps_zz","KL_gamma_xs");comp=names.index(field);return zvals,np.asarray(zf["strain"],float)[:,comp],field
        if field.startswith("KL_sig_") or field=="KL_tau_xs":
            names=("KL_sig_xx","KL_sig_ss","KL_tau_xs");comp=names.index(field);return zvals,np.asarray(zf["stress"],float)[:,comp],field
        raise ValueError(f"Field {field} has no through-thickness representation")

    def _preview_thickness_profile(self):
        try:
            step=int(self._v("result_step").get());field=self._v("result_field").get()
            # CUF midsurface coefficient w0/u0/v0 is a 2-D quantity.  For a
            # through-thickness profile the physical CUF displacement must be
            # reconstructed from all thickness coefficients.  Make the common
            # GUI action do that automatically instead of showing an empty/
            # unsupported profile when the default field is w0.
            if field in ("u0","v0","w0"):
                candidate=field[0]+"(z)"
                try:
                    if "q_coefficients" in self._result_derived_data(step):field=candidate
                except Exception:
                    pass
            zvals,a,label=self._result_volume_field(step,field)
            ix,js=self._result_selected_point(a.shape[-2:]);series=np.asarray(a[:,js,ix],float)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._result_figure_canvas is not None:self._result_figure_canvas.get_tk_widget().destroy();self._result_figure_canvas=None
            try:self.result_preview_label.pack_forget()
            except Exception:pass
            labels=self._gui_units();unit=self._online_field_unit(field)
            fig=Figure(figsize=(7,5));ax=fig.add_subplot(111);ax.plot(series,zvals,marker="o");ax.set_xlabel(f"{label} [{unit}]");ax.set_ylabel(f"z [{labels['length']}]");ax.set_title(f"Шаг {step}: профиль по толщине, is={js}, ix={ix}");ax.grid(True,alpha=.3);fig.tight_layout()
            self._result_current_figure=fig;self._result_figure_canvas=FigureCanvasTkAgg(fig,master=self.result_preview_frame);self._result_figure_canvas.draw();self._result_figure_canvas.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:messagebox.showerror("Профиль по толщине",str(exc))

    def _result_geometry_from_output(self):
        out=self._result_output_dir()
        for p in ((out/"REPORT"/"CALCULATION_SUMMARY.json") if out else None,(out/"FINAL_RUN_SUMMARY.json") if out else None,(out/"INPUT"/"FINAL_CASE.json") if out else None):
            if p is not None and p.exists():
                try:
                    j=json.loads(p.read_text(encoding="utf-8"));g=j.get("geometry") or {}
                    if g:return float(g.get("L")),float(g.get("R"))
                except Exception:pass
        return float(self._v("L").get()),float(self._v("R").get())

    def _result_volume_field(self,step,field):
        """Return a physical z-s-x field for result section viewers."""
        field=str(field)
        if field in ("u0","v0","w0"):
            data=self._result_derived_data(int(step))
            if "q_coefficients" in data:
                return self._numeric_3d_field(int(step),field[0]+"(z)")
            # Classical KL has w independent of z.  This is the only
            # through-thickness displacement reconstruction available from its
            # saved state; do not invent u/v thickness profiles.
            if field=="w0":
                vis=self._load_step_visual_state(int(step))
                if vis is None:raise ValueError("State file not found")
                zvals=np.asarray(data.get("z",[0.0]),float)
                w=np.asarray(vis["w"],float)
                return zvals,np.repeat(w[None,:,:],len(zvals),axis=0),"w (KL, independent of z)"
            raise ValueError("Для KL профиль u/v по толщине не сохраняется")
        return self._numeric_3d_field(int(step),field)

    def _preview_numeric_cut(self,kind):
        try:
            step=int(self._v("result_step").get());field=self._v("result_field").get();zvals,a,label=self._result_volume_field(step,field)
            ix,js=self._result_selected_point(a.shape[-2:]);L,_R=self._result_geometry_from_output()
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._result_figure_canvas is not None:self._result_figure_canvas.get_tk_widget().destroy();self._result_figure_canvas=None
            try:self.result_preview_label.pack_forget()
            except Exception:pass
            fig=Figure(figsize=(7.4,5),facecolor="#f7f9fc");ax=fig.add_subplot(111)
            import FINAL_VIDEO_EXPORT as fvideo
            finite=np.asarray(a,float);finite=finite[np.isfinite(finite)];mn=float(np.min(finite));mx=float(np.max(finite));norm,cmap=fvideo._engineering_norm_cmap(mn,mx)
            if kind=="x-z":
                sec=np.asarray(a[:,js,:],float);xx=np.linspace(0,L,sec.shape[1]);im=ax.pcolormesh(xx,zvals,sec,shading="auto",cmap=cmap,norm=norm);ax.set_xlabel(f"x [{self._gui_units()['length']}]");title=f"{label}: x-z at is={js}"
            else:
                sec=np.asarray(a[:,:,ix],float);th=np.linspace(0,2*np.pi,sec.shape[1],endpoint=False);im=ax.pcolormesh(th,zvals,sec,shading="auto",cmap=cmap,norm=norm);ax.set_xlabel("theta [rad]");title=f"{label}: s-z at ix={ix}"
            ax.set_ylabel(f"z [{self._gui_units()['length']}]");ax.set_title(f"Шаг {step}: {title}");cb=fig.colorbar(im,ax=ax);cb.set_label(f"[{self._online_field_unit(field)}]");fig.tight_layout()
            self._result_current_figure=fig;self._result_figure_canvas=FigureCanvasTkAgg(fig,master=self.result_preview_frame);self._result_figure_canvas.draw();self._result_figure_canvas.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:messagebox.showerror("Разрез по толщине",str(exc))

    def _preview_numeric_volume_3d(self):
        """Interactive 3-D view of already saved z-s-x result fields.

        This is postprocessing only.  No solver or MVI routine is called.
        """
        try:
            step=int(self._v("result_step").get())
            field=self._v("result_field").get()
            if field in ("plastic_thickness_fraction","birger_nonlinear_thickness_fraction","q_r") or field.startswith("R:"):
                raise ValueError("Для выбранного 2D-поля используйте «Показать поле» или «3D оболочка». Для объёмного 3D выберите GB_ratio, epsilon_i, sigma_i, detF, маску Биргера/пластичности, E/S или KL-поле.")
            zvals,a,label=self._result_volume_field(step,field)
            a=np.asarray(a,float)
            if a.ndim!=3:
                raise ValueError(f"Поле {field} не является объёмным z-s-x полем")
            L,R=self._result_geometry_from_output(); nz,ns,nx=a.shape
            # Keep interactive rendering responsive for very fine meshes.
            max_points=60000
            stride=max(1,int(np.ceil((nz*ns*nx/max_points)**0.5))) if nz*ns*nx>max_points else 1
            ixv=np.arange(0,nx,stride,dtype=int); jsv=np.arange(0,ns,stride,dtype=int)
            if ixv[-1] != nx-1: ixv=np.r_[ixv,nx-1]
            xx=np.linspace(0.0,float(L),nx)[ixv]
            th=np.linspace(0.0,2*np.pi,ns,endpoint=False)[jsv]
            X=[];Y=[];Z=[];C=[]
            for g,z in enumerate(np.asarray(zvals,float)):
                xg,tg=np.meshgrid(xx,th)
                rg=float(R)+float(z)
                X.append(xg.ravel());Y.append((rg*np.cos(tg)).ravel());Z.append((rg*np.sin(tg)).ravel())
                C.append(a[g][np.ix_(jsv,ixv)].ravel())
            X=np.concatenate(X);Y=np.concatenate(Y);Z=np.concatenate(Z);C=np.concatenate(C)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._result_figure_canvas is not None:
                self._result_figure_canvas.get_tk_widget().destroy();self._result_figure_canvas=None
            try:self.result_preview_label.pack_forget()
            except Exception:pass
            fig=Figure(figsize=(7.6,5.6));ax=fig.add_subplot(111,projection="3d")
            sc=ax.scatter(X,Y,Z,c=C,s=7,alpha=.85)
            unit=self._online_field_unit(field);lu=self._gui_units()["length"]
            cb=fig.colorbar(sc,ax=ax,shrink=.68,pad=.08);cb.set_label(f"{label} [{unit}]")
            ax.set_title(f"Шаг {step}: 3D поле {label}")
            ax.set_xlabel(f"x [{lu}]");ax.set_ylabel(f"Y [{lu}]");ax.set_zlabel(f"Z [{lu}]")
            try:ax.set_box_aspect((max(float(L),1e-12),max(2*float(R),1e-12),max(2*float(R),1e-12)))
            except Exception:pass
            fig.tight_layout()
            self._result_current_figure=fig
            self._result_figure_canvas=FigureCanvasTkAgg(fig,master=self.result_preview_frame)
            self._result_figure_canvas.draw();self._result_figure_canvas.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:
            messagebox.showerror("3D поле результата",str(exc))

    def _result_deformation_scale(self):
        try:
            txt=self._v("deformation_scale","1").get().strip(); return float(txt) if txt else 1.0
        except Exception:return 1.0

    def _result_visual_thickness_scale(self):
        """Display-only thickness exaggeration for 3-D cut pictures/video."""
        try:
            var=getattr(self,"_result_visual_thickness_var",None)
            val=float(var.get()) if var is not None else 5.0
        except Exception:
            val=5.0
        return max(0.1,min(val,50.0))

    def _preview_numeric_3d(self):
        """Presentation-style 3-D shell coloured by the selected saved field."""
        try:
            import FINAL_VIDEO_EXPORT as fvideo
            step=int(self._v("result_step").get());field=self._v("result_field").get()
            L,R=self._result_geometry_from_output();scale=self._result_deformation_scale();units=self._gui_units()
            # Prefer full z-s-x data.  For a native 2-D field, wrap the saved
            # midsurface into one display layer; no numerical value is changed.
            try:
                zvals,vol,label=self._numeric_3d_field(step,field);vol=np.asarray(vol,float)
                zi=self._selected_result_z(zvals)[1];zi=max(0,min(int(zi),len(zvals)-1))
                d=self._result_visual_frame_data(step,field);d["layer_index"]=zi
                finite=vol[np.isfinite(vol)]
            except Exception:
                layer,label=self._numeric_field_array(step,field);layer=np.asarray(layer,float)
                vis=self._load_step_visual_state(step)
                if vis is None:raise ValueError("State file not found")
                u,v,w=(np.asarray(vis[k],float) for k in ("u","v","w"))
                spec,load_value,load_label=self._result_visual_load_snapshot(step)
                vmeta=self._result_visual_metadata()
                d={**vmeta,"L":float(L),"R":float(R),"scale":float(scale),"length_unit":units["length"],
                   "field":field,"field_label":label,"field_unit":self._online_field_unit(field),"zvals":np.asarray([0.0]),
                   "field3d":layer[None,...],"color_values":layer,"uvol":u[None,...],"vvol":v[None,...],"wvol":w[None,...],
                   "layer_index":0,"global_zmax":0.0,"global_max_displacement":float(max(np.max(np.abs(u)),np.max(np.abs(v)),np.max(np.abs(w)))),
                   "load_spec":spec,"load_value":load_value,"load_label":load_label,"step":step}
                finite=layer[np.isfinite(layer)]
            if not finite.size:raise ValueError("В выбранном поле нет конечных значений")
            mn=float(np.min(finite));mx=float(np.max(finite));norm,cmap=fvideo._engineering_norm_cmap(mn,mx)
            d["_video_global_min"]=mn;d["_video_global_max"]=mx;d["step"]=step
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._result_figure_canvas is not None:self._result_figure_canvas.get_tk_widget().destroy();self._result_figure_canvas=None
            try:self.result_preview_label.pack_forget()
            except Exception:pass
            fig=Figure(figsize=(10.6,6.2),facecolor="#f7f9fc");ax=fvideo._presentation_3d_axis(fig)
            fvideo._plot_shell(ax,d,norm,cmap);fvideo._presentation_colorbar(fig,norm,cmap,d);fvideo._presentation_title(fig,d,step,"3D выбранное поле")
            self._result_current_figure=fig;self._result_figure_canvas=FigureCanvasTkAgg(fig,master=self.result_preview_frame);self._result_figure_canvas.draw();self._result_figure_canvas.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:messagebox.showerror("3D оболочка",str(exc))

    def _preview_numeric_cutaway_3d(self):
        """Presentation through-thickness cutaway of already saved z-s-x data."""
        try:
            import FINAL_VIDEO_EXPORT as fvideo
            step=int(self._v("result_step").get());field=self._v("result_field").get();d=self._result_visual_frame_data(step,field)
            finite=np.asarray(d["field3d"],float);finite=finite[np.isfinite(finite)]
            if not finite.size:raise ValueError("В выбранном поле нет конечных значений")
            mn=float(np.min(finite));mx=float(np.max(finite));norm,cmap=fvideo._engineering_norm_cmap(mn,mx)
            d["_video_global_min"]=mn;d["_video_global_max"]=mx
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._result_figure_canvas is not None:self._result_figure_canvas.get_tk_widget().destroy();self._result_figure_canvas=None
            try:self.result_preview_label.pack_forget()
            except Exception:pass
            fig=Figure(figsize=(10.6,6.2),facecolor="#f7f9fc");ax=fvideo._presentation_3d_axis(fig)
            fvideo._plot_cutaway(ax,d,norm,cmap);fvideo._presentation_colorbar(fig,norm,cmap,d);fvideo._presentation_title(fig,d,step,"3D разрез по толщине")
            self._result_current_figure=fig;self._result_figure_canvas=FigureCanvasTkAgg(fig,master=self.result_preview_frame);self._result_figure_canvas.draw();self._result_figure_canvas.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:messagebox.showerror("3D разрез по толщине",str(exc))

    def _result_visual_load_snapshot(self,step):
        """Display-only immutable load value/spec for pictures and VIDEO."""
        spec,_system=self._load_context_from_saved_run()
        load_value=None;load_label="load"
        try:
            hist=self._video_history_snapshot(self._video_available_steps())
            by_step={int(st):float(v) for st,v in zip(hist.get("steps",[]),hist.get("loads",[]))}
            load_value=by_step.get(int(step));load_label=str(hist.get("load_label") or "load")
        except Exception:pass
        return spec or {},load_value,load_label

    def _result_visual_metadata(self):
        """Compact immutable RUN metadata used only by pictures and VIDEO."""
        sm=self._saved_run_metadata() if hasattr(self,"_saved_run_metadata") else {}
        sm=sm if isinstance(sm,dict) else {}
        geo=sm.get("geometry") or {}; units=sm.get("units") or {}
        try:h=float(geo.get("h",0.0) or 0.0)
        except Exception:h=0.0
        return {
            "h":h,
            "theory":sm.get("theory"),"cuf_model":sm.get("cuf_model") or "",
            "geometry_model":sm.get("geometry_model"),"material_model":sm.get("material_model"),
            "method":sm.get("method"),"rank":sm.get("rank"),
            "unit_system":units.get("system"),
        }

    def _result_visual_frame_data(self,step,field):
        """Prepare already-saved z-s-x data for engineering 3-D pictures."""
        effective={"u0":"u(z)","v0":"v(z)","w0":"w(z)","u":"u(z)","v":"v(z)","w":"w(z)"}.get(str(field),str(field))
        zvals,a,label=self._numeric_3d_field(int(step),effective)
        zvals=np.asarray(zvals,float);a=np.asarray(a,float)
        if a.ndim!=3:raise ValueError(f"Поле {effective} не является объёмным z-s-x полем")
        uvol,vvol,wvol=self._result_volume_displacements(int(step),zvals)
        ix,js=self._result_selected_point(a.shape[-2:])
        section_note=""
        # Default 0/0 historically put every transverse section on the left end.
        # Treat untouched 0/0 as AUTO and use the critical |w|max location.
        try:
            if int(ix)==0 and int(js)==0:
                vis=self._load_step_visual_state(int(step))
                if vis is not None:
                    ww=np.asarray(vis.get("w"),float)
                    if ww.ndim==2 and ww.size:
                        js,ix=np.unravel_index(int(np.nanargmax(np.abs(ww))),ww.shape)
                        section_note="в месте |w|max / at |w|max"
        except Exception:pass
        L,R=self._result_geometry_from_output();scale=self._result_deformation_scale();units=self._gui_units()
        try:layer_index=int(self._selected_result_z(zvals)[1])
        except Exception:layer_index=len(zvals)//2
        spec,load_value,load_label=self._result_visual_load_snapshot(step)
        unit=(units.get("displacement","mm") if effective.endswith("(z)") else self._online_field_unit(effective))
        vmeta=self._result_visual_metadata()
        return {
            **vmeta,
            "L":float(L),"R":float(R),"scale":float(scale),"visual_thickness_scale":float(self._result_visual_thickness_scale()),"length_unit":units.get("length","mm"),
            "field":effective,"field_label":label,"field_unit":unit,"zvals":zvals,"field3d":a,
            "uvol":np.asarray(uvol,float),"vvol":np.asarray(vvol,float),"wvol":np.asarray(wvol,float),
            "layer_index":max(0,min(int(layer_index),len(zvals)-1)),"point_ix":int(ix),"point_is":int(js),
            "max_displacement":float(max(np.nanmax(np.abs(uvol)),np.nanmax(np.abs(vvol)),np.nanmax(np.abs(wvol)))),
            "zmax":float(np.max(np.abs(zvals))) if len(zvals) else 0.0,"global_zmax":float(np.max(np.abs(zvals))) if len(zvals) else 0.0,
            "global_max_displacement":float(max(np.nanmax(np.abs(uvol)),np.nanmax(np.abs(vvol)),np.nanmax(np.abs(wvol)))),
            "load_value":load_value,"load_label":load_label,"load_spec":spec,"step":int(step),"section_note":section_note,
        }

    def _preview_engineering_3d_cut(self,kind):
        """Presentation longitudinal/transverse/combined sections of saved data."""
        names={"longitudinal":"Продольный разрез","transverse":"Поперечный разрез","combined":"Комбинированный разрез"}
        modes={"longitudinal":"3D продольный разрез","transverse":"3D поперечный разрез","combined":"3D комбинированный разрез"}
        try:
            import FINAL_VIDEO_EXPORT as fvideo
            step=int(self._v("result_step").get());field=self._v("result_field").get();d=self._result_visual_frame_data(step,field)
            finite=np.asarray(d["field3d"],float);finite=finite[np.isfinite(finite)]
            if not finite.size:raise ValueError("В выбранном поле нет конечных значений")
            mn=float(np.min(finite));mx=float(np.max(finite));norm,cmap=fvideo._engineering_norm_cmap(mn,mx)
            d["_video_global_min"]=mn;d["_video_global_max"]=mx
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._result_figure_canvas is not None:self._result_figure_canvas.get_tk_widget().destroy();self._result_figure_canvas=None
            try:self.result_preview_label.pack_forget()
            except Exception:pass
            fig=Figure(figsize=(10.6,6.2),facecolor="#f7f9fc");ax=fvideo._presentation_3d_axis(fig)
            if kind=="longitudinal":fvideo._plot_longitudinal_cut(ax,d,norm,cmap)
            elif kind=="transverse":fvideo._plot_transverse_cut(ax,d,norm,cmap)
            elif kind=="combined":fvideo._plot_combined_cut(ax,d,norm,cmap)
            else:raise ValueError(f"Неизвестный 3D-разрез: {kind}")
            fvideo._presentation_colorbar(fig,norm,cmap,d);fvideo._presentation_title(fig,d,step,modes[kind])
            self._result_current_figure=fig;self._result_figure_canvas=FigureCanvasTkAgg(fig,master=self.result_preview_frame);self._result_figure_canvas.draw();self._result_figure_canvas.get_tk_widget().pack(fill="both",expand=True)
        except Exception as exc:messagebox.showerror(names.get(kind,"3D разрез"),str(exc))

    def _save_current_result_png(self):
        try:
            if self._result_current_figure is None:raise ValueError("Сначала постройте 3D/график результата")
            step=str(self._v("result_step","").get() or "STEP")
            field=str(self._v("result_field","field").get() or "field").replace("/","_").replace("\\","_")
            p=filedialog.asksaveasfilename(title="Сохранить текущую картинку",defaultextension=".png",initialfile=f"STEP_{step}__{field}.png",filetypes=[("PNG","*.png")])
            if p:self._result_current_figure.savefig(p,dpi=190,bbox_inches="tight")
        except Exception as exc:messagebox.showerror("Сохранить PNG",str(exc))

    def _video_available_steps(self):
        """Accepted steps available for optional video export."""
        out=self._result_output_dir()
        if out is None:return []
        vals=[]
        idx=self._numeric_result_index if isinstance(self._numeric_result_index,dict) else None
        if idx:
            for item in idx.get("steps",[]):
                try:vals.append(int(item.get("step")))
                except Exception:pass
        if not vals:
            for pat in ("STEP_*_STATE.npz","STEP_*.npz"):
                for fp in out.glob(pat):
                    try:
                        stem=fp.stem.replace("_STATE","");vals.append(int(stem.split("_")[-1]))
                    except Exception:pass
        return sorted(set(vals))

    def _video_history_snapshot(self,steps):
        """Immutable accepted-step load-deflection history for VIDEO.

        Use exactly the same reference point as FINAL_POSTPROCESS:
        x=L/2, theta=0.  It is fixed for every accepted step and is independent
        of the moving global |w|max location.  Both histories include the
        undeformed origin (0,0).
        """
        steps=[int(s) for s in steps]
        meta={}
        idx=self._numeric_result_index if isinstance(self._numeric_result_index,dict) else {}
        for item in idx.get("steps",[]):
            try:meta[int(item.get("step"))]=item
            except Exception:pass
        # Fixed physical control point, matching FINAL_POSTPROCESS:
        # x=L/2 on theta=0 generator.  It never follows |w|max.
        ref_is=0;ref_ix=None;ref_theta=0.0
        if steps:
            visf=self._load_step_visual_state(int(steps[-1]))
            if visf is not None:
                wf=np.asarray(visf["w"],float)
                if wf.ndim==2 and wf.size:
                    ref_ix=int(wf.shape[1]//2)
        factors=[];wref=[];wmax=[]
        for step in steps:
            item=meta.get(step,{})
            try:factor=float(item.get("load_factor"))
            except Exception:factor=float(self._online_factor_for_step(step,{}))
            factors.append(factor)
            vis=self._load_step_visual_state(step)
            if vis is None:raise ValueError(f"STEP {step}: состояние не найдено")
            w=np.asarray(vis["w"],float)
            js=max(0,min(int(ref_is),w.shape[0]-1));ix=(w.shape[1]//2 if ref_ix is None else max(0,min(int(ref_ix),w.shape[1]-1)))
            wref.append(abs(float(w[js,ix])));wmax.append(float(np.nanmax(np.abs(w))))
        loads,load_label,_key=self._current_load_axis(factors)
        du=self._gui_units()["displacement"]
        return {
            "steps":[0]+steps,"loads":[0.0]+[float(x) for x in loads],
            "wcenter":[0.0]+wref,"wref":[0.0]+wref,"wmax":[0.0]+wmax,
            "reference_is":int(ref_is),"reference_theta":float(ref_theta),
            "reference_label":f"|w|(x=L/2, θ={np.degrees(ref_theta):.1f}°) / контрольная точка",
            "load_label":load_label,"displacement_unit":du,
        }

    @staticmethod
    def _video_unit_for_field(field,units):
        c=str(field).lower()
        if c.startswith(("u","v","w")):return units.get("displacement","mm")
        if "sigma" in c or c.startswith("s_") or c.startswith("kl_sig") or c=="kl_tau_xs":return units.get("stress","MPa")
        return "-"

    def _video_run_has_birger(self):
        """True only for a saved run whose actual material mode is Birger."""
        meta=self._saved_run_metadata()
        try:
            tokens=[]
            if isinstance(meta,dict):
                for key in ("mode","material_model","constitutive_model"):
                    if meta.get(key) is not None:tokens.append(str(meta.get(key)))
                mat=meta.get("material")
                if isinstance(mat,dict):
                    for key in ("mode","model","material_model","constitutive_model"):
                        if mat.get(key) is not None:tokens.append(str(mat.get(key)))
            return any("BIRGER" in x.upper() for x in tokens)
        except Exception:
            return False

    @staticmethod
    def _video_pick_field(mode,requested,available_fields):
        fields=list(dict.fromkeys(str(x) for x in (available_fields or []) if str(x)))
        req=str(requested or "")
        if mode=="3D деформация":return "w"
        if mode=="Birger GB/G0":return "GB_ratio"
        if mode=="3D напряжения":
            if req=="sigma_i" or req.startswith("S_") or req.startswith("KL_sig_") or req=="KL_tau_xs":return req
            for x in fields:
                if x=="sigma_i" or x.startswith("S_") or x.startswith("KL_sig_") or x=="KL_tau_xs":return x
            raise ValueError("Для VIDEO не найдено доступное поле напряжений")
        if mode=="3D деформации":
            if req=="epsilon_i" or req.startswith("E_") or req.startswith("KL_eps_") or req=="KL_gamma_xs":return req
            for x in fields:
                if x=="epsilon_i" or x.startswith("E_") or x.startswith("KL_eps_") or x=="KL_gamma_xs":return x
            raise ValueError("Для VIDEO не найдено доступное поле деформаций")
        return req or "E_xx"

    def _video_build_jobs(self,requested_mode,requested_field,available_fields,has_birger=False):
        """Return a de-duplicated VIDEO job list without touching saved results."""
        import FINAL_VIDEO_EXPORT as fvideo
        requested_mode=str(requested_mode);requested_field=str(requested_field or "")
        available_fields=list(available_fields or [])
        has_birger=bool(has_birger)
        if requested_mode!=fvideo.ALL_VIDEO_MODE:
            if requested_mode==fvideo.BIRGER_VIDEO_MODE and not has_birger:
                raise ValueError("Birger-видео доступно только для расчёта с Birger")
            return [(requested_mode,self._video_pick_field(requested_mode,requested_field,available_fields))]
        stress=self._video_pick_field("3D напряжения",requested_field,available_fields)
        strain=self._video_pick_field("3D деформации",requested_field,available_fields)
        selected=requested_field or strain
        jobs=[("3D деформация","w"),("3D напряжения",stress),("3D деформации",strain)]
        represented={"w",stress,strain}
        if has_birger:represented.add("GB_ratio")
        if selected not in represented:
            jobs.append(("3D выбранное поле",selected))
        jobs.extend([
            ("3D разрез по толщине",selected),("3D продольный разрез",selected),
            ("3D поперечный разрез",selected),("3D комбинированный разрез",selected),
            ("3D + нагрузка–прогиб",selected),("Разрез x-z",selected),
            ("Разрез s-z",selected),("Профиль по толщине",selected),
        ])
        if has_birger:jobs.append((fvideo.BIRGER_VIDEO_MODE,"GB_ratio"))
        return jobs

    def _video_field_options(self,has_birger=False):
        vals=list(self.result_field_combo.cget("values")) if hasattr(self,"result_field_combo") else []
        allowed=[]
        for x in vals:
            x=str(x)
            if x in ("u0","v0","w0","u(z)","v(z)","w(z)","epsilon_i","sigma_i","detF") \
               or x.startswith(("E_","S_","KL_eps_","KL_sig_")) or x=="KL_gamma_xs" or x=="KL_tau_xs":
                allowed.append(x)
        meta=self._saved_run_metadata()
        theory=str(meta.get("theory","") if isinstance(meta,dict) else "").upper()
        if theory=="KL":
            defaults=("KL_eps_xx","KL_eps_ss","KL_eps_zz","KL_gamma_xs","KL_sig_xx","KL_sig_ss","KL_tau_xs","epsilon_i","sigma_i","detF")
        else:
            defaults=("E_xx","E_ss","E_rr","S_xx","S_ss","S_rr","epsilon_i","sigma_i","detF","w(z)")
        for x in defaults:
            if x not in allowed:allowed.append(x)
        if has_birger and "GB_ratio" not in allowed:allowed.append("GB_ratio")
        return allowed

    def _open_video_export_dialog(self):
        try:
            import FINAL_VIDEO_EXPORT as fvideo
            out=self._result_output_dir();steps=self._video_available_steps()
            if out is None or not steps:raise ValueError("Сначала откройте результаты с принятыми STEP")
            has_birger=self._video_run_has_birger()
            modes=fvideo.available_modes(has_birger=has_birger,include_all=True)
            fields=self._video_field_options(has_birger)
        except Exception as exc:
            messagebox.showerror("VIDEO",str(exc));return
        win=tk.Toplevel(self);win.title("VIDEO — отдельный экспорт из принятых шагов");win.geometry("655x545");win.resizable(False,False);win.transient(self)
        frm=ttk.Frame(win,padding=14);frm.pack(fill="both",expand=True)
        current=self._v("result_field","E_xx").get() or "E_xx"
        if current not in fields:current=fields[0]
        mode=tk.StringVar(value="3D + нагрузка–прогиб");field=tk.StringVar(value=current)
        every=tk.StringVar(value="1");fps=tk.StringVar(value="10");speed=tk.StringVar(value="Обычное");hold=tk.StringVar(value="0.35");res=tk.StringVar(value="1280x720");vth=tk.StringVar(value=f"{self._result_visual_thickness_scale():g}")
        rows=[("Режим",mode),("Поле / компонента",field),("Каждый N-й принятый шаг",every),("Кадров/с",fps),("Скорость",speed),("Секунд на принятый шаг",hold),("Разрешение",res),("Визуальная толщина срезов ×",vth)]
        widgets={}
        for r,(lab,var) in enumerate(rows):
            ttk.Label(frm,text=lab).grid(row=r,column=0,sticky="w",pady=5,padx=(0,12))
            if r==0:w=ttk.Combobox(frm,textvariable=var,state="readonly",width=39,values=modes)
            elif r==1:w=ttk.Combobox(frm,textvariable=var,state="readonly",width=39,values=fields)
            elif r==4:w=ttk.Combobox(frm,textvariable=var,state="readonly",width=39,values=("Обычное","Замедленное"))
            elif r==6:w=ttk.Combobox(frm,textvariable=var,state="readonly",width=39,values=("1280x720","1920x1080"))
            elif r==7:w=ttk.Combobox(frm,textvariable=var,state="readonly",width=39,values=("1","2","5","10"))
            else:w=ttk.Entry(frm,textvariable=var,width=42)
            w.grid(row=r,column=1,sticky="ew",pady=5);widgets[r]=w
        def sync_mode(_e=None):
            try:
                m=mode.get()
                if m not in (fvideo.ALL_VIDEO_MODE,"3D выбранное поле","3D разрез по толщине","3D продольный разрез","3D поперечный разрез","3D комбинированный разрез","3D + нагрузка–прогиб","Разрез x-z","Разрез s-z","Профиль по толщине"):
                    field.set(self._video_pick_field(m,field.get(),fields))
            except Exception:pass
        widgets[0].bind("<<ComboboxSelected>>",sync_mode)
        def sync_speed(_e=None):
            hold.set("1.00" if speed.get()=="Замедленное" else "0.35")
        widgets[4].bind("<<ComboboxSelected>>",sync_speed)
        ttk.Label(frm,text=f"Принятых шагов: {len(steps)} · первый {steps[0]} · последний {steps[-1]}",foreground="#444").grid(row=8,column=0,columnspan=2,sticky="w",pady=(12,3))
        birger_note=" · Birger-видео доступно" if has_birger else " · Birger не выбран — Birger-видео скрыто"
        ttk.Label(frm,text="VIDEO — только опция: ничего не создаётся автоматически"+birger_note,foreground="#285d38").grid(row=9,column=0,columnspan=2,sticky="w",pady=(2,3))
        ttk.Label(frm,text="Обычное = 0.35 с/шаг; Замедленное = 1.00 с/шаг. Экспорт пишет только RUN/VIDEO/*.mp4 и не меняет FULL/Excel/STEP.",wraplength=610,foreground="#555").grid(row=10,column=0,columnspan=2,sticky="w",pady=(2,12))
        btns=ttk.Frame(frm);btns.grid(row=11,column=0,columnspan=2,sticky="e")
        ttk.Button(btns,text="Отмена",command=win.destroy).pack(side="right",padx=(8,0))
        def launch(force_all=False):
            try:
                n=max(1,int(every.get()));fps_i=max(1,min(60,int(fps.get())));hold_f=max(.05,float(hold.get().replace(",",".")));vth_f=max(.1,min(50.0,float(vth.get().replace(",","."))))
                wh=res.get().lower().split("x");width,height=int(wh[0]),int(wh[1]);chosen=fvideo.select_steps(steps,n)
                if not chosen:raise ValueError("Нет шагов для VIDEO")
                m=fvideo.ALL_VIDEO_MODE if force_all else mode.get()
                cfg={"mode":m,"field":field.get(),"fps":fps_i,"hold":hold_f,"width":width,"height":height,"steps":chosen,"available_fields":fields,"has_birger":has_birger,"visual_thickness_scale":vth_f}
                win.destroy();self._start_video_export(cfg)
            except Exception as exc:messagebox.showerror("VIDEO",str(exc),parent=win)
        ttk.Button(btns,text="ВСЕ основные MP4",command=lambda:launch(True)).pack(side="right",padx=(0,8))
        ttk.Button(btns,text="Создать выбранное MP4",command=lambda:launch(False)).pack(side="right")
        frm.columnconfigure(1,weight=1)

    def _start_video_export(self,cfg):
        import FINAL_VIDEO_EXPORT as fvideo
        if getattr(self,"_video_export_thread",None) is not None and self._video_export_thread.is_alive():
            messagebox.showwarning("VIDEO","Экспорт VIDEO уже выполняется");return
        out=self._result_output_dir();steps=list(cfg["steps"]);requested_mode=str(cfg["mode"]);requested_field=str(cfg["field"])
        available_fields=list(cfg.get("available_fields") or self._video_field_options(bool(cfg.get("has_birger"))))
        has_birger=bool(cfg.get("has_birger"))
        try:
            jobs=self._video_build_jobs(requested_mode,requested_field,available_fields,has_birger)
        except Exception as exc:
            messagebox.showerror("VIDEO",str(exc));return
        history=self._video_history_snapshot(self._video_available_steps())
        L,R=self._result_geometry_from_output();scale=self._result_deformation_scale();units=dict(self._gui_units());visual_meta=self._result_visual_metadata()
        try:ix=int(float(self._v("result_ix","0").get() or 0))
        except Exception:ix=0
        try:js=int(float(self._v("result_is","0").get() or 0))
        except Exception:js=0
        section_note=""
        if ix==0 and js==0:
            try:
                vis_auto=self._load_step_visual_state(int(steps[-1]))
                ww_auto=np.asarray((vis_auto or {}).get("w"),float)
                if ww_auto.ndim==2 and ww_auto.size:
                    js,ix=np.unravel_index(int(np.nanargmax(np.abs(ww_auto))),ww_auto.shape)
                    section_note="в месте |w|max финального шага / at final |w|max"
            except Exception:pass
        ztext=str(self._v("result_z","0").get())
        load_by_step={int(s):float(v) for s,v in zip(history["steps"],history["loads"])}
        saved_load_spec,_saved_unit_system=self._load_context_from_saved_run()
        video_provider_cache={}
        def layer_index(zvals):
            zvals=np.asarray(zvals,float)
            if not len(zvals):return 0
            txt=ztext.strip()
            if ":" in txt and txt.split(":",1)[0].strip().lower().startswith("g"):
                try:return max(0,min(int(txt.split(":",1)[0].strip()[1:]),len(zvals)-1))
                except Exception:pass
            if txt.lower().startswith("z="):txt=txt.split("=",1)[1].strip()
            try:z=float(txt)
            except Exception:z=0.0
            return int(np.argmin(np.abs(zvals-z)))
        def provider(step,mode_name,field_name,scan_only):
            del scan_only
            effective_field=str(field_name)
            cache_key=(int(step),str(mode_name),effective_field)
            cached=video_provider_cache.get(cache_key)
            if cached is not None:
                return cached
            field_unit=self._video_unit_for_field(effective_field,units)
            displacement_key={"u":"u","v":"v","w":"w","u0":"u","v0":"v","w0":"w"}.get(effective_field)
            thickness_modes={"3D разрез по толщине","3D продольный разрез","3D поперечный разрез","3D комбинированный разрез","Разрез x-z","Разрез s-z","Профиль по толщине"}
            if mode_name=="3D деформация" or (displacement_key is not None and mode_name not in thickness_modes):
                vis=self._load_step_visual_state(step)
                if vis is None:raise ValueError(f"STEP {step}: состояние не найдено")
                u=np.asarray(vis["u"],float);v=np.asarray(vis["v"],float);w=np.asarray(vis["w"],float)
                component={"u":u,"v":v,"w":w}.get(displacement_key or "w",w)
                zvals=np.asarray([0.0]);a=component[None,:,:];uvol=u[None,:,:];vvol=v[None,:,:];wvol=w[None,:,:]
                label=("w (прогиб)" if mode_name=="3D деформация" else f"{displacement_key} (срединная поверхность)");unit=units.get("displacement","mm")
            else:
                field_for_volume={"u0":"u(z)","v0":"v(z)","w0":"w(z)","u":"u(z)","v":"v(z)","w":"w(z)"}.get(effective_field,effective_field)
                try:zvals,a,label=self._numeric_3d_field(step,field_for_volume)
                except Exception as exc:raise ValueError(f"STEP {step}: поле {effective_field} нельзя построить для VIDEO: {exc}") from exc
                zvals=np.asarray(zvals,float);a=np.asarray(a,float);uvol,vvol,wvol=self._result_volume_displacements(step,zvals);unit=(units.get("displacement","mm") if field_for_volume.endswith("(z)") else field_unit)
            payload={
                **visual_meta,
                "L":float(L),"R":float(R),"scale":float(scale),"visual_thickness_scale":float(cfg.get("visual_thickness_scale",5.0)),"length_unit":units.get("length","mm"),
                "field":effective_field,"field_label":label,"field_unit":unit,"zvals":np.asarray(zvals,float),"field3d":np.asarray(a,float),
                "color_values":np.asarray(a,float),
                "uvol":np.asarray(uvol,float),"vvol":np.asarray(vvol,float),"wvol":np.asarray(wvol,float),
                "layer_index":layer_index(zvals),"point_ix":ix,"point_is":js,
                "max_displacement":float(max(np.nanmax(np.abs(uvol)),np.nanmax(np.abs(vvol)),np.nanmax(np.abs(wvol)))),
                "zmax":float(np.max(np.abs(zvals))) if len(zvals) else 0.0,
                "load_value":load_by_step.get(int(step)),"load_label":history.get("load_label","load"),"load_spec":saved_load_spec or {},"step":int(step),"section_note":section_note,
            }
            video_provider_cache[cache_key]=payload
            return payload
        video_dir=out/"VIDEO";video_dir.mkdir(parents=True,exist_ok=True)
        targets=[video_dir/fvideo.output_name(m,f,steps[0],steps[-1]) for m,f in jobs]
        self._video_export_cancel=threading.Event();self._video_export_queue=queue.Queue()
        dlg=tk.Toplevel(self);dlg.title("VIDEO экспорт");dlg.geometry("575x190");dlg.resizable(False,False);dlg.transient(self)
        body=ttk.Frame(dlg,padding=14);body.pack(fill="both",expand=True)
        self._video_status_var=tk.StringVar(value="Подготовка VIDEO…");ttk.Label(body,textvariable=self._video_status_var,wraplength=535).pack(anchor="w",pady=(0,10))
        self._video_progress=ttk.Progressbar(body,mode="determinate",maximum=100,value=0,length=525);self._video_progress.pack(fill="x")
        target_text=(str(targets[0]) if len(targets)==1 else f"{video_dir} · будет создано MP4: {len(targets)}")
        ttk.Label(body,text=target_text,wraplength=535,foreground="#555").pack(anchor="w",pady=(10,6))
        ttk.Button(body,text="Отменить VIDEO",command=self._video_export_cancel.set).pack(anchor="e")
        self._video_export_dialog=dlg
        def worker():
            made=[];failures=[]
            nj=len(jobs)
            for ji,((m,f),target) in enumerate(zip(jobs,targets)):
                if self._video_export_cancel.is_set():
                    self._video_export_queue.put(("cancelled",made));return
                video_provider_cache.clear()
                try:
                    def progress(stage,done,total,step,_ji=ji,_m=m):
                        local=(0.15*done/max(total,1)) if stage=="scan" else (0.15+0.85*done/max(total,1))
                        overall=100.0*(_ji+min(max(local,0.0),1.0))/max(nj,1)
                        self._video_export_queue.put(("progress",overall,_ji+1,nj,_m,stage,int(done),int(total),int(step)))
                    path=fvideo.export_mp4(target,steps,provider,mode=m,field=f,fps=cfg["fps"],seconds_per_step=cfg["hold"],width=cfg["width"],height=cfg["height"],history=history,progress=progress,cancel_event=self._video_export_cancel)
                    made.append(str(path))
                except InterruptedError:
                    self._video_export_queue.put(("cancelled",made));return
                except Exception as exc:
                    failures.append({"mode":str(m),"field":str(f),"error":f"{type(exc).__name__}: {exc}"})
                    self._video_export_queue.put(("job_error",ji+1,nj,str(m),str(f),str(exc)))
                    continue
            self._video_export_queue.put(("done",made,failures))
        self._video_export_thread=threading.Thread(target=worker,name="CYL_VIDEO_EXPORT",daemon=True);self._video_export_thread.start();self.after(100,self._poll_video_export)

    def _poll_video_export(self):
        q=getattr(self,"_video_export_queue",None)
        if q is None:return
        terminal=False
        while True:
            try:item=q.get_nowait()
            except queue.Empty:break
            kind=item[0]
            if kind=="progress":
                _k,pc,job_i,job_n,mode,stage,done,total,step=item
                try:self._video_progress.configure(value=min(100.0,float(pc)))
                except Exception:pass
                part=("анализ шкалы" if stage=="scan" else "кодирование")
                try:self._video_status_var.set(f"VIDEO {job_i}/{job_n}: {mode} · {part} · STEP {step} ({done}/{total})")
                except Exception:pass
            elif kind=="job_error":
                try:self._video_status_var.set(f"VIDEO {item[1]}/{item[2]}: ошибка {item[3]} · {item[4]} — продолжаю очередь")
                except Exception:pass
            elif kind=="done":
                terminal=True;paths=list(item[1]);failures=list(item[2]) if len(item)>2 else []
                try:self._video_export_dialog.destroy()
                except Exception:pass
                folder=str(Path(paths[0]).parent) if paths else str(self._result_output_dir()/"VIDEO")
                if failures:
                    details="\n".join(f"• {x.get('mode')} / {x.get('field')}: {x.get('error')}" for x in failures[:8])
                    extra="" if len(failures)<=8 else f"\n… ещё {len(failures)-8}"
                    messagebox.showwarning("VIDEO",f"Экспорт завершён частично. Создано MP4: {len(paths)}; ошибок: {len(failures)}\n{folder}\n\n{details}{extra}")
                else:
                    messagebox.showinfo("VIDEO",f"Экспорт завершён. Создано MP4: {len(paths)}\n{folder}")
            elif kind=="cancelled":
                terminal=True;made=list(item[1])
                try:self._video_export_dialog.destroy()
                except Exception:pass
                messagebox.showinfo("VIDEO",f"Экспорт VIDEO отменён. Уже полностью созданных MP4: {len(made)}")
            elif kind=="error":
                terminal=True;msg=item[1];made=list(item[2])
                try:self._video_export_dialog.destroy()
                except Exception:pass
                messagebox.showerror("VIDEO",f"{msg}\nУспешно создано до ошибки: {len(made)}")
        if not terminal and getattr(self,"_video_export_thread",None) is not None and self._video_export_thread.is_alive():self.after(100,self._poll_video_export)

    def _set_top_running_state(self,running):
        try:
            if running and not bool(getattr(self,"_paused",False)): self.progress.start(90)
            else: self.progress.stop()
        except Exception:pass
        try:
            if hasattr(self,"progress_status_label"):
                self.progress_status_label.configure(text="Расчёт…" if running else "Готово")
        except Exception:pass
        try:self.top_run_btn.configure(state="disabled" if running else "normal")
        except Exception:pass
        try:self.top_stop_btn.configure(state="normal" if running else "disabled")
        except Exception:pass
        try:
            self.top_pause_btn.configure(
                state="normal" if running else "disabled",
                text="Продолжить" if (running and bool(getattr(self,"_paused",False))) else "Пауза",
            )
        except Exception:pass
        if not running:
            self._paused=False
            self._paused_pids=[]

    @staticmethod
    def _process_tree_pids(root_pid):
        """Return root PID plus currently visible descendants using stdlib only."""
        root_pid=int(root_pid)
        pairs=[]
        if os.name=="nt":
            import ctypes
            from ctypes import wintypes
            TH32CS_SNAPPROCESS=0x00000002
            INVALID_HANDLE_VALUE=ctypes.c_void_p(-1).value
            class PROCESSENTRY32W(ctypes.Structure):
                _fields_=[
                    ("dwSize",wintypes.DWORD),("cntUsage",wintypes.DWORD),
                    ("th32ProcessID",wintypes.DWORD),("th32DefaultHeapID",ctypes.c_size_t),
                    ("th32ModuleID",wintypes.DWORD),("cntThreads",wintypes.DWORD),
                    ("th32ParentProcessID",wintypes.DWORD),("pcPriClassBase",ctypes.c_long),
                    ("dwFlags",wintypes.DWORD),("szExeFile",wintypes.WCHAR*260),
                ]
            k32=ctypes.WinDLL("kernel32",use_last_error=True)
            k32.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
            k32.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
            k32.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESSENTRY32W)]
            k32.Process32FirstW.restype=wintypes.BOOL
            k32.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESSENTRY32W)]
            k32.Process32NextW.restype=wintypes.BOOL
            k32.CloseHandle.argtypes=[wintypes.HANDLE]
            k32.CloseHandle.restype=wintypes.BOOL
            snap=k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0)
            if snap==INVALID_HANDLE_VALUE:
                raise OSError(ctypes.get_last_error(),"CreateToolhelp32Snapshot(PROCESS) failed")
            try:
                pe=PROCESSENTRY32W(); pe.dwSize=ctypes.sizeof(PROCESSENTRY32W)
                ok=k32.Process32FirstW(snap,ctypes.byref(pe))
                while ok:
                    pairs.append((int(pe.th32ProcessID),int(pe.th32ParentProcessID)))
                    ok=k32.Process32NextW(snap,ctypes.byref(pe))
            finally:
                k32.CloseHandle(snap)
        else:
            procdir=Path("/proc")
            if procdir.exists():
                for ent in procdir.iterdir():
                    if not ent.name.isdigit():continue
                    try:
                        stat=(ent/"stat").read_text(encoding="utf-8",errors="replace")
                        r=stat.rfind(")")
                        fields=stat[r+2:].split()
                        pairs.append((int(ent.name),int(fields[1])))
                    except Exception:
                        pass
            else:
                pairs=[(root_pid,-1)]
        children={}
        for pid,ppid in pairs:children.setdefault(ppid,[]).append(pid)
        out=[]; stack=[root_pid]; seen=set()
        while stack:
            pid=stack.pop()
            if pid in seen:continue
            seen.add(pid);out.append(pid);stack.extend(children.get(pid,()))
        return out

    @staticmethod
    def _set_pid_suspended(pid,suspend):
        """Suspend/resume all threads in one process; no external packages required."""
        pid=int(pid)
        if os.name!="nt":
            import signal
            os.kill(pid,signal.SIGSTOP if suspend else signal.SIGCONT)
            return
        import ctypes
        from ctypes import wintypes
        TH32CS_SNAPTHREAD=0x00000004
        THREAD_SUSPEND_RESUME=0x0002
        INVALID_HANDLE_VALUE=ctypes.c_void_p(-1).value
        class THREADENTRY32(ctypes.Structure):
            _fields_=[
                ("dwSize",wintypes.DWORD),("cntUsage",wintypes.DWORD),
                ("th32ThreadID",wintypes.DWORD),("th32OwnerProcessID",wintypes.DWORD),
                ("tpBasePri",ctypes.c_long),("tpDeltaPri",ctypes.c_long),("dwFlags",wintypes.DWORD),
            ]
        k32=ctypes.WinDLL("kernel32",use_last_error=True)
        k32.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
        k32.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
        k32.Thread32First.argtypes=[wintypes.HANDLE,ctypes.POINTER(THREADENTRY32)]
        k32.Thread32First.restype=wintypes.BOOL
        k32.Thread32Next.argtypes=[wintypes.HANDLE,ctypes.POINTER(THREADENTRY32)]
        k32.Thread32Next.restype=wintypes.BOOL
        k32.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        k32.OpenThread.restype=wintypes.HANDLE
        k32.SuspendThread.argtypes=[wintypes.HANDLE]
        k32.SuspendThread.restype=wintypes.DWORD
        k32.ResumeThread.argtypes=[wintypes.HANDLE]
        k32.ResumeThread.restype=wintypes.DWORD
        k32.CloseHandle.argtypes=[wintypes.HANDLE]
        k32.CloseHandle.restype=wintypes.BOOL
        snap=k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0)
        if snap==INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(),"CreateToolhelp32Snapshot(THREAD) failed")
        touched=0
        try:
            te=THREADENTRY32();te.dwSize=ctypes.sizeof(THREADENTRY32)
            ok=k32.Thread32First(snap,ctypes.byref(te))
            while ok:
                if int(te.th32OwnerProcessID)==pid:
                    h=k32.OpenThread(THREAD_SUSPEND_RESUME,False,int(te.th32ThreadID))
                    if h:
                        try:
                            result=(k32.SuspendThread(h) if suspend else k32.ResumeThread(h))
                            if result!=0xFFFFFFFF:touched+=1
                        finally:k32.CloseHandle(h)
                ok=k32.Thread32Next(snap,ctypes.byref(te))
        finally:k32.CloseHandle(snap)
        if touched==0:
            raise ProcessLookupError(f"Не найдены активные потоки PID {pid}")

    def _pause_calculation(self):
        proc=getattr(self,"proc",None)
        if proc is None or proc.poll() is not None:
            self._set_top_running_state(False)
            return
        pids=self._process_tree_pids(proc.pid)
        suspended=[]
        try:
            # Freeze children first, launcher last, so no child keeps calculating.
            for pid in reversed(pids):
                try:self._set_pid_suspended(pid,True);suspended.append(pid)
                except ProcessLookupError:pass
            self._paused=True;self._paused_pids=list(pids);self._pause_started_at=time.monotonic()
            self.top_pause_btn.configure(text="Продолжить")
            try:self.progress.stop()
            except Exception:pass
            (self.progress_status_label.configure(text="Пауза") if hasattr(self,"progress_status_label") else self.progress_label.configure(text="Пауза"))
            # Refresh from disk after all solver threads are frozen.  Only
            # already accepted STEP files are shown; an in-progress step is
            # never injected into the load-deflection history.
            try:self._online_refresh_steps()
            except Exception:pass
            try:self._online_set("stage","PAUSED BY USER")
            except Exception:pass
            self._log("\n=== PAUSED BY USER; accepted-step results remain stable ===\n")
        except Exception:
            for pid in reversed(suspended):
                try:self._set_pid_suspended(pid,False)
                except Exception:pass
            raise

    def _resume_calculation(self,silent=False):
        proc=getattr(self,"proc",None)
        pids=list(getattr(self,"_paused_pids",[]) or [])
        if proc is not None and proc.poll() is None and proc.pid not in pids:pids.insert(0,proc.pid)
        errors=[]
        # Resume launcher first, then descendants. Repeated ResumeThread calls are
        # intentionally avoided: every thread was suspended exactly once by us.
        for pid in pids:
            try:self._set_pid_suspended(pid,False)
            except (ProcessLookupError,PermissionError):pass
            except Exception as exc:errors.append(str(exc))
        if errors and not silent:raise RuntimeError("; ".join(errors))
        if getattr(self,"_pause_started_at",None) is not None:
            self._paused_total=float(getattr(self,"_paused_total",0.0))+max(0.0,time.monotonic()-self._pause_started_at)
        self._pause_started_at=None;self._paused=False;self._paused_pids=[]
        try:self.top_pause_btn.configure(text="Пауза",state="normal" if proc is not None and proc.poll() is None else "disabled")
        except Exception:pass
        if proc is not None and proc.poll() is None:
            try:self.progress.start(90)
            except Exception:pass
            (self.progress_status_label.configure(text="Расчёт…") if hasattr(self,"progress_status_label") else self.progress_label.configure(text="Расчёт…"))
            try:self._online_set("stage","RUNNING")
            except Exception:pass
        if not silent:self._log("\n=== RESUMED BY USER ===\n")

    def toggle_pause_calculation(self):
        proc=getattr(self,"proc",None)
        if proc is None or proc.poll() is not None:
            self._set_top_running_state(False)
            messagebox.showinfo("Пауза","Активного расчёта нет.")
            return
        try:
            if bool(getattr(self,"_paused",False)):self._resume_calculation()
            else:self._pause_calculation()
        except Exception as exc:
            messagebox.showerror("Пауза",str(exc))
            self._log(f"PAUSE/RESUME FAIL: {type(exc).__name__}: {exc}\n")

    def _terminate_process_tree_background(self, proc, pids):
        """Terminate the solver tree without blocking Tk's main event loop."""
        try:
            if os.name=="nt":
                # taskkill may wait while Windows tears down a large MVI tree;
                # keep that wait outside the Tkinter thread.
                subprocess.run(
                    ["taskkill","/PID",str(proc.pid),"/T","/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                import signal
                for pid in reversed(list(pids)):
                    try:
                        os.kill(int(pid),signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        if int(pid)==int(proc.pid):
                            try: proc.terminate()
                            except Exception: pass
        except Exception as exc:
            # Tk widgets are touched only by the main thread.
            self.log_queue.put(("__STOP_FAIL__",f"{type(exc).__name__}: {exc}"))

    def stop_calculation(self):
        proc=getattr(self,"proc",None)
        if proc is None or proc.poll() is not None:
            self._set_top_running_state(False)
            messagebox.showinfo("Стоп","Активного расчёта нет.")
            return
        try:
            self._log("\n=== STOP REQUESTED BY USER ===\n")
            self._stop_requested=True
            if bool(getattr(self,"_paused",False)):
                self._resume_calculation(silent=True)
            # Stop the whole launcher/solver tree, not only the top process.
            # This prevents a child MVI/FDM worker from continuing to write into
            # the RUN folder while FULL postprocessing is reading accepted states.
            pids=self._process_tree_pids(proc.pid)
            (self.progress_status_label.configure(text="Остановка…") if hasattr(self,"progress_status_label") else self.progress_label.configure(text="Остановка…"))
            self.top_stop_btn.configure(state="disabled")
            # Windows taskkill and descendant termination run outside Tk's
            # event loop. The existing reader thread will receive process
            # completion and call _on_process_done.
            threading.Thread(
                target=self._terminate_process_tree_background,
                args=(proc,pids),
                daemon=True,
            ).start()
        except Exception as exc:
            messagebox.showerror("Стоп",str(exc))
            self._log(f"STOP FAIL: {type(exc).__name__}: {exc}\n")

    def _build_problem_tab(self):
        super()._build_problem_tab()
        # The legacy nx/ns/stencil controls are the independent FDM/MKR grid.
        stack=[self.tab_problem]
        while stack:
            ch=stack.pop()
            try:stack.extend(ch.winfo_children())
            except Exception:pass
            try:
                if isinstance(ch,ttk.LabelFrame) and ch.cget("text")=="Глобальная сетка и квадратура":
                    ch.configure(text="Сетка FDM / МКР и квадратура")
                    self.fdm_grid_frame=ch
            except Exception:
                pass
        if hasattr(self,"grid_info"):
            self.grid_info.configure(text="FDM: Δx = —; Δs = —; число неизвестных = —")
        if not self._v("stencil","7").get().strip(): self._v("stencil").set("7")
        if hasattr(self,"fdm_grid_frame"):
            self.fdm_stencil_combo=ttk.Combobox(self.fdm_grid_frame,textvariable=self._v("stencil"),values=("7",),state="readonly",width=14)
            self.fdm_stencil_combo.grid(row=2,column=1,sticky="w",padx=6,pady=3)

    def _build_numeric_tab(self):
        super()._build_numeric_tab()
        # Explicit independent MVI spatial grid.
        self.mvi_grid_frame=ttk.LabelFrame(self.tab_numeric,text="Отдельная пространственная сетка МВИ",padding=8)
        self.mvi_grid_frame.grid(row=1,column=0,sticky="nsew",padx=(0,8),pady=(10,0))
        e1=self._labeled_entry(self.mvi_grid_frame,0,"MVI nx","mvi_nx")[1]
        e2=self._labeled_entry(self.mvi_grid_frame,1,"MVI ns","mvi_ns")[1]
        ttk.Label(self.mvi_grid_frame,text="MVI FD stencil").grid(row=2,column=0,sticky="w",pady=3)
        if not self._v("mvi_stencil","7").get().strip(): self._v("mvi_stencil").set("7")
        self.mvi_stencil_combo=ttk.Combobox(self.mvi_grid_frame,textvariable=self._v("mvi_stencil"),values=("3","5","7"),state="readonly",width=14)
        self.mvi_stencil_combo.grid(row=2,column=1,sticky="w",padx=6,pady=3)
        ttk.Label(self.mvi_grid_frame,text="MVI formulation").grid(row=3,column=0,sticky="w",pady=3)
        self._v("mvi_formulation","STRONG")
        self.mvi_formulation_combo=ttk.Combobox(
            self.mvi_grid_frame,textvariable=self._v("mvi_formulation"),
            values=("STRONG","VARIATIONAL"),state="readonly",width=14
        )
        self.mvi_formulation_combo.grid(row=3,column=1,sticky="w",padx=6,pady=3)
        e1.bind("<KeyRelease>",lambda e:self._update_grid_info())
        e2.bind("<KeyRelease>",lambda e:self._update_grid_info())
        ttk.Label(self.mvi_grid_frame,text="VARIATIONAL — независимая weak-form проверка CUF; на первом этапе доступна для GL + HOOKE с геометрическими КУ на обоих торцах.",wraplength=500).grid(row=4,column=0,columnspan=2,sticky="w",pady=(7,0))
        self.mvi_grid_info=ttk.Label(self.mvi_grid_frame,text="MVI: Δx = —; Δs = —; узлов = —",wraplength=500)
        self.mvi_grid_info.grid(row=5,column=0,columnspan=2,sticky="w",pady=(5,0))

    def _build_birger_tab(self):
        # K5 tolerance is an internal validation setting, not a physical user input.
        self._v("initial_slope_rtol","0.05").set("0.05")
        f=self.tab_birger
        f.columnconfigure(0,weight=0); f.columnconfigure(1,weight=1); f.rowconfigure(0,weight=1)
        self.birger_frame=ttk.LabelFrame(f,text="Внешний цикл Биргера",padding=8)
        self.birger_frame.grid(row=0,column=0,sticky="nsw",padx=(0,8))
        for r,(lab,key) in enumerate((("ε_i tolerance","epsilon_tol"),("ω","omega"),("G_ref","G_ref"),("σ_ref","sigma_ref"),("tol_G","tol_G"),("tol_σ","tol_sigma"),("tol_u (изменение поля, В-6)","tol_u"),("Max outer iterations","max_outer"),("ε_i limit","eps_limit"),("|J−1| limit","dJ_limit"))):
            self._labeled_entry(self.birger_frame,r,lab,key)
        ttk.Label(self.birger_frame,text="K5: начальный наклон Φ(εᵢ) проверяется автоматически по E0 и ν0; вручную наклон/допуск вводить не требуется.",wraplength=330).grid(row=10,column=0,columnspan=2,sticky="w",pady=(8,0))
        ttk.Label(self.birger_frame,text="Шаг принимается только при εᵢ ≤ limit и |J−1| ≤ limit. Локальная разгрузка запрещена. При нарушении шаг не принимается; адаптивная ветвь делит шаг, остальные ветви останавливаются.",wraplength=330).grid(row=11,column=0,columnspan=2,sticky="w",pady=(8,0))

        curve=ttk.LabelFrame(f,text="Материальная кривая σ_i = Φ(ε_i)",padding=8)
        curve.grid(row=0,column=1,sticky="nsew"); curve.columnconfigure(0,weight=1); curve.rowconfigure(9,weight=1)
        head=ttk.Frame(curve); head.grid(row=0,column=0,sticky="ew"); head.columnconfigure(3,weight=1)
        ttk.Label(head,text="Способ задания").grid(row=0,column=0,sticky="w")
        cb=ttk.Combobox(head,textvariable=self._v("curve_mode","TABLE"),values=birgercurve.CURVE_MODES,state="readonly",width=18)
        cb.grid(row=0,column=1,sticky="w",padx=6); cb.bind("<<ComboboxSelected>>",lambda e:self._refresh_curve_visibility())
        ttk.Label(head,text="Экстраполяция").grid(row=0,column=2,sticky="w",padx=(12,0))
        self.curve_ext_combo=ttk.Combobox(head,textvariable=self._v("curve_extrap","ERROR"),values=birgercurve.EXTRAPOLATIONS,state="readonly",width=10)
        self.curve_ext_combo.grid(row=0,column=3,sticky="w",padx=6)

        tools=ttk.Frame(curve); tools.grid(row=1,column=0,sticky="ew",pady=(6,4))
        ttk.Button(tools,text="Импорт кривой JSON",command=self._import_birger_json).pack(side="left",padx=(0,4))
        ttk.Button(tools,text="Экспорт кривой JSON",command=self._export_birger_json).pack(side="left",padx=4)
        ttk.Button(tools,text="Импорт CSV",command=self._import_curve_csv).pack(side="left",padx=4)
        ttk.Button(tools,text="Экспорт CSV",command=self._export_curve_csv).pack(side="left",padx=4)
        ttk.Button(tools,text="Сохранить PNG",command=self._export_curve_png).pack(side="left",padx=4)
        ttk.Button(tools,text="Открыть график",command=self._open_curve_window).pack(side="left",padx=4)

        self.curve_table_frame=ttk.LabelFrame(curve,text="Точки: одна строка = ε_i, σ_i",padding=5)
        self.curve_table_frame.grid(row=2,column=0,sticky="ew",pady=4)
        self.curve_points=tk.Text(self.curve_table_frame,height=5,width=65); self.curve_points.pack(fill="both",expand=True)

        self.curve_csv_frame=ttk.Frame(curve); self.curve_csv_frame.grid(row=3,column=0,sticky="ew",pady=3); self.curve_csv_frame.columnconfigure(1,weight=1)
        ttk.Label(self.curve_csv_frame,text="CSV/TXT").grid(row=0,column=0,sticky="w")
        ttk.Entry(self.curve_csv_frame,textvariable=self._v("curve_csv"),width=48).grid(row=0,column=1,sticky="ew",padx=6)
        ttk.Button(self.curve_csv_frame,text="…",command=self._import_curve_csv).grid(row=0,column=2)

        self.curve_formula_frame=ttk.LabelFrame(curve,text="Формула Φ(eps) и её параметры",padding=5)
        self.curve_formula_frame.grid(row=4,column=0,sticky="ew",pady=3); self.curve_formula_frame.columnconfigure(1,weight=1)
        ttk.Label(self.curve_formula_frame,text="Φ(eps) =").grid(row=0,column=0,sticky="w")
        ttk.Entry(self.curve_formula_frame,textvariable=self._v("curve_formula"),width=50).grid(row=0,column=1,sticky="ew",padx=6)
        ttk.Label(self.curve_formula_frame,text="Параметры, по одному на строку: A=..., B=..., E=...",wraplength=620).grid(row=1,column=0,columnspan=2,sticky="w",pady=(4,2))
        self.curve_parameters=tk.Text(self.curve_formula_frame,height=4,width=55)
        self.curve_parameters.grid(row=2,column=0,columnspan=2,sticky="ew")

        self.curve_piece_frame=ttk.LabelFrame(curve,text="Кусочная функция: epsilon_max | formula; последняя граница None",padding=5)
        self.curve_piece_frame.grid(row=5,column=0,sticky="ew",pady=3)
        self.curve_piecewise=tk.Text(self.curve_piece_frame,height=4,width=65); self.curve_piecewise.pack(fill="both",expand=True)

        self.curve_external_frame=ttk.LabelFrame(curve,text="Внешний Python callable",padding=5)
        self.curve_external_frame.grid(row=6,column=0,sticky="ew",pady=3)
        self._labeled_entry(self.curve_external_frame,0,"Module","phi_module"); self._labeled_entry(self.curve_external_frame,1,"Callable","phi_callable")
        ttk.Label(self.curve_external_frame,text="Python path").grid(row=2,column=0,sticky="w")
        ttk.Entry(self.curve_external_frame,textvariable=self._v("phi_path"),width=38).grid(row=2,column=1,sticky="ew",padx=6)
        ttk.Button(self.curve_external_frame,text="…",command=lambda:self._choose_dir("phi_path")).grid(row=2,column=2)

        options=ttk.Frame(curve); options.grid(row=7,column=0,sticky="ew",pady=(4,2))
        self._labeled_entry(options,0,"σ_T (опц.)","yield_stress",column=0,width=13)
        self._labeled_entry(options,0,"ε_T (опц.)","yield_strain",column=2,width=13)
        self._labeled_entry(options,1,"ε max графика","curve_preview_max",column=0,width=13)
        ttk.Button(options,text="Обновить график",command=self._preview_birger_curve).grid(row=1,column=2,columnspan=2,sticky="w",padx=6)
        self.curve_status=ttk.Label(curve,text="",wraplength=760); self.curve_status.grid(row=8,column=0,sticky="w",pady=(3,3))

        self.curve_plot_frame=ttk.LabelFrame(curve,text="Предпросмотр кривой — именно эта Φ(ε_i) используется расчётом",padding=4)
        self.curve_plot_frame.grid(row=9,column=0,sticky="nsew"); self._curve_canvas=None; self._curve_figure=None; self._curve_last_xy=None
        self._refresh_curve_visibility()

    def _build_files_tab(self):
        super()._build_files_tab()
        f=self.tab_files

        # The selected directory is a RESULTS ROOT.  Each press of Start creates
        # a new self-contained child folder named from the calculation label +
        # theory/method/mode/geometry/load/grid.  This prevents one run from
        # overwriting another and means the user chooses the root only once.
        for row in (3,4,5,6,7,8):
            for child in list(f.grid_slaves(row=row)):
                try: child.destroy()
                except Exception: pass

        ttk.Label(f,text="Корневая папка результатов").grid(row=3,column=0,sticky="w",pady=3)
        self.output_dir_entry=ttk.Entry(f,textvariable=self._v("output_dir"),width=72,state="readonly")
        self.output_dir_entry.grid(row=3,column=1,sticky="ew",padx=6,pady=3)
        self.output_dir_choose_btn=ttk.Button(f,text="Выбрать папку...",command=lambda:self._choose_dir("output_dir"))
        self.output_dir_choose_btn.grid(row=3,column=2,sticky="ew",padx=(0,4),pady=3)

        ttk.Label(f,text="Название расчёта (опционально)").grid(row=4,column=0,sticky="w",pady=3)
        ttk.Entry(f,textvariable=self._v("run_label"),width=48).grid(row=4,column=1,sticky="ew",padx=6,pady=3)
        ttk.Label(f,text=r"По умолчанию в Windows: C:\CYL_RESULTS. Каждый запуск сохраняется сразу как C:\CYL_RESULTS\RUN_001__TE1_MVI1_NL_Q10000_g41x41, без промежуточной папки расчёта. Полные параметры лежат в RUN_INFO.json, PARAMETERS.txt и RUNS_INDEX.csv.",wraplength=900).grid(row=5,column=0,columnspan=3,sticky="w",pady=(2,7))

        casebox=ttk.LabelFrame(f,text="CASE (ручное сохранение/загрузка при необходимости)",padding=7)
        casebox.grid(row=6,column=0,columnspan=3,sticky="ew",pady=(4,4)); casebox.columnconfigure(1,weight=1)
        ttk.Label(casebox,text="FINAL_CASE.json").grid(row=0,column=0,sticky="w")
        ttk.Entry(casebox,textvariable=self._v("case_path"),width=60).grid(row=0,column=1,sticky="ew",padx=6)
        ttk.Button(casebox,text="…",command=lambda:self._choose_file("case_path")).grid(row=0,column=2)
        cbuttons=ttk.Frame(casebox); cbuttons.grid(row=1,column=0,columnspan=3,sticky="w",pady=(7,0))
        ttk.Button(cbuttons,text="Сохранить CASE",command=self.save_case).pack(side="left",padx=(0,5))
        ttk.Button(cbuttons,text="Загрузить CASE",command=self.load_case).pack(side="left",padx=5)

        pars=ttk.LabelFrame(f,text="Preset интерфейса",padding=7)
        pars.grid(row=7,column=0,columnspan=3,sticky="ew",pady=(4,4)); pars.columnconfigure(1,weight=1)
        ttk.Label(pars,text="Файл параметров").grid(row=0,column=0,sticky="w")
        ttk.Entry(pars,textvariable=self._v("parameters_path"),width=60).grid(row=0,column=1,sticky="ew",padx=6)
        buttons=ttk.Frame(pars); buttons.grid(row=1,column=0,columnspan=3,sticky="w",pady=(7,0))
        ttk.Button(buttons,text="Открыть параметры",command=self.open_parameters).pack(side="left",padx=(0,5))
        ttk.Button(buttons,text="Сохранить",command=self.save_parameters).pack(side="left",padx=5)
        ttk.Button(buttons,text="Сохранить как…",command=self.save_parameters_as).pack(side="left",padx=5)
        ttk.Button(buttons,text="Сохранить в папку…",command=self.save_parameters_to_folder).pack(side="left",padx=5)
        ttk.Label(pars,text="При каждом Пуске такой preset автоматически сохраняется ещё и внутри папки конкретного расчёта.",wraplength=850).grid(row=2,column=0,columnspan=3,sticky="w",pady=(5,0))

    def _auto_run_bundle(self,cfg):
        root=cfg.get("output_dir")
        if root is None or not str(root).strip():
            raise ValueError("Выберите корневую папку результатов")
        cp=self.vars.get("case_path")
        cp_txt=cp.get().strip() if cp is not None else ""
        source_base=(Path(cp_txt).resolve().parent if cp_txt and Path(cp_txt).exists() else APP_DIR)
        bundle=runbundle.prepare_run_bundle(
            cfg, root, run_label=self._v("run_label").get().strip(), source_base=source_base
        )
        run_dir=Path(bundle["run_dir"]).resolve(); case_path=Path(bundle["case_path"]).resolve()
        # Full GUI preset is saved with the calculation so this exact variant
        # can be reopened without manually entering parameters again.
        try:
            (Path(bundle["input_dir"])/"GUI_PRESET.json").write_text(
                json.dumps(self._gui_state(),indent=2,ensure_ascii=False),encoding="utf-8"
            )
        except Exception as exc:
            self._log(f"GUI preset snapshot warning: {exc}\n")
        self._log(f"Automatic calculation folder: {run_dir}\n")
        return bundle,case_path

    def _run_case_auto_bundle(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Расчёт","Расчёт уже выполняется");return
        try:
            cfg=self._case_dict()
            bundle,p=self._auto_run_bundle(cfg)
            validate_final_case(p)
            self.run_output_dir=Path(bundle["run_dir"]).resolve()
            bc_cfg=bundle.get("cfg",cfg); ld=bc_cfg.get("load_definition") or {}
            self._online_run_load_definition=dict(ld) if isinstance(ld,dict) else {}
            self._online_run_unit_system=str(((bc_cfg.get("units") or {}).get("system") or self._v("unit_system","N_MM_MPA").get()))
            if str(ld.get("type","")).upper()=="EXTERNAL_NPZ":
                lp=Path(str(ld.get("npz") or bc_cfg.get("loads_npz")))
                if not lp.is_absolute(): lp=p.parent/lp
                with np.load(lp,allow_pickle=False) as z:self.expected_steps=int(np.asarray(z["loads"]).shape[0])
            else:
                self.expected_steps=len(ld.get("load_factors") or [])
            self.run_kind="CASE"
            cmd=[sys.executable,str(APP_DIR/"RUN_FINAL_CASE.py"),str(p)]
            self._log("\n=== START ===\n"+" ".join(cmd)+"\n")
            self.run_btn.configure(state="disabled");self._progress_reset(self.expected_steps)
            self.proc=subprocess.Popen(cmd,cwd=str(APP_DIR),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding="utf-8",errors="replace")
            self._set_top_running_state(True);threading.Thread(target=self._read_process_output,daemon=True).start()
        except Exception as exc:
            self._set_top_running_state(False);messagebox.showerror("Запуск — ошибка",str(exc));self._log(f"START FAIL: {type(exc).__name__}: {exc}\n")

    def run_mvi_series(self):
        if self.vars["method"].get()!="MVI":
            messagebox.showwarning("МВИ-1…8","Серия доступна только при method=MVI");return
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Расчёт","Расчёт уже выполняется");return
        try:
            cfg=self._case_dict();bundle,p=self._auto_run_bundle(cfg);validate_final_case(p)
            run_dir=Path(bundle["run_dir"]).resolve()
            self.run_output_dir=run_dir/"MVI_1_8_SERIES"
            self.expected_steps=8;self.run_kind="MVI_SERIES"
            self._paused=False;self._paused_pids=[];self._pause_started_at=None;self._paused_total=0.0
            cmd=[sys.executable,str(APP_DIR/"RUN_MVI_1_8.py"),str(p)]
            self._log("\n=== START MVI-1…8 ===\n"+" ".join(cmd)+"\n")
            self.run_btn.configure(state="disabled");self.series_btn.configure(state="disabled");self._progress_reset(8)
            self.proc=subprocess.Popen(cmd,cwd=str(APP_DIR),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding="utf-8",errors="replace")
            self._set_top_running_state(True);threading.Thread(target=self._read_process_output,daemon=True).start()
        except Exception as exc:
            self._set_top_running_state(False);messagebox.showerror("МВИ-1…8 — ошибка",str(exc));self._log(f"MVI SERIES START FAIL: {type(exc).__name__}: {exc}\n")

    def _refresh_curve_visibility(self):
        if not hasattr(self,"curve_table_frame"):return
        mode=self.vars.get("curve_mode",tk.StringVar(value="TABLE")).get()
        for fr,m in ((self.curve_table_frame,"TABLE"),(self.curve_csv_frame,"CSV"),(self.curve_formula_frame,"FORMULA"),(self.curve_piece_frame,"PIECEWISE"),(self.curve_external_frame,"EXTERNAL_PYTHON")):
            if mode==m:fr.grid()
            else:fr.grid_remove()
        try:self.curve_ext_combo.configure(state="readonly" if mode in ("TABLE","CSV") else "disabled")
        except Exception:pass

    def _parse_curve_parameters(self):
        out={}
        if not hasattr(self,"curve_parameters"):return out
        for ln in self.curve_parameters.get("1.0","end").splitlines():
            s=ln.strip()
            if not s or s.startswith("#"):continue
            if "=" not in s:raise ValueError(f"Параметр формулы должен быть NAME=value: {ln}")
            name,val=s.split("=",1); name=name.strip(); val=val.strip()
            if not name.isidentifier():raise ValueError(f"Недопустимое имя параметра: {name}")
            out[name]=float(val)
        return out

    def _curve_spec(self):
        spec=super()._curve_spec()
        if spec.get("mode") in ("FORMULA","PIECEWISE"):
            params=self._parse_curve_parameters()
            if params:spec["parameters"]=params
        return spec

    def _current_curve_base_dir(self):
        cp=self.vars.get("case_path",tk.StringVar()).get().strip()
        return Path(cp).parent if cp else APP_DIR

    def _preview_birger_curve(self):
        try:
            spec=self._curve_spec()
            if spec["mode"]=="EXTERNAL_PYTHON":raise ValueError("Внешний Python callable проверяется dispatcher; встроенный график строится для FORMULA/TABLE/CSV/PIECEWISE.")
            mx=self.vars.get("curve_preview_max",tk.StringVar()).get().strip(); mx=None if not mx else float(mx)
            x,y=birgercurve.sample_curve(spec,base_dir=self._current_curve_base_dir(),epsilon_max=mx,n=400)
            # K5/K6 audit check: Phi is the intensity diagram sigma_i(epsilon_i),
            # with elastic slope 3*G0 and a true origin.  Preview fails before
            # a case can be launched if the curve is an ordinary sigma-epsilon
            # diagram or is unsafe near epsilon_i=0.
            E0=_float(self.vars["E0"].get(),"E0",positive=True)
            nu0=_float(self.vars["nu0"].get(),"nu0")
            if not (-1.0 < nu0 < 0.5):raise ValueError("nu0 must satisfy -1 < nu0 < 0.5")
            eps_tol=_float(self.vars["epsilon_tol"].get(),"epsilon_tol",nonnegative=True)
            G0=E0/(2.0*(1.0+nu0))
            fn=birgercurve.build_curve(spec,base_dir=self._current_curve_base_dir())
            diag=b5.validate_material_diagram_near_origin(
                fn,G0=G0,epsilon_tol=eps_tol,
                initial_slope_rtol=0.05
            )
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self._curve_canvas is not None:self._curve_canvas.get_tk_widget().destroy()
            fig=Figure(figsize=(7.3,3.6)); ax=fig.add_subplot(111); ax.plot(x,y); ax.set_xlabel("ε_i [-]"); ax.set_ylabel(f"σ_i [{self._gui_units()['stress']}]"); ax.grid(True,alpha=.3); ax.set_title("σ_i = Φ(ε_i)"); fig.tight_layout()
            self._curve_canvas=FigureCanvasTkAgg(fig,master=self.curve_plot_frame); self._curve_canvas.draw(); self._curve_canvas.get_tk_widget().pack(fill="both",expand=True)
            self._curve_figure=fig; self._curve_last_xy=(np.asarray(x,float),np.asarray(y,float))
            self.curve_status.configure(text=(
                f"Кривая PASS (K5/K6): ε∈[{x.min():.6g}, {x.max():.6g}], "
                f"σ∈[{y.min():.6g}, {y.max():.6g}]; "
                f"начальный наклон={diag['initial_slope']:.6g}, "
                f"3G0={diag['expected_initial_slope']:.6g}, "
                f"ошибка={diag['initial_slope_relative_error']:.2%}"
            ))
        except Exception as exc:
            self.curve_status.configure(text=f"FAIL: {exc}"); messagebox.showerror("Кривая Биргера",str(exc))

    def _set_curve_from_spec(self,spec):
        if not isinstance(spec,dict):raise ValueError("Файл кривой должен содержать JSON-объект")
        if "curve" in spec and isinstance(spec["curve"],dict):spec=spec["curve"]
        mode=str(spec.get("mode","TABLE")).upper(); self.vars["curve_mode"].set(mode); self.vars["curve_extrap"].set(str(spec.get("extrapolation","ERROR")))
        self.vars["yield_stress"].set("" if spec.get("yield_stress") is None else str(spec.get("yield_stress"))); self.vars["yield_strain"].set("" if spec.get("yield_strain") is None else str(spec.get("yield_strain")))
        self.curve_points.delete("1.0","end")
        if mode=="TABLE":self.curve_points.insert("1.0","\n".join(f"{a[0]}, {a[1]}" for a in spec.get("points",[])))
        self.vars["curve_csv"].set(str(spec.get("path","") or "")); self.vars["curve_formula"].set(str(spec.get("formula","") or ""))
        self.curve_parameters.delete("1.0","end"); self.curve_parameters.insert("1.0","\n".join(f"{k}={v}" for k,v in (spec.get("parameters") or {}).items()))
        self.curve_piecewise.delete("1.0","end")
        if mode=="PIECEWISE":self.curve_piecewise.insert("1.0","\n".join(f"{('None' if q.get('epsilon_max') is None else q.get('epsilon_max'))} | {q.get('formula','')}" for q in spec.get("segments",[])))
        md=spec.get("material_diagram",{}) if mode=="EXTERNAL_PYTHON" else {}; self.vars["phi_module"].set(str(md.get("module","") or "")); self.vars["phi_callable"].set(str(md.get("callable","") or "")); self.vars["phi_path"].set(str(md.get("python_path","") or ""))
        self._refresh_curve_visibility()

    def _import_birger_json(self):
        p=filedialog.askopenfilename(title="Импорт кривой Биргера",filetypes=[("Birger curve JSON","*.json"),("All files","*.*")])
        if not p:return
        try:
            data=json.loads(Path(p).read_text(encoding="utf-8")); self._set_curve_from_spec(data)
            pv=data.get("preview",{}) if isinstance(data,dict) else {}; mx=pv.get("epsilon_max") if isinstance(pv,dict) else None
            if mx is not None:self.vars["curve_preview_max"].set(str(mx))
            self._preview_birger_curve(); self._log(f"Birger curve imported: {p}\n")
        except Exception as exc:messagebox.showerror("Импорт кривой",str(exc))

    def _export_birger_json(self):
        try:spec=self._curve_spec()
        except Exception as exc:messagebox.showerror("Экспорт кривой",str(exc));return
        p=filedialog.asksaveasfilename(title="Экспорт кривой Биргера",defaultextension=".json",initialfile="BIRGER_CURVE.json",filetypes=[("JSON","*.json")])
        if not p:return
        data={"format":"BIRGER_CURVE_V1","curve":spec,"preview":{"epsilon_max":self.vars.get("curve_preview_max",tk.StringVar()).get() or None},"units":self.vars.get("unit_system",tk.StringVar()).get()}
        Path(p).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8"); self._log(f"Birger curve exported: {p}\n")

    def _import_curve_csv(self):
        p=filedialog.askopenfilename(title="Импорт ε_i–σ_i из CSV/TXT",filetypes=[("CSV/TXT","*.csv *.txt"),("All files","*.*")])
        if not p:return
        try:
            pts=birgercurve.read_csv_points(p); self.vars["curve_mode"].set("CSV"); self.vars["curve_csv"].set(p); self.curve_points.delete("1.0","end"); self.curve_points.insert("1.0","\n".join(f"{a}, {b}" for a,b in pts)); self._refresh_curve_visibility(); self._preview_birger_curve()
        except Exception as exc:messagebox.showerror("Импорт CSV",str(exc))

    def _export_curve_csv(self):
        try:
            self._preview_birger_curve()
            if self._curve_last_xy is None:return
            x,y=self._curve_last_xy
        except Exception:return
        p=filedialog.asksaveasfilename(title="Экспорт кривой CSV",defaultextension=".csv",initialfile="BIRGER_CURVE.csv",filetypes=[("CSV","*.csv")])
        if not p:return
        np.savetxt(p,np.column_stack([x,y]),delimiter=",",header="epsilon_i,sigma_i",comments="")

    def _export_curve_png(self):
        if self._curve_figure is None:
            self._preview_birger_curve()
            if self._curve_figure is None:return
        p=filedialog.asksaveasfilename(title="Сохранить график кривой",defaultextension=".png",initialfile="BIRGER_CURVE.png",filetypes=[("PNG","*.png")])
        if p:self._curve_figure.savefig(p,dpi=180,bbox_inches="tight")

    def _open_curve_window(self):
        try:
            if self._curve_figure is None:
                self._preview_birger_curve()
                if self._curve_figure is None:return
            self._open_figure_in_window(
                self._curve_figure,
                "Кривая Биргера σ_i = Φ(ε_i) — отдельное окно"
            )
        except Exception as exc:
            messagebox.showerror("График кривой Биргера",str(exc))

    def _update_grid_info(self):
        super()._update_grid_info()
        try:
            L=float(self.vars["L"].get()); R=float(self.vars["R"].get()); nx=int(self.vars["mvi_nx"].get()); ns=int(self.vars["mvi_ns"].get()); dx=L/(nx-1); ds=2*np.pi*R/ns
            self.mvi_grid_info.configure(text=f"MVI: Δx={dx:.6g}; Δs={ds:.6g}; узлов={nx*ns}")
        except Exception:
            if hasattr(self,"mvi_grid_info"):self.mvi_grid_info.configure(text="MVI: Δx = —; Δs = —; узлов = —")
        try:
            if hasattr(self,"grid_info") and self.vars.get("nx") and self.vars["nx"].get() and self.vars["ns"].get():
                old=self.grid_info.cget("text"); self.grid_info.configure(text="FDM: "+old.replace("FDM: ",""))
        except Exception:pass

    def _collect_preflight_missing(self):
        missing=[]
        def need(key,label):
            v=self.vars.get(key)
            if v is None or not str(v.get()).strip(): missing.append(label)
        for key,label in (("L","Длина L"),("R","Радиус R"),("h","Толщина h"),("E0","Модуль E0"),("nu0","Коэффициент Пуассона ν0"),("n_gauss","Gauss-точек по толщине")):
            need(key,label)
        if self._v("quadrature_type","GAUSS").get()=="LAYERED":
            need("layer_bounds","Границы слоёв квадратуры z")
        method=self.vars.get("method",tk.StringVar(value="FDM")).get()
        if method=="FDM":
            for key,label in (("nx","FDM nx"),("ns","FDM ns"),("stencil","FDM stencil")): need(key,label)
        else:
            for key,label in (("mvi_nx","MVI nx"),("mvi_ns","MVI ns"),("mvi_stencil","MVI stencil"),("rank","МВИ-N"),("mvi_proj","MVI projected tolerance"),("mvi_inc","MVI increment tolerance"),("mvi_fd","MVI tangent FD step"),("mvi_half","MVI max halfstep"),("mvi_cycles","MVI max X↔Y cycles")): need(key,label)
        if self.vars.get("bc_mode",tk.StringVar(value="SIMPLE")).get()=="SIMPLE":
            need("bc_left","КУ левого торца"); need("bc_right","КУ правого торца")
            elastic_sides=[side for side,key in (("left","bc_left"),("right","bc_right")) if self._v(key).get().upper()=="ELASTIC_ENGINEERING"]
            if elastic_sides:
                eng=getattr(self,"engineering_bc",None)
                if not isinstance(eng,dict):
                    missing.append("Инженерные жёсткости k для: "+", ".join(elastic_sides))
                else:
                    theory=self._v("theory","CUF").get().upper().strip()
                    def check_item(side,label,item):
                        if not isinstance(item,dict):
                            missing.append(f"Инженерное КУ {side}.{label}")
                            return
                        if "stiffness" not in item:
                            missing.append(f"Жёсткость k {side}.{label}")
                            return
                        try:
                            k=float(item["stiffness"]); q0=float(item.get("reference",0.0)); tb=float(item.get("traction",item.get("value",0.0)))
                            if not np.isfinite(k) or k<0.0 or not np.isfinite(q0) or not np.isfinite(tb):
                                raise ValueError
                        except Exception:
                            missing.append(f"Корректные k/q0/T̄ {side}.{label}")
                    for side in elastic_sides:
                        sed=eng.get(side)
                        if not isinstance(sed,dict):
                            missing.append(f"Инженерные жёсткости k для: {side}")
                            continue
                        positive=False
                        if theory=="KL":
                            for field in ("u","v","w","slope"):
                                item=sed.get(field)
                                check_item(side,field,item)
                                try: positive |= float(item.get("stiffness",0.0))>0.0
                                except Exception: pass
                        else:
                            m=self._te_m()
                            for comp in ("u","v","w"):
                                vals=sed.get(comp)
                                if not isinstance(vals,list) or len(vals)!=m:
                                    missing.append(f"Инженерное КУ {side}.{comp}: требуется {m} коэффициентов τ")
                                    continue
                                for tau,item in enumerate(vals):
                                    check_item(side,f"{comp}[{tau}]",item)
                                    try: positive |= float(item.get("stiffness",0.0))>0.0
                                    except Exception: pass
                        if not positive:
                            missing.append(f"Хотя бы одна k>0 на инженерном торце {side}")
        else:
            adv=getattr(self,"advanced_bc",None)
            if not isinstance(adv,dict):
                missing.append("Ручные / Custom краевые условия")
            else:
                theory=self._v("theory","CUF").get().upper().strip()
                def check_custom_item(side,label,item,data_key):
                    if not isinstance(item,dict):
                        missing.append(f"Custom КУ {side}.{label}")
                        return
                    kind=str(item.get("kind","")).upper().strip()
                    if kind not in ("GEOMETRIC","NATURAL","ELASTIC"):
                        missing.append(f"Тип Custom КУ {side}.{label}")
                        return
                    try:
                        val=float(item.get(data_key,0.0))
                        if not np.isfinite(val): raise ValueError
                        if kind=="ELASTIC":
                            k=float(item["stiffness"]); q0=float(item.get("reference",0.0))
                            if not np.isfinite(k) or k<0.0 or not np.isfinite(q0): raise ValueError
                    except Exception:
                        missing.append(f"Числовые параметры Custom КУ {side}.{label}")
                for side in ("left","right"):
                    sed=adv.get(side)
                    if not isinstance(sed,dict):
                        missing.append(f"Custom КУ торца {side}")
                        continue
                    if theory=="KL":
                        for field in ("u","v","w","slope"):
                            check_custom_item(side,field,sed.get(field),"value")
                    else:
                        m=self._te_m()
                        for comp in ("u","v","w"):
                            vals=sed.get(comp)
                            if not isinstance(vals,list) or len(vals)!=m:
                                missing.append(f"Custom КУ {side}.{comp}: требуется {m} коэффициентов τ")
                                continue
                            for tau,item in enumerate(vals):
                                check_custom_item(side,f"{comp}[{tau}]",item,"data")
        typ=self.vars.get("load_type",tk.StringVar(value="EXTERNAL_NPZ")).get()
        if typ=="EXTERNAL_NPZ": need("loads_npz","Файл нагрузки NPZ")
        else:
            if typ in ("UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE","PATCH_RADIAL_FACE"): need("load_q","Величина нагрузки q")
            if typ=="HARMONIC_RADIAL_FACE": need("load_harmonic_m","Гармоника m по x"); need("load_harmonic_n","Гармоника n по окружности")
            if typ=="POINT_RADIAL_FORCE": need("load_point_P","Сила P"); need("load_point_x0","x0 точки"); need("load_point_theta0","theta0 точки")
            if typ=="LINE_RADIAL_GENERATOR": need("load_p_line","Линейная нагрузка p_line"); need("load_c0","Координата линии нагрузки"); need("load_band_width","Физическая ширина полосы b")
            if typ=="PATCH_RADIAL_FACE" and not bool(getattr(self,"load_full_var",tk.BooleanVar(value=False)).get()):
                need("load_c1","Начальная окружная координата площадки"); need("load_c2","Конечная окружная координата площадки")
            step_mode=self.vars.get("load_step_mode",tk.StringVar(value="MANUAL_LIST")).get()
            if step_mode=="RANGE":
                for key,label in (("load_lambda_start","λ start"),("load_lambda_step","Δλ"),("load_lambda_end","λ end")): need(key,label)
            elif step_mode=="COUNT":
                for key,label in (("load_lambda_end","λ end"),("load_step_count","Число шагов")): need(key,label)
            else: need("load_factors","Список шагов нагрузки λ")
        mode=self.vars.get("mode",tk.StringVar(value="LINEAR_ELASTIC")).get()
        nonlinear=mode.startswith("NONLINEAR"); birger=mode.endswith("BIRGER")
        if method=="FDM" and mode=="LINEAR_ELASTIC":
            need("linear_res","Linear FDM relative residual tolerance")
        if method=="FDM" and nonlinear:
            for key,label in (("newton_res","Newton relative residual tolerance"),("newton_inc","Newton increment tolerance"),("newton_fd","Newton relative FD step"),("newton_max","Newton max iterations")): need(key,label)
        if birger:
            for key,label in (("epsilon_tol","Birger ε tolerance"),("omega","Birger ω"),("G_ref","Birger G_ref"),("sigma_ref","Birger σ_ref"),("tol_G","Birger tol_G"),("tol_sigma","Birger tol_σ"),("tol_u","Birger tol_u (изменение поля)"),("max_outer","Birger max outer iterations"),("eps_limit","Birger ε_i limit"),("dJ_limit","Birger |J−1| limit")): need(key,label)
            cm=self.vars.get("curve_mode",tk.StringVar(value="TABLE")).get()
            if cm=="TABLE" and not self.curve_points.get("1.0","end-1c").strip(): missing.append("Точки кривой Биргера")
            elif cm=="CSV": need("curve_csv","CSV/TXT кривой Биргера")
            elif cm=="FORMULA": need("curve_formula","Формула Биргера Φ(eps)")
            elif cm=="PIECEWISE" and not self.curve_piecewise.get("1.0","end-1c").strip(): missing.append("Кусочная функция Биргера")
            elif cm=="EXTERNAL_PYTHON": need("phi_module","Модуль внешней функции Биргера"); need("phi_callable","Callable внешней функции Биргера")
        need("output_dir","Папка результатов")
        return missing

    def _preflight_or_show(self):
        self._prepare_ordinary_hidden_values()
        if not self._v("stencil","7").get().strip(): self._v("stencil").set("7")
        if not self._v("mvi_stencil","7").get().strip(): self._v("mvi_stencil").set("7")
        missing=self._collect_preflight_missing()
        if missing:
            messagebox.showerror("Не заполнены параметры","Перед запуском заполните обязательные поля:\n\n"+"\n".join(f"• {x}" for x in missing))
            self._log("PREFLIGHT FAIL: required input is missing (see dialog)\n")
            return False
        try:
            capability=capreg.case_capability(self._case_dict())
            if capability.status==capreg.UNAVAILABLE:
                messagebox.showerror(
                    "Недоступное сочетание",
                    f"{capability.message_ru}\n\nКод: {capability.code}"
                )
                self._log(f"CAPABILITY FAIL [{capability.code}] (see dialog)\n")
                return False
            if capability.status==capreg.LIMITED:
                messagebox.showwarning(
                    "Ограниченная постановка",
                    f"{capability.message_ru}\n\nРезультат будет явно помечен как LIMITED."
                )
                self._log(f"CAPABILITY WARNING [{capability.code}] (see dialog)\n")
        except Exception as exc:
            messagebox.showerror("Проверка сочетания",str(exc))
            self._log(f"CAPABILITY CHECK FAIL: {exc}\n")
            return False
        # V-7: the periodic centred D1s has an exact sawtooth null mode for even ns.
        try:
            ns_key="ns" if self._v("method","FDM").get()=="FDM" else "mvi_ns"
            ns_val=int(self._v(ns_key).get())
            if ns_val % 2 == 0:
                messagebox.showerror(
                    "Сетка по окружности (В-7)",
                    "ns должно быть нечётным. При чётном ns центральный периодический "
                    "оператор D1s обнуляет пилообразную моду (-1)^j."
                )
                self._log(f"PREFLIGHT FAIL V-7: {ns_key}={ns_val} is even\n")
                return False
        except ValueError:
            pass
        # Reviewer V-9: boundary-layer resolution is a warning, never a block.
        try:
            L=float(self._v("L").get()); R=float(self._v("R").get())
            h=float(self._v("h").get()); nu0=float(self._v("nu0").get())
            nx_key="nx" if self._v("method","FDM").get()=="FDM" else "mvi_nx"
            nx_val=int(self._v(nx_key).get())
            beta=(3.0*(1.0-nu0**2)/(R**2*h**2))**0.25
            decay=1.0/beta
            dx=L/(nx_val-1)
            dx_required=0.5*decay
            if dx>dx_required:
                nx_rec=int(np.ceil(L*2.0*beta))+1
                msg=(
                    f"BOUNDARY_LAYER_UNRESOLVED: dx={dx:.4g} > {dx_required:.4g}. "
                    f"Длина затухания краевого эффекта 1/beta={decay:.4g}. "
                    f"w в середине сойдётся, wmax у заделки — нет. "
                    f"Рекомендуемое nx >= {nx_rec}."
                )
                self._log(f"V-9 BOUNDARY LAYER WARNING: dx={dx:.4g} > {dx_required:.4g}; decay length 1/beta={decay:.4g}; recommended nx >= {nx_rec}.\n")
                messagebox.showwarning("Разрешение краевого эффекта (В-9)",msg)
        except Exception:
            # Numeric validity itself is checked by _case_dict/preflight.
            pass

        # V5 audit rule: conspicuous unit-scale mismatches are warnings only,
        # never hidden material restrictions.  The user may continue.
        try:
            warns=funits.plausibility_warnings(
                self._v("unit_system","N_MM_MPA").get(),
                L=float(self._v("L").get()), R=float(self._v("R").get()),
                h=float(self._v("h").get()), E0=float(self._v("E0").get()),
                load_max=(float(self._v("load_q").get()) if self._v("load_q").get().strip() else None),
            )
            if warns:
                text="\n\n".join(warns)
                self._log("V5 UNIT WARNING (see dialog)\n")
                messagebox.showwarning("Проверка единиц (V5)",text)
        except Exception:
            # Numeric validity itself is checked by _case_dict; V5 does not
            # replace those input validators.
            pass
        try:
            if self._v("mode").get().endswith("BIRGER") and self._v("quadrature_type","GAUSS").get()=="GAUSS":
                ng=int(self._v("n_gauss").get())
                if ng<7:
                    msg=(f"V6: n_gauss={ng} для расчёта Биргера может недостаточно разрешать "
                         "упругопластическую границу по толщине. Рекомендуется >=7 либо LAYERED. "
                         "Это предупреждение, не запрет запуска.")
                    self._log(f"V6 QUADRATURE WARNING: n_gauss={ng}; Birger thickness resolution may be insufficient. Recommended: n_gauss >= 7 or LAYERED.\n"); messagebox.showwarning("Квадратура по толщине (V6)",msg)
        except Exception:
            pass
        return True

    def run_case(self):
        if self.post_proc is not None and self.post_proc.poll() is None:
            messagebox.showwarning("Постобработка","Дождитесь окончания постобработки предыдущего расчёта или выберите другую папку результатов.")
            return
        if not self._preflight_or_show(): return
        self._stop_requested=False
        self._paused=False;self._paused_pids=[];self._pause_started_at=None;self._paused_total=0.0
        self._reset_online_view(); self._online_wall_start=time.monotonic()
        try:self.notebook.select(self.tab_online)
        except Exception:pass
        # Use the output-bundle launcher defined in this final GUI class, not
        # the legacy parent launcher.  Each press of Start therefore gets a
        # unique self-contained run folder + GUI preset + independent Excel.
        return self._run_case_auto_bundle()

    def validate_case(self):
        if not self._preflight_or_show(): return
        return super().validate_case()

    def _j2_case_from_gui(self):
        kind=self._v("j2_hardening_kind","LINEAR").get().upper().strip()
        if kind not in ("PERFECT","LINEAR","TABLE"):
            raise ValueError("J2 hardening_kind должен быть PERFECT, LINEAR или TABLE")
        data={
            "hardening_kind":kind,
            "sigma_y0":_float(self._v("j2_sigma_y0").get(),"J2 σy0",positive=True),
            "initial_history":{
                "source":self._v("j2_history_source","ELASTIC_ZERO").get().upper().strip()
            },
        }
        if data["initial_history"]["source"]=="NPZ":
            path=self._v("j2_history_npz").get().strip()
            if not path:
                raise ValueError("Для J2 initial_history=NPZ укажите файл истории")
            data["initial_history"]["npz"]=path
        elif data["initial_history"]["source"]!="ELASTIC_ZERO":
            raise ValueError("J2 initial_history.source должен быть ELASTIC_ZERO или NPZ")
        for key,target,positive in (
            ("j2_H","H",False),
            ("j2_yield_tolerance","yield_tolerance",False),
            ("j2_local_tolerance","local_tolerance",True),
        ):
            text=self._v(key).get().strip()
            if text:
                data[target]=_float(text,target,positive=positive,nonnegative=not positive)
        raw=self._v("j2_max_local_iterations").get().strip()
        if raw:
            data["max_local_iterations"]=_int(raw,"J2 max_local_iterations",1)
        if kind=="TABLE":
            rows=[]
            for number,line in enumerate(self.j2_hardening_table.get("1.0","end").splitlines(),1):
                text=line.strip()
                if not text or text.startswith("#"):
                    continue
                values=[part.strip() for part in text.replace(";"," ").replace(","," ").split()]
                if len(values)!=2:
                    raise ValueError(f"J2 table line {number}: expected κ, σy")
                rows.append([
                    _float(values[0],f"J2 table κ line {number}",nonnegative=True),
                    _float(values[1],f"J2 table σy line {number}",positive=True),
                ])
            if len(rows)<2:
                raise ValueError("J2 TABLE требует минимум две строки κ, σy")
            data["hardening_table"]=rows
        return data

    def _case_dict(self):
        theory=self._v("theory","CUF").get()
        material=self._v("material_model","HOOKE").get()
        if theory in capreg.PLANNED_THEORIES:
            raise ValueError(f"{theory}: расчёт недоступен до верификации математической постановки")
        if material!="HOOKE":
            raise ValueError("В cylinder-flexo1 отключены Биргер и J2, доступна только упругость")
        self._v("initial_slope_rtol","0.05").set("0.05")
        method=self.vars.get("method",tk.StringVar(value="FDM")).get()
        # Enhanced base expects nx/ns/stencil. For MVI, allow the independent MVI grid even if FDM grid is not filled yet.
        original={k:self.vars[k].get() if k in self.vars else "" for k in ("nx","ns","stencil")}
        if method=="MVI" and (not original["nx"].strip() or not original["ns"].strip() or not original["stencil"].strip()):
            for src,dst in (("mvi_nx","nx"),("mvi_ns","ns"),("mvi_stencil","stencil")):
                self._v(dst).set(self._v(src).get())
        try:cfg=super()._case_dict()
        finally:
            for k,v in original.items():self._v(k).set(v)
        # Independent grids. FDM values fall back to active base grid only when they are not entered and method=MVI.
        def grid_from(keys,fallback):
            vals=[]
            for key,fb in zip(keys,(fallback["nx"],fallback["ns"],fallback["stencil"])):
                txt=self._v(key).get().strip()
                vals.append(int(txt) if txt else int(fb))
            if vals[0]<7 or vals[1]<7 or vals[2]<3:raise ValueError("Сетка должна иметь nx, ns >= 7 и stencil >= 3")
            return {"nx":vals[0],"ns":vals[1],"stencil":vals[2]}
        base=dict(cfg["grid"])
        fdm=grid_from(("nx","ns","stencil"),base)
        # For an actual MVI calculation, its separate grid must be entered explicitly.
        if method=="MVI" and any(not self._v(k).get().strip() for k in ("mvi_nx","mvi_ns","mvi_stencil")):
            raise ValueError("Введите отдельную сетку МВИ: MVI nx, MVI ns и MVI FD stencil")
        mvi=grid_from(("mvi_nx","mvi_ns","mvi_stencil"),fdm)
        cfg["fdm_grid"]=fdm; cfg["mvi_grid"]=mvi; cfg["grid"]=dict(fdm if method=="FDM" else mvi)
        cfg["geometry_model"]=self._v("geometry_model","VON_KARMAN").get()
        cfg["material_model"]=self._v("material_model","HOOKE").get()
        cfg["scale_model"]=self._v("scale_model","MACRO").get()
        if cfg["scale_model"]=="MICRO":
            cfg["micro"]={
                "gradient_model":"G3D",
                "ell_m":self._v("micro_ell_m").get().strip(),
                "mu1_c_per_m":self._v("micro_mu1").get().strip(),
                "mu2_c_per_m":self._v("micro_mu2").get().strip(),
                "kappa_f_per_m":self._v("micro_kappa").get().strip(),
                "rho_kg_m3":self._v("micro_rho").get().strip(),
                "eta_s":self._v("micro_eta").get().strip(),
                "potential_order":self._v("micro_potential_order").get().strip(),
            }
            flexoscale.micro_material_from_case(cfg)
        if method=="MVI":
            form=self._v("mvi_formulation","STRONG").get().upper().strip()
            if form not in ("STRONG","VARIATIONAL"):
                raise ValueError("MVI formulation должна быть STRONG или VARIATIONAL")
            cfg.setdefault("inner_mvi",{})["formulation"]=form
            if form=="VARIATIONAL":
                if str(cfg.get("theory","")).upper()!="CUF":
                    raise ValueError("VARIATIONAL MVI сейчас предназначен только для CUF")
                if cfg["geometry_model"]!="GREEN_LAGRANGE" or cfg["material_model"]!="HOOKE":
                    raise ValueError("VARIATIONAL MVI сейчас предназначен только для GREEN_LAGRANGE + HOOKE")
        if cfg["material_model"]=="J2_INCREMENTAL":
            cfg["j2"]=self._j2_case_from_gui()
        return cfg

    def _gui_state(self):
        state={"format":"CYLINDER_GUI_PARAMETERS_V2","vars":{k:v.get() for k,v in self.vars.items() if k!="initial_slope_rtol"},"advanced_bc":self.advanced_bc,"engineering_bc":self.engineering_bc,"load_full_circumference":bool(getattr(self,"load_full_var",tk.BooleanVar(value=False)).get())}
        for name,widget in (("curve_points",getattr(self,"curve_points",None)),("curve_parameters",getattr(self,"curve_parameters",None)),("curve_piecewise",getattr(self,"curve_piecewise",None)),("control_points",getattr(self,"control_points_text",None))):
            if widget is not None:state[name]=widget.get("1.0","end-1c")
        return state

    def _apply_gui_state(self,state):
        if not isinstance(state,dict):
            raise ValueError("Некорректный файл параметров: ожидается JSON-объект")
        fmt=str(state.get("format") or "").strip()
        if fmt!="CYLINDER_GUI_PARAMETERS_V2":
            if "geometry" in state or "load_definition" in state or "boundary_conditions" in state:
                raise ValueError(
                    "Выбран FINAL_CASE.json/шаблон CASE, а не файл параметров интерфейса. "
                    "Для CASE используйте «Загрузить CASE»."
                )
            raise ValueError(
                "Неподдерживаемый формат файла параметров. "
                "Ожидается format=CYLINDER_GUI_PARAMETERS_V2."
            )
        if not isinstance(state.get("vars"),dict):
            raise ValueError(
                "Файл параметров интерфейса повреждён: отсутствует объект vars."
            )
        for k,val in state["vars"].items():
            if k!="initial_slope_rtol":
                if k=="output_dir" and _is_legacy_program_results_root(val):val=str(_default_results_root())
                self._v(k).set("" if val is None else str(val))
        if _is_legacy_program_results_root(self._v("output_dir").get()):
            self._v("output_dir").set(str(_default_results_root()))
        self._v("initial_slope_rtol","0.05").set("0.05")
        self.advanced_bc=state.get("advanced_bc")
        self.engineering_bc=state.get("engineering_bc")
        if hasattr(self,"load_full_var"):self.load_full_var.set(bool(state.get("load_full_circumference",False)))
        for name,widget in (("curve_points",getattr(self,"curve_points",None)),("curve_parameters",getattr(self,"curve_parameters",None)),("curve_piecewise",getattr(self,"curve_piecewise",None)),("control_points",getattr(self,"control_points_text",None))):
            if widget is not None:
                widget.delete("1.0","end"); widget.insert("1.0",str(state.get(name,"")))
        if not self._v("stencil","7").get().strip(): self._v("stencil").set("7")
        if not self._v("mvi_stencil","7").get().strip(): self._v("mvi_stencil").set("7")
        # Saved GUI parameters already carry values in their saved unit system;
        # do not reinterpret opening the file as a selector conversion event.
        self._last_unit_system=self._v("unit_system","N_MM_MPA").get()
        self._refresh_material_mode(); self._refresh_visibility(); self._refresh_load_step_mode(); self._refresh_curve_visibility(); self._update_units(); self._update_grid_info(); self._schedule_problem_scheme_redraw(20)
        try:
            if self.vars.get("curve_preview_max") and self.vars["curve_preview_max"].get().strip():self._preview_birger_curve()
        except Exception:pass

    def save_parameters_as(self):
        p=filedialog.asksaveasfilename(title="Сохранить параметры интерфейса",defaultextension=".json",initialfile="CYLINDER_GUI_PARAMETERS.json",filetypes=[("JSON","*.json")])
        if not p:return
        self.vars["parameters_path"].set(p); Path(p).write_text(json.dumps(self._gui_state(),indent=2,ensure_ascii=False),encoding="utf-8"); self._log(f"GUI parameters saved: {p}\n")

    def save_parameters_to_folder(self):
        folder=filedialog.askdirectory(title="Выберите папку для параметров расчёта")
        if not folder:return
        path=Path(folder)/"CYLINDER_GUI_PARAMETERS.json"
        path.write_text(json.dumps(self._gui_state(),indent=2,ensure_ascii=False),encoding="utf-8")
        self.vars["parameters_path"].set(str(path))
        self._log(f"GUI parameters saved in folder: {path}\n")
        messagebox.showinfo("Параметры сохранены",f"Все введённые параметры сохранены в:\n{path}")

    def save_parameters(self):
        p=self.vars.get("parameters_path",tk.StringVar()).get().strip()
        if not p:return self.save_parameters_as()
        Path(p).write_text(json.dumps(self._gui_state(),indent=2,ensure_ascii=False),encoding="utf-8"); self._log(f"GUI parameters saved: {p}\n")

    def open_parameters(self):
        p=filedialog.askopenfilename(title="Открыть параметры интерфейса",filetypes=[("JSON","*.json"),("All files","*.*")])
        if not p:return
        try:
            state=json.loads(Path(p).read_text(encoding="utf-8")); self.vars["parameters_path"].set(p); self._apply_gui_state(state); self._log(f"GUI parameters loaded: {p}\n")
        except Exception as exc:messagebox.showerror("Открытие параметров",str(exc))

    def load_case(self):
        """Restore a saved FINAL_CASE without changing its physical meaning."""
        p=filedialog.askopenfilename(title="Загрузить CASE",filetypes=[("JSON","*.json"),("All files","*.*")])
        if not p:return
        try:
            cfg=json.loads(Path(p).read_text(encoding="utf-8"))
            if not isinstance(cfg,dict):raise ValueError("CASE должен содержать JSON-объект")
            if cfg.get("format")=="CYLINDER_GUI_PARAMETERS_V2":
                raise ValueError(
                    "Выбран файл параметров интерфейса, а не FINAL_CASE.json. "
                    "Для параметров используйте «Открыть параметры интерфейса»."
                )
            if isinstance(cfg.get("newton"),dict) and "residual_tolerance" in cfg["newton"]:
                raise ValueError("V2: старое поле newton.residual_tolerance (абсолютное) запрещено; используйте residual_tolerance_relative")

            def get(obj,*path,default=""):
                cur=obj
                for key in path:
                    if not isinstance(cur,dict) or key not in cur:return default
                    cur=cur[key]
                return default if cur is None else cur

            def put(key,value):
                self._v(key).set("" if value is None else str(value))

            # CASE values already use the declared system.  Set this first and
            # mark it as current so _update_units updates labels only and never
            # rescales a just-loaded physical problem.
            unit_system=str(get(cfg,"units","system",default="N_MM_MPA")).upper()
            if unit_system not in UNIT_SYSTEMS:
                raise ValueError(f"CASE.units.system должен быть одним из {tuple(UNIT_SYSTEMS)}")
            self._v("unit_system").set(unit_system)
            self._last_unit_system=unit_system
            self._v("case_path").set(p)

            theory=str(get(cfg,"theory",default="CUF")).upper()
            if theory=="CUF_TE1":theory="CUF"
            put("theory",theory)
            put("cuf_model",str(get(cfg,"cuf_model",default="TE1")).upper())
            put("method",str(get(cfg,"method",default="FDM")).upper())
            put("mode",str(get(cfg,"mode",default="LINEAR_ELASTIC")).upper())
            put("rank",get(cfg,"rank",default=""))

            for key,path in {
                "L":("geometry","L"),"R":("geometry","R"),"h":("geometry","h"),
                "E0":("material","E0"),"nu0":("material","nu0"),
                "output_dir":("output_dir",),"profiles_npz":("initial_profiles_npz",),
                "loads_npz":("loads_npz",),
                "state_source":("initial_state","source"),"state_npz":("initial_state","npz"),
                "gb_source":("initial_GB","source"),"gb_npz":("initial_GB","npz"),
                "mvi_proj":("inner_mvi","projected_tolerance"),
                "mvi_inc":("inner_mvi","increment_tolerance"),
                "mvi_fd":("inner_mvi","tangent_fd_step"),
                "mvi_half":("inner_mvi","max_halfstep_iterations"),
                "mvi_cycles":("inner_mvi","max_xy_cycles"),
                "mvi_formulation":("inner_mvi","formulation"),
                "step_predictor":("step_predictor","mode"),
                "linear_res":("linear","residual_tolerance_relative"),
                "newton_res":("newton","residual_tolerance_relative"),
                "newton_inc":("newton","increment_tolerance"),
                "newton_fd":("newton","fd_step"),
                "newton_max":("newton","max_iterations"),
                "epsilon_tol":("birger","epsilon_tol"),"omega":("birger","omega"),
                "G_ref":("birger","G_ref"),"sigma_ref":("birger","sigma_ref"),
                "tol_G":("birger","tol_G"),"tol_sigma":("birger","tol_sigma"),
                "tol_u":("birger","tol_u"),"max_outer":("birger","max_outer_iterations"),
                "eps_limit":("birger","epsilon_i_limit"),"dJ_limit":("birger","delta_J_limit"),
            }.items():
                put(key,get(cfg,*path))

            mode=self._v("mode").get()
            geom=str(get(cfg,"geometry_model",default=("LINEAR" if mode.startswith("LINEAR") else ("GREEN_LAGRANGE" if theory=="CUF" else "VON_KARMAN")))).upper()
            material_model=str(get(
                cfg,"material_model",
                default=("J2_INCREMENTAL" if mode.endswith("J2") else
                         ("BIRGER" if mode.endswith("BIRGER") else "HOOKE"))
            )).upper()
            put("geometry_model",geom);put("material_model",material_model)
            put("scale_model",cfg.get("scale_model","MACRO"))
            mdata=cfg.get("micro",{}) if isinstance(cfg.get("micro"),dict) else {}
            for key,field in (("micro_ell_m","ell_m"),("micro_mu1","mu1_c_per_m"),
                 ("micro_mu2","mu2_c_per_m"),("micro_kappa","kappa_f_per_m"),
                 ("micro_rho","rho_kg_m3"),("micro_eta","eta_s"),
                 ("micro_potential_order","potential_order")):
                put(key,mdata.get(field,""))
            j2=cfg.get("j2") if isinstance(cfg.get("j2"),dict) else {}
            put("j2_hardening_kind",j2.get("hardening_kind","LINEAR"))
            put("j2_sigma_y0",j2.get("sigma_y0",""))
            put("j2_H",j2.get("H",""))
            put("j2_yield_tolerance",j2.get("yield_tolerance",""))
            put("j2_local_tolerance",j2.get("local_tolerance",""))
            put("j2_max_local_iterations",j2.get("max_local_iterations",""))
            initial_j2=j2.get("initial_history") if isinstance(j2.get("initial_history"),dict) else {}
            put("j2_history_source",initial_j2.get("source","ELASTIC_ZERO"))
            put("j2_history_npz",initial_j2.get("npz",""))
            if hasattr(self,"j2_hardening_table"):
                rows=j2.get("hardening_table",[])
                text="\n".join(
                    f"{row[0]}, {row[1]}" for row in rows
                    if isinstance(row,(list,tuple)) and len(row)>=2
                )
                self.j2_hardening_table.delete("1.0","end")
                self.j2_hardening_table.insert("1.0",text)

            # Material metadata is part of the CASE contract.  A missing local
            # library preset falls back to USER while retaining its exact E0/nu0.
            material=cfg.get("material") if isinstance(cfg.get("material"),dict) else {}
            source=str(material.get("source","USER") or "USER").upper()
            name=str(material.get("name","") or "")
            if source=="REFERENCE" and name not in self._load_material_library():
                self._log(f"CASE material preset not found locally; restored as USER: {name}\n")
                source="USER"
            if source not in ("USER","REFERENCE"):source="USER"
            put("material_source",source)
            put("material_name",name if source=="REFERENCE" else "")
            put("user_material_name",name if source=="USER" else "")

            fg=cfg.get("fdm_grid") or cfg.get("grid") or {}
            mg=cfg.get("mvi_grid") or cfg.get("grid") or fg
            for key,grid_key in (("nx","nx"),("ns","ns"),("stencil","stencil")):
                put(key,fg.get(grid_key,"") if isinstance(fg,dict) else "")
            for key,grid_key in (("mvi_nx","nx"),("mvi_ns","ns"),("mvi_stencil","stencil")):
                put(key,mg.get(grid_key,"") if isinstance(mg,dict) else "")

            quadrature=cfg.get("quadrature") if isinstance(cfg.get("quadrature"),dict) else {}
            qtype=str(quadrature.get("type","GAUSS") or "GAUSS").upper()
            put("quadrature_type",qtype)
            put("n_gauss",quadrature.get("n_gauss",""))
            bounds=quadrature.get("layer_bounds",[])
            put("layer_bounds",", ".join(str(v) for v in bounds) if isinstance(bounds,list) else "")

            bc=cfg.get("boundary_conditions") if isinstance(cfg.get("boundary_conditions"),dict) else {}
            advanced=bc.get("advanced")
            if isinstance(advanced,dict):
                put("bc_mode","INDIVIDUAL")
                self.advanced_bc=copy.deepcopy(advanced);self.engineering_bc=None
                put("bc_left","");put("bc_right","")
            else:
                put("bc_mode","SIMPLE")
                put("bc_left",bc.get("left","") or "");put("bc_right",bc.get("right","") or "")
                self.advanced_bc=None
                self.engineering_bc=copy.deepcopy(bc.get("engineering")) if isinstance(bc.get("engineering"),dict) else None

            # Restore every input that contributes to load_definition, including
            # its sign convention and continuation schedule.
            ld=cfg.get("load_definition") if isinstance(cfg.get("load_definition"),dict) else {}
            for key in ("load_q","load_p_line","load_point_P","load_point_x0","load_point_theta0",
                        "load_x1","load_x2","load_c0","load_c1","load_c2","load_band_width"):
                put(key,"")
            if ld:
                put("load_type",str(ld.get("type","UNIFORM_RADIAL_FACE")).upper())
                put("loads_npz",ld.get("npz",get(cfg,"loads_npz")))
                put("load_face",ld.get("surface",ld.get("face","OUTER")))
                put("load_coord",ld.get("circumferential_coordinate","THETA"))
                put("load_x1",ld.get("x1",""));put("load_x2",ld.get("x2",""))
                def magnitude(prefix):
                    signed=ld.get(prefix)
                    raw=ld.get(prefix+"_magnitude",abs(float(signed)) if signed is not None else "")
                    direction=str(ld.get("pressure_direction") or ("EXTERNAL" if signed is not None and float(signed)<0.0 else "INTERNAL")).upper()
                    return raw,direction
                if ld.get("type") in ("UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE","PATCH_RADIAL_FACE"):
                    raw,direction=magnitude("q");put("load_q",raw);put("load_pressure_direction",direction)
                if ld.get("type")=="POINT_RADIAL_FORCE":
                    raw,direction=magnitude("P");put("load_point_P",raw);put("load_pressure_direction",direction)
                    put("load_point_x0",ld.get("x0",""));put("load_point_theta0",ld.get("theta0",""))
                if ld.get("type")=="LINE_RADIAL_GENERATOR":
                    raw,direction=magnitude("p_line");put("load_p_line",raw);put("load_pressure_direction",direction)
                    put("load_c0",ld.get("c0",""));put("load_band_width",ld.get("band_width",""))
                put("load_harmonic_m",ld.get("m",self._v("load_harmonic_m").get()))
                put("load_harmonic_n",ld.get("n",self._v("load_harmonic_n").get()))
                put("load_c1",ld.get("c1",""));put("load_c2",ld.get("c2",""))
                if hasattr(self,"load_full_var"):self.load_full_var.set(bool(ld.get("full_circumference",False)))
                factors=ld.get("load_factors",[])
                put("load_factors",", ".join(f"{float(v):.15g}" for v in factors) if isinstance(factors,list) else "")
                steps=ld.get("step_definition") if isinstance(ld.get("step_definition"),dict) else {}
                step_mode=str(steps.get("mode","MANUAL_LIST" if factors else "COUNT")).upper()
                put("load_step_mode",step_mode)
                put("load_lambda_start",steps.get("lambda_start",""))
                put("load_lambda_step",steps.get("lambda_step",""))
                put("load_lambda_end",steps.get("lambda_end",""))
                put("load_step_count",steps.get("step_count",""))
            else:
                put("load_type","EXTERNAL_NPZ" if get(cfg,"loads_npz") else "UNIFORM_RADIAL_FACE")
                if hasattr(self,"load_full_var"):self.load_full_var.set(False)

            birger=cfg.get("birger") if isinstance(cfg.get("birger"),dict) else {}
            curve=birger.get("curve")
            if isinstance(curve,dict):self._set_curve_from_spec(curve)

            post=cfg.get("postprocess") if isinstance(cfg.get("postprocess"),dict) else {}
            put("deformation_scale",post.get("deformation_scale",self._v("deformation_scale").get()))
            thresholds=post.get("gb_ratio_thresholds",[])
            put("gb_thresholds",", ".join(str(v) for v in thresholds) if isinstance(thresholds,list) else "")
            if hasattr(self,"control_points_text") and isinstance(post.get("control_points"),list):
                rows=[]
                for item in post["control_points"]:
                    if not isinstance(item,dict):continue
                    rows.append(f"{item.get('label','P')}, {item.get('x','')}, {float(item.get('theta',0.0))*180.0/np.pi:.12g}")
                self.control_points_text.delete("1.0","end");self.control_points_text.insert("1.0","\n".join(rows))

            self._v("initial_slope_rtol","0.05").set("0.05")
            if _is_legacy_program_results_root(self._v("output_dir").get()):
                self._v("output_dir").set(str(_default_results_root()))
            self._refresh_material_mode()
            self._update_kinematics_labels();self._refresh_visibility()
            self._refresh_load_step_mode();self._refresh_curve_visibility()
            self._update_units();self._update_grid_info();self._schedule_problem_scheme_redraw(20)
            self._log(f"CASE fully loaded: {p}\n")
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки CASE",str(exc))
            self._log(f"CASE LOAD FAIL: {type(exc).__name__}: {exc}\n")

    # ------------------------------------------------------------------
    # Human-facing Stage12 shell
    # ------------------------------------------------------------------
    # The methods below intentionally override the earlier incremental UI
    # builders in this class.  They only reorganise widgets and translate
    # selector captions; all variables, CASE generation and solver launchers
    # remain the same.

    _DISPLAY_MAPS={
        "theory":{
            "Carrera CUF":"CUF",
            "Kirchhoff–Love":"KL",
            "Тимошенко (FSDT) · в разработке":"FSDT",
            "Редди (TSDT) · в разработке":"REDDY",
        },
        "method":{
            "МКР / конечные разности":"FDM",
            "МВИ":"MVI",
        },
        "mode":{
            "Линейный · упругий":"LINEAR_ELASTIC",
            "Нелинейный · упругий":"NONLINEAR_ELASTIC",
        },
        "unit_system":{
            "мм · Н · МПа":"N_MM_MPA",
            "м · Н · Па":"SI_N_M_PA",
        },
        "material_source":{
            "Ввести вручную":"USER",
            "Выбрать из справочника":"REFERENCE",
        },
        "bc_mode":{
            "Готовые условия / Preset":"SIMPLE",
            "Ручные условия / Custom":"INDIVIDUAL",
        },
        "bc_left":{
            "Выберите…":"",
            "Заделка":"CLAMPED",
            "Шарнир S3 (неподвижный)":"SIMPLY_SUPPORTED",
            "Шарнир S1 (осевое скольжение)":"PINNED_S1",
            "Шарнир S2 (окружное скольжение)":"PINNED_S2",
            "Шарнир S4 (подвижный в плоскости)":"PINNED_S4",
            "Свободный торец":"FREE",
            "Симметрия":"SYMMETRY",
            "Упругая / инженерная":"ELASTIC_ENGINEERING",
        },
        "bc_right":{
            "Выберите…":"",
            "Заделка":"CLAMPED",
            "Шарнир S3 (неподвижный)":"SIMPLY_SUPPORTED",
            "Шарнир S1 (осевое скольжение)":"PINNED_S1",
            "Шарнир S2 (окружное скольжение)":"PINNED_S2",
            "Шарнир S4 (подвижный в плоскости)":"PINNED_S4",
            "Свободный торец":"FREE",
            "Симметрия":"SYMMETRY",
            "Упругая / инженерная":"ELASTIC_ENGINEERING",
        },
        "load_type":{
            "Равномерное радиальное давление":"UNIFORM_RADIAL_FACE",
            "Гармоническое радиальное давление":"HARMONIC_RADIAL_FACE",
            "Давление на площадке":"PATCH_RADIAL_FACE",
            "Линейная радиальная нагрузка":"LINE_RADIAL_GENERATOR",
            "Сосредоточенная радиальная сила":"POINT_RADIAL_FORCE",
            "Нагрузка из NPZ":"EXTERNAL_NPZ",
        },
        "load_step_mode":{
            "Диапазон λ":"RANGE",
            "По числу шагов":"COUNT",
            "Список λ":"MANUAL_LIST",
        },
        "load_coord":{
            "Угол θ":"THETA",
            "Длина дуги s":"S",
        },
        "quadrature_type":{
            "Гаусс":"GAUSS",
            "По слоям":"LAYERED",
        },
    }

    _STANDARD_DEFAULTS={
        # Fixed/discretisation defaults.  Physical geometry, material and load
        # are deliberately not invented.
        "theory":"CUF", "cuf_model":"TE1", "method":"MVI",
        "mode":"NONLINEAR_ELASTIC", "geometry_model":"VON_KARMAN",
        "material_model":"HOOKE", "scale_model":"MACRO", "unit_system":"N_MM_MPA",
        "material_source":"USER", "bc_mode":"SIMPLE",
        "load_type":"UNIFORM_RADIAL_FACE", "load_coord":"THETA",
        "load_face":"OUTER", "curve_mode":"TABLE", "curve_extrap":"ERROR",
        "nx":"41", "ns":"41", "mvi_nx":"41", "mvi_ns":"41",
        "stencil":"7", "mvi_stencil":"7", "n_gauss":"5", "rank":"1",
        "linear_res":"1e-10",
        "newton_res":"1e-9", "newton_inc":"1e-9",
        "newton_fd":"1e-8", "newton_max":"30",
        "mvi_proj":"1e-7", "mvi_inc":"1e-9", "mvi_fd":"1e-7",
        "mvi_half":"80", "mvi_cycles":"30", "step_predictor":"ERROR",
        "epsilon_tol":"1e-12", "omega":"1.0",
        "tol_G":"1e-8", "tol_sigma":"1e-7", "tol_u":"1e-7",
        "max_outer":"30", "eps_limit":"1.0", "dJ_limit":"1.0",
        "state_source":"ZERO", "gb_source":"ELASTIC_G0",
        "j2_hardening_kind":"LINEAR", "j2_yield_tolerance":"1e-10",
        "j2_local_tolerance":"1e-12", "j2_max_local_iterations":"48",
        "j2_history_source":"ELASTIC_ZERO",
        "deformation_scale":"1.0", "load_factors":"1",
        "load_step_mode":"COUNT", "load_pressure_direction":"INTERNAL", "load_lambda_start":"0.01",
        "load_lambda_step":"0.01", "load_lambda_end":"1.0",
        "load_step_count":"20",
        "quadrature_type":"GAUSS",
    }

    def _configure_human_style(self):
        style=ttk.Style(self)
        try:style.theme_use("clam")
        except Exception:pass
        style.configure(".",font=("Segoe UI",10),background="#f4f6f8")
        style.configure("TFrame",background="#f4f6f8")
        style.configure("TLabel",background="#f4f6f8",foreground="#243447")
        style.configure("TLabelframe",background="#f4f6f8",bordercolor="#d9e0e7",relief="solid")
        style.configure("TLabelframe.Label",background="#f4f6f8",foreground="#17324d",font=("Segoe UI Semibold",10))
        style.configure("TNotebook",background="#f4f6f8",borderwidth=0)
        style.configure("TNotebook.Tab",padding=(15,8),font=("Segoe UI Semibold",9))
        style.map("TNotebook.Tab",background=[("selected","#ffffff"),("active","#eaf1f7")],foreground=[("selected","#0d4f7c")])
        style.configure("TButton",padding=(11,6))
        style.configure("Accent.TButton",padding=(18,8),font=("Segoe UI Semibold",10),foreground="white",background="#0877bd",bordercolor="#0877bd")
        style.map("Accent.TButton",background=[("active","#075f96"),("disabled","#9fb8c8")])
        style.configure("Stop.TButton",padding=(14,8),font=("Segoe UI Semibold",10),foreground="white",background="#c43d3d",bordercolor="#c43d3d")
        style.map("Stop.TButton",background=[("active","#9f2f2f"),("disabled","#d5b7b7")])
        style.configure("Mode.TRadiobutton",font=("Segoe UI Semibold",9),padding=(7,4))
        style.configure("Treeview",rowheight=25,background="white",fieldbackground="white")
        style.configure("Treeview.Heading",font=("Segoe UI Semibold",9))
        style.configure("Green.Vertical.TScrollbar",background="#20a847",troughcolor="#e5f2e9",bordercolor="#9bcbaa",arrowcolor="#ffffff")
        style.map("Green.Vertical.TScrollbar",background=[("active","#17883a"),("pressed","#126f30")])

    def _install_standard_defaults(self):
        for key,value in self._STANDARD_DEFAULTS.items():
            if not self._v(key,value).get().strip():self._v(key).set(value)
        if _is_legacy_program_results_root(self._v("output_dir").get()):
            self._v("output_dir").set(str(_default_results_root()))
        self._v("ui_level","BASIC")

    def _find_bound_combo(self,parent,key):
        target=str(self.vars.get(key,""))
        stack=[parent]
        while stack:
            w=stack.pop()
            try:stack.extend(w.winfo_children())
            except Exception:pass
            if isinstance(w,ttk.Combobox):
                try:
                    if str(w.cget("textvariable"))==target:return w
                except Exception:pass
        return None

    def _bind_display_combo(self,combo,key,callback=None):
        mapping=self._DISPLAY_MAPS[key]
        reverse={v:k for k,v in mapping.items()}
        display=tk.StringVar(value=reverse.get(self._v(key).get(),next(iter(mapping))))
        self._display_bindings[key]=(display,mapping)
        combo.configure(textvariable=display,values=tuple(mapping),state="readonly")
        def selected(_event=None):
            active_mapping=self._display_bindings.get(key,(display,mapping))[1]
            self._v(key).set(active_mapping.get(display.get(),self._v(key).get()))
            if callback is not None:callback()
        combo.bind("<<ComboboxSelected>>",selected)
        return combo

    def _sync_display_selectors(self):
        for key,(display,mapping) in getattr(self,"_display_bindings",{}).items():
            reverse={v:k for k,v in mapping.items()}
            val=self._v(key).get()
            if val in reverse and display.get()!=reverse[val]:display.set(reverse[val])

    def _selector(self,parent,column,title,key,width,callback=None,row=0):
        box=ttk.Frame(parent)
        box.grid(row=row,column=column,sticky="ew",padx=(0,10),pady=(0,6))
        ttk.Label(box,text=title,font=("Segoe UI Semibold",9)).pack(anchor="w")
        combo=ttk.Combobox(box,width=width,state="readonly")
        combo.pack(fill="x",pady=(3,0))
        self._bind_display_combo(combo,key,callback)
        return combo

    def _build_ui(self):
        # Fit the main window to the usable monitor area. Fixed 1380x900 and
        # minsize 1120x720 extended beyond smaller/scaled laptop screens.
        screen_w=max(640,int(self.winfo_screenwidth()))
        screen_h=max(480,int(self.winfo_screenheight()))
        window_w=min(1380,max(640,screen_w-40))
        window_h=min(900,max(480,screen_h-80))
        pos_x=max(0,(screen_w-window_w)//2)
        pos_y=max(0,(screen_h-window_h)//2)
        self.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")
        self.minsize(min(900,window_w),min(600,window_h))
        self.configure(background="#f4f6f8")
        self._configure_human_style()
        self._display_bindings={}
        self._install_standard_defaults()

        banner=tk.Frame(self,background="#12344d",height=70)
        banner.pack(fill="x")
        banner.pack_propagate(False)
        titlebox=tk.Frame(banner,background="#12344d")
        titlebox.pack(side="left",fill="y",padx=18,pady=10)
        tk.Label(titlebox,text="CYL_SOLVER",font=("Segoe UI Semibold",17),foreground="white",background="#12344d").pack(anchor="w")
        tk.Label(titlebox,text="Статический расчёт замкнутой цилиндрической оболочки",font=("Segoe UI",10),foreground="#cfe3f1",background="#12344d").pack(anchor="w")
        levelbox=tk.Frame(banner,background="#12344d")
        levelbox.pack(side="right",padx=18,pady=13)
        tk.Label(levelbox,text="Интерфейс",font=("Segoe UI",9),foreground="#cfe3f1",background="#12344d").pack(anchor="w")
        levelrow=tk.Frame(levelbox,background="#12344d");levelrow.pack(anchor="e")
        for text,value in (("Обычный","BASIC"),("Экспертный","EXPERT")):
            ttk.Radiobutton(levelrow,text=text,value=value,variable=self._v("ui_level"),style="Mode.TRadiobutton",command=self._apply_ui_level_layout).pack(side="left",padx=(0,4))

        model=ttk.LabelFrame(self,text="Расчётная модель",padding=(12,9))
        model.pack(fill="x",padx=12,pady=(10,7))
        # Two compact rows prevent the selector strip from forcing the window wider than the monitor.
        for c in range(4):model.columnconfigure(c,weight=1)
        self.theory_combo=self._selector(model,0,"Теория","theory",22,self._on_theory_change)

        cufbox=ttk.Frame(model);cufbox.grid(row=0,column=1,sticky="ew",padx=(0,10))
        self.cuf_selector_box=cufbox
        ttk.Label(cufbox,text="Модель CUF",font=("Segoe UI Semibold",9)).pack(anchor="w")
        self.cuf_model_combo=ttk.Combobox(cufbox,textvariable=self._v("cuf_model","TE1"),values=CUF_MODELS,state="readonly",width=8)
        self.cuf_model_combo.pack(fill="x",pady=(3,0));self.cuf_model_combo.bind("<<ComboboxSelected>>",lambda _e:self._on_model_change())

        geombox=ttk.Frame(model);geombox.grid(row=0,column=2,sticky="ew",padx=(0,10))
        ttk.Label(geombox,text="Геометрическая модель",font=("Segoe UI Semibold",9)).pack(anchor="w")
        self.geometry_display=tk.StringVar(value="Линейная")
        self.geometry_combo=ttk.Combobox(geombox,textvariable=self.geometry_display,state="readonly",width=22)
        self.geometry_combo.pack(fill="x",pady=(3,0))
        self.geometry_combo.bind("<<ComboboxSelected>>",lambda _e:self._on_geometry_material_change())

        matbox=ttk.Frame(model);matbox.grid(row=0,column=3,sticky="ew",padx=(0,10))
        ttk.Label(matbox,text="Материал",font=("Segoe UI Semibold",9)).pack(anchor="w")
        self.material_model_display=tk.StringVar(value="Закон Гука")
        self.material_model_combo=ttk.Combobox(matbox,textvariable=self.material_model_display,values=("Закон Гука",),state="readonly",width=24)
        self.material_model_combo.pack(fill="x",pady=(3,0))
        self.material_model_combo.bind("<<ComboboxSelected>>",lambda _e:self._on_geometry_material_change())

        self.method_combo=self._selector(model,0,"Численный метод","method",25,self._refresh_visibility,row=1)

        rankbox=ttk.Frame(model);rankbox.grid(row=1,column=1,sticky="ew",padx=(0,10),pady=(0,6))
        self.rank_selector_box=rankbox
        ttk.Label(rankbox,text="Порядок МВИ",font=("Segoe UI Semibold",9)).pack(anchor="w")
        self.rank_combo=ttk.Combobox(rankbox,textvariable=self._v("rank","1"),values=[str(i) for i in range(1,9)],state="readonly",width=8)
        self.rank_combo.pack(fill="x",pady=(3,0))

        self.unit_combo=self._selector(model,2,"Единицы","unit_system",14,self._update_units,row=1)
        namebox=ttk.Frame(model);namebox.grid(row=1,column=3,sticky="ew",pady=(0,6))
        ttk.Label(namebox,text="Название расчёта",font=("Segoe UI Semibold",9)).pack(anchor="w")
        ttk.Entry(namebox,textvariable=self._v("run_label"),width=25).pack(fill="x",pady=(3,0))
        self.model_info=ttk.Label(model,text="",foreground="#52687a")
        self.model_info.grid(row=2,column=0,columnspan=4,sticky="w",pady=(7,0))

        actions=ttk.Frame(self,padding=(12,0,12,7));actions.pack(fill="x")
        ttk.Button(actions,text="Открыть параметры",command=self.open_parameters).pack(side="left",padx=(0,5))
        ttk.Button(actions,text="Сохранить параметры в папку…",command=self.save_parameters_to_folder).pack(side="left",padx=5)
        ttk.Button(actions,text="Проверить",command=self.validate_case).pack(side="left",padx=(14,5))
        self.series_btn=ttk.Button(actions,text="МВИ 1–8",command=self.run_mvi_series)
        self.series_btn.pack(side="left",padx=5)
        self.top_results_btn=ttk.Button(actions,text="Открыть результаты",command=self.open_results)
        self.top_results_btn.pack(side="right",padx=(5,0))
        self.top_stop_btn=ttk.Button(actions,text="Стоп",command=self.stop_calculation,state="disabled",style="Stop.TButton")
        self.top_stop_btn.pack(side="right",padx=5)
        self.top_pause_btn=ttk.Button(actions,text="Пауза",command=self.toggle_pause_calculation,state="disabled")
        self.top_pause_btn.pack(side="right",padx=5)
        self.top_run_btn=ttk.Button(actions,text="Рассчитать",command=self.run_case,style="Accent.TButton")
        self.top_run_btn.pack(side="right",padx=5)
        self.run_btn=self.top_run_btn
        # Keep calculation progress on its own row.  VIDEO/3D and other action
        # buttons can make the toolbar wider; the old inline label was clipped
        # by Tk, which made step/percent information disappear.
        self.parameter_toolbar=actions
        progress_row=ttk.Frame(self,padding=(12,0,12,7))
        progress_row.pack(fill="x")
        self.progress_label=ttk.Label(
            progress_row,text="Принято 0/0 · 0%",width=30,anchor="w"
        )
        self.progress_label.pack(side="left",padx=(0,8))
        self.progress=GreenStepProgressbar(progress_row,mode="determinate",length=205,height=18)
        self.progress.set_text("Принято 0/0 · 0%")
        self.progress.pack(side="left",fill="x",expand=True)
        self.progress_status_label=ttk.Label(progress_row,text="Готово",width=22,anchor="e")
        self.progress_status_label.pack(side="right",padx=(8,0))
        self.progress_row=progress_row

        self.notebook=ttk.Notebook(self)
        self.notebook.pack(fill="both",expand=True,padx=12,pady=(0,10))
        self.tab_problem=ttk.Frame(self.notebook,padding=10)
        self.tab_load=ttk.Frame(self.notebook,padding=10)
        self.tab_numeric=ttk.Frame(self.notebook,padding=10)
        self.tab_birger=ttk.Frame(self.notebook,padding=10)
        self.tab_j2=ttk.Frame(self.notebook,padding=10)
        self.tab_online=ttk.Frame(self.notebook,padding=8)
        self.tab_results=ttk.Frame(self.notebook,padding=10)
        self.tab_files=ttk.Frame(self.notebook,padding=10)
        self.tab_run=ttk.Frame(self.notebook,padding=10)

        self._build_problem_tab()
        self._build_load_tab()
        self._build_numeric_tab()
        self._build_birger_tab()
        self._build_j2_tab()
        self._build_online_tab()
        self._build_results_tab()
        self._build_files_tab()
        self._build_run_tab()
        self._prepare_basic_problem_grid()
        self._apply_basic_grid_preset()
        self._polish_existing_controls()
        self._update_kinematics_labels()
        self._apply_ui_level_layout()
        self._refresh_material_mode()
        self._update_units()
        self._install_global_scroll_dispatch()

    @staticmethod
    def _widget_is_inside(widget,ancestor):
        if widget is None or ancestor is None:return False
        cur=widget
        while cur is not None:
            if cur is ancestor:return True
            cur=getattr(cur,"master",None)
        return False

    def _dispatch_scrollable_mousewheel(self,event):
        """Route the wheel to the scrollable parameter pane under the pointer.

        Binding only the Canvas itself does not work when the pointer is over an
        Entry/Combobox/Label.  This dispatcher fixes that without changing any
        calculation logic.
        """
        try:target=self.winfo_containing(event.x_root,event.y_root)
        except Exception:target=getattr(event,"widget",None)
        pairs=((getattr(self,"problem_left_canvas",None),getattr(self,"problem_left_inner",None)),
               (getattr(self,"problem_right_canvas",None),getattr(self,"problem_right_inner",None)),
               (getattr(self,"load_left_canvas",None),getattr(self,"load_left_inner",None)))
        for canvas,inner in pairs:
            if canvas is None or inner is None:continue
            if not (self._widget_is_inside(target,inner) or target is canvas):continue
            try:
                if getattr(event,"num",None)==4:steps=-1
                elif getattr(event,"num",None)==5:steps=1
                else:
                    delta=float(getattr(event,"delta",0) or 0)
                    if delta==0:return "break"
                    magnitude=max(1,int(abs(delta)/120.0))
                    steps=-magnitude if delta>0 else magnitude
                canvas.yview_scroll(int(steps),"units")
                return "break"
            except Exception:
                return "break"
        return None

    def _install_global_scroll_dispatch(self):
        if bool(getattr(self,"_global_scroll_dispatch_installed",False)):return
        self._global_scroll_dispatch_installed=True
        # Child widgets (Entry/Combobox/etc.) receive the wheel event before the
        # Tk class binding, so scrolling does not accidentally change a combo.
        for canvas,inner in ((getattr(self,"problem_left_canvas",None),getattr(self,"problem_left_inner",None)),
                             (getattr(self,"problem_right_canvas",None),getattr(self,"problem_right_inner",None)),
                             (getattr(self,"load_left_canvas",None),getattr(self,"load_left_inner",None))):
            if canvas is None or inner is None:continue
            try:canvas.unbind("<MouseWheel>");inner.unbind("<MouseWheel>")
            except Exception:pass
            stack=[canvas,inner]
            seen=set()
            while stack:
                w=stack.pop()
                if id(w) in seen:continue
                seen.add(id(w))
                try:
                    w.bind("<MouseWheel>",self._dispatch_scrollable_mousewheel,add="+")
                    w.bind("<Button-4>",self._dispatch_scrollable_mousewheel,add="+")
                    w.bind("<Button-5>",self._dispatch_scrollable_mousewheel,add="+")
                    stack.extend(w.winfo_children())
                except Exception:pass
        # Fallback for dynamically shown children and platform-specific bindtags.
        self.bind_all("<MouseWheel>",self._dispatch_scrollable_mousewheel,add="+")
        self.bind_all("<Button-4>",self._dispatch_scrollable_mousewheel,add="+")
        self.bind_all("<Button-5>",self._dispatch_scrollable_mousewheel,add="+")

    def _build_run_tab(self):
        top=ttk.Frame(self.tab_run);top.pack(fill="x")
        ttk.Label(top,text="Технический журнал расчёта",font=("Segoe UI Semibold",11)).pack(side="left")
        ttk.Button(top,text="Очистить журнал",command=self._clear_logs).pack(side="right")
        ttk.Label(self.tab_run,text="Эта вкладка нужна для диагностики. В обычном режиме ход расчёта показывается на вкладке «Онлайн».",foreground="#5d7182").pack(anchor="w",pady=(5,7))
        self.log=tk.Text(self.tab_run,wrap="word",height=28,font=("Consolas",9),background="#fbfcfd",foreground="#21313f")
        self.log.pack(fill="both",expand=True)

    def _clear_logs(self):
        for name in ("log","online_log","online_log_detached"):
            w=getattr(self,name,None)
            if w is not None:
                try:w.delete("1.0","end")
                except Exception:pass

    def _log(self,text):
        for name in ("log","online_log","online_log_detached"):
            w=getattr(self,name,None)
            if w is None:continue
            try:
                w.insert("end",text)
                if bool(getattr(self,"_log_autoscroll_states",{}).get(name,True)):
                    w.see("end")
            except Exception:pass

    def _prepare_basic_problem_grid(self):
        if not hasattr(self,"fdm_grid_frame"):return
        parent=self.fdm_grid_frame.master
        self.basic_grid_frame=ttk.LabelFrame(parent,text="Сетки МВИ и МКР",padding=8)
        before_widget=getattr(getattr(self,"material_note",None),"master",self.fdm_grid_frame)
        self._basic_grid_before=before_widget
        self.basic_grid_frame.pack(fill="x",before=before_widget,pady=(0,8))
        self.basic_grid_frame.columnconfigure(0,weight=1)
        self.basic_grid_frame.columnconfigure(1,weight=1)

        presetrow=ttk.Frame(self.basic_grid_frame);presetrow.grid(row=0,column=0,columnspan=2,sticky="ew",pady=(0,5))
        ttk.Label(presetrow,text="Быстрый выбор").pack(side="left")
        self.basic_grid_preset=ttk.Combobox(presetrow,textvariable=self._v("basic_grid_preset","41×41 · рабочая"),values=("21×21 · быстро","41×41 · рабочая","61×61 · подробно"),state="readonly",width=21)
        self.basic_grid_preset.pack(side="left",padx=7)
        self.basic_grid_preset.bind("<<ComboboxSelected>>",lambda _e:self._apply_basic_grid_preset())
        ttk.Label(presetrow,text="Пресет заполняет обе сетки; затем их можно менять независимо.",foreground="#5d7182").pack(side="left",padx=7)

        ttk.Label(self.basic_grid_frame,text="nz — точек Гаусса по толщине").grid(row=1,column=0,sticky="w",pady=3)
        ttk.Entry(self.basic_grid_frame,textvariable=self._v("n_gauss","5"),width=12).grid(row=1,column=1,sticky="w",padx=6,pady=3)

        self.basic_fdm_controls=ttk.LabelFrame(self.basic_grid_frame,text="Сетка МКР / FDM",padding=6)
        self.basic_mvi_controls=ttk.LabelFrame(self.basic_grid_frame,text="Сетка МВИ",padding=6)
        for fr,prefix in ((self.basic_fdm_controls,""),(self.basic_mvi_controls,"mvi_")):
            ttk.Label(fr,text="Узлов вдоль оси nx").grid(row=0,column=0,sticky="w",pady=3)
            ex=ttk.Entry(fr,textvariable=self._v(prefix+"nx"),width=12);ex.grid(row=0,column=1,sticky="w",padx=6,pady=3)
            ttk.Label(fr,text="Узлов по окружности ns").grid(row=1,column=0,sticky="w",pady=3)
            es=ttk.Entry(fr,textvariable=self._v(prefix+"ns"),width=12);es.grid(row=1,column=1,sticky="w",padx=6,pady=3)
            ex.bind("<KeyRelease>",lambda _e:self._update_grid_info());es.bind("<KeyRelease>",lambda _e:self._update_grid_info())
        self.basic_fdm_controls.grid(row=2,column=0,sticky="nsew",padx=(0,4),pady=(4,0))
        self.basic_mvi_controls.grid(row=2,column=1,sticky="nsew",padx=(4,0),pady=(4,0))

        self.basic_profile_row=ttk.Frame(self.basic_grid_frame);self.basic_profile_row.grid(row=3,column=0,columnspan=2,sticky="ew",pady=(6,0));self.basic_profile_row.columnconfigure(1,weight=1)
        ttk.Label(self.basic_profile_row,text="Начальное приближение МВИ").grid(row=0,column=0,sticky="w")
        ttk.Label(
            self.basic_profile_row,
            text="Создаётся автоматически по сетке, толщине и числу мод",
            foreground="#24733b",
        ).grid(row=0,column=1,columnspan=2,sticky="w",padx=6)
        self.basic_grid_info=ttk.Label(self.basic_grid_frame,text="",foreground="#5d7182",wraplength=470)
        self.basic_grid_info.grid(row=4,column=0,columnspan=2,sticky="w",pady=(6,0))

    def _apply_basic_grid_preset(self):
        try:n=int(self._v("basic_grid_preset").get().split("×",1)[0])
        except Exception:return
        for key in ("nx","ns","mvi_nx","mvi_ns"):self._v(key).set(str(n))
        self._update_grid_info()

    def _polish_existing_controls(self):
        # Translate high-frequency choices while retaining the exact internal
        # strings consumed by CASE validation and the dispatcher.
        callbacks={
            "material_source":self._refresh_material_mode,
            "bc_mode":self._refresh_visibility,
            "bc_left":self._refresh_visibility,
            "bc_right":self._refresh_visibility,
            "load_type":self._refresh_load_visibility,
            "load_step_mode":self._refresh_load_step_mode,
            "load_coord":self._update_load_units,
            "quadrature_type":self._refresh_visibility,
        }
        for key,callback in callbacks.items():
            combo=self._find_bound_combo(self,key)
            if combo is not None:self._bind_display_combo(combo,key,callback)
        self._prepare_user_material_controls()

        # Remember the advanced-only result card so the viewer can use the full
        # width in ordinary mode.
        self.postprocess_options_frame=None
        for child in self.tab_results.winfo_children():
            try:
                if isinstance(child,ttk.LabelFrame) and child.cget("text")=="Автоматическая постобработка":
                    self.postprocess_options_frame=child
            except Exception:pass

        self.birger_curve_frame=None
        for child in self.tab_birger.winfo_children():
            try:
                if isinstance(child,ttk.LabelFrame) and str(child.cget("text")).startswith("Материальная кривая"):
                    self.birger_curve_frame=child
            except Exception:pass

        # Shorter human captions on the load page.  Detailed method notes stay
        # available in expert mode but no longer dominate ordinary input.
        self._basic_hidden_labels=[]
        for root in (self.tab_load,self.tab_problem):
            stack=[root]
            while stack:
                w=stack.pop()
                try:stack.extend(w.winfo_children())
                except Exception:pass
                if not isinstance(w,ttk.Label):continue
                try:text=str(w.cget("text"))
                except Exception:continue
                if text.startswith("V8:") or text.startswith("Встроенные radial") or text.startswith("Для CUF редактор") or text.startswith("Справочные значения"):
                    self._basic_hidden_labels.append(w)

    def _prepare_user_material_controls(self):
        self.material_save_frame=None
        stack=[self.tab_problem]
        material_frame=None
        while stack:
            w=stack.pop()
            try:stack.extend(w.winfo_children())
            except Exception:pass
            try:
                if isinstance(w,ttk.LabelFrame) and w.cget("text")=="Материал":material_frame=w;break
            except Exception:pass
        if material_frame is None:return
        fr=ttk.LabelFrame(material_frame,text="Сохранить свой материал",padding=6)
        fr.grid(row=6,column=0,columnspan=2,sticky="ew",pady=(8,0));fr.columnconfigure(1,weight=1)
        ttk.Label(fr,text="Название").grid(row=0,column=0,sticky="w")
        self.user_material_name_entry=ttk.Entry(fr,textvariable=self._v("user_material_name"),width=28)
        self.user_material_name_entry.grid(row=0,column=1,sticky="ew",padx=6)
        self.user_material_save_btn=ttk.Button(fr,text="Добавить в библиотеку",command=self._save_user_material_to_library)
        self.user_material_save_btn.grid(row=0,column=2,sticky="e")
        ttk.Label(fr,text="Сохраняются название, E0 и ν0. Материал сразу появится в списке после сохранения.",wraplength=440,foreground="#5d7182").grid(row=1,column=0,columnspan=3,sticky="w",pady=(5,0))
        self.material_save_frame=fr

    def _save_user_material_to_library(self):
        try:
            name=self._v("user_material_name").get().strip()
            if not name:raise ValueError("Введите название материала")
            if len(name)>96 or "\n" in name or "\r" in name:raise ValueError("Название материала должно быть одной строкой не длиннее 96 символов")
            E=_float(self._v("E0").get(),"E0",positive=True)
            nu=_float(self._v("nu0").get(),"ν0")
            if not (-1.0<nu<0.5):raise ValueError("Для изотропного материала требуется -1 < ν0 < 0.5")
            Epa=E*1.0e6 if self._v("unit_system","N_MM_MPA").get()=="N_MM_MPA" else E
            path=APP_DIR/"MATERIAL_LIBRARY.json"
            data=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version":1,"materials":[]}
            items=list(data.get("materials") or [])
            matches=[i for i,m in enumerate(items) if str(m.get("name","")).casefold()==name.casefold()]
            if matches and not messagebox.askyesno("Материал уже существует",f"Материал «{name}» уже есть в библиотеке. Заменить его?"):
                return
            item={"name":name,"E_Pa":float(Epa),"nu":float(nu),"note":"Пользовательский материал.","source":"USER"}
            if matches:items[matches[0]]=item
            else:items.append(item)
            data["version"]=max(int(data.get("version",1)),1);data["materials"]=items
            tmp=path.with_name(path.name+".tmp")
            tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            tmp.replace(path)
            if hasattr(self,"_material_library_cache"):delattr(self,"_material_library_cache")
            self._v("material_source").set("REFERENCE");self._v("material_name").set(name)
            self._refresh_material_mode();self._sync_display_selectors()
            self._log(f"User material saved to library: {name}\n")
            messagebox.showinfo("Материал сохранён",f"«{name}» добавлен в библиотеку материалов.")
        except Exception as exc:
            messagebox.showerror("Сохранение материала",str(exc))

    def _refresh_material_mode(self):
        EnhancedSolverGUI._refresh_material_mode(self)
        user=self._v("material_source","USER").get()=="USER"
        try:
            self.user_material_name_entry.configure(state="normal" if user else "disabled")
            self.user_material_save_btn.configure(state="normal" if user else "disabled")
        except Exception:pass

    def _rebuild_visible_tabs(self):
        if not hasattr(self,"notebook"):return
        try:selected=str(self.notebook.select())
        except Exception:selected=""
        for tab in (self.tab_problem,self.tab_load,self.tab_numeric,self.tab_birger,self.tab_j2,self.tab_online,self.tab_results,self.tab_files,self.tab_run):
            try:self.notebook.forget(tab)
            except Exception:pass
        expert=self._v("ui_level","BASIC").get()=="EXPERT"
        # Historical Birger/J2 widgets are not available in the flexo interface.
        tabs=[(self.tab_problem,"1 · Постановка"),(self.tab_load,"2 · Нагрузка")]
        if expert:tabs.append((self.tab_numeric,"Численные настройки"))
        tabs.extend(((self.tab_online,"Онлайн-расчёт"),(self.tab_results,"Результаты")))
        if expert:tabs.extend(((self.tab_files,"Файлы и начальные поля"),(self.tab_run,"Журнал")))
        for tab,title in tabs:self.notebook.add(tab,text=title)
        visible={str(tab) for tab,_ in tabs}
        try:
            if selected in visible:self.notebook.select(selected)
        except Exception:pass

    def _apply_ui_level_layout(self):
        if not hasattr(self,"notebook"):return
        expert=self._v("ui_level","BASIC").get()=="EXPERT"
        self._rebuild_visible_tabs()

        if hasattr(self,"basic_grid_frame"):
            if expert:
                self.basic_grid_frame.pack_forget()
                if not self.fdm_grid_frame.winfo_manager():self.fdm_grid_frame.pack(fill="x")
            else:
                self.fdm_grid_frame.pack_forget()
                if not self.basic_grid_frame.winfo_manager():
                    self.basic_grid_frame.pack(
                        fill="x",before=getattr(self,"_basic_grid_before",self.fdm_grid_frame),pady=(0,8)
                    )

        for w in getattr(self,"_basic_hidden_labels",[]):
            try:
                if expert:w.grid()
                else:w.grid_remove()
            except Exception:pass

        # The ordinary Birger page contains only the material curve.  Reference
        # scales, tolerances and cycle limits are generated from documented
        # defaults and remain inspectable/editable in expert mode.
        if hasattr(self,"birger_frame"):
            curve=getattr(self,"birger_curve_frame",None)
            if expert:
                self.birger_frame.grid()
                if curve is not None:curve.grid_configure(column=1,columnspan=1)
            else:
                self.birger_frame.grid_remove()
                if curve is not None:curve.grid_configure(column=0,columnspan=2)

        opts=getattr(self,"postprocess_options_frame",None)
        viewer=getattr(self,"result_viewer",None)
        if opts is not None and viewer is not None:
            if expert:
                opts.grid();viewer.grid_configure(column=1,columnspan=1)
            else:
                opts.grid_remove();viewer.grid_configure(column=0,columnspan=2)

        self._sync_display_selectors()
        self._refresh_basic_grid_visibility()

    def _refresh_basic_grid_visibility(self):
        if not hasattr(self,"basic_fdm_controls"):return
        mvi_active=self._v("method","FDM").get()=="MVI"
        cuf_active=self._v("theory","CUF").get()=="CUF"
        expert=self._v("ui_level","BASIC").get()=="EXPERT"
        self.basic_grid_frame.configure(text="Сетки МВИ и МКР")
        self.basic_fdm_controls.grid();self.basic_mvi_controls.grid()
        if mvi_active:self.basic_profile_row.grid()
        else:self.basic_profile_row.grid_remove()
        try:
            self.series_btn.configure(state="normal" if mvi_active else "disabled")
            self.rank_combo.configure(state="readonly" if mvi_active else "disabled")
            if mvi_active or expert:
                if not self.series_btn.winfo_manager():self.series_btn.pack(side="left",padx=5)
                self.rank_selector_box.grid()
            else:
                self.series_btn.pack_forget();self.rank_selector_box.grid_remove()
            if cuf_active:self.cuf_selector_box.grid()
            else:self.cuf_selector_box.grid_remove()
        except Exception:pass

    def _prepare_ordinary_hidden_values(self):
        """Fill only service/scaling values hidden by the ordinary UI."""
        self._install_standard_defaults()
        if self._v("ui_level","BASIC").get()!="BASIC":return
        if not self._v("mode","LINEAR_ELASTIC").get().endswith("BIRGER"):return
        try:
            E0=float(self._v("E0").get());nu0=float(self._v("nu0").get())
            G0=E0/(2.0*(1.0+nu0))
            if not self._v("G_ref").get().strip():self._v("G_ref").set(f"{G0:.12g}")
            if not self._v("sigma_ref").get().strip():
                sigma_scale=None
                try:
                    spec=self._curve_spec()
                    cm=str(spec.get("mode","")).upper()
                    eps_max=None if cm in ("TABLE","CSV") else float(self._v("eps_limit","1.0").get())
                    _x,y=birgercurve.sample_curve(spec,base_dir=self._current_curve_base_dir(),epsilon_max=eps_max,n=200)
                    sigma_scale=float(np.max(np.asarray(y,float)))
                except Exception:pass
                if not sigma_scale or not np.isfinite(sigma_scale) or sigma_scale<=0:sigma_scale=E0
                self._v("sigma_ref").set(f"{sigma_scale:.12g}")
        except Exception:
            # Geometry/material validation below will report the actual missing
            # physical field.  Do not replace that message with a scale error.
            pass

    def _refresh_visibility(self):
        EnhancedSolverGUI._refresh_visibility(self)
        self._update_kinematics_labels()
        self._sync_display_selectors()
        self._refresh_basic_grid_visibility()
        self._rebuild_visible_tabs()
        pending=self._v("theory","CUF").get() in capreg.PLANNED_THEORIES
        micro_pending=self._v("scale_model","MACRO").get()=="MICRO"
        retired=self._v("material_model","HOOKE").get()!="HOOKE"
        if pending or retired or micro_pending:
            for name in ("top_run_btn","series_btn","bc_left_combo","bc_right_combo","bc_edit_btn"):
                w=getattr(self,name,None)
                if w is not None:
                    try:w.configure(state="disabled")
                    except Exception:pass
        else:
            try:self.top_run_btn.configure(state="normal")
            except Exception:pass

    def _on_theory_change(self):
        self.advanced_bc=None
        self.engineering_bc=None
        self._v("bc_mode","SIMPLE").set("SIMPLE")
        self._update_kinematics_labels()
        self._refresh_visibility()

    def _on_geometry_material_change(self):
        """Store the explicit physics and keep the legacy mode in sync."""
        geom=getattr(self,"geometry_display",tk.StringVar(value="Линейная")).get()
        material=getattr(self,"material_model_display",tk.StringVar(value="Закон Гука")).get()
        geometry_code={
            "Линейная":"LINEAR",
            "von Kármán":"VON_KARMAN",
            "Green–Lagrange":"GREEN_LAGRANGE",
            "Green–Lagrange (полная классическая)":"GREEN_LAGRANGE",
            "Green–Lagrange (пока ограниченная)":"GREEN_LAGRANGE",
        }.get(geom)
        if geometry_code is None:return
        if material not in ("Закон Гука","Гук S–E (StVK)"):return
        nonlinear=(geometry_code!="LINEAR")
        self._v("geometry_model","VON_KARMAN").set(geometry_code)
        self._v("material_model","HOOKE").set("HOOKE")
        mode=("NONLINEAR" if nonlinear else "LINEAR")+"_ELASTIC"
        self._v("mode","LINEAR_ELASTIC").set(mode)
        self._refresh_visibility()
        # Refresh the compact model caption immediately.  Previously the
        # selector already held the new mode while the line below it still
        # displayed the previous kinematics until another grid/UI event.
        self._update_grid_info()

    def _update_kinematics_labels(self):
        theory=self._v("theory","CUF").get()
        gl_caption="Green–Lagrange (полная классическая)"
        geom_values=("Линейная","von Kármán",gl_caption)
        try:self.geometry_combo.configure(values=geom_values)
        except Exception:return

        mode=self._v("mode","LINEAR_ELASTIC").get()
        geometry_code=self._v("geometry_model","").get().upper().strip()
        if not geometry_code:
            geometry_code=("LINEAR" if mode.startswith("LINEAR") else
                           ("GREEN_LAGRANGE" if theory=="CUF" else "VON_KARMAN"))
            self._v("geometry_model").set(geometry_code)
        geom_value={"LINEAR":"Линейная","VON_KARMAN":"von Kármán",
                    "GREEN_LAGRANGE":gl_caption}[geometry_code]
        if self.geometry_display.get()!=geom_value:self.geometry_display.set(geom_value)
        material_code=self._v("material_model","").get().upper().strip()
        if not material_code:
            material_code=("J2_INCREMENTAL" if mode.endswith("J2") else
                           ("BIRGER" if mode.endswith("BIRGER") else "HOOKE"))
            self._v("material_model").set(material_code)
        material_values=("Гук S–E (StVK)",) if geometry_code=="GREEN_LAGRANGE" else ("Закон Гука",)
        self.material_model_combo.configure(values=material_values,width=32)
        mat_value=material_values[0] if material_code=="HOOKE" else "Материал исключён из FLEXO"
        if self.material_model_display.get()!=mat_value:self.material_model_display.set(mat_value)
        if theory in capreg.PLANNED_THEORIES:
            for key,combo in (("bc_left",self.bc_left_combo),("bc_right",self.bc_right_combo)):
                display_bc,_old=self._display_bindings[key]
                note="ГУ ещё не реализованы"
                self._display_bindings[key]=(display_bc,{note:""})
                combo.configure(values=(note,),state="disabled")
                display_bc.set(note)
            return

        # Boundary choices are theory/model dependent.  The same physical
        # support name is expanded differently for KL and CUF TE1..TE4.
        cuf_model=self._v("cuf_model","TE1").get()
        for key,combo in (("bc_left",self.bc_left_combo),("bc_right",self.bc_right_combo)):
            display_bc,_old_bc=self._display_bindings[key]
            bc_mapping=dict(bcreg.gui_caption_map(theory,cuf_model))
            current=self._v(key).get().upper().strip()
            if current and current not in bc_mapping.values():
                if current in bcreg.LEGACY_PRESETS:
                    cap=f"Legacy {current}: {bcreg.formula_text(theory,cuf_model,current)}"
                    bc_mapping[cap]=current
                else:
                    self._v(key).set(""); current=""
            self._display_bindings[key]=(display_bc,bc_mapping)
            combo.configure(values=tuple(bc_mapping),width=52)
            reverse_bc={v:k for k,v in bc_mapping.items()}
            display_bc.set(reverse_bc.get(self._v(key).get(),"Выберите…"))
        self._update_bc_display()

    def _update_grid_info(self):
        # Preserve all existing FDM/MVI diagnostics, then mirror only the active
        # grid into the compact ordinary-mode card.
        super()._update_grid_info()
        if self._v("ui_level","BASIC").get()=="BASIC":
            try:
                theory_code=self._v("theory").get()
                geom=self._v("geometry_model","VON_KARMAN").get()
                material=self._v("material_model","HOOKE").get()
                if theory_code=="CUF":
                    gname={"LINEAR":"Линейная","VON_KARMAN":"von Kármán","GREEN_LAGRANGE":"Green–Lagrange (полная классическая)"}.get(geom,geom)
                    theory="Carrera CUF · "+gname
                else:
                    theory=capreg.THEORY_LABELS.get(theory_code,theory_code)+" · "+{"LINEAR":"Линейная","VON_KARMAN":"von Kármán","GREEN_LAGRANGE":"Green–Lagrange (полная классическая)"}.get(geom,geom)
                model=(f" · {self._v('cuf_model').get()}" if self._v("theory").get()=="CUF" else "")
                method="МВИ" if self._v("method").get()=="MVI" else "МКР / конечные разности"
                matname=("J2 incremental (finite strain)" if geom=="GREEN_LAGRANGE" and material=="J2_INCREMENTAL" else
                         "J2 incremental (small strain)" if material=="J2_INCREMENTAL" else
                         "Биргер S–E" if geom=="GREEN_LAGRANGE" and material=="BIRGER" else
                         "Гук S–E (StVK)" if geom=="GREEN_LAGRANGE" else
                         "Биргер" if material=="BIRGER" else "Закон Гука")
                rank_value=(int(self._v("rank").get()) if method=="МВИ" else None)
                rank=(f"-{rank_value}" if method=="МВИ" else "")
                capability=capreg.physics_capability(
                    theory_code,geom,material,self._v("method").get(),
                    cuf_model=self._v("cuf_model").get(),rank=rank_value,
                )
                status_text={
                    capreg.SUPPORTED:"ПОЛНОСТЬЮ ПОДДЕРЖИВАЕТСЯ",
                    capreg.LIMITED:"ОГРАНИЧЕННАЯ ПОСТАНОВКА",
                    capreg.UNAVAILABLE:"НЕДОСТУПНО",
                }[capability.status]
                status_color={
                    capreg.SUPPORTED:"#176b3a",
                    capreg.LIMITED:"#9a6700",
                    capreg.UNAVAILABLE:"#b42318",
                }[capability.status]
                if self._v("scale_model","MACRO").get()=="MICRO":
                    status_text="MICRO G3D · связанный решатель ещё не реализован"
                    status_color="#b42318"
                self.model_info.configure(
                    text=f"{theory}{model} · {matname} · {method}{rank}. {status_text}: {capability.message_ru}",
                    foreground=status_color,
                )
            except Exception:pass
        if not hasattr(self,"basic_grid_info"):return
        try:
            if self._v("method","FDM").get()=="MVI":
                nx=int(self._v("mvi_nx").get());ns=int(self._v("mvi_ns").get());name="МВИ"
            else:
                nx=int(self._v("nx").get());ns=int(self._v("ns").get());name="МКР"
            L=float(self._v("L").get());R=float(self._v("R").get())
            self.basic_grid_info.configure(text=f"{name}: Δx={L/(nx-1):.6g}; Δs={2*np.pi*R/ns:.6g}; узлов={nx*ns}.  ns должно быть нечётным.")
        except Exception:
            self.basic_grid_info.configure(text="Введите геометрию и нечётные nx/ns; служебный FD-шаблон установлен автоматически.")

def main():
    app=FinalSolverGUI()
    app.mainloop()


if __name__=="__main__":
    main()
