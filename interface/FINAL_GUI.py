#!/usr/bin/env python3
"""G3D/Gz elastic shell GUI: KL, FSDT, Reddy and CUF TE1--TE4."""
from __future__ import annotations
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from queue import Queue,Empty
from threading import Thread
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from cylinder_reference_gui import ReferenceCaseEditor, edit_reference_case
from cylinder_dynamic_sweep_gui import DynamicSweepDialog
from cylinder_dynamic_parameter_sweep import run_parameter_sweep, plan_parameter_sweep
from cylinder_final_stage11a_dispatcher import (
    validate_final_case,run_final_case,run_mvi_rank_series
)
from cylinder_time_integrator_gui import (
    TIME_INTEGRATOR_CHOICES,plan_time_integrator_run,execute_time_integrator_plan
)
ROOT=Path(__file__).resolve().parent.parent
TEMPLATES={
 ("G3D","STATIC"):"CASE_REFERENCE_CUF_G3D_STATIC.json",
 ("GZ","STATIC"):"CASE_REFERENCE_CUF_GZ_STATIC.json",
 ("G3D","DYNAMIC"):"CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json",
 ("GZ","DYNAMIC"):"CASE_REFERENCE_CUF_GZ_MVI_DYNAMIC.json",
}
class ElasticFlexoGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Цилиндр: G3D / Gz — полный интерфейс")
        sw,sh=self.winfo_screenwidth(),self.winfo_screenheight()
        self.geometry(f"{min(1400,max(950,sw-70))}x{min(920,max(650,sh-100))}")
        self.vars={};self.case=None;self.case_path=None
        self.events=Queue();self.running=False
        self.sweep_status_path=None;self.sweep_last_status=None
        self.sweep_last_snapshot=None;self.latest_result_dir=None
        self.gradient=tk.StringVar(value="G3D")
        self.kind=tk.StringVar(value="STATIC")
        self.time_integrator=tk.StringVar(value=TIME_INTEGRATOR_CHOICES[0])
        self.form_vars={};self.json_widgets={};self._loading=False
        main=ttk.Frame(self,padding=10);main.pack(fill="both",expand=True)
        header=ttk.Frame(main);header.pack(fill="x",pady=(0,8))
        ttk.Label(header,text="ЦИЛИНДРИЧЕСКАЯ ОБОЛОЧКА · G3D / Gz",
                  font=("Segoe UI",15,"bold")).pack(anchor="w")
        ttk.Label(header,text="KL / Тимошенко / Редди / CUF TE1–TE4 · МВИ / МКР · статика / динамика · флексоэлектричество").pack(anchor="w")
        selectors=ttk.Frame(main);selectors.pack(fill="x",pady=5)
        for label,var,values,width in (("Градиенты",self.gradient,("G3D","Gz"),8),
                                        ("Задача",self.kind,("STATIC","DYNAMIC"),11),
                                        ("Интегратор",self.time_integrator,TIME_INTEGRATOR_CHOICES,36)):
            ttk.Label(selectors,text=label).pack(side="left",padx=(7,3))
            ttk.Combobox(selectors,textvariable=var,values=values,state="readonly",width=width).pack(side="left",padx=(0,13))
        toolbar=ttk.Frame(main);toolbar.pack(fill="x",pady=5)
        self.actions=[]
        for label,fn in (("Новый CASE",self.new_case),("Открыть CASE",self.open_case),
                         ("Сохранить CASE",self.save_case),("Расширенный редактор",self.edit_case),
                         ("Проверить",self.validate_case),("Рассчитать",lambda:self.start(False)),
                         ("МВИ 1–8",lambda:self.start(True)),
                         ("Бифуркации / динамика",self.open_dynamic_sweep)):
            b=ttk.Button(toolbar,text=label,command=fn)
            b.pack(side="left",padx=3)
            self.actions.append(b)
        self.path_label=ttk.Label(main,text="CASE не выбран")
        self.path_label.pack(anchor="w",pady=(2,8))
        self.notebook=ttk.Notebook(main);self.notebook.pack(fill="both",expand=True)
        self.tab_problem=self._tab("1 · Постановка")
        self.tab_flexo=self._tab("2 · Материал / флексо")
        self.tab_load=self._tab("3 · Нагрузка")
        self.tab_bc=self._tab("4 · Граничные условия")
        self.tab_dynamic=self._tab("5 · Динамика")
        self.tab_numeric=self._tab("6 · Численные настройки")
        self.tab_results=self._tab("7 · Результаты")
        self.tab_online=self._tab("8 · Онлайн / журнал")
        self._section(self.tab_problem,"Теория и пространственная схема",[
          ("Теория","theory",("KL","FSDT","REDDY","CUF")),
          ("CUF","cuf_model",("TE1","TE2","TE3","TE4")),
          ("Метод","method",("MVI","FDM")),("Ранг МВИ","rank",tuple(str(i) for i in range(1,9))),
          ("Узлы nx","nx",None),("Узлы ns","ns",None),
          ("Операторы","operator_scheme",("LEGACY_REFERENCE","SBP21_PERIODIC_ODD","SBP21_PERIODIC_ODD_CSR")),
          ("МКР","fdm_backend",("DENSE_REFERENCE","MATRIX_FREE_GMRES"))])
        self._section(self.tab_problem,"Геометрия оболочки",[("Длина L, м","L",None),
          ("Радиус R, м","R",None),("Толщина h, м","h",None)])
        self._section(self.tab_flexo,"Материал и модель градиентов",[
          ("Масштаб","scale_model",("MACRO","MICRO")),
          ("E, Па","E0",None),("ν","nu0",None),
          ("Внутренняя длина ℓ, м","ell_m",None),
          ("μ1, Кл/м","mu1",None),("μ2, Кл/м","mu2",None),
          ("κ, Ф/м","kappa",None),("Порядок потенциала","potential_order",None),
          ("Плотность, кг/м³","rho_kg_m3",None),("Вязкость η, с","eta_s",None)])
        self.flexo=tk.BooleanVar(value=True)
        ttk.Checkbutton(self.tab_flexo,text="Флексоэлектричество включено",variable=self.flexo).pack(anchor="w",pady=6)
        self._section(self.tab_load,"Радиальное нагружение",[
          ("Тип","load_type",("UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE","PATCH_RADIAL_FACE")),
          ("Поверхность","load_face",("MIDDLE","INNER","OUTER")),("Давление q","load_q",None)])
        self._json(self.tab_load,"load","Полный JSON нагрузки и шагов")
        self._section(self.tab_bc,"Опоры цилиндра",[
          ("Левый торец","left_support",("CLAMPED","SIMPLY_SUPPORTED","FREE","SYMMETRY")),
          ("Правый торец","right_support",("CLAMPED","SIMPLY_SUPPORTED","FREE","SYMMETRY"))])
        self._json(self.tab_bc,"bc","Механические, градиентные и электрические ГУ")
        self._section(self.tab_dynamic,"Временная схема",[
          ("Δt, с","dt",None),("Шагов","n_steps",None),
          ("β Ньюмарка","beta",None),("γ Ньюмарка","gamma",None),
          ("Плотность в динамике","dyn_rho",None)])
        self._json(self.tab_dynamic,"dynamics","Динамика, F(t) и начальные условия")
        self._section(self.tab_numeric,"Метод и точность",[
          ("Точек Гаусса","n_gauss",None),("Циклы МВИ","max_sweeps",None),
          ("Папка результатов","output_dir",None)])
        self._json(self.tab_numeric,"newton","Полные параметры Ньютона")
        ttk.Button(self.tab_results,text="Открыть папку результатов",command=self.open_results).pack(anchor="w",pady=5)
        self.result_text=tk.Text(self.tab_results,height=14,wrap="word",state="disabled")
        self.result_text.pack(fill="both",expand=True)
        status=ttk.Frame(self.tab_online);status.pack(fill="x",pady=8)
        self.sweep_status_text=tk.StringVar(value="Расчёт не запущен")
        ttk.Label(status,textvariable=self.sweep_status_text).pack(side="left",padx=5)
        self.sweep_progress=ttk.Progressbar(status,mode="determinate",length=200,maximum=1)
        self.sweep_progress.pack(side="left",padx=10)
        self.results_button=ttk.Button(status,text="Открыть результаты",command=self.open_results,state="disabled")
        self.results_button.pack(side="right")
        self.log=tk.Text(self.tab_online,height=18,wrap="word",state="disabled",font=("Consolas",10))
        self.log.pack(fill="both",expand=True)
        self.after(100,self.poll)

    def _tab(self,title):
        page=ttk.Frame(self.notebook,padding=12)
        self.notebook.add(page,text=title)
        return page

    def _section(self,parent,title,rows):
        group=ttk.LabelFrame(parent,text=title,padding=8)
        group.pack(fill="x",pady=7)
        for idx,(label,key,choices) in enumerate(rows):
            r,c=divmod(idx,2)
            ttk.Label(group,text=label).grid(row=r,column=c*2,sticky="w",padx=(0,8),pady=5)
            var=tk.StringVar()
            self.form_vars[key]=var
            if choices:
                widget=ttk.Combobox(group,textvariable=var,values=choices,state="readonly",width=27)
            else:
                widget=ttk.Entry(group,textvariable=var,width=30)
            widget.grid(row=r,column=c*2+1,sticky="ew",padx=(0,22),pady=5)
        group.columnconfigure(1,weight=1);group.columnconfigure(3,weight=1)

    def _json(self,parent,key,title):
        frame=ttk.LabelFrame(parent,text=title,padding=8)
        frame.pack(fill="both",expand=True,pady=8)
        t=tk.Text(frame,height=14,wrap="none",font=("Consolas",10),undo=True)
        scroll=ttk.Scrollbar(frame,orient="vertical",command=t.yview)
        t.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right",fill="y");t.pack(side="left",fill="both",expand=True)
        self.json_widgets[key]=t

    def _put_json(self,key,value):
        t=self.json_widgets[key]
        t.delete("1.0","end")
        t.insert("1.0",json.dumps(value,ensure_ascii=False,indent=2))

    def _get_json(self,key):
        try:out=json.loads(self.json_widgets[key].get("1.0","end-1c"))
        except json.JSONDecodeError as exc:raise ValueError(f"Ошибка JSON {key}: {exc}") from exc
        if not isinstance(out,dict):raise ValueError(f"{key} должен быть JSON-объектом")
        return out

    def _number(self,key):
        try:return float(self.form_vars[key].get().strip().replace(",","."))
        except ValueError as exc:raise ValueError(f"{key}: требуется число") from exc

    def _integer(self,key):
        value=self.form_vars[key].get().strip()
        if not value.isdecimal():raise ValueError(f"{key}: требуется целое число")
        return int(value)
    def _v(self,key,default=""):
        if key not in self.vars:self.vars[key]=tk.StringVar(value=str(default))
        return self.vars[key]
    def _log(self,message):
        self.log.configure(state="normal")
        self.log.insert("end",str(message));self.log.see("end")
        self.log.configure(state="disabled")
    def adopt(self,case,path=None):
        self.case=copy.deepcopy(case)
        self.case_path=Path(path).resolve() if path else None
        self._v("case_path").set(str(self.case_path) if self.case_path else "")
        self.path_label.configure(text=str(self.case_path or "Новый CASE (не сохранён)"))
        micro=case.get("micro",{});geo=case.get("geometry",{});mat=case.get("material",{})
        kind=str(case.get("calculation_kind","STATIC")).upper()
        self.kind.set(kind)
        self.gradient.set("Gz" if str(micro.get("gradient_model","G3D")).upper()=="GZ" else "G3D")
        grid=case.get("mvi_grid" if case.get("method")=="MVI" else "fdm_grid",case.get("grid",{}))
        load=case.get("reference_loads",case.get("load_definition",{}))
        bc=case.get("boundary_conditions",{}).get("variational",{})
        dyn=case.get("dynamics",case.get("editor_saved_dynamics",{}))
        values={"theory":case.get("theory","CUF"),"cuf_model":case.get("cuf_model","TE1"),
          "method":case.get("method","MVI"),"rank":case.get("rank",1),
          "nx":grid.get("nx",4),"ns":grid.get("ns",4),
          "operator_scheme":grid.get("operator_scheme","LEGACY_REFERENCE"),
          "fdm_backend":case.get("fdm_backend","DENSE_REFERENCE"),
          "L":geo.get("L",""),"R":geo.get("R",""),"h":geo.get("h",""),
          "E0":mat.get("E0",""),"nu0":mat.get("nu0",""),
          "scale_model":case.get("scale_model","MICRO"),
          "ell_m":micro.get("ell_m",0),"mu1":micro.get("mu1_c_per_m",0),
          "mu2":micro.get("mu2_c_per_m",0),"kappa":micro.get("kappa_f_per_m",0),
          "potential_order":micro.get("potential_order",1),
          "rho_kg_m3":micro.get("rho_kg_m3",1300),"eta_s":micro.get("eta_s",0),
          "load_type":load.get("type","UNIFORM_RADIAL_FACE"),
          "load_face":load.get("face","MIDDLE"),"load_q":load.get("q",0),
          "left_support":bc.get("mechanical",{}).get("left",{}).get("support","CLAMPED"),
          "right_support":bc.get("mechanical",{}).get("right",{}).get("support","CLAMPED"),
          "dt":dyn.get("time_step_s",0.01),"n_steps":dyn.get("n_steps",20),
          "beta":dyn.get("newmark_beta",0.25),"gamma":dyn.get("newmark_gamma",0.5),
          "dyn_rho":dyn.get("density_kg_m3",1300),
          "n_gauss":case.get("quadrature",{}).get("n_gauss",3),
          "max_sweeps":case.get("reference_mvi",{}).get("max_sweeps",30),
          "output_dir":case.get("output_dir","")}
        for key,value in values.items():self.form_vars[key].set(str(value))
        self.flexo.set(bool(micro.get("enable_flexoelectric",False)))
        self._put_json("load",load)
        self._put_json("bc",bc)
        self._put_json("dynamics",dyn or {"time_step_s":0.01,"n_steps":20,
             "density_kg_m3":1300,"newmark_beta":0.25,"newmark_gamma":0.5,
             "time_law":{"kind":"HARMONIC","amplitude":1,"frequency_hz":1}})
        self._put_json("newton",case.get("reference_newton",{}))
        self._field_snapshot={key:var.get() for key,var in self.form_vars.items()}
        self._saved_form_state=self._form_state()

    def _form_state(self):
        return (
            tuple(sorted((key,var.get()) for key,var in self.form_vars.items())),
            self.kind.get(),self.gradient.get(),self.flexo.get(),
            tuple(sorted((key,widget.get("1.0","end-1c"))
                         for key,widget in self.json_widgets.items())),
        )

    def _field_changed(self,key):
        return self.form_vars[key].get()!=getattr(self,"_field_snapshot",{}).get(key)

    def new_case(self):
        try:
            name=TEMPLATES[(self.gradient.get().upper(),self.kind.get().upper())]
            self.adopt(json.loads((ROOT/"config"/name).read_text(encoding="utf-8")))
            self.notebook.select(self.tab_problem)
        except (KeyError,ValueError,OSError) as exc:
            messagebox.showerror("Новый CASE",str(exc),parent=self)

    def open_case(self):
        filename=filedialog.askopenfilename(parent=self,filetypes=[("JSON","*.json")])
        if not filename:return
        try:
            case=json.loads(Path(filename).read_text(encoding="utf-8-sig"))
            if not isinstance(case,dict) or case.get("solver_backend")!="VARIATIONAL_REFERENCE":
                raise ValueError("Откройте упругий вариационный CASE")
            self.adopt(case,filename)
        except (ValueError,OSError) as exc:
            messagebox.showerror("CASE",str(exc),parent=self)

    def collect(self,base_dir=None):
        if self.case is None:raise ValueError("Сначала создайте CASE")
        c=copy.deepcopy(self.case)
        values={key:var.get() for key,var in self.form_vars.items()}
        fields=dict(values)
        fields.update({"calculation_kind":self.kind.get(),
                       "gradient_model":self.gradient.get(),"flexo":self.flexo.get()})
        load=self._get_json("load");bc=self._get_json("bc")
        dyn=self._get_json("dynamics")
        for key,target in (("load_type","type"),("load_face","face")):
            if self._field_changed(key):load[target]=values[key]
        if self._field_changed("load_q"):load["q"]=self._number("load_q")
        for key,side in (("left_support","left"),("right_support","right")):
            if self._field_changed(key):
                bc.setdefault("mechanical",{}).setdefault(side,{})["support"]=values[key]
        if self.kind.get()=="DYNAMIC":
            for key,target in (
                ("dt","time_step_s"),("n_steps","n_steps"),
                ("beta","newmark_beta"),("gamma","newmark_gamma"),
                ("dyn_rho","density_kg_m3"),
            ):
                if self._field_changed(key):
                    dyn[target]=(self._integer(key) if key=="n_steps"
                                 else self._number(key))
        return edit_reference_case(c,fields,json.dumps(bc),json.dumps(load),
              self.json_widgets["newton"].get("1.0","end-1c"),json.dumps(dyn),base_dir=base_dir)

    def save_case(self):
        path=filedialog.asksaveasfilename(parent=self,defaultextension=".json",
             initialfile=self.case_path.name if self.case_path else "CASE_REFERENCE.json",
             filetypes=[("JSON","*.json")])
        if not path:return None
        target=Path(path).resolve()
        try:
            case=self.collect(base_dir=target.parent)
            target.write_text(json.dumps(case,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            self.adopt(case,target)
            self._log(f"CASE сохранён: {target}\n")
            return target
        except Exception as exc:
            messagebox.showerror("Сохранение CASE",str(exc),parent=self)
            return None

    def edit_case(self):
        if self.case is None:
            self.new_case();return
        try:
            case=self.collect(base_dir=self.case_path.parent if self.case_path else ROOT/"config")
            ReferenceCaseEditor(self,case=case,path=self.case_path)
        except Exception as exc:messagebox.showerror("Редактор CASE",str(exc),parent=self)

    def saved_path(self):
        value=self._v("case_path").get()
        if not value:raise ValueError("Сначала сохраните CASE")
        if self._form_state()!=getattr(self,"_saved_form_state",None):
            raise ValueError("Есть несохранённые изменения. Сначала сохраните CASE")
        return Path(value).resolve()

    def validate_case(self):
        try:
            self.collect(base_dir=self.case_path.parent if self.case_path else ROOT/"config")
            self._log("Параметры формы проверены без расчёта\n")
            messagebox.showinfo("CASE","Параметры формы приняты. Расчёт не запускался.",parent=self)
        except Exception as exc:messagebox.showerror("CASE",str(exc),parent=self)
    def _launch_reference_case(self,path,case=None,series=False):
        if case is not None:self.adopt(case,path)
        self.start(series)
    def open_dynamic_sweep(self):
        """Open the study controls for a saved, independently validated CASE."""
        if self.running:
            return
        try:
            path=self.saved_path()
            info=validate_final_case(path)
            if info["calculation_kind"]!="DYNAMIC":
                raise ValueError("Сначала выберите и сохраните динамический CASE")
            DynamicSweepDialog(self,path)
        except Exception as exc:
            messagebox.showerror("Динамическая серия",str(exc),parent=self)

    def start_parameter_sweep(self,spec_file,*,preflight_only=False):
        """Run the existing full-shell sweep engine outside the Tk event loop."""
        if self.running:
            raise ValueError("Другой расчёт уже выполняется")
        plan=plan_parameter_sweep(spec_file)
        count=len(plan["jobs"])
        self.running=True
        self.sweep_last_snapshot=None
        self.sweep_status_path=(None if preflight_only
                                else plan["output_dir"]/"SWEEP_STATUS.json")
        self.sweep_progress.configure(maximum=count,value=0)
        self.sweep_status_text.set(
            ("Проверка" if preflight_only else "Расчёт")+" серии: 0 / "+str(count))
        for button in self.actions:
            button.configure(state="disabled")
        self._log(f"Серия: {spec_file} | точек: {count}\n")
        def worker():
            try:
                report=run_parameter_sweep(spec_file,preflight_only=preflight_only)
                self.events.put(("sweep_result",(report,preflight_only)))
            except Exception as exc:
                self.events.put(("error",f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("done",None))
        Thread(target=worker,daemon=True).start()

    def _poll_sweep_status(self):
        """Read only atomically persisted progress; never access Tk from a worker."""
        path=self.sweep_status_path
        if path is None or not path.is_file():
            return
        try:
            status=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,ValueError):
            return
        completed=int(status.get("completed_runs",0))
        running=next((r.get("index") for r in status.get("runs",[])
                      if r.get("status")=="RUNNING"),None)
        state=(completed,status.get("status"),running)
        if state==self.sweep_last_snapshot:
            return
        self.sweep_last_snapshot=state
        self.sweep_progress.configure(value=completed)
        suffix="" if running is None else f" | выполняется {running}"
        self.sweep_status_text.set(
            f"Серия: {completed}/{len(status.get('runs',[]))} | "
            f"{status.get('status','UNKNOWN')}{suffix}")
        self._log(self.sweep_status_text.get()+"\n")
        self.latest_result_dir=path.parent
        self.results_button.configure(state="normal")

    def open_results(self):
        """Open an existing results folder on Windows/macOS/Linux."""
        folder=self.latest_result_dir
        if folder is None or not Path(folder).is_dir():
            messagebox.showerror("Результаты","Папка результатов пока не создана",parent=self)
            return
        try:
            folder=Path(folder).resolve()
            if sys.platform=="win32":
                os.startfile(str(folder))
            elif sys.platform=="darwin":
                subprocess.Popen(["open",str(folder)])
            else:
                subprocess.Popen(["xdg-open",str(folder)])
        except (OSError,ValueError) as exc:
            messagebox.showerror("Результаты",str(exc),parent=self)

    def start(self,series):
        if self.running:return
        try:
            path=self.saved_path()
            info=validate_final_case(path)
            if series and info["method"]!="MVI":
                raise ValueError("Для серии рангов выберите MVI")
        except Exception as exc:
            messagebox.showerror("Расчёт",str(exc),parent=self);return
        self.running=True
        self.sweep_status_path=None
        for b in self.actions:b.configure(state="disabled")
        try:
            plan=plan_time_integrator_run(path,self.time_integrator.get(),series=series)
        except Exception as exc:
            self.running=False
            for b in self.actions:b.configure(state="normal")
            messagebox.showerror("Интегратор",str(exc),parent=self)
            return
        self._log(f"Старт: {path} | {plan.integrator} | {plan.output_dir}\n")
        def worker():
            try:
                result=execute_time_integrator_plan(plan)
                self.events.put(("result",result))
            except Exception as exc:
                self.events.put(("error",f"{type(exc).__name__}: {exc}"))
            finally:self.events.put(("done",None))
        Thread(target=worker,daemon=True).start()
    def open_parameter_sweep(self):
        self.open_dynamic_sweep()

    def poll(self):
        if self.running and self.sweep_status_path is not None:
            try:
                if self.sweep_status_path.is_file():
                    status=json.loads(self.sweep_status_path.read_text(encoding="utf-8"))
                    progress=(status.get("status"),status.get("completed_runs"),
                              len(status.get("runs",())))
                    if progress!=self.sweep_last_status:
                        self.sweep_last_status=progress
                        self._log("Серия: {} | готово: {} / {}\n".format(*progress))
            except (OSError,ValueError):
                # Status is updated atomically by the worker; keep GUI alive
                # if a status file is temporarily unavailable.
                pass
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=="result":
                    self._log("Результат:\n"+json.dumps(value,ensure_ascii=False,
                             default=str,indent=2)+"\n")
                    self.result_text.configure(state="normal")
                    self.result_text.delete("1.0","end")
                    self.result_text.insert("1.0",json.dumps(value,ensure_ascii=False,default=str,indent=2))
                    self.result_text.configure(state="disabled")
                    path=value.get("output_dir") if isinstance(value,dict) else None
                    if path and Path(path).is_dir():
                        self.latest_result_dir=Path(path)
                        self.results_button.configure(state="normal")
                elif kind=="sweep_result":
                    report,preflight_only=value
                    done=report.get("preflighted_cases",0) if preflight_only else report.get("completed_runs",0)
                    self.sweep_progress.configure(value=done)
                    self.sweep_status_text.set(
                        f"{'Проверено' if preflight_only else 'Завершено'}: "
                        f"{done}/{len(report.get('values',[]))} | {report.get('status')}")
                    self._log(self.sweep_status_text.get()+"\n")
                    if not preflight_only:
                        self.latest_result_dir=Path(report["output_dir"])
                        self.results_button.configure(state="normal")
                        self._log("Бифуркационные Excel и графики: "+
                                  str(self.latest_result_dir/"BIFURCATION_ANALYTICS")+"\n")
                elif kind=="error":
                    self._log("Ошибка: "+value+"\n")
                    messagebox.showerror("Расчёт",value,parent=self)
                elif kind=="done":
                    self._poll_sweep_status()
                    self.running=False
                    for b in self.actions:b.configure(state="normal")
                    self.sweep_status_path=None
                    self.sweep_status_path=None
        except Empty:pass
        self._poll_sweep_status()
        self.after(100,self.poll)
def main():
    ElasticFlexoGUI().mainloop()
if __name__=="__main__":main()