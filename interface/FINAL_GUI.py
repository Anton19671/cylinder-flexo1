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
from cylinder_reference_gui import ReferenceCaseEditor
from cylinder_dynamic_sweep_gui import DynamicSweepDialog
from cylinder_dynamic_parameter_sweep import run_parameter_sweep, plan_parameter_sweep
from cylinder_dynamic_sweep_gui import DynamicSweepDialog
from cylinder_dynamic_parameter_sweep import plan_parameter_sweep, run_parameter_sweep
from cylinder_final_stage11a_dispatcher import (
    validate_final_case,run_final_case,run_mvi_rank_series
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
        self.title("Цилиндр: упругость и флексоэлектричество G3D / Gz")
        self.geometry("1000x680")
        self.vars={};self.case=None;self.case_path=None
        self.events=Queue();self.running=False
        self.sweep_status_path=None;self.sweep_last_status=None
        self.latest_result_dir=None
        self.sweep_status_path=None
        self.sweep_last_snapshot=None
        box=ttk.Frame(self,padding=12);box.pack(fill="both",expand=True)
        ttk.Label(box,text="KL / Тимошенко / Редди / Carrera TE1--TE4  |  МВИ / МКР  |  Ньюмарк",
                  font=("Segoe UI",12,"bold")).pack(anchor="w",pady=8)
        opts=ttk.Frame(box);opts.pack(fill="x",pady=6)
        ttk.Label(opts,text="Градиентная модель").pack(side="left")
        self.gradient=tk.StringVar(value="G3D")
        ttk.Combobox(opts,textvariable=self.gradient,values=("G3D","Gz"),
                     state="readonly",width=8).pack(side="left",padx=8)
        ttk.Label(opts,text="Задача").pack(side="left",padx=(12,0))
        self.kind=tk.StringVar(value="STATIC")
        ttk.Combobox(opts,textvariable=self.kind,values=("STATIC","DYNAMIC"),
                     state="readonly",width=10).pack(side="left",padx=8)
        ttk.Button(opts,text="Новый CASE",command=self.new_case).pack(side="left",padx=5)
        ttk.Button(opts,text="Открыть CASE",command=self.open_case).pack(side="left",padx=5)
        self.path_label=ttk.Label(box,text="CASE не выбран",wraplength=870)
        self.path_label.pack(anchor="w",pady=8)
        actions=ttk.Frame(box);actions.pack(fill="x",pady=5)
        self.actions=[]
        for name,command in (("Редактировать",self.edit_case),
                             ("Проверить",self.validate_case),
                             ("Рассчитать",lambda:self.start(False)),
                             ("МВИ 1--8",lambda:self.start(True)),
                             ("Бифуркации",self.open_parameter_sweep),
                             ("Бифуркации / динамика",self.open_dynamic_sweep)):
            button=ttk.Button(actions,text=name,command=command)
            button.pack(side="left",padx=6);self.actions.append(button)
        status=ttk.Frame(box)
        status.pack(fill="x",pady=(8,2))
        self.sweep_status_text=tk.StringVar(value="Серия динамики не запущена")
        ttk.Label(status,textvariable=self.sweep_status_text,wraplength=670).pack(side="left",padx=(0,10))
        self.sweep_progress=ttk.Progressbar(status,orient="horizontal",
                                           mode="determinate",length=160,maximum=1)
        self.sweep_progress.pack(side="left",padx=8)
        self.results_button=ttk.Button(status,text="Открыть результаты",
                                       command=self.open_results,state="disabled")
        self.results_button.pack(side="right")
        ttk.Label(box,text="Журнал",font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(12,3))
        self.log=tk.Text(box,height=13,wrap="word",state="disabled")
        self.log.pack(fill="both",expand=True)
        self.after(100,self.poll)
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
        self._v("calculation_kind","STATIC").set(case.get("calculation_kind","STATIC"))
        self.path_label.configure(text=str(self.case_path or "Новый CASE не сохранён"))
        self.gradient.set(str(case.get("micro",{}).get("gradient_model","G3D")))
        self.kind.set(str(case.get("calculation_kind","STATIC")).upper())
    def new_case(self):
        try:
            filename=TEMPLATES[(self.gradient.get().upper(),self.kind.get().upper())]
            case=json.loads((ROOT/"config"/filename).read_text(encoding="utf-8"))
            self.adopt(case);self.edit_case()
        except (ValueError,OSError) as exc:
            messagebox.showerror("Новый CASE",str(exc),parent=self)
    def open_case(self):
        filename=filedialog.askopenfilename(parent=self,filetypes=[("JSON","*.json")])
        if not filename:return
        try:
            case=json.loads(Path(filename).read_text(encoding="utf-8"))
            if not isinstance(case,dict) or case.get("solver_backend")!="VARIATIONAL_REFERENCE":
                raise ValueError("Откройте только упругий вариационный CASE")
            self.adopt(case,filename)
        except (ValueError,OSError) as exc:
            messagebox.showerror("CASE",str(exc),parent=self)
    def edit_case(self):
        if self.case is None:self.new_case();return
        ReferenceCaseEditor(self,case=self.case,path=self.case_path)
    def saved_path(self):
        value=self._v("case_path").get()
        if not value:raise ValueError("Сначала сохраните CASE в редакторе")
        return Path(value).resolve()
    def validate_case(self):
        try:
            info=validate_final_case(self.saved_path())
            self._log(json.dumps(info,ensure_ascii=False)+"\n")
            messagebox.showinfo("CASE","Параметры приняты",parent=self)
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
        self._log(f"Старт: {path}\n")
        def worker():
            try:
                result=(run_mvi_rank_series(path) if series else run_final_case(path))
                self.events.put(("result",result))
            except Exception as exc:
                self.events.put(("error",f"{type(exc).__name__}: {exc}"))
            finally:self.events.put(("done",None))
        Thread(target=worker,daemon=True).start()
    def open_parameter_sweep(self):
        if self.running:
            return
        try:
            path=self.saved_path()
            info=validate_final_case(path)
            if info["calculation_kind"]!="DYNAMIC":
                raise ValueError("Для бифуркационной серии нужен сохранённый DYNAMIC CASE")
            DynamicSweepDialog(self,path)
        except Exception as exc:
            messagebox.showerror("Бифуркационная серия",str(exc),parent=self)

    def start_parameter_sweep(self,spec_file,*,preflight_only=False):
        """Run validated parameter studies without blocking Tk's event loop."""
        if self.running:
            return
        plan=plan_parameter_sweep(spec_file)
        self.sweep_status_path=(None if preflight_only else plan["output_dir"]/"SWEEP_STATUS.json")
        self.sweep_last_status=None
        self.running=True
        for button in self.actions:
            button.configure(state="disabled")
        self._log(f"Проверка серии: {spec_file}\n" if preflight_only
                  else f"Запуск бифуркационной серии: {spec_file}\n")
        def worker():
            try:
                result=run_parameter_sweep(spec_file,preflight_only=preflight_only)
                self.events.put(("result",result))
            except Exception as exc:
                self.events.put(("error",f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("done",None))
        Thread(target=worker,daemon=True).start()

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
