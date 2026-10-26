#!/usr/bin/env python3
"""G3D/Gz elastic shell GUI: KL, FSDT, Reddy and CUF TE1--TE4."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from queue import Queue,Empty
from threading import Thread
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from cylinder_reference_gui import ReferenceCaseEditor
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
        self.geometry("940x610")
        self.vars={};self.case=None;self.case_path=None
        self.events=Queue();self.running=False
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
                             ("МВИ 1--8",lambda:self.start(True))):
            button=ttk.Button(actions,text=name,command=command)
            button.pack(side="left",padx=6);self.actions.append(button)
        ttk.Label(box,text="Журнал",font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(12,3))
        self.log=tk.Text(box,height=17,wrap="word",state="disabled")
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
    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=="result":
                    self._log("Результат:\n"+json.dumps(value,ensure_ascii=False,
                             default=str,indent=2)+"\n")
                elif kind=="error":
                    self._log("Ошибка: "+value+"\n")
                    messagebox.showerror("Расчёт",value,parent=self)
                else:
                    self.running=False
                    for b in self.actions:b.configure(state="normal")
        except Empty:pass
        self.after(100,self.poll)
def main():
    ElasticFlexoGUI().mainloop()
if __name__=="__main__":main()
