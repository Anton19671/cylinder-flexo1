"""Separate MVI STATIC GUI: own case, own process, no dynamic fallback.

An explicit user action is the sole way to launch a stationary solve.
Validation and preview never construct the numerical solver or start a
subprocess. The other spatial method is neither imported nor dispatched.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4
import numpy as np

from independent_mvi.linear_models import VariationalModelCase
from independent_mvi import run_static as own_runner

ROOT=Path(__file__).resolve().parents[1]
TEMPLATES=ROOT/"config"/"verification"/"independent_static"
THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
OWN_SCHEMA="INDEPENDENT_MVI_STATIC_SEVEN_MODELS_V1"
OWN_SUMMARY="MVI_STATIC_SUMMARY.json"
OWN_MODULE="independent_mvi.run_static"


def _template_path(theory,rank=1):
    if theory not in THEORIES:
        raise ValueError("MVI_STATIC_GUI_UNKNOWN_THEORY")
    if type(rank) is not int or not 1<=rank<=8:
        raise ValueError("МВИ: ранг должен быть 1–8")
    return TEMPLATES/f"MVI_R{rank}_{theory}.json"


def inspect_static_document(doc):
    """Check the method's own static schema without writing or solving."""
    if not isinstance(doc,dict) or set(doc)!={"schema","physics","static","rank","output_dir"}:
        raise ValueError("MVI_STATIC_GUI_CASE_KEYS_INVALID")
    if doc["schema"]!=OWN_SCHEMA:
        raise ValueError("MVI_STATIC_GUI_FOREIGN_SOLVER_SCHEMA")
    if not isinstance(doc["physics"],dict):
        raise ValueError("MVI_STATIC_GUI_PHYSICS_OBJECT_REQUIRED")
    try:
        model=VariationalModelCase(**doc["physics"])
    except TypeError as exc:
        raise ValueError("MVI_STATIC_GUI_BAD_PHYSICS_KEYS") from exc
    settings=doc["static"]
    if not isinstance(settings,dict) or set(settings)!={"load_factor","max_sweeps"}:
        raise ValueError("MVI_STATIC_GUI_LOAD_KEYS_INVALID")
    factor=settings["load_factor"]
    if type(factor) not in (float,int) or not np.isfinite(factor) or factor<0:
        raise ValueError("MVI_STATIC_GUI_LOAD_FACTOR_INVALID")
    if type(settings["max_sweeps"]) is not int or not 1<=settings["max_sweeps"]<=200:
        raise ValueError("МВИ: X/Y циклы 1..200")
    rank=doc["rank"]
    if type(rank) is not int or not 1<=rank<=8:
        raise ValueError("МВИ: ранг должен быть 1..8")
    if rank>min(model.ns,model.nx-(4 if model.theory=="KL" else 2)):
        raise ValueError("МВИ: ранг превышает доступное закреплённое пространство")
    output=doc["output_dir"]
    if not isinstance(output,str) or not output.strip() or output in (".","/"):
        raise ValueError("MVI_STATIC_GUI_OUTPUT_REQUIRED")
    return model


class IndependentStaticWindow(tk.Toplevel):
    """Independent MVI stationary solver control, not dynamic Newmark."""

    def __init__(self,parent):
        super().__init__(parent)
        self.title("Независимый MVI: линейная статика (7 теорий)")
        self.geometry("940x760")
        self.minsize(680,560)
        self.theory=tk.StringVar(self,value="CUF_TE4")
        self.rank=tk.StringVar(self,value="1")
        self.root_dir=tk.StringVar(
            self,value=str(Path.home()/"CYL_RESULTS"/"INDEPENDENT_MVI"))
        self.progress=tk.StringVar(self,value="Статика: CASE не сохранён")
        self.saved_path=None
        self.saved_output=None
        self.fingerprint=None
        self.proc=None
        self.lines=queue.Queue()
        self._stop_requested=False

        ttk.Label(self,text=(
            "Независимая линейная статика MVI, LINEAR/HOOKE/MACRO. "
            "Градиентная G3D/Gz, нелинейность, флексоэлектричество "
            "пока не включены; численная достоверность не аттестована."),
            padding=10,wraplength=900).pack(fill="x")
        controls=ttk.Frame(self,padding=7)
        controls.pack(fill="x")
        ttk.Label(controls,text="Теория").pack(side="left")
        ttk.Combobox(controls,textvariable=self.theory,values=THEORIES,
                     state="readonly",width=13).pack(side="left",padx=5)
        ttk.Label(controls,text="Ранг МВИ").pack(side="left")
        ttk.Combobox(controls,textvariable=self.rank,
                     values=[str(i) for i in range(1,9)],
                     state="readonly",width=4).pack(side="left",padx=5)
        ttk.Label(controls,text="Папка").pack(side="left")
        ttk.Entry(controls,textvariable=self.root_dir).pack(
            side="left",fill="x",expand=True,padx=5)
        ttk.Button(controls,text="…",width=3,command=self.choose_root).pack(
            side="left")
        ttk.Label(self,text="Собственный STATIC CASE (JSON, SI)",
                  padding=(10,5)).pack(fill="x")
        frame=ttk.Frame(self,padding=8)
        frame.pack(fill="both",expand=True)
        self.editor=tk.Text(frame,wrap="none",height=18,undo=True)
        scroll=ttk.Scrollbar(frame,command=self.editor.yview)
        self.editor.configure(yscrollcommand=scroll.set)
        self.editor.pack(side="left",fill="both",expand=True)
        scroll.pack(side="right",fill="y")
        actions=ttk.Frame(self,padding=8)
        actions.pack(fill="x")
        ttk.Button(actions,text="Шаблон",command=self.template).pack(
            side="left",padx=3)
        ttk.Button(actions,text="Загрузить CASE",command=self.load).pack(
            side="left",padx=3)
        ttk.Button(actions,text="Проверить",command=self.validate).pack(
            side="left",padx=3)
        ttk.Button(actions,text="Сохранить CASE",command=self.save).pack(
            side="left",padx=3)
        self.run_button=ttk.Button(actions,text="РАССЧИТАТЬ",
                                   command=self.run)
        self.run_button.pack(side="left",padx=7)
        self.stop_button=ttk.Button(actions,text="СТОП",
                                    command=self.stop,state="disabled")
        self.stop_button.pack(side="left",padx=3)
        ttk.Button(actions,text="Результаты",command=self.view_results).pack(
            side="left",padx=3)
        ttk.Button(actions,text="Открыть папку",command=self.open_output).pack(
            side="left",padx=3)
        ttk.Label(self,textvariable=self.progress,padding=6).pack(fill="x")
        self.journal=tk.Text(self,height=7,state="disabled",wrap="word")
        self.journal.pack(fill="both",padx=8,pady=5)
        self.template(force=True)
        self.after(175,self._poll)
        self.protocol("WM_DELETE_WINDOW",self._close)

    def _note(self,message):
        self.journal.configure(state="normal")
        self.journal.insert("end",str(message)+"\n")
        self.journal.see("end")
        self.journal.configure(state="disabled")

    def choose_root(self):
        initial=Path(self.root_dir.get()).expanduser()
        selected=filedialog.askdirectory(parent=self,initialdir=str(
            initial if initial.exists() else Path.home()))
        if selected:self.root_dir.set(selected)

    def template(self,force=False):
        if not force and self.editor.get("1.0","end-1c").strip():
            if not messagebox.askyesno("Заменить CASE?",
                    "Несохранённые изменения будут потеряны.",parent=self):
                return
        path=_template_path(self.theory.get(),int(self.rank.get()))
        doc=json.loads(path.read_text(encoding="utf-8"))
        inspect_static_document(doc)
        self.editor.delete("1.0","end")
        self.editor.insert("1.0",json.dumps(doc,indent=2,ensure_ascii=False))
        self.saved_path=self.saved_output=self.fingerprint=None
        self.progress.set("Статика: новый шаблон, не рассчитан")
        self._note(f"Загружен только статический шаблон {path.name}")

    def _draft(self):
        doc=json.loads(self.editor.get("1.0","end-1c"))
        inspect_static_document(doc)
        if doc["physics"]["theory"]!=self.theory.get():
            raise ValueError("Теория не совпадает с CASE; обновите шаблон")
        if doc["rank"]!=int(self.rank.get()):
            raise ValueError("Ранг МВИ не совпадает с CASE")
        return doc

    def load(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("MVI","Во время счёта нельзя менять CASE",
                                   parent=self)
            return
        selected=filedialog.askopenfilename(parent=self,
            title="Статический MVI CASE",filetypes=(("JSON","*.json"),))
        if not selected:return
        try:
            doc=json.loads(Path(selected).read_text(encoding="utf-8"))
            inspect_static_document(doc)
            self.theory.set(doc["physics"]["theory"])
            self.rank.set(str(doc["rank"]))
            self.editor.delete("1.0","end")
            self.editor.insert("1.0",json.dumps(doc,indent=2,ensure_ascii=False))
            self.saved_path=self.saved_output=self.fingerprint=None
            self.progress.set("Статика: загружен CASE, счёт не запускался")
        except Exception as exc:
            messagebox.showerror("MVI","Ошибка CASE: "+str(exc),parent=self)

    def validate(self):
        try:
            doc=self._draft()
            self._note(f"Проверка MVI STATIC принята: "
                       f"{doc['physics']['theory']}; расчётов не было")
            return True
        except Exception as exc:
            messagebox.showerror("MVI: CASE",str(exc),parent=self)
            return False

    def save(self):
        try:
            draft=self._draft()
            root=Path(self.root_dir.get()).expanduser().resolve()
            if root==ROOT or root==Path(root.anchor):
                raise ValueError("Результаты нельзя сохранять в корне диска")
            name=("MVI_STATIC_"+self.theory.get()+f"_R{self.rank.get()}"+
                  "_"+datetime.now().strftime("%Y%m%d_%H%M%S")+
                  "_"+uuid4().hex[:8])
            case_dir=root/"CASES"
            case_dir.mkdir(parents=True,exist_ok=True)
            output=root/"RUNS"/name
            if output.exists():
                raise FileExistsError("Каталог расчёта уже существует")
            document=json.loads(json.dumps(draft))
            document["output_dir"]=str(output)
            path=case_dir/(name+".json")
            with path.open("x",encoding="utf-8") as stream:
                json.dump(document,stream,indent=2,ensure_ascii=False,
                          allow_nan=False)
                stream.write("\n")
            try:
                own_runner.load_case(path)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            self.saved_path=path
            self.saved_output=output
            self.fingerprint=json.dumps(
                draft,sort_keys=True,ensure_ascii=True,allow_nan=False)
            self.progress.set("Статика: сохранён CASE, готов к запуску")
            self._note(f"CASE: {path}")
            self._note(f"Результаты MVI: {output}")
            return path
        except Exception as exc:
            messagebox.showerror("MVI: сохранение",str(exc),parent=self)
            return None

    def run(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("MVI","Расчёт уже запущен",parent=self)
            return
        try:
            draft=self._draft()
            if self.saved_path is None:
                if self.save() is None:return
            elif json.dumps(draft,sort_keys=True,ensure_ascii=True,
                            allow_nan=False)!=self.fingerprint:
                raise ValueError("CASE изменён; сохраните его снова")
            if self.saved_output.exists():
                raise FileExistsError("Результаты существуют; сохраните новый CASE")
            command=[sys.executable,"-m",OWN_MODULE,
                     "--case",str(self.saved_path),"--run"]
            self._stop_requested=False
            self.proc=subprocess.Popen(command,cwd=str(ROOT),
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                bufsize=1,text=True,encoding="utf-8",errors="replace")
            self.run_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.progress.set("Статика: выполнение")
            threading.Thread(target=self._reader,args=(self.proc,),
                             daemon=True).start()
        except Exception as exc:
            messagebox.showerror("MVI: запуск",str(exc),parent=self)

    def _reader(self,process):
        try:
            for line in process.stdout:
                self.lines.put(line.rstrip())
        except Exception as exc:
            self.lines.put("Ошибка журнала: "+str(exc))

    def _poll(self):
        while True:
            try:self._note(self.lines.get_nowait())
            except queue.Empty:break
        if self.proc is not None and self.saved_output is not None:
            summary=self.saved_output/OWN_SUMMARY
            if summary.is_file():
                try:
                    report=json.loads(summary.read_text(encoding="utf-8"))
                    self.progress.set("Статика: "+str(report.get("status")))
                except (OSError,ValueError,TypeError):
                    pass
        if self.proc is not None and self.proc.poll() is not None:
            code=self.proc.returncode
            if self._stop_requested and self.saved_output:
                summary=self.saved_output/OWN_SUMMARY
                if summary.is_file():
                    try:
                        report=json.loads(summary.read_text(encoding="utf-8"))
                        if report.get("status")=="IN_PROGRESS":
                            report.update(status="STOPPED_BY_USER",
                                          converged=False)
                            temporary=summary.with_suffix(".tmp")
                            temporary.write_text(json.dumps(report,indent=2)+"\n",
                                                 encoding="utf-8")
                            temporary.replace(summary)
                    except (OSError,ValueError,TypeError) as exc:
                        self._note("Ошибка статуса СТОП: "+str(exc))
            self.proc=None
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            file=(self.saved_output/OWN_SUMMARY
                  if self.saved_output is not None else None)
            if file is not None and file.is_file():
                report=json.loads(file.read_text(encoding="utf-8"))
                success=(code==0 and report.get("status")=="CONVERGED" and
                         report.get("converged") is True)
                self.progress.set("Статика: "+(
                    "ГОТОВО" if success else str(report.get("status"))))
                self._note(f"MVI STATIC: "+
                           ("ЗАВЕРШЕНО" if success else "НЕ ЗАВЕРШЕНО"))
            else:
                self.progress.set("Статика: нет итогового файла")
                self._note(f"Процесс закончен (код {code}), итоговый файл отсутствует")
        if self.winfo_exists():
            self.after(200,self._poll)

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self._stop_requested=True
            self.proc.terminate()
            self._note("MVI STATIC: остановка запрошена пользователем")

    def view_results(self):
        folder=self.saved_output
        if folder is None or not folder.is_dir():
            messagebox.showinfo("MVI","Нет результатов",parent=self)
            return
        path=folder/OWN_SUMMARY
        if not path.is_file():
            messagebox.showinfo("MVI","Статический итог ещё не создан",
                                parent=self)
            return
        try:
            report=json.loads(path.read_text(encoding="utf-8"))
            if (report.get("schema")!=OWN_SCHEMA or
                    report.get("independent_method")!="MVI"):
                raise ValueError("Это результаты другого решателя")
            window=tk.Toplevel(self)
            window.title("MVI: сохранённая статика")
            window.geometry("680x440")
            for key in ("theory","status","rank","load_factor","central_w0_m",
                        "central_outer_w_m","maximum_abs_outer_w_m",
                        "completed_xy_halfsteps","solver_module"):
                if key in report:
                    ttk.Label(window,text=f"{key}: {report[key]}",
                              padding=7,anchor="w").pack(fill="x")
            ttk.Label(window,text="Полные 2D-поля и отдельные профили в NPZ",
                      padding=10).pack(fill="x")
        except Exception as exc:
            messagebox.showerror("MVI: результаты",str(exc),parent=self)

    def open_output(self):
        target=self.saved_output
        if target is None or not target.is_dir():
            messagebox.showinfo("MVI","Папка результатов не создана",
                                parent=self)
            return
        try:
            if sys.platform=="win32":os.startfile(str(target))
            elif sys.platform=="darwin":subprocess.Popen(["open",str(target)])
            else:subprocess.Popen(["xdg-open",str(target)])
        except Exception as exc:
            messagebox.showerror("MVI: папка",str(exc),parent=self)

    def _close(self):
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("MVI","Остановить расчёт?",
                                       parent=self):
                return
            self.stop()
            self.after(200,self._close)
            return
        self.destroy()


def open_window(parent):
    return IndependentStaticWindow(parent)
