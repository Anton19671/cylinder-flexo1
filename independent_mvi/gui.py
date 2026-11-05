"""Own МВИ seven-theory experimental CASE editor and launcher.

User action ONLY starts a computation. Import/preview/validate/save never
execute physical shell equations. This GUI does not import the other solver,
legacy shared functional, or the old reference CASE dispatcher.
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
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
from uuid import uuid4

from independent_mvi.linear_models import VariationalModelCase
from independent_mvi.run_models import SCHEMA, load_case

ROOT=Path(__file__).resolve().parents[1]
THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
SCHEMES=("BATHE","NEWMARK","GAUSS_IRK4")
TEMPLATE_DIR=ROOT/"config"/"verification"/"independent_all_models"
MODULE="independent_mvi.run_models"
OWN_SUMMARY="MVI_OWN_RUN_SUMMARY.json"


def inspect_document(document):
    """Independent validation; no files, subprocesses or numerical solves."""
    if not isinstance(document,dict) or set(document)!={"schema","physics","time","rank","output_dir"}:
        raise ValueError("МВИ: требуется собственный CASE без чужих полей")
    if document["schema"]!=SCHEMA:
        raise ValueError("МВИ: несовместимый CASE другого метода")
    if not isinstance(document["physics"],dict):
        raise ValueError("МВИ: physics должен быть объектом")
    physics=VariationalModelCase(**document["physics"])
    if physics.theory not in THEORIES:
        raise ValueError("МВИ: неподдерживаемая теория")
    timing=document["time"]
    if not isinstance(timing,dict) or set(timing)!={"integrator","steps","dt_s"}:
        raise ValueError("МВИ: time должен содержать integrator, steps, dt_s")
    if timing["integrator"] not in SCHEMES:
        raise ValueError("МВИ: интегратор не поддерживается")
    if type(timing["steps"]) is not int or not 1<=timing["steps"]<=120:
        raise ValueError("МВИ: нужно 1..120 временных шагов")
    from math import isfinite
    if (isinstance(timing["dt_s"],bool) or not isinstance(timing["dt_s"],(int,float))
        or not isfinite(timing["dt_s"]) or not 0<timing["dt_s"]<=.1):
        raise ValueError("МВИ: dt_s должен быть положительным конечным числом")
    if type(document.get("rank")) is not int or not 1 <= document["rank"] <= 8:
        raise ValueError("МВИ: ранг должен быть от 1 до 8")
    available=min(physics.ns,physics.nx-(4 if physics.theory=="KL" else 2))
    if document["rank"]>available:
        raise ValueError("МВИ: ранг превышает допустимое пространство с закреплением")
    if not isinstance(document["output_dir"],str) or not document["output_dir"].strip():
        raise ValueError("МВИ: нужна отдельная папка результатов")
    return physics


class IndependentMVIWindow(tk.Toplevel):
    """Separate GUI controls and background process owned by МВИ only."""

    def __init__(self,parent):
        super().__init__(parent)
        self.title("Независимый МВИ — 7 теорий, линейная динамика (эксперимент)")
        self.geometry("930x730")
        self.minsize(720,580)
        self.theory=tk.StringVar(self,value="CUF_TE4")
        self.integrator=tk.StringVar(self,value="NEWMARK")
        self.rank=tk.StringVar(self,value="1")
        self.root_dir=tk.StringVar(self,value=str(
            Path.home()/"CYL_RESULTS"/"INDEPENDENT_MVI"))
        self.saved_path=None
        self.saved_fingerprint=None
        self.saved_output=None
        self.proc=None
        self.pause_signal=None
        self.lines=queue.Queue()

        ttk.Label(self,text=(
            "Отдельный МВИ: собственная сборка, масса, нагрузка и "
            "временной интегратор. Только LINEAR, HOOKE, MACRO. "
            "VK/GL, G3D/Gz, флексо и модальный режим здесь НЕ подключены."),
            wraplength=850,foreground="#9a6700").pack(
                fill="x",padx=12,pady=(10,6))
        selectors=ttk.Frame(self)
        selectors.pack(fill="x",padx=12,pady=5)
        ttk.Label(selectors,text="Теория").pack(side="left")
        ttk.Combobox(selectors,textvariable=self.theory,values=THEORIES,
                     state="readonly",width=13).pack(side="left",padx=6)
        ttk.Label(selectors,text="По времени").pack(side="left",padx=(12,3))
        ttk.Combobox(selectors,textvariable=self.integrator,values=SCHEMES,
                     state="readonly",width=14).pack(side="left",padx=6)
        ttk.Label(selectors,text="Ранг").pack(side="left",padx=(12,3))
        ttk.Combobox(selectors,textvariable=self.rank,values=[str(i) for i in range(1,9)],
                     state="readonly",width=4).pack(side="left")

        directory=ttk.Frame(self)
        directory.pack(fill="x",padx=12,pady=5)
        ttk.Label(directory,text="Корень результатов").pack(side="left")
        ttk.Entry(directory,textvariable=self.root_dir).pack(
            side="left",fill="x",expand=True,padx=6)
        ttk.Button(directory,text="…",width=3,
                   command=self.choose_root).pack(side="left")
        ttk.Label(self,text="Собственный CASE (JSON; SI): измените геометрию, сетку, нагрузку, dt, шаги.",
                  anchor="w").pack(fill="x",padx=12,pady=(9,2))
        self.editor=tk.Text(self,height=23,wrap="none",undo=True)
        self.editor.pack(fill="both",expand=True,padx=12)
        actions=ttk.Frame(self)
        actions.pack(fill="x",padx=12,pady=(9,7))
        ttk.Button(actions,text="Шаблон",command=self.template).pack(side="left",padx=3)
        ttk.Button(actions,text="Проверить",command=self.validate).pack(side="left",padx=3)
        ttk.Button(actions,text="Сохранить CASE",command=self.save).pack(side="left",padx=3)
        self.run_button=ttk.Button(actions,text="РАССЧИТАТЬ",command=self.run)
        self.run_button.pack(side="left",padx=10)
        self.pause_button=ttk.Button(actions,text="ПАУЗА",command=self.toggle_pause,state="disabled")
        self.pause_button.pack(side="left",padx=3)
        self.stop_button=ttk.Button(actions,text="СТОП",command=self.stop,state="disabled")
        self.stop_button.pack(side="left",padx=3)
        ttk.Button(actions,text="Папка результатов",
                   command=self.open_result).pack(side="left",padx=3)
        self.log=tk.Text(self,height=8,wrap="word",state="disabled")
        self.log.pack(fill="both",padx=12,pady=(0,10))
        self.template(force=True)
        self.after(150,self._poll)
        self.protocol("WM_DELETE_WINDOW",self._close)

    def _note(self,message):
        self.log.configure(state="normal")
        self.log.insert("end",str(message)+"\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def choose_root(self):
        initial=Path(self.root_dir.get()).expanduser()
        result=filedialog.askdirectory(parent=self,initialdir=str(
            initial if initial.is_dir() else Path.home()),
            title="МВИ: собственная папка результатов")
        if result:self.root_dir.set(result)

    def template(self,force=False):
        if not force and self.editor.get("1.0","end-1c").strip():
            if not messagebox.askyesno("Заменить текст CASE?",
                                       "Несохранённые изменения будут потеряны.",
                                       parent=self):
                return
        path=TEMPLATE_DIR/f"MVI_{self.theory.get()}_{self.integrator.get()}.json"
        document=json.loads(path.read_text(encoding="utf-8"))
        self.rank.set(str(document.get("rank",1)))
        self.editor.delete("1.0","end")
        self.editor.insert("1.0",json.dumps(document,ensure_ascii=False,indent=2))
        self.saved_path=None
        self.saved_output=None
        self.saved_fingerprint=None
        self._note(f"Загружен шаблон: {path.name} (расчёт не запускался)")

    def _draft(self):
        document=json.loads(self.editor.get("1.0","end-1c"))
        document["rank"]=int(self.rank.get())
        inspect_document(document)
        if document["physics"]["theory"]!=self.theory.get():
            raise ValueError("Теория в JSON отличается от выбора; загрузите нужный шаблон")
        if document["time"]["integrator"]!=self.integrator.get():
            raise ValueError("Интегратор в JSON отличается от выбора; загрузите шаблон")
        return document

    @staticmethod
    def _fingerprint(document):
        return json.dumps(document,sort_keys=True,allow_nan=False,ensure_ascii=True)

    def validate(self):
        try:
            document=self._draft()
            physics=inspect_document(document)
            self._note("Проверка принята: "+physics.theory+" / "+
                       document["time"]["integrator"]+
                       ". Физические расчёты НЕ выполнялись.")
            messagebox.showinfo("Проверка МВИ","CASE допустим для отдельного "+
                                "МВИ; физическая точность требует верификации.",
                                parent=self)
            return True
        except Exception as exc:
            messagebox.showerror("Ошибка независимого МВИ",str(exc),parent=self)
            return False

    def save(self):
        try:
            document=self._draft()
            root=Path(self.root_dir.get()).expanduser().resolve()
            if root==ROOT or root==Path(root.anchor):
                raise ValueError("Выберите отдельный каталог результатов, не корень диска")
            suffix=datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:8]
            stem="MVI_"+self.theory.get()+"_"+self.integrator.get()+"_"+suffix
            case_dir=root/"CASES"
            # No source CASE is ever overwritten or duplicated on every run.
            case_dir.mkdir(parents=True,exist_ok=True)
            output=root/"RUNS"/stem
            document["output_dir"]=str(output)
            case_file=case_dir/(stem+".json")
            if output.exists():
                raise FileExistsError("Папка результатов уже существует")
            with case_file.open("x",encoding="utf-8") as stream:
                json.dump(document,stream,ensure_ascii=False,indent=2,allow_nan=False)
                stream.write("\n")
            try:
                load_case(case_file)  # uses ONLY the method-specific runner
            except BaseException:
                case_file.unlink(missing_ok=True)
                raise
            self.saved_path=case_file
            self.saved_output=output
            self.saved_fingerprint=self._fingerprint(self._draft())
            self._note(f"Сохранён один CASE: {case_file}")
            self._note(f"Отдельные результаты: {output}")
            return case_file
        except Exception as exc:
            messagebox.showerror("Сохранение МВИ",str(exc),parent=self)
            return None

    def run(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("МВИ","Процесс уже выполняется",parent=self)
            return
        try:
            draft=self._draft()
            if self.saved_path is None:
                if self.save() is None:return
            elif self._fingerprint(draft)!=self.saved_fingerprint:
                messagebox.showerror("МВИ","CASE изменён после сохранения. "+
                                     "Нажмите «Сохранить CASE» перед расчётом.",
                                     parent=self)
                return
            if self.saved_output.exists():
                raise FileExistsError("Результаты уже созданы; сохраните новый CASE")
            command=[sys.executable,"-m",MODULE,"--case",str(self.saved_path),"--run"]
            self.pause_signal=self.saved_path.with_suffix(".pause")
            if self.pause_signal.exists():
                raise FileExistsError("Найден старый сигнал паузы рядом с CASE")
            self.proc=subprocess.Popen(command,cwd=str(ROOT),
                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                 text=True,bufsize=1,encoding="utf-8",errors="replace")
            self.run_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.pause_button.configure(state="normal",text="ПАУЗА")
            self._note("Запущен только отдельный МВИ по нажатию пользователя")
            threading.Thread(target=self._reader,args=(self.proc,),daemon=True).start()
        except Exception as exc:
            messagebox.showerror("МВИ: запуск",str(exc),parent=self)

    def _reader(self,process):
        try:
            for line in process.stdout:
                self.lines.put(line.rstrip())
        except Exception as exc:
            self.lines.put("Ошибка чтения лога: "+str(exc))

    def _poll(self):
        while True:
            try:self._note(self.lines.get_nowait())
            except queue.Empty:break
        if self.proc is not None and self.proc.poll() is not None:
            code=self.proc.returncode
            self.proc=None
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.pause_button.configure(state="disabled",text="ПАУЗА")
            if self.pause_signal is not None:
                self.pause_signal.unlink(missing_ok=True)
            summary_path=self.saved_output/OWN_SUMMARY if self.saved_output else None
            if summary_path is not None and summary_path.exists():
                try:
                    summary=json.loads(summary_path.read_text(encoding="utf-8"))
                    accepted=summary.get("accepted_steps",0)
                    requested=summary.get("requested_steps",None)
                    done=(code==0 and summary.get("status")=="CONVERGED"
                          and type(requested) is int and requested>0
                          and accepted==requested)
                    self._note(f"Итог МВИ: статус={summary.get('status')}, "+
                               f"шаги={accepted}/{requested}, code={code}; "+
                               ("ПРИНЯТО" if done else "НЕ ЗАВЕРШЕНО"))
                except Exception as exc:self._note("Ошибка итогового файла: "+str(exc))
            else:self._note(f"МВИ: процесс завершён, код={code}; нет итогового файла")
        if self.winfo_exists():self.after(200,self._poll)

    def toggle_pause(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        signal=self.pause_signal
        if signal is None:
            return
        if signal.is_file():
            signal.unlink()
            self.pause_button.configure(text="ПАУЗА")
            self._note("MVI: возобновление между шагами")
        else:
            signal.write_text("PAUSE_BY_USER\\n",encoding="ascii")
            self.pause_button.configure(text="ПРОДОЛЖИТЬ")
            self._note("MVI: пауза после принятого шага")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self._note("МВИ: пользователь запросил остановку")
            self.proc.terminate()
            if self.pause_signal is not None:
                self.pause_signal.unlink(missing_ok=True)

    def open_result(self):
        target=self.saved_output
        if target is None or not target.is_dir():
            messagebox.showinfo("МВИ","Папка результатов пока не создана",parent=self)
            return
        try:
            if sys.platform=="win32":os.startfile(str(target))
            elif sys.platform=="darwin":subprocess.Popen(["open",str(target)])
            else:subprocess.Popen(["xdg-open",str(target)])
        except Exception as exc:messagebox.showerror("Папка",str(exc),parent=self)

    def _close(self):
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("МВИ","Остановить выполняющийся расчёт?",
                                       parent=self):
                return
            self.proc.terminate()
            if self.pause_signal is not None:
                self.pause_signal.unlink(missing_ok=True)
        self.destroy()


def open_window(parent):
    return IndependentMVIWindow(parent)


__all__=("inspect_document","IndependentMVIWindow","open_window")
