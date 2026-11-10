"""Separate Tk research CASE manager for fully independent nonlinear MKR/MVI.

Available from the existing cylinder GUI without changing legacy tabs.
Only explicit native JSON CASEs; no hidden physics defaults or solver
substitution. Validation is non-solving. Real execution uses the
public final-case command in a child process. Pause is cooperative
BETWEEN accepted physical steps. Stop terminates the child if needed.
The mini online graph reads already-accepted native NPZ results only.
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
from tkinter import ttk,filedialog,messagebox

import numpy as np

from cylinder_final_stage11a_dispatcher import validate_final_case

SCHEMAS=frozenset((
    "INDEPENDENT_MKR_COUPLED_RESEARCH_V1",
    "INDEPENDENT_MVI_COUPLED_RESEARCH_V1",
))


def _load_native(path):
    path=Path(path).expanduser().resolve()
    if path.suffix.lower()!=".json":
        raise ValueError("Выберите нативный JSON CASE")
    data=json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data,dict) or data.get("schema") not in SCHEMAS:
        raise ValueError("Нужен связанный исследовательский CASE МКР или МВИ")
    return path,data


class NativeResearchWindow(tk.Toplevel):
    """Owns its child solver process and user-selected saved CASE file."""

    def __init__(self,parent):
        super().__init__(parent)
        self.title("Нативные МКР / МВИ — связанный решатель G3D/Gz")
        self.geometry("1050x800")
        self.minsize(760,560)
        self.case_path=None
        self.process=None
        self.read_queue=queue.Queue()
        self.results_dir=None
        self.expected_steps=0
        self.seen_steps=set()
        self.history=[]
        self.stop_requested=False
        self.closed=False
        self.case_description=tk.StringVar(value="CASE не выбран")
        self.state_description=tk.StringVar(value="Ожидание")
        self.path_description=tk.StringVar(value="Папка результатов не выбрана")
        self._build()
        self.protocol("WM_DELETE_WINDOW",self._close)
        self.after(180,self._poll)

    def _build(self):
        bar=ttk.Frame(self,padding=8)
        bar.pack(fill="x")
        ttk.Button(bar,text="Открыть CASE",command=self.open_case).pack(
            side="left",padx=2)
        ttk.Button(bar,text="Сохранить CASE",command=self.save_case).pack(
            side="left",padx=2)
        ttk.Button(bar,text="Сохранить как",command=self.save_as).pack(
            side="left",padx=2)
        ttk.Button(bar,text="Папка результатов",command=self.choose_results).pack(
            side="left",padx=2)
        ttk.Button(bar,text="Открыть результаты",command=self.open_results).pack(
            side="left",padx=2)
        ttk.Label(self,textvariable=self.case_description).pack(
            anchor="w",padx=12)
        ttk.Label(self,textvariable=self.path_description).pack(
            anchor="w",padx=12)
        tabs=ttk.Notebook(self)
        tabs.pack(fill="both",expand=True,padx=8,pady=6)
        editor=ttk.Frame(tabs)
        execution=ttk.Frame(tabs)
        tabs.add(editor,text="CASE JSON / постановка")
        tabs.add(execution,text="Расчёт / график / журнал")
        ttk.Label(editor,text=(
            "Явная постановка. Изменения нужно сохранить перед запуском. "
            "Неподдержанные физические условия отклоняются.")).pack(
            anchor="w",padx=6,pady=4)
        shell=ttk.Frame(editor)
        shell.pack(fill="both",expand=True)
        self.editor=tk.Text(shell,wrap="none",undo=True,font=("Consolas",10))
        sx=ttk.Scrollbar(shell,orient="horizontal",command=self.editor.xview)
        sy=ttk.Scrollbar(shell,orient="vertical",command=self.editor.yview)
        self.editor.configure(xscrollcommand=sx.set,yscrollcommand=sy.set)
        self.editor.grid(row=0,column=0,sticky="nsew")
        sy.grid(row=0,column=1,sticky="ns")
        sx.grid(row=1,column=0,sticky="ew")
        shell.columnconfigure(0,weight=1)
        shell.rowconfigure(0,weight=1)
        toolbar=ttk.Frame(execution,padding=8)
        toolbar.pack(fill="x")
        self.validate_button=ttk.Button(
            toolbar,text="Проверить CASE",command=self.validate)
        self.validate_button.pack(side="left",padx=3)
        self.run_button=ttk.Button(toolbar,text="Запустить",
                                   command=self.run_solver)
        self.run_button.pack(side="left",padx=3)
        ttk.Button(toolbar,text="Пауза между шагами",
                   command=self.pause).pack(side="left",padx=3)
        ttk.Button(toolbar,text="Продолжить",
                   command=self.resume).pack(side="left",padx=3)
        ttk.Button(toolbar,text="Стоп",
                   command=self.stop).pack(side="left",padx=3)
        ttk.Label(execution,textvariable=self.state_description).pack(
            anchor="w",padx=12)
        self.progress=ttk.Progressbar(execution,mode="determinate",
                                     maximum=100,value=0)
        self.progress.pack(fill="x",padx=12,pady=4)
        self.graph=tk.Canvas(execution,height=260,bg="white",
                             highlightbackground="#a0a0a0",
                             highlightthickness=1)
        self.graph.pack(fill="x",padx=12,pady=(8,6))
        self.graph.bind("<Configure>",lambda _e:self._draw_graph())
        self.journal=tk.Text(execution,height=12,wrap="word",
                             state="disabled",font=("Consolas",9))
        self.journal.pack(fill="both",expand=True,padx=12,pady=6)

    def _log(self,message):
        self.journal.configure(state="normal")
        self.journal.insert("end",str(message).rstrip()+"\n")
        self.journal.see("end")
        self.journal.configure(state="disabled")

    def _edited(self):
        try:
            case=json.loads(self.editor.get("1.0","end-1c"))
        except json.JSONDecodeError as exc:
            raise ValueError("Ошибка JSON CASE: "+str(exc)) from exc
        if not isinstance(case,dict) or case.get("schema") not in SCHEMAS:
            raise ValueError("Нужна схема связанного нативного МКР или МВИ")
        if case.get("research_opt_in") is not True:
            raise ValueError("Нативный экспериментальный CASE требует research_opt_in=true")
        return case

    def _save_to(self,destination):
        if self.process is not None and self.process.poll() is None:
            raise ValueError("Нельзя изменять CASE во время расчёта")
        case=self._edited()
        destination=Path(destination).expanduser().resolve()
        if destination.suffix.lower()!=".json":
            raise ValueError("CASE сохраняется в формате JSON")
        destination.write_text(
            json.dumps(case,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        self.case_path=destination
        self.case_description.set(str(destination))
        self._log("CASE сохранён: "+str(destination))
        return destination

    def open_case(self):
        try:
            path=filedialog.askopenfilename(
                parent=self,title="Нативный CASE",
                filetypes=[("CASE JSON","*.json")])
            if not path:
                return
            chosen,data=_load_native(path)
            self.case_path=chosen
            self.editor.delete("1.0","end")
            self.editor.insert("1.0",json.dumps(
                data,ensure_ascii=False,indent=2))
            self.editor.edit_modified(False)
            self.case_description.set(str(chosen))
            self._show_output(data,chosen.parent)
            self._log("Открыт: "+str(chosen))
        except Exception as exc:
            messagebox.showerror("Открытие CASE",str(exc),parent=self)

    def save_case(self):
        try:
            if self.case_path is None:
                self.save_as()
            else:
                self._save_to(self.case_path)
        except Exception as exc:
            messagebox.showerror("Сохранение CASE",str(exc),parent=self)

    def save_as(self):
        try:
            path=filedialog.asksaveasfilename(
                parent=self,title="Сохранить нативный CASE",
                defaultextension=".json",filetypes=[("CASE JSON","*.json")])
            if path:
                self._save_to(path)
        except Exception as exc:
            messagebox.showerror("Сохранение CASE",str(exc),parent=self)

    def _show_output(self,case,base):
        output=Path(case.get("output_dir",""))
        self.results_dir=(output if output.is_absolute() else base/output).resolve()
        self.path_description.set(str(self.results_dir))
        self.expected_steps=(int(case["solver"].get("steps",0))
                             if case.get("solver",{}).get("kind")=="DYNAMIC"
                             else 1)

    def choose_results(self):
        try:
            case=self._edited()
            folder=filedialog.askdirectory(parent=self,title="Новая папка результатов")
            if not folder:
                return
            case["output_dir"]=folder
            self.editor.delete("1.0","end")
            self.editor.insert("1.0",json.dumps(case,ensure_ascii=False,indent=2))
            self._show_output(case,self.case_path.parent if self.case_path else Path.cwd())
            self._log("Выбрана папка; сохраните CASE перед запуском.")
        except Exception as exc:
            messagebox.showerror("Папка результатов",str(exc),parent=self)

    def open_results(self):
        try:
            if self.results_dir is None or not self.results_dir.exists():
                raise ValueError("Папка результатов пока не создана")
            if os.name=="nt":
                os.startfile(str(self.results_dir))
            elif sys.platform=="darwin":
                subprocess.Popen(["open",str(self.results_dir)])
            else:
                subprocess.Popen(["xdg-open",str(self.results_dir)])
        except Exception as exc:
            messagebox.showerror("Результаты",str(exc),parent=self)

    def _require_saved(self):
        if self.case_path is None:
            raise ValueError("Сначала сохраните CASE")
        current=self._edited()
        saved=json.loads(self.case_path.read_text(encoding="utf-8-sig"))
        if current!=saved:
            raise ValueError("Изменения CASE не сохранены")
        self._show_output(saved,self.case_path.parent)
        return saved

    def validate(self):
        try:
            self._require_saved()
            result=validate_final_case(self.case_path,dry_run=True)
            self._log("Проверка входных данных успешна: "+
                      result.get("solver_backend",""))
            self.state_description.set("CASE проверен; физический расчёт не запускался")
        except Exception as exc:
            self.state_description.set("Проверка CASE не прошла")
            self._log("ОШИБКА: "+str(exc))
            messagebox.showerror("Проверка CASE",str(exc),parent=self)

    def run_solver(self):
        try:
            if self.process is not None and self.process.poll() is None:
                raise ValueError("Расчёт уже выполняется")
            self._require_saved()
            validate_final_case(self.case_path,dry_run=True)
            if self.results_dir.exists():
                raise ValueError("Папка результатов уже существует: "+str(self.results_dir))
            self.seen_steps=set()
            self.history=[]
            self.stop_requested=False
            self.progress["value"]=0
            self._draw_graph()
            for name in ("CONTROL_PAUSE","CONTROL_STOP"):
                # No stale control file is ever removed from an earlier run.
                if (self.results_dir/name).exists():
                    raise ValueError("Обнаружена управляющая метка старого запуска")
            command=[sys.executable,str(
                Path(__file__).resolve().parent/"RUN_FINAL_CASE.py"),
                str(self.case_path)]
            env=dict(os.environ)
            env["PYTHONIOENCODING"]="utf-8"
            self.process=subprocess.Popen(
                command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding="utf-8",errors="replace",
                bufsize=1,env=env)
            threading.Thread(target=self._reader,
                             args=(self.process,),daemon=True).start()
            self.run_button.configure(state="disabled")
            self.state_description.set("Расчёт выполняется")
            self._log("Начат нативный расчёт (отдельный процесс).")
        except Exception as exc:
            self._log("ОШИБКА запуска: "+str(exc))
            messagebox.showerror("Запуск",str(exc),parent=self)

    def _reader(self,process):
        try:
            for line in process.stdout:
                self.read_queue.put(line)
        finally:
            self.read_queue.put(("finished",process))

    def _control(self,name):
        if self.process is None or self.process.poll() is not None:
            self._log("Нет запущенного процесса")
            return False
        if self.results_dir is None or not self.results_dir.is_dir():
            self._log("Папка расчёта ещё не создана")
            return False
        (self.results_dir/name).write_text("requested\n",encoding="utf-8")
        return True

    def pause(self):
        if self._control("CONTROL_PAUSE"):
            self.state_description.set("Пауза запрошена — между шагами")
            self._log("Пауза вступит в силу между принятыми шагами.")

    def resume(self):
        if self.results_dir is not None:
            (self.results_dir/"CONTROL_PAUSE").unlink(missing_ok=True)
            self.state_description.set("Продолжение")
            self._log("Продолжить расчёт")

    def stop(self):
        self.stop_requested=True
        self._control("CONTROL_STOP")
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self._log("Остановлен процесс расчёта; уже принятые шаги сохранены.")
        self.state_description.set("Расчёт остановлен")

    def _radial_index(self,theory):
        if theory.startswith("CUF_TE"):
            return 2*(int(theory[-1])+1)
        return 2

    def _scan_history(self):
        if self.results_dir is None or not self.results_dir.exists():
            return
        paths=list(self.results_dir.glob("STEP_????.npz"))
        if (self.results_dir/"STATIC_ACCEPTED.npz").exists():
            paths.append(self.results_dir/"STATIC_ACCEPTED.npz")
        for path in sorted(paths):
            if str(path) in self.seen_steps:
                continue
            try:
                with np.load(path,allow_pickle=False) as data:
                    q=np.asarray(data["mechanical"])
                    saved=json.loads(self.case_path.read_text(encoding="utf-8-sig"))
                    index=self._radial_index(saved["physics"]["theory"])
                    sample=float(q[index,q.shape[1]//2,q.shape[2]//2])
                    if not np.isfinite(sample):
                        continue
                    moment=(float(data["time_s"]) if "time_s" in data
                            else float(len(self.history)))
                self.history.append((moment,sample))
                self.seen_steps.add(str(path))
            except (OSError,ValueError,KeyError,IndexError,TypeError):
                continue
        completed=max(0,len(self.history)-1) if self.expected_steps>1 else len(self.history)
        self.progress["value"]=(100*min(completed,self.expected_steps)/
                                max(1,self.expected_steps))
        self._draw_graph()

    def _draw_graph(self):
        canvas=self.graph
        canvas.delete("all")
        width=max(canvas.winfo_width(),400)
        height=max(canvas.winfo_height(),230)
        canvas.create_text(12,12,anchor="nw",
                           text="Прогиб в центре, м — онлайн по принятым шагам",
                           fill="#202020")
        x0,y0,x1,y1=52,height-35,width-20,35
        canvas.create_line(x0,y0,x1,y0,fill="#333333")
        canvas.create_line(x0,y0,x0,y1,fill="#333333")
        if not self.history:
            return
        points=sorted(self.history)
        ts=np.asarray([x[0] for x in points])
        ws=np.asarray([x[1] for x in points])
        xmin,xmax=float(min(ts)),float(max(ts))
        ymin,ymax=float(min(ws)),float(max(ws))
        if xmax<=xmin:xmax=xmin+1.
        if ymax<=ymin:
            span=max(1e-12,abs(ymin)*.1)
            ymin-=span
            ymax+=span
        xs=x0+(ts-xmin)/(xmax-xmin)*(x1-x0)
        ys=y0-(ws-ymin)/(ymax-ymin)*(y0-y1)
        coords=[coord for xx,yy in zip(xs,ys) for coord in (float(xx),float(yy))]
        if len(coords)>=4:
            canvas.create_line(*coords,fill="#006fc9",width=2)
        for xx,yy in zip(xs,ys):
            canvas.create_oval(xx-2,yy-2,xx+2,yy+2,fill="#006fc9",outline="")
        canvas.create_text(x0,y1-4,anchor="sw",text=f"{ymax:.4g}")
        canvas.create_text(x0,y0+3,anchor="nw",text=f"{ymin:.4g}")
        canvas.create_text(x1,y0+3,anchor="ne",text=f"{xmax:.3g} c")

    def _poll(self):
        if self.closed:return
        while True:
            try:message=self.read_queue.get_nowait()
            except queue.Empty:break
            if isinstance(message,tuple):
                _,process=message
                if process is self.process:
                    code=process.poll()
                    self.run_button.configure(state="normal")
                    if self.stop_requested:
                        status="Остановлен пользователем"
                    elif code==0:
                        status="Экспериментальный расчёт завершён; физическая верификация не подтверждена"
                    else:
                        status=f"Расчёт завершён с ошибкой: код {code}"
                    self.state_description.set(status)
                    self._log(status)
                continue
            self._log(message)
        try:self._scan_history()
        except Exception as exc:self._log("Ошибка чтения онлайн-графика: "+str(exc))
        self.after(300,self._poll)

    def _close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno(
                    "Закрыть","Остановить текущий нативный расчёт и закрыть окно?",
                    parent=self):
                return
            self.stop()
        self.closed=True
        self.destroy()


def open_native_research_window(parent):
    return NativeResearchWindow(parent)


__all__=("NativeResearchWindow","open_native_research_window")
