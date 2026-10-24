#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Editor for optional conservative boundary passport (reference-only until wired)."""
import copy
import json
import tkinter as tk
from tkinter import ttk, messagebox
from cylinder_variational_boundary import (
    boundary_passport, validate_boundary_passport, MECHANICAL_SUPPORTS,
    HIGHER_TRACES, ELECTRIC_MODES,
)


def open_boundary_editor(gui):
    spec=copy.deepcopy(getattr(gui,"variational_bc",None) or boundary_passport())
    win=tk.Toplevel(gui)
    win.title("Механические, G3D и электрические граничные условия")
    win.geometry("790x680")
    win.transient(gui)
    win.columnconfigure(0,weight=1)
    win.rowconfigure(0,weight=1)
    book=ttk.Notebook(win)
    book.grid(row=0,column=0,sticky="nsew",padx=8,pady=8)
    mech=ttk.Frame(book,padding=10)
    electric=ttk.Frame(book,padding=10)
    full=ttk.Frame(book,padding=10)
    book.add(mech,text="Механика / G3D")
    book.add(electric,text="Электрика")
    book.add(full,text="Полный JSON и время")
    m={}
    for i,side in enumerate(("left","right")):
        row=ttk.LabelFrame(mech,text=("x=0" if side=="left" else "x=L"),padding=8)
        row.grid(row=i,column=0,sticky="ew",pady=6)
        item=spec.get("mechanical",{}).get(side,{})
        support=tk.StringVar(value=item.get("support","FREE"))
        higher=tk.StringVar(value=item.get("higher","NATURAL"))
        ttk.Label(row,text="Опора").grid(row=0,column=0,sticky="w")
        ttk.Combobox(row,textvariable=support,values=MECHANICAL_SUPPORTS,
                     state="readonly",width=34).grid(row=0,column=1,padx=8)
        ttk.Label(row,text="Высшие нормальные следы").grid(row=1,column=0,sticky="w")
        ttk.Combobox(row,textvariable=higher,values=HIGHER_TRACES,
                     state="readonly",width=34).grid(row=1,column=1,padx=8)
        m[side]=(support,higher)
    ttk.Label(mech,text="Нагрузки, пружины и CUSTOM-компоненты задаются на вкладке JSON.\n"
              "Это паспорт вариационных КУ, а не подтверждение численного расчёта.").grid(row=3,column=0,sticky="w",pady=12)
    e={}
    for i,side in enumerate(("left","right","outer","inner")):
        item=spec.get("electrical",{}).get(side,{})
        mode=tk.StringVar(value=item.get("mode","INSULATED"))
        value=tk.StringVar(value=str(item.get("value","")))
        ttk.Label(electric,text=side).grid(row=i,column=0,sticky="w",pady=7)
        ttk.Combobox(electric,textvariable=mode,values=ELECTRIC_MODES,
                     state="readonly",width=22).grid(row=i,column=1,padx=8)
        ttk.Entry(electric,textvariable=value,width=24).grid(row=i,column=2,padx=8)
        e[side]=(mode,value)
    ttk.Label(electric,text="POTENTIAL [В]; CHARGE [Кл/м²]; INSULATED — пустое значение.\n"
              "Электрические ГУ требуют MICRO + флексо ВКЛ.").grid(row=5,column=0,columnspan=3,sticky="w",pady=12)
    full.columnconfigure(0,weight=1)
    full.rowconfigure(0,weight=1)
    editor=tk.Text(full,wrap="none",undo=True)
    editor.grid(row=0,column=0,sticky="nsew")
    sb=ttk.Scrollbar(full,command=editor.yview)
    sb.grid(row=0,column=1,sticky="ns")
    editor.configure(yscrollcommand=sb.set)
    editor.insert("1.0",json.dumps(spec,ensure_ascii=False,indent=2))
    ttk.Label(full,text="В этом поле можно добавить time_law, traction, double_traction, spring и CUSTOM.").grid(row=1,column=0,sticky="w",pady=6)

    def sync():
        try:
            data=json.loads(editor.get("1.0","end"))
            for side,(support,higher) in m.items():
                item=data.setdefault("mechanical",{}).setdefault(side,{})
                item.update(support=support.get(),higher=higher.get())
            for side,(mode,value) in e.items():
                item=data.setdefault("electrical",{}).setdefault(side,{})
                item["mode"]=mode.get()
                if mode.get()=="INSULATED":
                    item.pop("value",None)
                    item.pop("time_law",None)
                else:
                    if value.get().strip():item["value"]=float(value.get())
                    elif "value" not in item:raise ValueError(side+": введите значение")
            editor.delete("1.0","end")
            editor.insert("1.0",json.dumps(data,ensure_ascii=False,indent=2))
        except Exception as exc:
            messagebox.showerror("ГУ",str(exc),parent=win)

    def apply():
        try:
            data=json.loads(editor.get("1.0","end"))
            theory=gui._v("theory","CUF").get().upper()
            if theory=="CUF":theory="CUF_"+gui._v("cuf_model","TE1").get().upper()
            enabled=gui._v("scale_model","MACRO").get()=="MICRO" and gui._v("micro_flexoelectric","OFF").get()=="ON"
            validate_boundary_passport(data,electrical_enabled=enabled,
                                       calculation_kind=gui._v("calculation_kind","STATIC").get(),
                                       theory=theory)
            gui.variational_bc=copy.deepcopy(data)
            win.destroy()
        except Exception as exc:
            messagebox.showerror("Некорректные ГУ",str(exc),parent=win)

    buttons=ttk.Frame(win,padding=8)
    buttons.grid(row=1,column=0,sticky="ew")
    ttk.Button(buttons,text="Перенести выбор в JSON",command=sync).pack(side="left",padx=4)
    ttk.Button(buttons,text="Применить JSON",command=apply).pack(side="left",padx=4)
    ttk.Button(buttons,text="Убрать дополнительный паспорт",
               command=lambda:(setattr(gui,"variational_bc",None),win.destroy())).pack(side="left",padx=4)
    ttk.Button(buttons,text="Отмена",command=win.destroy).pack(side="right")
