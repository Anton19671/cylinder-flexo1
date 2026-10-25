#!/usr/bin/env python3
"""Tk editor for the explicit small-grid variational CASE (no legacy conversion)."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from cylinder_reference_case import BACKEND, prepare_reference_case

TEMPLATE=Path(__file__).resolve().parent.parent/'config/CASE_REFERENCE_CUF_G3D_STATIC.json'


def edit_reference_case(original, fields, boundary, load, newton):
    """Make one lossless CASE update; fail rather than discard unknown settings."""
    case=copy.deepcopy(original)
    if case.get('solver_backend')!=BACKEND:raise ValueError('Откройте вариационный CASE')
    def numeric(key):
        try:return float(str(fields[key]).strip().replace(',','.'))
        except (TypeError,ValueError) as exc:raise ValueError(f'{key}: требуется число') from exc
    def integer(key):
        value=str(fields[key]).strip()
        if not value.isdecimal():raise ValueError(f'{key}: требуется целое число')
        return int(value)
    def document(raw,label):
        try:data=json.loads(raw)
        except json.JSONDecodeError as exc:raise ValueError(f'{label}: {exc}') from exc
        if not isinstance(data,dict):raise ValueError(f'{label}: нужен JSON-объект')
        return data

    theory=str(fields['theory']).upper()
    if theory not in ('KL','FSDT','REDDY','CUF'):
        raise ValueError('Теория: KL / FSDT / REDDY / CUF')
    case['theory']=theory
    if theory=='CUF':case['cuf_model']=str(fields['cuf_model']).upper()
    else:case.pop('cuf_model',None)
    method=str(fields['method']).upper()
    if method not in ('MVI','FDM'):raise ValueError('Метод: MVI / FDM')
    case['method']=method
    case.pop('fdm_grid',None);case.pop('mvi_grid',None);case.pop('grid',None)
    case['mvi_grid' if method=='MVI' else 'fdm_grid']={'nx':integer('nx'),'ns':integer('ns')}
    if method=='MVI':
        case['rank']=integer('rank')
        case['reference_mvi']={'max_sweeps':integer('max_sweeps')}
    else:
        case.pop('rank',None);case.pop('reference_mvi',None)
    case['scale_model']=str(fields['scale_model']).upper()
    for key in ('L','R','h'):case.setdefault('geometry',{})[key]=numeric(key)
    for key in ('E0','nu0'):case.setdefault('material',{})[key]=numeric(key)
    case['quadrature']={'type':'GAUSS','n_gauss':integer('n_gauss')}
    case['output_dir']=str(fields['output_dir']).strip()
    if case['scale_model']=='MICRO':
        micro=case.setdefault('micro',{})
        micro['gradient_model']='G3D'
        micro['enable_flexoelectric']=bool(fields['flexo'])
        micro['ell_m']=numeric('ell_m')
        if not micro['enable_flexoelectric']:
            # Keep the user's input for later re-enabling, while the physical
            # OFF case retains exactly zero coupling and no electric unknowns.
            micro['editor_saved_flexo']={key:numeric(key) for key in ('mu1','mu2','kappa')}
            micro['editor_saved_flexo']['potential_order']=integer('potential_order')
        for field,key in (('mu1','mu1_c_per_m'),('mu2','mu2_c_per_m'),
                          ('kappa','kappa_f_per_m')):
            micro[key]=numeric(field) if micro['enable_flexoelectric'] else 0.
        micro['potential_order']=integer('potential_order') if micro['enable_flexoelectric'] else 0
        if micro['enable_flexoelectric']:micro.pop('editor_saved_flexo',None)
        micro['rho_kg_m3']=numeric('rho_kg_m3')
        micro['eta_s']=numeric('eta_s')
    bc=document(boundary,'Граничные условия')
    electric_on=(case['scale_model']=='MICRO' and case['micro']['enable_flexoelectric'])
    if not electric_on:
        if bc.get('electrical'):
            case.setdefault('micro',{})['editor_saved_electrical_bc']=copy.deepcopy(bc['electrical'])
        bc['electrical']={}
    elif not bc.get('electrical'):
        saved_bc=case.get('micro',{}).get('editor_saved_electrical_bc')
        if saved_bc:bc['electrical']=copy.deepcopy(saved_bc)
    case.setdefault('boundary_conditions',{})['variational']=bc
    if 'reference_loads' in case:
        case['reference_loads']=document(load,'Нагрузка')
    else:
        case['load_definition']=document(load,'Нагрузка')
    case['reference_newton']=document(newton,'Настройки Ньютона')
    prepare_reference_case(case)
    return case


class ReferenceCaseEditor(tk.Toplevel):
    def __init__(self, parent, case=None, path=None):
        super().__init__(parent)
        self.parent=parent
        self.path=Path(path) if path else None
        self.case=copy.deepcopy(case) if case is not None else json.loads(TEMPLATE.read_text(encoding='utf-8'))
        self.title('Вариационный CASE: G3D / флексо / МВИ / МКР')
        self.geometry('930x780')
        self.transient(parent)
        self.fields={}
        outer=ttk.Frame(self,padding=10);outer.pack(fill='both',expand=True)
        canvas=tk.Canvas(outer,highlightthickness=0)
        scrollbar=ttk.Scrollbar(outer,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        body=ttk.Frame(canvas,padding=6)
        window=canvas.create_window((0,0),window=body,anchor='nw')
        body.bind('<Configure>',lambda _event:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda event:canvas.itemconfigure(window,width=event.width))
        ttk.Label(body,text='Статика · GREEN_LAGRANGE · HOOKE · СИ. Динамика через CASE пока не подключена.',
                  foreground='#825400').pack(anchor='w',pady=(0,7))
        def group(title,items):
            box=ttk.LabelFrame(body,text=title,padding=7);box.pack(fill='x',pady=4)
            for index,(label,key,value,choices) in enumerate(items):
                row,col=divmod(index,2)
                ttk.Label(box,text=label).grid(row=row,column=2*col,sticky='w',padx=(0,5),pady=3)
                var=tk.StringVar(value=str(value))
                self.fields[key]=var
                if choices:
                    widget=ttk.Combobox(box,textvariable=var,values=choices,state='readonly',width=15)
                else:widget=ttk.Entry(box,textvariable=var,width=17)
                widget.grid(row=row,column=2*col+1,sticky='ew',padx=(0,12),pady=3)
            box.columnconfigure(1,weight=1);box.columnconfigure(3,weight=1)
            return box
        c=self.case
        geo=c.get('geometry',{});mat=c.get('material',{});micro=c.get('micro',{})
        saved=micro.get('editor_saved_flexo',{})
        method=c.get('method','FDM');grid=c.get('mvi_grid' if method=='MVI' else 'fdm_grid',{})
        group('Теория и метод',[
            ('Теория','theory',c.get('theory','CUF'),('KL','FSDT','REDDY','CUF')),
            ('Порядок CUF','cuf_model',c.get('cuf_model','TE1'),('TE1','TE2','TE3','TE4')),
            ('Пространственный метод','method',method,('FDM','MVI')),
            ('Ранг МВИ (1–8)','rank',c.get('rank',1),None),
            ('Узлы nx','nx',grid.get('nx',4),None),('Узлы ns','ns',grid.get('ns',4),None),
            ('Циклы МВИ','max_sweeps',c.get('reference_mvi',{}).get('max_sweeps',30),None),
            ('Точек Гаусса','n_gauss',c.get('quadrature',{}).get('n_gauss',3),None),
        ])
        group('Геометрия и материал (СИ)',[
            ('Длина L, м','L',geo.get('L',''),None),('Радиус R, м','R',geo.get('R',''),None),
            ('Толщина h, м','h',geo.get('h',''),None),('Модуль E, Па','E0',mat.get('E0',''),None),
            ('Пуассон ν','nu0',mat.get('nu0',''),None),
            ('Масштаб','scale_model',c.get('scale_model','MICRO'),('MACRO','MICRO')),
        ])
        flexo=group('G3D и флексоэлектричество',[
            ('Длина ℓ, м','ell_m',micro.get('ell_m',0),None),
            ('Коэффициент μ₁, Кл/м','mu1',saved.get('mu1',micro.get('mu1_c_per_m',0)),None),
            ('Коэффициент μ₂, Кл/м','mu2',saved.get('mu2',micro.get('mu2_c_per_m',0)),None),
            ('Диэлектрик κ, Ф/м','kappa',saved.get('kappa',micro.get('kappa_f_per_m',0)),None),
            ('Порядок φ','potential_order',saved.get('potential_order',micro.get('potential_order',1)),None),
            ('Плотность, кг/м³','rho_kg_m3',micro.get('rho_kg_m3',1300),None),
            ('Вязкость η, с','eta_s',micro.get('eta_s',0),None),
        ])
        self.flexo=tk.BooleanVar(value=bool(micro.get('enable_flexoelectric',True)))
        ttk.Checkbutton(flexo,text='Флексоэффект ВКЛ',variable=self.flexo).grid(row=4,column=0,columnspan=2,sticky='w')
        group('Результаты', [('Папка результатов','output_dir',c.get('output_dir',''),None)])
        def json_box(title,obj):
            box=ttk.LabelFrame(body,text=title,padding=6);box.pack(fill='both',expand=True,pady=4)
            widget=tk.Text(box,height=9,wrap='none',font=('Consolas',10))
            widget.pack(fill='both',expand=True)
            widget.insert('1.0',json.dumps(obj,ensure_ascii=False,indent=2))
            return widget
        bc=copy.deepcopy(c.get('boundary_conditions',{}).get('variational',{}))
        if not micro.get('enable_flexoelectric',True) and micro.get('editor_saved_electrical_bc'):
            bc['electrical']=copy.deepcopy(micro['editor_saved_electrical_bc'])
        self.bc_text=json_box('Граничные условия: механические / G3D / электрические',bc)
        self.load_text=json_box('Нагрузка и коэффициенты шагов (JSON)',
                                c.get('reference_loads',c.get('load_definition',{})))
        self.newton_text=json_box('Параметры связанного Ньютона (JSON)',c.get('reference_newton',{}))
        buttons=ttk.Frame(self,padding=8);buttons.pack(fill='x')
        ttk.Button(buttons,text='Проверить',command=self.validate).pack(side='left',padx=4)
        ttk.Button(buttons,text='Сохранить CASE…',command=self.save).pack(side='left',padx=4)
        ttk.Button(buttons,text='Рассчитать',command=self.run).pack(side='left',padx=4)
        ttk.Button(buttons,text='МВИ 1–8',command=lambda:self.run(series=True)).pack(side='left',padx=4)
        ttk.Button(buttons,text='Закрыть',command=self.destroy).pack(side='right',padx=4)

    def collect(self):
        fields={key:value.get() for key,value in self.fields.items()}
        fields['flexo']=self.flexo.get()
        return edit_reference_case(self.case,fields,
            self.bc_text.get('1.0','end-1c'),self.load_text.get('1.0','end-1c'),
            self.newton_text.get('1.0','end-1c'))

    def validate(self):
        try:
            self.collect()
            messagebox.showinfo('Проверка CASE','Параметры, нагрузка и ГУ приняты проверочным решателем.',parent=self)
        except Exception as exc:messagebox.showerror('Ошибка CASE',str(exc),parent=self)

    def save(self):
        try:case=self.collect()
        except Exception as exc:
            messagebox.showerror('Ошибка CASE',str(exc),parent=self);return None
        path=filedialog.asksaveasfilename(parent=self,title='Сохранить вариационный CASE',
            initialfile=self.path.name if self.path else 'CASE_REFERENCE.json',
            initialdir=str(self.path.parent) if self.path else None,
            defaultextension='.json',filetypes=[('JSON','*.json')])
        if not path:return None
        target=Path(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(case,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        self.path=target;self.case=case
        self.parent._v('case_path').set(str(target))
        self.parent._log(f'Вариационный CASE сохранён: {target}\n')
        return target

    def run(self,series=False):
        path=self.save()
        if path is None:return
        try:self.parent._launch_reference_case(path,self.case,series=series)
        except Exception as exc:messagebox.showerror('Запуск CASE',str(exc),parent=self)
