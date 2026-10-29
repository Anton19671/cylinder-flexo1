#!/usr/bin/env python3
"""Tk editor for the explicit small-grid variational CASE (no legacy conversion)."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from cylinder_reference_case import BACKEND, prepare_reference_case
from cylinder_time_integrator_gui import TIME_INTEGRATOR_CHOICES
from cylinder_thickness_locking_diagnostic import thickness_locking_diagnostic
from cylinder_director_rotation_audit import director_rotation_diagnostic
from cylinder_boundary_ui_model import boundary_ui_model

TEMPLATE=Path(__file__).resolve().parent.parent/'config/CASE_REFERENCE_CUF_G3D_STATIC.json'
DYNAMIC_TEMPLATE=Path(__file__).resolve().parent.parent/'config/CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json'


def edit_reference_case(original, fields, boundary, load, newton, dynamics=None, *, base_dir=None):
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
    old_method=str(original.get('method','')).upper()
    old_grid=original.get('mvi_grid' if old_method=='MVI' else 'fdm_grid',
                          original.get('grid',{}))
    if not isinstance(old_grid,dict):raise ValueError('Сетка должна быть JSON-объектом')
    grid=copy.deepcopy(old_grid)
    if method!=old_method and (str(original.get('fdm_backend','DENSE_REFERENCE')).upper()
                               =='MATRIX_FREE_GMRES' or
                               str(grid.get('operator_scheme','')).upper()=='SBP21_PERIODIC_ODD_CSR'):
        raise ValueError('Для смены метода в matrix-free CASE сначала создайте отдельный CASE')
    scheme=str(fields.get('operator_scheme',grid.get('operator_scheme','LEGACY_REFERENCE'))).upper()
    if scheme not in ('LEGACY_REFERENCE','SBP21_PERIODIC_ODD','SBP21_PERIODIC_ODD_CSR','SBP21_FOURIER_ODD'):
        raise ValueError('Неизвестная схема пространственных операторов')
    fdm_backend=str(fields.get('fdm_backend',case.get('fdm_backend','DENSE_REFERENCE'))).upper()
    if fdm_backend not in ('DENSE_REFERENCE','MATRIX_FREE_GMRES'):
        raise ValueError('Неизвестный вариант МКР')
    case['fdm_backend']=fdm_backend
    case['method']=method
    kind=str(fields.get('calculation_kind',case.get('calculation_kind','STATIC'))).upper()
    if kind not in ('STATIC','DYNAMIC'):raise ValueError('Тип задачи: STATIC / DYNAMIC')
    case['calculation_kind']=kind
    case.pop('fdm_grid',None);case.pop('mvi_grid',None);case.pop('grid',None)
    grid.update(nx=integer('nx'),ns=integer('ns'),operator_scheme=scheme)
    case['mvi_grid' if method=='MVI' else 'fdm_grid']=grid
    if method=='MVI':
        case['rank']=integer('rank')
        case.setdefault('reference_mvi',{})['max_sweeps']=integer('max_sweeps')
    else:
        case.pop('rank',None);case.pop('reference_mvi',None)
    case['scale_model']=str(fields['scale_model']).upper()
    for key in ('L','R','h'):case.setdefault('geometry',{})[key]=numeric(key)
    for key in ('E0','nu0'):case.setdefault('material',{})[key]=numeric(key)
    case['quadrature']={'type':'GAUSS','n_gauss':integer('n_gauss')}
    case['output_dir']=str(fields['output_dir']).strip()
    if case['scale_model']=='MICRO':
        micro=case.setdefault('micro',{})
        model=str(fields.get('gradient_model',micro.get('gradient_model','G3D'))).strip().upper()
        if model not in ('G3D','GZ'):
            raise ValueError('Градиентная теория: G3D (18) или Gz (6)')
        micro['gradient_model']='G3D' if model=='G3D' else 'Gz'
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
    load_data=document(load,'Нагрузка')
    if kind=='DYNAMIC':
        factors=load_data.pop('load_factors',None)
        if factors is not None:case['editor_saved_static_load_factors']=factors
        raw=dynamics if dynamics is not None else json.dumps(case.get('dynamics',case.get('editor_saved_dynamics',{})))
        case['dynamics']=document(raw,'Динамика')
    else:
        if 'load_factors' not in load_data:
            load_data['load_factors']=copy.deepcopy(case.get('editor_saved_static_load_factors',[1.0]))
        if isinstance(case.get('dynamics'),dict):case['editor_saved_dynamics']=copy.deepcopy(case['dynamics'])
        case.pop('dynamics',None)
    if 'reference_loads' in case:case['reference_loads']=load_data
    else:case['load_definition']=load_data
    case['reference_newton']=document(newton,'Настройки Ньютона')
    prepare_reference_case(case,base_dir=base_dir)
    return case


def reference_locking_message(case):
    """Display both bending and finite-rotation limitations without changing physics."""
    theory=str(case.get('theory','')).upper()
    if theory=='CUF':theory='CUF_'+str(case.get('cuf_model','')).upper()
    info=thickness_locking_diagnostic(theory,case['material']['nu0'])
    warnings=[]
    if info['thickness_locking']=='YES':
        warnings.append('Полная 3D-упругость и данная кинематика могут завышать '
            'энергию чистого изгиба: коэффициент плоского предела '
            f"{info['flat_limit_pure_bending_energy_ratio']:.5g}. "
            'Автоматическая редукция материала не выполняется.')
    if theory in ('FSDT','REDDY'):
        warnings.append('V-4: линейный директор НЕ инвариантен к конечным '
            'поворотам; Ezz=0.5*(theta_x**2+theta_s**2). '
            '0.1 рад — ориентир для предупреждения, а не гарантия точности.')
    return ' '.join(warnings) if warnings else None


class ReferenceCaseEditor(tk.Toplevel):
    def __init__(self, parent, case=None, path=None):
        super().__init__(parent)
        self.parent=parent
        self.path=Path(path) if path else None
        if case is not None:self.case=copy.deepcopy(case)
        else:
            dynamic=str(parent._v('calculation_kind','STATIC').get()).upper()=='DYNAMIC'
            self.case=json.loads((DYNAMIC_TEMPLATE if dynamic else TEMPLATE).read_text(encoding='utf-8'))
        self.title('Вариационный CASE: G3D / Gz / флексо / МВИ / МКР')
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
        ttk.Label(body,text='GREEN_LAGRANGE · HOOKE · СИ. Динамика: Ньюмарк или экспериментальный IRK4 (только МВИ).',
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
        ttk.Label(body,text='Интегратор времени (выбор не изменяет физический CASE)').pack(anchor='w')
        self.time_integrator=self.parent.time_integrator
        ttk.Combobox(body,textvariable=self.time_integrator,
                     values=TIME_INTEGRATOR_CHOICES,state='readonly',
                     width=42).pack(anchor='w',pady=(1,7))
        geo=c.get('geometry',{});mat=c.get('material',{});micro=c.get('micro',{})
        saved=micro.get('editor_saved_flexo',{})
        method=c.get('method','FDM');grid=c.get('mvi_grid' if method=='MVI' else 'fdm_grid',{})
        group('Теория и метод',[
            ('Тип задачи','calculation_kind',c.get('calculation_kind','STATIC'),('STATIC','DYNAMIC')),
            ('Теория','theory',c.get('theory','CUF'),('KL','FSDT','REDDY','CUF')),
            ('Порядок CUF','cuf_model',c.get('cuf_model','TE1'),('TE1','TE2','TE3','TE4')),
            ('Пространственный метод','method',method,('FDM','MVI')),
            ('Схема операторов','operator_scheme',grid.get('operator_scheme','LEGACY_REFERENCE'),
             ('LEGACY_REFERENCE','SBP21_PERIODIC_ODD','SBP21_PERIODIC_ODD_CSR','SBP21_FOURIER_ODD')),
            ('Вариант МКР','fdm_backend',c.get('fdm_backend','DENSE_REFERENCE'),
             ('DENSE_REFERENCE','MATRIX_FREE_GMRES')),
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
        flexo=group('Градиентная теория G3D/Gz и флексоэлектричество',[
            ('Теория градиентов','gradient_model',
             'Gz' if str(micro.get('gradient_model','G3D')).upper()=='GZ' else 'G3D',
             ('G3D','Gz')),
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
        self.bc_text=json_box('Граничные условия: механические / градиентные / электрические',bc)
        ttk.Button(body,text='Краевые условия по выбранной теории…',
                   command=self.open_theory_boundary_editor).pack(anchor='w',pady=4)
        self.bc_theory_info=ttk.Label(body,text='',wraplength=840)
        self.bc_theory_info.pack(anchor='w')
        for key in ('theory','cuf_model','scale_model'):
            self.fields[key].trace_add('write',lambda *_:self._refresh_boundary_theory())
        self.flexo.trace_add('write',lambda *_:self._refresh_boundary_theory())
        self._refresh_boundary_theory()
        self.load_text=json_box('Нагрузка и коэффициенты шагов (JSON)',
                                c.get('reference_loads',c.get('load_definition',{})))
        self.newton_text=json_box('Параметры связанного Ньютона (JSON)',c.get('reference_newton',{}))
        self.dynamics_text=json_box('Динамика Ньюмарка и закон F(t) (JSON)',
            c.get('dynamics',c.get('editor_saved_dynamics',{
                'time_step_s':0.01,'n_steps':20,'density_kg_m3':1300.0,
                'newmark_beta':0.25,'newmark_gamma':0.5,'start_time_s':0.0,
                'damping':'NONE','initial_conditions':{'type':'ZERO'},
                'time_law':{'kind':'HARMONIC','amplitude':1.0,'frequency_hz':1.0,
                            'phase_rad':0.0,'start_time_s':0.0}})))
        buttons=ttk.Frame(self,padding=8);buttons.pack(fill='x')
        ttk.Button(buttons,text='Проверить',command=self.validate).pack(side='left',padx=4)
        ttk.Button(buttons,text='Сохранить CASE…',command=self.save).pack(side='left',padx=4)
        ttk.Button(buttons,text='Рассчитать',command=self.run).pack(side='left',padx=4)
        ttk.Button(buttons,text='МВИ 1–8',command=lambda:self.run(series=True)).pack(side='left',padx=4)
        ttk.Button(buttons,text='Закрыть',command=self.destroy).pack(side='right',padx=4)

    def _current_boundary_model(self):
        fields={key:var.get() for key,var in self.fields.items()}
        return boundary_ui_model({
            'theory':fields['theory'],'cuf_model':fields['cuf_model'],
            'scale_model':fields['scale_model'],
            'micro':{'enable_flexoelectric':bool(self.flexo.get())}})

    def _refresh_boundary_theory(self):
        try:
            info=self._current_boundary_model()
            modes=' / '.join(info['electric_modes'])
            self.bc_theory_info.configure(
                text=f"{info['theory']}: {info['n_components']} механических компонент "
                     f"({', '.join(info['components'])}); высшие следы: "
                     f"{', '.join(info['higher_choices'])}; электрические ГУ: {modes}")
        except Exception as exc:
            self.bc_theory_info.configure(text=f"Проверьте теорию: {exc}")

    def open_theory_boundary_editor(self):
        """Live theory-dependent BC controls, synchronized into this CASE JSON."""
        try:
            info=self._current_boundary_model()
            spec=json.loads(self.bc_text.get('1.0','end-1c'))
            if not isinstance(spec,dict):raise ValueError('Нужен JSON-объект ГУ')
        except Exception as exc:
            messagebox.showerror('Краевые условия',str(exc),parent=self);return
        from cylinder_variational_boundary import (
            MECHANICAL_SUPPORTS,validate_boundary_passport,_fixed_components)
        win=tk.Toplevel(self);win.title('ГУ: '+info['theory'])
        win.geometry('900x720');win.transient(self)
        outer=ttk.Frame(win,padding=10);outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='Теория '+info['theory']+' — '+', '.join(info['components']),
                  wraplength=830).pack(anchor='w',pady=5)
        tab=ttk.Notebook(outer);tab.pack(fill='both',expand=True)
        mechanical=ttk.Frame(tab,padding=8);electric=ttk.Frame(tab,padding=8)
        tab.add(mechanical,text='Механические и высшие ГУ')
        tab.add(electric,text='Электрические ГУ')
        mechanical.columnconfigure(0,weight=1)
        controls={}
        for rowno,side in enumerate(('left','right')):
            saved=spec.get('mechanical',{}).get(side,{})
            group=ttk.LabelFrame(mechanical,text='x=0' if side=='left' else 'x=L',padding=8)
            group.grid(row=rowno,column=0,sticky='ew',pady=6)
            support=tk.StringVar(value=saved.get('support','FREE'))
            higher=tk.StringVar(value=saved.get('higher','NATURAL'))
            ttk.Label(group,text='Закрепление').grid(row=0,column=0,sticky='w')
            ttk.Combobox(group,textvariable=support,values=MECHANICAL_SUPPORTS,
                         state='readonly').grid(row=0,column=1,sticky='ew')
            ttk.Label(group,text='Градиентные следы').grid(row=1,column=0,sticky='w')
            ttk.Combobox(group,textvariable=higher,values=info['higher_choices'],
                         state='readonly',width=32).grid(row=1,column=1,sticky='ew')
            caption=ttk.Label(group,text='',wraplength=720)
            caption.grid(row=2,column=0,columnspan=2,sticky='w',pady=4)
            custom=tk.StringVar(value=','.join(str(i) for i in saved.get('fixed_components',[])))
            ttk.Label(group,text='CUSTOM: индексы компонент').grid(row=3,column=0,sticky='w')
            ttk.Entry(group,textvariable=custom).grid(row=3,column=1,sticky='ew')
            def update_label(*_,support=support,caption=caption,custom=custom):
                try:
                    mode=support.get()
                    if mode=='CUSTOM':
                        caption.configure(text='Выберите индексы из: '+
                                          ', '.join(f'{i}={v}' for i,v in enumerate(info['components'])))
                    else:
                        names=info['support_components'][mode]
                        caption.configure(text='Фиксируются: '+(', '.join(names) if names else 'нет'))
                except Exception as exc:caption.configure(text=str(exc))
            support.trace_add('write',update_label);update_label()
            controls[side]=(support,higher,custom)
        econtrols={}
        ttk.Label(electric,text='Электрическая часть выключена' if not info['electric_enabled']
                  else 'POTENTIAL — В; CHARGE — Кл/м²; INSULATED — изолировано').grid(
                      row=0,column=0,columnspan=3,sticky='w',pady=8)
        for j,side in enumerate(('left','right','outer','inner'),start=1):
            previous=spec.get('electrical',{}).get(side,{})
            mode=tk.StringVar(value=previous.get('mode','INSULATED') if info['electric_enabled'] else 'INSULATED')
            val=tk.StringVar(value=str(previous.get('value','')))
            ttk.Label(electric,text=side).grid(row=j,column=0,sticky='w')
            sel=ttk.Combobox(electric,textvariable=mode,values=info['electric_modes'],state='readonly')
            sel.grid(row=j,column=1,padx=5,pady=5)
            entry=ttk.Entry(electric,textvariable=val)
            entry.grid(row=j,column=2,padx=5,pady=5)
            def refresh(*_,entry=entry,mode=mode):
                entry.configure(state='normal' if info['electric_enabled']
                                and mode.get()!='INSULATED' else 'disabled')
            mode.trace_add('write',refresh);refresh()
            econtrols[side]=(mode,val)
        def apply():
            from copy import deepcopy
            try:
                new=deepcopy(spec)
                mech=new.setdefault('mechanical',{})
                for side,(support,higher,custom) in controls.items():
                    obj=mech.setdefault(side,{})
                    obj['support']=support.get();obj['higher']=higher.get()
                    if support.get()=='CUSTOM':
                        obj['fixed_components']=[int(x.strip()) for x in custom.get().split(',') if x.strip()]
                    else:obj.pop('fixed_components',None)
                electrical=new.setdefault('electrical',{})
                for side,(mode,val) in econtrols.items():
                    obj=electrical.setdefault(side,{})
                    obj['mode']=mode.get()
                    if obj['mode']=='INSULATED':
                        obj.pop('value',None);obj.pop('time_law',None)
                    else:obj['value']=float(val.get().replace(',','.'))
                validate_boundary_passport(new,electrical_enabled=info['electric_enabled'],
                    calculation_kind=self.fields['calculation_kind'].get(),
                    theory=info['theory'])
                self.bc_text.delete('1.0','end')
                self.bc_text.insert('1.0',json.dumps(new,ensure_ascii=False,indent=2))
                win.destroy()
            except Exception as exc:
                messagebox.showerror('Некорректные краевые условия',str(exc),parent=win)
        actions=ttk.Frame(outer);actions.pack(fill='x',pady=6)
        ttk.Button(actions,text='Применить к CASE',command=apply).pack(side='left')
        ttk.Button(actions,text='Отмена',command=win.destroy).pack(side='right')

    def collect(self, *, base_dir=None):
        fields={key:value.get() for key,value in self.fields.items()}
        fields['flexo']=self.flexo.get()
        return edit_reference_case(self.case,fields,
            self.bc_text.get('1.0','end-1c'),self.load_text.get('1.0','end-1c'),
            self.newton_text.get('1.0','end-1c'),self.dynamics_text.get('1.0','end-1c'),
            base_dir=self.path.parent if base_dir is None and self.path else base_dir)

    def validate(self):
        try:
            case=self.collect()
            warning=reference_locking_message(case)
            if warning:messagebox.showwarning('Ограничение изгибной модели',warning,parent=self)
            else:messagebox.showinfo('Проверка CASE','Параметры, нагрузка и ГУ приняты проверочным решателем.',parent=self)
        except Exception as exc:messagebox.showerror('Ошибка CASE',str(exc),parent=self)

    def save(self):
        path=filedialog.asksaveasfilename(parent=self,title='Сохранить вариационный CASE',
            initialfile=self.path.name if self.path else 'CASE_REFERENCE.json',
            initialdir=str(self.path.parent) if self.path else None,
            defaultextension='.json',filetypes=[('JSON','*.json')])
        if not path:return None
        target=Path(path)
        try:case=self.collect(base_dir=target.parent)
        except Exception as exc:
            messagebox.showerror('Ошибка CASE',str(exc),parent=self);return None
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(case,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        self.path=target;self.case=case
        self.parent.adopt(case,target)
        self.parent._log(f'Вариационный CASE сохранён: {target}\n')
        return target

    def run(self,series=False):
        path=self.save()
        if path is None:return
        warning=reference_locking_message(self.case)
        if warning:messagebox.showwarning('Ограничение изгибной модели',warning,parent=self)
        try:self.parent._launch_reference_case(path,self.case,series=series)
        except Exception as exc:messagebox.showerror('Запуск CASE',str(exc),parent=self)

