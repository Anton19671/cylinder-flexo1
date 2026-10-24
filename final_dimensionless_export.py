#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dimensionless partner Excel export for CYLINDER SOLVER saved results.

Output-only. Reads saved JSON/CSV/NPZ files and never evaluates mechanics.
All normalization rules are written into every workbook in English and Russian.
"""
from __future__ import annotations

import csv, json, math
from pathlib import Path
import numpy as np

MAX_ROWS=900_000


def _json(path, default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:return default


def _summary(root:Path):
    for name in ('FINAL_RUN_SUMMARY.json','CASE_RUN_SUMMARY.json','RUN_RESULT.json','SERIES_SUMMARY.json'):
        j=_json(root/name,None)
        if isinstance(j,dict):return j
    return {}


def _case(root:Path):
    for p in (root/'INPUT'/'FINAL_CASE.json',root/'FINAL_CASE.json',root/'RUN_PARAMETERS.json'):
        j=_json(p,None)
        if isinstance(j,dict):return j
    return {}


def _refs(root:Path):
    sm=_summary(root); case=_case(root)
    g=(sm.get('geometry') or case.get('geometry') or {})
    m=(sm.get('material') or case.get('material') or {})
    L=float(g.get('L',1.0) or 1.0); R=float(g.get('R',1.0) or 1.0); h=float(g.get('h',1.0) or 1.0)
    E=float(m.get('E0',m.get('E',1.0)) or 1.0); nu=float(m.get('nu0',m.get('nu',0.3)) or 0.3)
    G=E/(2.0*(1.0+nu))
    return sm,case,{'L':L,'R':R,'h':h,'E':E,'G':G,'nu':nu}


def _states(root:Path):
    q=sorted(root.glob('STEP_*_STATE.npz'))
    return q if q else sorted(p for p in root.glob('STEP_*.npz') if 'STATE' not in p.name)


def _formats(wb):
    return {
        'title':wb.add_format({'bold':True,'font_size':14}),
        'head':wb.add_format({'bold':True,'bg_color':'#D9EAF7','border':1,'text_wrap':True}),
        'sub':wb.add_format({'bold':True,'bg_color':'#EDEDED'}),
        'num':wb.add_format({'num_format':'0.0000000000E+00'}),
        'wrap':wb.add_format({'text_wrap':True,'valign':'top'}),
    }


def _open_book(path:Path,title:str,refs:dict,sm:dict):
    try:import xlsxwriter
    except Exception as exc:raise RuntimeError('XlsxWriter is required for dimensionless Excel export') from exc
    path.parent.mkdir(parents=True,exist_ok=True)
    wb=xlsxwriter.Workbook(str(path),{'constant_memory':True})
    wb.set_properties({'title':title,'subject':'Dimensionless cylinder-shell result data','author':'Cylinder Solver','comments':'Output-only export from saved calculation results.'})
    f=_formats(wb); ws=wb.add_worksheet('Normalization_Нормировка')
    ws.write('A1',title,f['title'])
    rows=[
        ('Purpose / Назначение','Reference-scale dimensionless copy of saved physical results. / Безразмерная копия сохранённых физических результатов.'),
        ('x̄','x / L'),('s̄','s / (2πR) = θ/(2π)'),('z̄','z / h'),
        ('ū, v̄, w̄','u/h, v/h, w/h'),('ε̄','strain components are already dimensionless / деформации уже безразмерны'),
        ('σ̄, S̄','stress / E_ref'),('Ḡ_B','G_B / E_ref'),('G_B/G0','already dimensionless / уже безразмерно'),
        ('q̄ pressure / давление','q / E_ref'),('p̄ line load / погонная нагрузка','p / (E_ref h)'),
        ('N̄ membrane resultant / мембранное усилие','N / (E_ref h)'),('M̄ bending resultant / момент','M / (E_ref h²)'),
        ('CUF q̄τ','qτ · h^(τ−1), because [qτ]=L^(1−τ) / поскольку [qτ]=L^(1−τ)'),
        ('CUF Āτ','Aτ / (E_ref h^(τ+1))'),('CUF C̄τ','Cτ / (E_ref h^τ)'),
        ('Reference L / Опорная L',refs['L']),('Reference R / Опорный R',refs['R']),('Reference h / Опорная h',refs['h']),
        ('Reference E / Опорный E',refs['E']),('Reference G0 / Опорный G0',refs['G']),
        ('Theory / Теория',sm.get('theory')),('Geometry model / Геометрическая модель',sm.get('geometry_model')),('Material model / Модель материала',sm.get('material_model')),
    ]
    for r,(a,b) in enumerate(rows,2):ws.write(r,0,a,f['sub']);ws.write(r,1,b,f['wrap'])
    ws.set_column(0,0,40);ws.set_column(1,1,95);ws.freeze_panes(2,0)
    return wb,f


def _sheet(wb,f,name,headers,width=18):
    ws=wb.add_worksheet(name[:31])
    for c,h in enumerate(headers):ws.write(0,c,h,f['head']);ws.set_column(c,c,width)
    ws.freeze_panes(1,0)
    return ws


def _csv_rows(path:Path):
    if not path.exists():return []
    with path.open('r',encoding='utf-8-sig',newline='') as fh:return list(csv.DictReader(fh))


def _f(v,default=0.0):
    try:return float(v)
    except Exception:return default


def _write_load_deflection(root,target,refs,sm):
    rows=_csv_rows(root/'NUMERIC'/'LOAD_DEFLECTION.csv')
    if not rows:return None
    path=target/'01_Load_Deflection_Нагрузка_Прогиб_DIMLESS.xlsx';wb,f=_open_book(path,'Dimensionless Load–Deflection / Безразмерная нагрузка–прогиб',refs,sm)
    ws=_sheet(wb,f,'Load_Deflection_НагрПрог',['step / шаг [-]','w̄max=|w|max/h [-]','w̄ref=|wref|/h [-]','w̄center=wcenter/h [-]','|w̄center| [-]','θref [rad]','load factor λ [-]','q̄ or p̄ / безразм. нагрузка [-]'],22)
    for i,r in enumerate(rows,1):
        vals=list(r.values()); load=_f(vals[-1]) if vals else 0.0
        last_header=list(r.keys())[-1].lower()
        if any(x in last_header for x in ('n/mm²','n/mm^2','mpa','pa','pressure','q [')): lbar=load/refs['E']
        elif any(x in last_header for x in ('n/mm','line')): lbar=load/(refs['E']*refs['h'])
        else:lbar=load
        out=[_f(vals[0]),_f(vals[1])/refs['h'],_f(vals[2])/refs['h'],_f(vals[3])/refs['h'],_f(vals[4])/refs['h'],_f(vals[5]),_f(vals[6]),lbar]
        ws.write_row(i,0,out,f['num'])
    wb.close();return path


def _last_state(root):
    st=_states(root)
    if not st:return None,None
    p=st[-1]
    with np.load(p,allow_pickle=False) as z:return p,{k:np.asarray(z[k]) for k in z.files}


def _mid_displacements(st):
    if st is None:return None
    if 'q' in st:
        q=np.asarray(st['q'],float);return q[0,0],q[1,0],q[2,0]
    if 'U' in st:
        U=np.asarray(st['U'],float);return U[0],U[1],U[2]
    if {'X','Y'}.issubset(st):
        X=np.asarray(st['X'],float);Y=np.asarray(st['Y'],float)
        if X.ndim==4:
            q=np.einsum('cmrx,cmrs->cmsx',X,Y);return q[0,0],q[1,0],q[2,0]
        if X.ndim==3:
            U=np.einsum('crx,crs->csx',X,Y);return U[0],U[1],U[2]
    return None


def _write_surface(root,target,refs,sm):
    _,st=_last_state(root); disp=_mid_displacements(st)
    if disp is None:return None
    u,v,w=(np.asarray(x,float) for x in disp);ns,nx=w.shape
    path=target/'02_Surface_Fields_Поверхностные_Поля_DIMLESS.xlsx';wb,f=_open_book(path,'Dimensionless Midsurface Fields / Безразмерные поля средней поверхности',refs,sm)
    ws=_sheet(wb,f,'Surface_Fields_Поля',['is [-]','ix [-]','x̄=x/L [-]','s̄=s/(2πR) [-]','θ [rad]','ū=u/h [-]','v̄=v/h [-]','w̄=w/h [-]'],18)
    rr=1
    for j in range(ns):
        th=2*math.pi*j/ns
        for i in range(nx):
            ws.write_row(rr,0,[j,i,i/max(nx-1,1),j/ns,th,float(u[j,i])/refs['h'],float(v[j,i])/refs['h'],float(w[j,i])/refs['h']],f['num']);rr+=1
    wb.close();return path


def _derived(root):
    p=root/'NUMERIC'/'DERIVED_FINAL_FIELDS.npz'
    if not p.exists():return None
    with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}


def _field_components(d,mode):
    all_fields=[]
    def add(name,a,kind):
        a=np.asarray(a)
        if a.ndim==3:all_fields.append((name,a,kind))
    if 'epsilon_i' in d:add('epsilon_i',d['epsilon_i'],'strain')
    if 'sigma_i' in d:add('sigma_i',d['sigma_i'],'stress')
    for name in ('detF','GB_ratio','birger_nonlinear_mask','plastic_mask'):
        if name in d:add(name,d[name],'dimensionless')
    if 'GB' in d:add('GB',d['GB'],'stress')
    if 'E' in d and d['E'].ndim==4:
        for c,n in enumerate(('E_xx','E_ss','E_rr','E_xs','E_xr','E_sr')):add(n,d['E'][:,c],'strain')
    if 'S' in d and d['S'].ndim==4:
        for c,n in enumerate(('S_xx','S_ss','S_rr','S_xs','S_xr','S_sr')):add(n,d['S'][:,c],'stress')
    if 'strain' in d and d['strain'].ndim==4:
        for c,n in enumerate(('eps_xx','eps_ss','eps_zz','gamma_xs')):add(n,d['strain'][:,c],'strain')
    if 'stress' in d and d['stress'].ndim==4:
        for c,n in enumerate(('sig_xx','sig_ss','tau_xs')):add(n,d['stress'][:,c],'stress')
    if str(mode).upper()=='STANDARD':
        priority={'epsilon_i','sigma_i','detF','GB_ratio','GB','E_xx','S_xx','eps_xx','sig_xx','plastic_mask','birger_nonlinear_mask'}
        all_fields=[q for q in all_fields if q[0] in priority]
    return all_fields


def _write_thickness(root,target,refs,sm,mode):
    d=_derived(root)
    if not d or 'z' not in d:return None
    fields=_field_components(d,mode)
    if not fields:return None
    z=np.asarray(d['z'],float)
    path=target/'03_Thickness_Fields_Поля_По_Толщине_DIMLESS.xlsx';wb,f=_open_book(path,'Dimensionless Through-thickness Fields / Безразмерные поля по толщине',refs,sm)
    for name,a,kind in fields:
        ns,nx=a.shape[1],a.shape[2]
        ws=_sheet(wb,f,(name+'_bar')[:31],['g [-]','is [-]','ix [-]','z̄=z/h [-]','x̄=x/L [-]','s̄=s/(2πR) [-]',f'{name} normalized / нормированное [-]'],18)
        rr=1
        for g in range(a.shape[0]):
            for j in range(ns):
                for i in range(nx):
                    val=float(a[g,j,i]); val=val/refs['E'] if kind=='stress' else val
                    ws.write_row(rr,0,[g,j,i,float(z[g])/refs['h'],i/max(nx-1,1),j/ns,val],f['num']);rr+=1
                    if rr>=MAX_ROWS:break
                if rr>=MAX_ROWS:break
            if rr>=MAX_ROWS:break
    wb.close();return path


def _write_resultants(root,target,refs,sm):
    d=_derived(root)
    if not d:return None
    keys=[k for k in d if k.startswith('resultant_')]
    if not keys:return None
    path=target/'04_Resultants_Результанты_DIMLESS.xlsx';wb,f=_open_book(path,'Dimensionless Resultants / Безразмерные результанты',refs,sm)
    for key in keys:
        a=np.asarray(d[key],float); name=key[len('resultant_'):]
        if a.ndim==2:a=a[None,...]
        if a.ndim!=3:continue
        ws=_sheet(wb,f,(name+'_bar')[:31],['τ / family index [-]','is [-]','ix [-]','x̄ [-]','s̄ [-]',f'{name} normalized / нормированное [-]','normalization / нормировка'],19)
        rr=1
        for tau in range(a.shape[0]):
            if name in ('Nx','Ns','Nxs'):den=refs['E']*refs['h'];formula='E_ref h'
            elif name in ('Mx','Ms','Mxs'):den=refs['E']*refs['h']**2;formula='E_ref h²'
            elif name.startswith(('A','B')):den=refs['E']*refs['h']**(tau+1);formula=f'E_ref h^{tau+1}'
            elif name.startswith('C'):den=refs['E']*refs['h']**tau;formula=f'E_ref h^{tau}'
            else:den=refs['E']*refs['h']**(tau+1);formula=f'E_ref h^{tau+1}'
            ns,nx=a.shape[1],a.shape[2]
            for j in range(ns):
                for i in range(nx):
                    ws.write_row(rr,0,[tau,j,i,i/max(nx-1,1),j/ns,float(a[tau,j,i])/den,formula],f['num']);rr+=1
    wb.close();return path


def _write_convergence(root,target,refs,sm):
    rows=_csv_rows(root/'NUMERIC'/'ONLINE_HISTORY.csv')
    if not rows:return None
    path=target/'05_Convergence_Сходимость_DIMLESS.xlsx';wb,f=_open_book(path,'Dimensionless Solver Convergence / Безразмерная сходимость решателя',refs,sm)
    keys=list(rows[0].keys()); keep=[k for k in keys if any(t in k.lower() for t in ('event','step','iteration','cycle','relative','eta_','projected','status','converg','outer'))]
    ws=_sheet(wb,f,'Convergence_Сходимость',[f'{k} [-]' for k in keep],20)
    for rix,r in enumerate(rows,1):
        vals=[]
        for k in keep:
            v=r.get(k,'')
            try:v=float(v)
            except Exception:pass
            vals.append(v)
        ws.write_row(rix,0,vals)
    wb.close();return path


def _write_control_points(root,target,refs,sm):
    report=_json(root/'REPORT'/'CALCULATION_SUMMARY.json',{}) or {}
    rows=report.get('control_points') or []
    if not rows:return None
    path=target/'06_Control_Points_Контрольные_Точки_DIMLESS.xlsx';wb,f=_open_book(path,'Dimensionless Control Points / Безразмерные контрольные точки',refs,sm)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    ws=_sheet(wb,f,'Control_Points_Точки',keys,20)
    for rr,r in enumerate(rows,1):
        out=[]
        for k in keys:
            v=r.get(k)
            lk=k.lower()
            try:
                x=float(v)
                if lk in ('x','x_coord','reference_x'):x/=refs['L']
                elif lk in ('s','s_coord','reference_s'):x/=(2*math.pi*refs['R'])
                elif lk=='z':x/=refs['h']
                elif lk.startswith(('u','v','w')):x/=refs['h']
                elif 'sigma' in lk or 'stress' in lk or lk.startswith(('s_','gb')):x/=refs['E']
                v=x
            except Exception:pass
            out.append(v)
        ws.write_row(rr,0,out)
    wb.close();return path


def _write_cuf_coeff(root,target,refs,sm):
    if str(sm.get('theory','')).upper()!='CUF':return None
    _,st=_last_state(root)
    if st is None:return None
    if 'q' in st:q=np.asarray(st['q'],float)
    elif {'X','Y'}.issubset(st):
        X=np.asarray(st['X'],float);Y=np.asarray(st['Y'],float)
        if X.ndim!=4:return None
        q=np.einsum('cmrx,cmrs->cmsx',X,Y)
    else:return None
    if q.ndim!=4:return None
    path=target/'07_CUF_Coefficients_Коэффициенты_CUF_DIMLESS.xlsx';wb,f=_open_book(path,'Dimensionless CUF Coefficients / Безразмерные коэффициенты CUF',refs,sm)
    ws=_sheet(wb,f,'CUF_Coefficients_Коэфф',['component / компонента','τ [-]','is [-]','ix [-]','x̄ [-]','s̄ [-]','q̄τ=qτ h^(τ−1) [-]'],20)
    rr=1; comps=('u','v','w')
    for c in range(min(3,q.shape[0])):
        for tau in range(q.shape[1]):
            scale=refs['h']**(tau-1)
            for j in range(q.shape[2]):
                for i in range(q.shape[3]):
                    ws.write_row(rr,0,[comps[c],tau,j,i,i/max(q.shape[3]-1,1),j/q.shape[2],float(q[c,tau,j,i])*scale],f['num']);rr+=1
    wb.close();return path


def export_dimensionless_excels(output_dir:str|Path,target_dir:str|Path,mode='FULL'):
    root=Path(output_dir).resolve(); target=Path(target_dir).resolve();target.mkdir(parents=True,exist_ok=True)
    sm,_case,refs=_refs(root);mode=str(mode).upper()
    funcs=[_write_load_deflection]
    if mode!='FAST':funcs += [_write_surface]
    out={}
    for fn in funcs:
        p=fn(root,target,refs,sm)
        if p:out[p.stem]=p
    if mode!='FAST':
        for fn,args in ((_write_thickness,(mode,)),(_write_resultants,()),(_write_convergence,()),(_write_control_points,()),(_write_cuf_coeff,())):
            p=fn(root,target,refs,sm,*args) if fn is _write_thickness else fn(root,target,refs,sm)
            if p:out[p.stem]=p
    return out

if __name__=='__main__':
    import sys
    if len(sys.argv)<2:raise SystemExit('Usage: final_dimensionless_export.py RUN_DIR [TARGET_DIR] [FAST|STANDARD|FULL]')
    root=Path(sys.argv[1]);target=Path(sys.argv[2]) if len(sys.argv)>2 else root/'DIMENSIONLESS_EXCEL_БЕЗРАЗМЕРНЫЕ_EXCEL';mode=sys.argv[3] if len(sys.argv)>3 else 'FULL'
    r=export_dimensionless_excels(root,target,mode);print('DIMENSIONLESS EXCEL PASS:',len(r))
