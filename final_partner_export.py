#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a clean bilingual partner-facing results package for one saved RUN.

Output-only. Internal solver folder names remain untouched for GUI compatibility.
The partner package contains only calculation results and neutral engineering documentation.
"""
from __future__ import annotations

import json, shutil
from pathlib import Path

PARTNER_ROOT='PARTNER_RESULTS_РЕЗУЛЬТАТЫ_ДЛЯ_ПАРТНЕРОВ'

FOLDERS={
 'readme':'00_README_ИНСТРУКЦИЯ',
 'summary':'01_ENGINEERING_SUMMARY_ИНЖЕНЕРНАЯ_СВОДКА',
 'excel_dim':'02_EXCEL_DIMENSIONAL_РАЗМЕРНЫЕ',
 'excel_nd':'03_EXCEL_DIMENSIONLESS_БЕЗРАЗМЕРНЫЕ',
 'g2d':'04_GRAPHS_2D_ГРАФИКИ_2D',
 'g3d':'05_GRAPHICS_3D_ГРАФИКА_3D',
 'sections':'06_SECTIONS_СЕЧЕНИЯ',
 'thickness':'07_THROUGH_THICKNESS_ПО_ТОЛЩИНЕ',
 'conv':'08_CONVERGENCE_СХОДИМОСТЬ',
 'plastic':'09_PLASTICITY_ПЛАСТИЧНОСТЬ',
 'numeric':'10_NUMERICAL_DATA_ЧИСЛЕННЫЕ_ДАННЫЕ',
 'vtk':'11_VTK_PARAVIEW_ВИЗУАЛИЗАЦИЯ',
 'video':'12_VIDEO_ANIMATIONS_ВИДЕО_АНИМАЦИИ',
 'inputs':'13_INPUT_PARAMETERS_ИСХОДНЫЕ_ПАРАМЕТРЫ',
}

EXCEL_NAMES={
 'FULL_RESULTS':'00_Full_Results_Полные_Результаты.xlsx',
 'LOAD_DEFLECTION':'01_Load_Deflection_Нагрузка_Прогиб.xlsx',
 'DISPLACEMENTS':'02_Displacements_Перемещения.xlsx',
 'STRAINS_STRESSES':'03_Strains_Stresses_Деформации_Напряжения.xlsx',
 'RESULTANTS':'04_Resultants_Результанты.xlsx',
 'MOMENTS':'05_Moments_Моменты.xlsx',
 'CONVERGENCE':'06_Convergence_Сходимость.xlsx',
 'CONTROL_POINTS':'07_Control_Points_Контрольные_Точки.xlsx',
 'CUF_COEFFICIENTS':'08_CUF_Coefficients_Коэффициенты_CUF.xlsx',
 'THEORY_COMPARISON':'09_Theory_Comparison_Сравнение_Теорий.xlsx',
}

GRAPH_DIRS={
 'LOAD_DEFLECTION':'01_LOAD_DEFLECTION_НАГРУЗКА_ПРОГИБ',
 'DISPLACEMENTS':'02_DISPLACEMENTS_ПЕРЕМЕЩЕНИЯ',
 'STRAINS':'03_STRAINS_ДЕФОРМАЦИИ',
 'STRESSES':'04_STRESSES_НАПРЯЖЕНИЯ',
 'RESULTANTS':'05_RESULTANTS_РЕЗУЛЬТАНТЫ',
 'CONTROL_POINTS':'06_CONTROL_POINTS_КОНТРОЛЬНЫЕ_ТОЧКИ',
 'CUF_DIAGNOSTICS':'07_CUF_DIAGNOSTICS_ДИАГНОСТИКА_CUF',
 'LOAD_BC':'08_LOAD_AND_BC_НАГРУЗКА_И_ГУ',
}

GRAPH3D_DIRS={
 'DEFORMED_SHELL':'01_DEFORMED_SHELL_ДЕФОРМИРОВАННАЯ_ОБОЛОЧКА',
 'STRESS_FIELDS':'02_STRESS_STRAIN_FIELDS_ПОЛЯ_НАПРЯЖЕНИЙ_ДЕФОРМАЦИЙ',
 'BIRGER_ZONES':'03_BIRGER_ZONES_ЗОНЫ_БИРГЕРА',
 'LOAD_AND_BC':'04_LOAD_AND_BC_НАГРУЗКА_И_ГУ',
 'CUTS':'05_CUTS_РАЗРЕЗЫ',
}

GRAPH_STEMS={
 'LOAD_DEFLECTION':'Load_Deflection_Нагрузка_Прогиб',
 'LOAD_WMAX_DIAGNOSTIC':'Maximum_Deflection_Diagnostic_Диагностика_Max_Прогиба',
 'LOAD_COMPARE_CENTER_WMAX':'Center_vs_Max_Deflection_Центр_и_Max_Прогиб',
 'FINAL_DEFORMED_SHELL':'Deformed_Shell_Деформированная_Оболочка',
 'FINAL_LOAD_AND_BC_3D':'Load_and_Boundary_Conditions_Нагрузка_и_ГУ_3D',
 'FINAL_SIGMA_I_SURFACE_3D':'Stress_Intensity_Интенсивность_Напряжений_3D',
 'FINAL_EPS_I_SURFACE_3D':'Strain_Intensity_Интенсивность_Деформаций_3D',
 'FINAL_GB_RATIO_3D':'Birger_GB_Ratio_Отношение_GB_Биргера_3D',
 'FINAL_U':'Axial_Displacement_u_Продольное_Перемещение_u',
 'FINAL_V':'Circumferential_Displacement_v_Окружное_Перемещение_v',
 'FINAL_W':'Radial_Deflection_w_Радиальный_Прогиб_w',
 'FINAL_W_PROFILES':'Deflection_Profiles_Профили_Прогиба',
 'FINAL_W_PROFILES_TRUE_SCALE':'Deflection_Profiles_True_Scale_Профили_Прогиба_Истинный_Масштаб',
 'NEWTON_RESIDUAL_HISTORY':'Newton_Residual_Невязка_Newton',
 'NEWTON_INCREMENT_HISTORY':'Newton_Increment_Приращение_Newton',
 'MVI_RPROJ_HISTORY':'MVI_Rproj_Невязка_МВИ',
 'MVI_FIELD_CHANGE_HISTORY':'MVI_Field_Change_Изменение_Поля_МВИ',
 'BIRGER_ETA_HISTORY':'Birger_Convergence_Сходимость_Биргера',
 'DETF_MIN_HISTORY':'Minimum_detF_Минимум_detF',
 'GB_RATIO_MIN_HISTORY':'Minimum_GB_Ratio_Минимум_GB',
 'GB_RATIO_MAX_HISTORY':'Maximum_GB_Ratio_Максимум_GB',
}


def _read_json(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:return default


def _copy_tree(src:Path,dst:Path):
    if not src.exists():return 0
    dst.mkdir(parents=True,exist_ok=True);n=0
    for p in src.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(src);q=dst/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);n+=1
    return n


def _write_readme(folder:Path,en:str,ru:str):
    folder.mkdir(parents=True,exist_ok=True)
    text=(
        'ENGLISH\n=======\n'+en.strip()+'\n\n'
        'РУССКИЙ\n=======\n'+ru.strip()+'\n'
    )
    (folder/'README_EN_RU.txt').write_text(text,encoding='utf-8')


def _friendly_graph_name(path:Path):
    stem=path.stem
    new=GRAPH_STEMS.get(stem)
    if new is None:
        if stem.startswith('FINAL_E_') or stem.startswith('FINAL_EPS_'):new='Strain_Деформация_'+stem
        elif stem.startswith('FINAL_S_') or stem.startswith('FINAL_SIGMA_'):new='Stress_Напряжение_'+stem
        elif stem.startswith('FINAL_') and any(x in stem for x in ('Nx','Ns','Nxs','Mx','Ms','Mxs','Au_','Av_','Aw_','Cu','Cv','Cw')):new='Resultant_Результанта_'+stem
        elif 'PLASTIC' in stem:new='Plasticity_Пластичность_'+stem
        elif 'BIRGER' in stem:new='Birger_Биргер_'+stem
        else:return path.name
    return new+path.suffix.lower()


def _rename_graph_files(folder:Path):
    if not folder.exists():return
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in ('.png','.svg','.pdf'):continue
        name=_friendly_graph_name(p)
        if name!=p.name:
            q=p.with_name(name)
            if not q.exists():p.rename(q)


def _write_extrema(run:Path,dest:Path,sm:dict):
    import csv, numpy as np, math
    p=run/'NUMERIC'/'DERIVED_FINAL_FIELDS.npz'
    if not p.exists():return None
    g=sm.get('geometry') or {};L=float(g.get('L',1.0) or 1.0);R=float(g.get('R',1.0) or 1.0)
    rows=[]
    with np.load(p,allow_pickle=False) as z:
        data={k:np.asarray(z[k]) for k in z.files};zz=np.asarray(data.get('z',[]),float)
        fields=[]
        def add(name,a):
            a=np.asarray(a,float)
            if a.ndim in (2,3):fields.append((name,a))
        for name in ('epsilon_i','sigma_i','detF','GB','GB_ratio','birger_nonlinear_thickness_fraction','plastic_thickness_fraction'):
            if name in data:add(name,data[name])
        if 'E' in data and data['E'].ndim==4:
            for c,n in enumerate(('E_xx','E_ss','E_rr','E_xs','E_xr','E_sr')):add(n,data['E'][:,c])
        if 'S' in data and data['S'].ndim==4:
            for c,n in enumerate(('S_xx','S_ss','S_rr','S_xs','S_xr','S_sr')):add(n,data['S'][:,c])
        if 'strain' in data and data['strain'].ndim==4:
            for c,n in enumerate(('eps_xx','eps_ss','eps_zz','gamma_xs')):add(n,data['strain'][:,c])
        if 'stress' in data and data['stress'].ndim==4:
            for c,n in enumerate(('sig_xx','sig_ss','tau_xs')):add(n,data['stress'][:,c])
        for key,a in data.items():
            if key.startswith('resultant_'):add(key[len('resultant_'):],a[0] if np.asarray(a).ndim==3 else a)
        for name,a in fields:
            if not np.isfinite(a).any():continue
            imin=np.unravel_index(int(np.nanargmin(a)),a.shape);imax=np.unravel_index(int(np.nanargmax(a)),a.shape)
            def loc(idx):
                if a.ndim==3:gi,j,i=idx;zv=float(zz[gi]) if zz.size>a.shape[0]-1 else None
                else:j,i=idx;gi=None;zv=None
                ns=a.shape[-2];nx=a.shape[-1];theta=2*math.pi*j/max(ns,1)
                return {'g':gi,'is':int(j),'ix':int(i),'x':L*i/max(nx-1,1),'theta':theta,'s':R*theta,'z':zv}
            lo=loc(imin);hi=loc(imax)
            rows.append({'field / поле':name,'min':float(np.nanmin(a)),**{f'min_{k}':v for k,v in lo.items()},'max':float(np.nanmax(a)),**{f'max_{k}':v for k,v in hi.items()}})
    if not rows:return None
    out=dest/'FIELD_EXTREMA_WITH_COORDINATES_ЭКСТРЕМУМЫ_С_КООРДИНАТАМИ.csv'
    keys=list(rows[0].keys())
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return out


def _summary(root:Path):
    for p in (root/'REPORT'/'CALCULATION_SUMMARY.json',root/'FINAL_RUN_SUMMARY.json',root/'CASE_RUN_SUMMARY.json',root/'RUN_RESULT.json'):
        j=_read_json(p,None)
        if isinstance(j,dict):return j
    return {}


def _engineering_summary_text(sm:dict,mode:str):
    g=sm.get('geometry') or {};m=sm.get('material') or {};diag=sm.get('solver_diagnostics') or {}
    def v(x):return '—' if x is None else x
    en=[
        'CYLINDER SOLVER — ENGINEERING SUMMARY',
        f'Post-processing mode: {mode}',f'Theory: {v(sm.get("theory"))}',f'CUF model: {v(sm.get("cuf_model"))}',
        f'Geometry model: {v(sm.get("geometry_model"))}',f'Material model: {v(sm.get("material_model"))}',f'Method: {v(sm.get("method"))}',
        f'Geometry: L={v(g.get("L"))}, R={v(g.get("R"))}, h={v(g.get("h"))}',f'Material: E={v(m.get("E0",m.get("E")))}, nu={v(m.get("nu0",m.get("nu")))}',
        f'Status: {v(sm.get("status"))}; converged={v(sm.get("converged"))}',f'Accepted steps: {v(sm.get("steps",sm.get("accepted_steps")))}',
        f'Final load: {v(sm.get("final_load_value"))} {v(sm.get("final_load_label"))}',f'Final max |w|: {v(sm.get("w_abs_max_final"))}',
        f'Final reference |w|: {v(sm.get("w_abs_reference_final"))}',f'Max stress intensity / sigma diagnostic: {v(diag.get("sigma_max",sm.get("sigma_max")))}',
        f'Derived-field error: {v(sm.get("derived_error"))}',
        '', 'For independent verification use the dimensional/normalized Excel books, CSV/NPZ arrays, extrema/control-point data, and VTK files in this package.'
    ]
    ru=[
        'CYLINDER SOLVER — ИНЖЕНЕРНАЯ СВОДКА',
        f'Режим постобработки: {mode}',f'Теория: {v(sm.get("theory"))}',f'Модель CUF: {v(sm.get("cuf_model"))}',
        f'Геометрическая модель: {v(sm.get("geometry_model"))}',f'Модель материала: {v(sm.get("material_model"))}',f'Метод: {v(sm.get("method"))}',
        f'Геометрия: L={v(g.get("L"))}, R={v(g.get("R"))}, h={v(g.get("h"))}',f'Материал: E={v(m.get("E0",m.get("E")))}, nu={v(m.get("nu0",m.get("nu")))}',
        f'Статус: {v(sm.get("status"))}; сходимость={v(sm.get("converged"))}',f'Принятые шаги: {v(sm.get("steps",sm.get("accepted_steps")))}',
        f'Конечная нагрузка: {v(sm.get("final_load_value"))} {v(sm.get("final_load_label"))}',f'Конечный max |w|: {v(sm.get("w_abs_max_final"))}',
        f'Конечный контрольный |w|: {v(sm.get("w_abs_reference_final"))}',f'Максимальная интенсивность напряжений / диагностическое sigma: {v(diag.get("sigma_max",sm.get("sigma_max")))}',
        f'Ошибка восстановления производных полей: {v(sm.get("derived_error"))}',
        '', 'Для независимой проверки используйте размерные/безразмерные Excel, массивы CSV/NPZ, данные экстремумов/контрольных точек и VTK из этого пакета.'
    ]
    return '\n'.join(en)+'\n\nРУССКИЙ\n=======\n'+'\n'.join(ru)+'\n'


def publish_partner_package(output_dir:str|Path,mode='FULL'):
    run=Path(output_dir).resolve();mode=str(mode).upper().strip();dest=run/PARTNER_ROOT
    if dest.exists():shutil.rmtree(dest,ignore_errors=True)
    dirs={k:dest/v for k,v in FOLDERS.items()}
    for p in dirs.values():p.mkdir(parents=True,exist_ok=True)

    _write_readme(dirs['readme'],
        'Start with the Engineering Summary. Excel is separated into dimensional and dimensionless branches. Plot titles, axes and legends are bilingual with English first. SVG/PDF are publication/vector formats; PNG is the quick preview. FAST, STANDARD and FULL are launched manually from the GUI and never start automatically.',
        'Начните с инженерной сводки. Excel разделён на размерную и безразмерную ветви. Заголовки, оси и легенды графиков двуязычные, английский указан первым. SVG/PDF — форматы для публикаций/векторного вывода, PNG — быстрый просмотр. FAST, STANDARD и FULL запускаются вручную из GUI и никогда не стартуют автоматически.'
    )
    sm=_summary(run)
    (dirs['summary']/'ENGINEERING_SUMMARY_EN_RU.txt').write_text(_engineering_summary_text(sm,mode),encoding='utf-8')
    _write_extrema(run,dirs['summary'],sm)
    for p in (run/'REPORT').glob('*') if (run/'REPORT').exists() else ():
        if p.is_file():shutil.copy2(p,dirs['summary']/p.name)
    _write_readme(dirs['summary'],'Key calculation status, extrema/diagnostics and machine-readable report files.','Ключевой статус расчёта, экстремумы/диагностика и машиночитаемые файлы отчёта.')

    # Dimensional Excel: preserve originals internally, publish clear bilingual filenames here.
    xdir=run/'EXCEL'
    if xdir.exists():
        for p in sorted(xdir.glob('*.xlsx')):
            upper=p.stem.upper();new=None
            for token,name in EXCEL_NAMES.items():
                if upper.endswith('__'+token) or token in upper:new=name;break
            if new is None:new=p.name
            shutil.copy2(p,dirs['excel_dim']/new)
    _write_readme(dirs['excel_dim'],'Dimensional Excel workbooks in the calculation unit system. File names are English first / Russian second.','Размерные книги Excel в системе единиц расчёта. Имена файлов: сначала английский, затем русский.')

    try:
        import final_dimensionless_export
        final_dimensionless_export.export_dimensionless_excels(run,dirs['excel_nd'],mode=mode)
    except Exception as exc:
        (dirs['excel_nd']/'DIMENSIONLESS_EXPORT_ERROR_EN_RU.txt').write_text(
            f'ENGLISH\nDimensionless Excel export error: {type(exc).__name__}: {exc}\n\nРУССКИЙ\nОшибка экспорта безразмерных Excel: {type(exc).__name__}: {exc}\n',encoding='utf-8')
    _write_readme(dirs['excel_nd'],'Reference-scale normalized Excel workbooks. Open the Normalization sheet in each workbook for the exact formulas and reference values.','Безразмерные книги Excel с нормировкой по опорным величинам. Точные формулы и опорные значения приведены на листе Normalization каждой книги.')

    graphs=run/'GRAPHS'
    for srcname,dstname in GRAPH_DIRS.items():_copy_tree(graphs/srcname,dirs['g2d']/dstname)
    _rename_graph_files(dirs['g2d'])
    _write_readme(dirs['g2d'],'Two-dimensional engineering plots. PNG is preview; SVG and PDF are saved next to each plot for publication/vector use.','Двумерные инженерные графики. PNG предназначен для просмотра; рядом сохраняются SVG и PDF для публикаций и векторного использования.')

    src3d=run/'3D'
    if src3d.exists():
        for p3 in src3d.iterdir():
            if p3.is_dir():_copy_tree(p3,dirs['g3d']/GRAPH3D_DIRS.get(p3.name,p3.name))
            elif p3.is_file():shutil.copy2(p3,dirs['g3d']/p3.name)
    _rename_graph_files(dirs['g3d'])
    _write_readme(dirs['g3d'],'Three-dimensional cylinder-shell visualization based on calculated fields. Visual smoothing/presentation does not alter the numerical arrays.','Трёхмерная визуализация цилиндрической оболочки по расчётным полям. Визуальное оформление/сглаживание не изменяет численные массивы.')

    _copy_tree(graphs/'SECTIONS',dirs['sections']);_write_readme(dirs['sections'],'x–z and s–z sections through the saved physical fields.','Сечения x–z и s–z по сохранённым физическим полям.')
    _copy_tree(graphs/'THROUGH_THICKNESS',dirs['thickness']);_write_readme(dirs['thickness'],'Through-thickness z profiles at the documented control/max-deflection location. STANDARD contains the engineering subset; FULL contains all available fields.','Профили по толщине z в указанной контрольной точке/точке максимального прогиба. STANDARD содержит инженерный набор, FULL — все доступные поля.')
    _copy_tree(graphs/'CONVERGENCE',dirs['conv']);_rename_graph_files(dirs['conv']);_write_readme(dirs['conv'],'Solver, MVI, Newton and Birger convergence histories when available.','Истории сходимости решателя, МВИ, Newton и Birger при наличии.')
    _copy_tree(graphs/'PLASTICITY',dirs['plastic']);_copy_tree(run/'3D'/'BIRGER_ZONES',dirs['plastic']/'3D_BIRGER_ЗОНЫ_БИРГЕРА');_rename_graph_files(dirs['plastic']);_write_readme(dirs['plastic'],'Plasticity/Birger nonlinear-zone diagnostics and evolution plots.','Диагностика пластичности/нелинейных зон Birger и графики их развития.')

    _copy_tree(run/'NUMERIC',dirs['numeric'])
    for p in (run/'NUMERIC_RESULTS.zip',):
        if p.exists():shutil.copy2(p,dirs['numeric']/p.name)
    _write_readme(dirs['numeric'],'Machine-readable CSV and NPZ arrays. These are the authoritative numerical data used by post-processing.','Машиночитаемые массивы CSV и NPZ. Это исходные численные данные, используемые постпроцессором.')
    _copy_tree(run/'VTK',dirs['vtk']);_write_readme(dirs['vtk'],'VTK files for independent visualization in ParaView or compatible software.','Файлы VTK для независимой визуализации в ParaView или совместимых программах.')
    _copy_tree(run/'VIDEO',dirs['video']);_write_readme(dirs['video'],'Optional animations/videos created only when explicitly requested.','Необязательные анимации/видео, создаваемые только при явном запуске пользователем.')

    _copy_tree(run/'INPUT',dirs['inputs'])
    for name in ('FINAL_CASE.json','RUN_PARAMETERS.json','PARAMETERS.txt','RUN_INFO.json','LOAD_DEFINITION.json','GENERATED_LOADS.npz'):
        p=run/name
        if p.exists():shutil.copy2(p,dirs['inputs']/p.name)
    _write_readme(dirs['inputs'],'Input parameters, load definition and run identity needed to reproduce the calculation.','Исходные параметры, определение нагрузки и паспорт запуска, необходимые для воспроизведения расчёта.')

    manifest={'format':'CYLINDER_PARTNER_RESULTS_V1','postprocess_mode':mode,'folders':FOLDERS}
    (dest/'PARTNER_RESULTS_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return dest

if __name__=='__main__':
    import sys
    if len(sys.argv)<2:raise SystemExit('Usage: final_partner_export.py RUN_DIR [FAST|STANDARD|FULL]')
    p=publish_partner_package(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'FULL');print('PARTNER PACKAGE PASS:',p)
