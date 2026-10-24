#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import csv, json, shutil, zipfile, re
import numpy as np

import final_run_bundle as rb
import final_excel_export as xe
import FINAL_POSTPROCESS as pp
from cylinder_final_stage11a_dispatcher import validate_final_case, run_final_case

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'OUTPUT_BUNDLE_EXCEL_SELF_CHECK_WORK'

def case_cfg(root):
    return {
      'theory':'KL','method':'FDM','mode':'LINEAR_ELASTIC','rank':None,
      'units':{'system':'N_MM_MPA'},
      'geometry':{'L':100.0,'R':50.0,'h':1.0},
      'material':{'E0':70000.0,'nu0':0.3},
      'grid':{'nx':7,'ns':7,'stencil':7},
      'fdm_grid':{'nx':7,'ns':7,'stencil':7},
      'mvi_grid':{'nx':7,'ns':7,'stencil':7},
      'quadrature':{'type':'GAUSS','n_gauss':3},
      'boundary_conditions':{'left':'CLAMPED','right':'CLAMPED','advanced':None},
      'load_definition':{'type':'UNIFORM_RADIAL_FACE','q':1e-4,'load_factors':[1.0]},
      'initial_profiles_npz':None,
      'initial_state':{'source':None,'npz':None},
      'initial_GB':{'source':None,'npz':None},
      'inner_mvi':{},'newton':{},'birger':{},
      'linear':{'residual_tolerance_relative':1e-10},
      'postprocess':{},'output_dir':str(root),
    }

def sheet_names(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        xml=z.read('xl/workbook.xml').decode('utf-8')
        return re.findall(r'<sheet name="([^"]+)"',xml)

def main():
    shutil.rmtree(WORK,ignore_errors=True); WORK.mkdir(parents=True)
    cfg=case_cfg(WORK)
    b1=rb.prepare_run_bundle(cfg,WORK,run_label='Variant A',source_base=ROOT)
    b2=rb.prepare_run_bundle(cfg,WORK,run_label='Variant A',source_base=ROOT)
    assert Path(b1['run_dir'])!=Path(b2['run_dir'])
    assert Path(b1['run_dir']).parent==Path(b2['run_dir']).parent
    assert Path(b1['run_dir']).parent==WORK.resolve()
    assert Path(b1['run_dir']).name.startswith('RUN_001__')
    assert Path(b2['run_dir']).name.startswith('RUN_002__')
    calc=Path(b1['calculation_dir'])
    assert (calc/'README_RESULTS_RU.txt').exists()
    assert (calc/'README_RESULTS_EN_RU.txt').exists()
    rt=(calc/'README_RESULTS_EN_RU.txt').read_text(encoding='utf-8')
    assert 'ENGLISH' in rt and 'РУССКИЙ' in rt
    assert Path(b1['runs_index_csv']).exists()
    validate_final_case(b1['case_path'])
    r=run_final_case(b1['case_path']); assert r.get('converged')
    # Minimal independent export can run before graph postprocessing; it uses
    # solver state + summary + progress directly.
    xp=xe.export_calculation_excel(b1['run_dir'])
    assert xp.exists() and xp.parent==Path(b1['run_dir'])/'EXCEL'
    named=xe.export_named_excels(b1['run_dir'])
    for key in ('LOAD_DEFLECTION','DISPLACEMENTS','STRAINS_STRESSES','RESULTANTS','CONVERGENCE','CONTROL_POINTS'):
        assert key in named and Path(named[key]).exists(),key
    names=sheet_names(xp)
    for need in ('README','Parameters','Load_Deflection','Online_History','Surface_Fields'):
        assert need in names,(need,names)
    processed=pp.process_output(Path(b1['run_dir']))
    readable_csv=sorted((Path(b1['run_dir'])/'CSV').glob('*__LOAD_DEFLECTION.csv'))
    assert len(readable_csv)==1,readable_csv
    with readable_csv[0].open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.reader(f))
    assert rows[1][:3]==['0','0.0','0.0'],rows[:2]
    assert Path(processed['excel_file']).exists()
    assert (Path(b1['run_dir'])/'README_RESULTS_RU.txt').exists()
    assert (Path(b1['run_dir'])/'README_RESULTS_EN_RU.txt').exists()
    rt=(Path(b1['run_dir'])/'README_RESULTS_EN_RU.txt').read_text(encoding='utf-8')
    assert 'ENGLISH' in rt and 'РУССКИЙ' in rt
    assert (Path(b1['run_dir'])/'INPUT'/'FINAL_CASE.json').exists()
    assert (Path(b1['run_dir'])/'INPUT'/'RUN_PARAMETERS.json').exists()
    print('OUTPUT_BUNDLE_EXCEL_SELF_CHECK: PASS')
    print('  independent run directories:',Path(b1['run_dir']).name,'|',Path(b2['run_dir']).name)
    print('  workbook:',xp.name)
    print('  sheets:',', '.join(names))

if __name__=='__main__': main()
