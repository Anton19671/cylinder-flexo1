#!/usr/bin/env python3
"""CASE GUI mapping: fields, switches and boundary data survive round trips."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

from cylinder_reference_gui import edit_reference_case
from FINAL_GUI import FinalSolverGUI

ROOT=Path(__file__).resolve().parent.parent


class ReferenceGuiTest(unittest.TestCase):
    def sample(self):
        c=json.loads((ROOT/'config/CASE_REFERENCE_CUF_G3D_STATIC.json').read_text())
        fields={
            'theory':'CUF','cuf_model':'TE1','method':'MVI','rank':'1','max_sweeps':'30',
            'nx':'4','ns':'4','n_gauss':'3','L':'1.2','R':'2','h':'0.2',
            'E0':'100','nu0':'.25','scale_model':'MICRO','flexo':True,
            'ell_m':'.08','mu1':'.001','mu2':'-.002','kappa':'1.5',
            'potential_order':'1','rho_kg_m3':'1300','eta_s':'0',
            'output_dir':'../GUI_REFERENCE_RUN',
        }
        return c,fields

    def edit(self,c,f,bc=None,load=None,newton=None):
        return edit_reference_case(c,f,json.dumps(bc if bc is not None else c['boundary_conditions']['variational']),
            json.dumps(load if load is not None else c['load_definition']),
            json.dumps(newton if newton is not None else c['reference_newton']))

    def test_mvi_fields_and_nested_documents_preserved(self):
        c,f=self.sample()
        c['custom_extension']={'retained':True}
        bc=copy.deepcopy(c['boundary_conditions']['variational'])
        bc['mechanical']['left']['higher']='FIX_FIRST_NORMAL'
        load=copy.deepcopy(c['load_definition']);load['q']=.03;load['load_factors']=[.4,1]
        result=self.edit(c,f,bc,load)
        self.assertEqual(result['method'],'MVI')
        self.assertEqual(result['mvi_grid'],{'nx':4,'ns':4})
        self.assertNotIn('fdm_grid',result)
        self.assertEqual(result['boundary_conditions']['variational'],bc)
        self.assertEqual(result['load_definition'],load)
        self.assertEqual(result['custom_extension'],c['custom_extension'])
        self.assertEqual(c['method'],'FDM')

    def test_flexo_off_on_restores_coefficients_and_electrode(self):
        c,f=self.sample()
        f['flexo']=False
        off=self.edit(c,f)
        self.assertFalse(off['micro']['enable_flexoelectric'])
        self.assertEqual([off['micro'][key] for key in ('mu1_c_per_m','mu2_c_per_m','kappa_f_per_m')],[0,0,0])
        self.assertEqual(off['boundary_conditions']['variational']['electrical'],{})
        self.assertEqual(off['micro']['editor_saved_electrical_bc'],c['boundary_conditions']['variational']['electrical'])
        f['flexo']=True
        restored=self.edit(off,f)
        self.assertEqual(restored['boundary_conditions']['variational']['electrical'],c['boundary_conditions']['variational']['electrical'])
        self.assertEqual(restored['micro']['mu1_c_per_m'],.001)
        f['method']='FDM';f['scale_model']='MACRO'
        macro=self.edit(restored,f)
        self.assertNotIn('rank',macro)
        self.assertEqual(macro['fdm_grid'],{'nx':4,'ns':4})
        self.assertEqual(macro['boundary_conditions']['variational']['electrical'],{})

    def test_unrepresentable_or_bad_json_rejected_before_save(self):
        c,f=self.sample();f['rank']='5'
        with self.assertRaisesRegex(ValueError,'grid dimensions'):self.edit(c,f)
        f['rank']='1'
        with self.assertRaisesRegex(ValueError,'Граничные условия'):
            edit_reference_case(c,f,'{',json.dumps(c['load_definition']),json.dumps(c['reference_newton']))
        f['method']='DYNAMIC'
        with self.assertRaisesRegex(ValueError,'Метод'):self.edit(c,f)

    def test_load_case_opens_reference_editor_without_touching_classical_fields(self):
        from tempfile import TemporaryDirectory
        c,_=self.sample()
        with TemporaryDirectory() as d:
            path=Path(d)/'reference.json';path.write_text(json.dumps(c))
            fake=Mock()
            fake._v.return_value=Mock()
            with patch('FINAL_GUI.filedialog.askopenfilename',return_value=str(path)):
                FinalSolverGUI.load_case(fake)
            fake._v.assert_called_once_with('case_path')
            fake.open_reference_case.assert_called_once_with(c,str(path))
            fake._log.assert_called_once()

    def test_gui_launches_saved_reference_case_and_preflights_rank_series(self):
        from tempfile import TemporaryDirectory
        c,f=self.sample()
        c=self.edit(c,f)
        with TemporaryDirectory() as d:
            path=Path(d)/'case.json';path.write_text(json.dumps(c))
            c['output_dir']=str(Path(d)/'result')
            fake=Mock();fake.proc=None
            with patch('FINAL_GUI.subprocess.Popen') as process,patch('FINAL_GUI.threading.Thread') as thread:
                FinalSolverGUI._launch_reference_case(fake,path,c)
            self.assertEqual(fake.run_kind,'REFERENCE_CASE')
            self.assertEqual(fake.run_output_dir,Path(c['output_dir']))
            self.assertEqual(process.call_args.args[0][-1],str(path))
            thread.return_value.start.assert_called_once()
            with patch('FINAL_GUI.subprocess.Popen') as process:
                with self.assertRaisesRegex(ValueError,'grid dimensions'):
                    FinalSolverGUI._launch_reference_case(fake,path,c,series=True)
                process.assert_not_called()

    def test_dynamic_editor_switches_time_contract_without_static_factors(self):
        c=json.loads((ROOT/'config/CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json').read_text())
        _,fields=self.sample();fields['calculation_kind']='DYNAMIC'
        dynamics=copy.deepcopy(c['dynamics']);dynamics['n_steps']=7
        result=edit_reference_case(c,fields,
            json.dumps(c['boundary_conditions']['variational']),json.dumps(c['load_definition']),
            json.dumps(c['reference_newton']),json.dumps(dynamics))
        self.assertEqual(result['calculation_kind'],'DYNAMIC')
        self.assertEqual(result['dynamics']['n_steps'],7)
        self.assertNotIn('load_factors',result['load_definition'])
        fields['calculation_kind']='STATIC'
        restored=edit_reference_case(result,fields,
            json.dumps(result['boundary_conditions']['variational']),json.dumps(result['load_definition']),
            json.dumps(result['reference_newton']),json.dumps(dynamics))
        self.assertEqual(restored['load_definition']['load_factors'],[1.0])
        self.assertNotIn('dynamics',restored)


if __name__=='__main__':unittest.main()
