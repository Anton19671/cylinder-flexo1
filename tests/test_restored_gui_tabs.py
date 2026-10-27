"""UI wiring regressions. No mechanics runs; every external process is blocked."""
import json
import os
import tempfile
import time
import tkinter as tk
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import FINAL_GUI as gui
from cylinder_time_integrator_gui import TIME_INTEGRATOR_CHOICES


@skipUnless(os.environ.get("DISPLAY"), "Requires a display or Xvfb")
class RestoredGuiTabsTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process_guard=patch("subprocess.Popen",side_effect=AssertionError("Numerical/external processes forbidden in UI tests"))
        cls.process_guard.start()
        cls.dialog_guards=[patch.object(gui.messagebox,name,return_value=None) for name in ("showerror","showwarning","showinfo")]
        for guard in cls.dialog_guards:guard.start()
        cls.app=gui.FinalSolverGUI()
        cls.errors=[]
        cls.app.report_callback_exception=lambda kind,error,tb:cls.errors.append(str(error))
        cls.pump()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()
        for guard in cls.dialog_guards:guard.stop()
        cls.process_guard.stop()

    @classmethod
    def pump(cls):
        for _ in range(12):cls.app.update();time.sleep(.005)

    def setUp(self):
        self.errors.clear()
        self.app._v("case_path").set("")
        self.app._v("ui_level").set("EXPERT")
        self.app._v("calculation_kind").set("STATIC")
        self.app._v("method").set("MVI")
        self.app.time_integrator.set(TIME_INTEGRATOR_CHOICES[0])
        self.app._apply_ui_level_layout();self.app._refresh_visibility()
        self.app.geometry("1640x1000")
        self.pump()

    @staticmethod
    def walk(widget):
        yield widget
        for child in widget.winfo_children():yield from RestoredGuiTabsTests.walk(child)

    def test_all_tabs_and_menu_targets_in_each_mode(self):
        app=self.app
        records=[]
        for level in ("BASIC","EXPERT"):
            for kind in ("STATIC","DYNAMIC"):
                app._v("ui_level").set(level)
                app._v("calculation_kind").set(kind)
                app._apply_ui_level_layout();app._on_calculation_kind_change();self.pump()
                tabs=app.notebook.tabs()
                self.assertEqual(len(tabs), (6 if level=="BASIC" else 9)+(kind=="DYNAMIC"))
                self.assertEqual(app.tab_menu.index("end")+1,len(tabs))
                for index,page in enumerate(tabs):
                    app.tab_menu.invoke(index);self.pump()
                    self.assertEqual(app.notebook.select(),page)
                    self.assertTrue(app.nametowidget(page).winfo_ismapped())
                    buttons=[w for w in self.walk(app.nametowidget(page)) if isinstance(w,gui.ttk.Button)]
                    for button in buttons:
                        command=str(button.cget("command"))
                        self.assertTrue(command,(app.notebook.tab(page,"text"),button.cget("text")))
                        self.assertTrue(app.tk.call("info","commands",command))
                    records.append((level,kind,app.notebook.tab(page,"text"),len(buttons)))
        self.assertFalse(self.errors,self.errors)
        print("TAB_AUDIT",json.dumps(records,ensure_ascii=False))

    def test_integrator_widget_persistence_and_series_state(self):
        app=self.app
        source=Path(__file__).resolve().parents[1]/"config/CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json"
        app._v("case_path").set(str(source))
        app._v("calculation_kind").set("DYNAMIC")
        app._on_calculation_kind_change();self.pump()
        self.assertEqual(str(app.time_integrator_combo.cget("textvariable")),str(app.time_integrator))
        self.assertEqual(tuple(app.time_integrator_combo.cget("values")),TIME_INTEGRATOR_CHOICES)
        app.time_integrator_combo.current(1)
        app.time_integrator_combo.event_generate("<<ComboboxSelected>>");self.pump()
        self.assertEqual(app.time_integrator.get(),TIME_INTEGRATOR_CHOICES[1])
        self.assertIn("disabled",app.series_btn.state())
        saved=app._gui_state()
        app.time_integrator.set(TIME_INTEGRATOR_CHOICES[0])
        self.assertNotIn("disabled",app.series_btn.state())
        app._apply_gui_state(saved);self.pump()
        self.assertEqual(app.time_integrator.get(),TIME_INTEGRATOR_CHOICES[1])
        old=dict(saved);old.pop("time_integrator")
        app._apply_gui_state(old);self.pump()
        self.assertEqual(app.time_integrator.get(),TIME_INTEGRATOR_CHOICES[0])
        self.assertFalse(self.errors,self.errors)

    def test_navigation_popup_and_form_width_at_multiple_window_sizes(self):
        app=self.app
        for width in (1640,1280,960):
            app.geometry(f"{width}x1000")
            app.notebook.select(app.tab_problem);self.pump()
            self.assertGreaterEqual(app.problem_left_inner.winfo_width(),app.problem_left_inner.winfo_reqwidth())
            for w in self.walk(app.problem_left_inner):
                if isinstance(w,(gui.ttk.Entry,gui.ttk.Combobox)) and w.winfo_ismapped():
                    self.assertLessEqual(w.winfo_x()+w.winfo_width(),w.master.winfo_width()+2,str(w))
            app._show_cylinder_tab();self.pump()
            self.assertEqual(app.notebook.select(),str(app.tab_cylinder))
            for title,target in (("Нагрузка",app.tab_load),("Постановка",app.tab_problem)):
                buttons=[w for w in self.walk(app.tab_cylinder) if isinstance(w,gui.ttk.Button) and w.cget("text")==title]
                self.assertEqual(len(buttons),1)
                buttons[0].invoke();self.pump()
                self.assertEqual(app.notebook.select(),str(target))
        app._open_cylinder_popup();self.pump()
        self.assertTrue(app._cylinder_popup.winfo_exists())
        app._cylinder_popup.destroy()
        self.assertFalse(self.errors,self.errors)

    def test_scroll_pages_and_file_parameters_round_trip(self):
        app=self.app
        self.assertEqual(len(app._scroll_pages),3)
        for canvas,content in app._scroll_pages:
            page=canvas.master
            app.notebook.select(page);self.pump()
            self.assertTrue(canvas.cget("yscrollcommand"))
            self.assertTrue(canvas.cget("xscrollcommand"))
            self.assertEqual(str(content.master),str(canvas))
        with tempfile.TemporaryDirectory() as folder:
            app.time_integrator.set(TIME_INTEGRATOR_CHOICES[1])
            app._v("L").set("321.5")
            with patch.object(gui.filedialog,"askdirectory",return_value=folder):app.save_parameters_to_folder()
            file=Path(folder)/"CYLINDER_GUI_PARAMETERS.json"
            self.assertEqual(json.loads(file.read_text())["vars"]["L"],"321.5")
            app._v("L").set("1")
            app.time_integrator.set(TIME_INTEGRATOR_CHOICES[0])
            with patch.object(gui.filedialog,"askopenfilename",return_value=str(file)):app.open_parameters()
            self.assertEqual(app._v("L").get(),"321.5")
            self.assertEqual(app.time_integrator.get(),TIME_INTEGRATOR_CHOICES[1])
        self.assertFalse(self.errors,self.errors)

    def test_reference_launch_arguments_without_starting_process(self):
        app=self.app
        config=Path(__file__).resolve().parents[1]/"config"
        app.time_integrator.set(TIME_INTEGRATOR_CHOICES[1])
        with tempfile.TemporaryDirectory() as folder:
            for kind,filename,code in (("STATIC","CASE_REFERENCE_CUF_G3D_STATIC.json","NEWMARK"),
                                       ("DYNAMIC","CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json","GAUSS2_IRK4")):
                case=json.loads((config/filename).read_text())
                case["output_dir"]=str(Path(folder)/kind)
                path=Path(folder)/(kind+".json")
                path.write_text(json.dumps(case))
                with patch.object(gui.subprocess,"Popen",side_effect=AssertionError("BLOCKED")) as blocked:
                    with self.assertRaisesRegex(AssertionError,"BLOCKED"):
                        app._launch_reference_case(path,case)
                    self.assertEqual(blocked.call_args.args[0][-1],code)
                if kind=="DYNAMIC":self.assertEqual(app.expected_steps,case["dynamics"]["n_steps"])
                self.assertFalse(Path(case["output_dir"]).exists())

    def test_schematic_captions_do_not_overlap(self):
        from cylinder_live_preview import render_cylinder_view
        app=self.app
        win=tk.Toplevel(app)
        canvas=tk.Canvas(win,width=420,height=245)
        canvas.pack();self.pump()
        render_cylinder_view(app,canvas)
        labels={canvas.itemcget(i,"text"):canvas.bbox(i) for i in canvas.find_all() if canvas.type(i)=="text"}
        bottom=[bounds for text,bounds in labels.items() if text.startswith(("x=0:","x=L:"))]
        dimensions=next(bounds for text,bounds in labels.items() if text=="Размеры не заданы" or text.startswith("L="))
        self.assertLess(max(bounds[3] for bounds in bottom),dimensions[1])
        win.destroy()

    def test_reference_case_edit_is_saved_and_no_unrelated_fields_are_editable(self):
        app=self.app
        template=Path(__file__).resolve().parents[1]/"config/CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json"
        original=json.loads(template.read_text())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"input.json"
            path.write_text(json.dumps(original))
            app._v("case_path").set(str(path));self.pump()
            self.assertTrue(app.reference_bar.winfo_ismapped())
            self.assertEqual(float(app._v("L").get()),original["geometry"]["L"])
            self.assertEqual(float(app._v("E0").get()),original["material"]["E0"])
            self.assertEqual(app._v("unit_system").get(),"SI_N_M_PA")
            entries=[w for w in self.walk(app.problem_left_inner) if isinstance(w,gui.ttk.Entry) and str(w.cget("textvariable"))==str(app._v("L"))]
            self.assertEqual(len(entries),1)
            self.assertIn("disabled",entries[0].state())
            self.assertNotIn("disabled",app.time_integrator_combo.state())
            from cylinder_reference_gui import ReferenceCaseEditor
            editor=app._edit_active_reference_case();self.pump()
            self.assertIsInstance(editor,ReferenceCaseEditor)
            self.assertEqual(float(editor.fields["L"].get()),original["geometry"]["L"])
            editor.fields["L"].set("1.5")
            loads=json.loads(editor.load_text.get("1.0","end-1c"));loads["q"]=0.02
            editor.load_text.delete("1.0","end");editor.load_text.insert("1.0",json.dumps(loads))
            target=Path(folder)/"edited.json"
            with patch.object(gui.filedialog,"asksaveasfilename",return_value=str(target)):
                self.assertEqual(editor.save(),target)
            self.pump()
            edited=json.loads(target.read_text())
            self.assertEqual(edited["geometry"]["L"],1.5)
            self.assertEqual(edited["load_definition"]["q"],0.02)
            self.assertEqual(json.loads(path.read_text()),original)
            self.assertEqual(float(app._v("L").get()),1.5)
            self.assertEqual(float(app._v("load_q").get()),0.02)
            self.assertIn("disabled",entries[0].state())
            editor.destroy()
            app._close_active_reference_case();self.pump()
            self.assertFalse(app.reference_bar.winfo_ismapped())
            self.assertNotIn("disabled",entries[0].state())
            self.assertFalse(self.errors,self.errors)
