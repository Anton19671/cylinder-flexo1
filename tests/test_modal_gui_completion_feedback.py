"""GUI success feedback must be grounded in complete saved modal output."""
from pathlib import Path
import ast
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interface"))
from cylinder_gui_run_feedback import utf8_solver_process_env, reference_completion_notice


def test_utf8_environment_for_windows_and_posix_pipe():
    original = {"USER_VAR": "kept", "PYTHONIOENCODING": "cp1251"}
    setting = utf8_solver_process_env(original)
    assert setting["PYTHONIOENCODING"] == "utf-8"
    assert setting["USER_VAR"] == "kept"
    assert original["PYTHONIOENCODING"] == "cp1251"


def test_modal_completion_requires_durable_full_spectrum():
    report = {"success": True, "accepted": 12, "requested": 12,
              "status": "MODAL_COMPLETED", "detail": ""}
    ok, title, body = reference_completion_notice(
        report, kind="MODAL", method="MVI", rank=6)
    assert ok and "сошлось" in title.lower()
    assert "МВИ-6" in body and "12 из 12" in body

    for override in ({"accepted": 11}, {"requested": 13}, {"success": False},
                     {"status": "INCOMPLETE"}, {"accepted": None}):
        rec = {**report, **override}
        success, title, message = reference_completion_notice(
            rec, kind="MODAL", method="MVI", rank=6)
        assert not success and "не завершён" in title.lower()
        assert "Сходимость не подтверждена" in message


def test_mkr_and_nonmodal_success_are_labelled():
    ok, title, body = reference_completion_notice(
        {"success": True, "accepted": 5, "requested": 5, "status": "CONVERGED"},
        kind="STATIC", method="FDM")
    assert ok and "МКР/FDM" in body and "Принято шагов: 5 из 5" in body


def test_reference_gui_is_syntactically_valid_and_uses_verified_popup():
    source = (ROOT / "interface" / "FINAL_GUI.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "reference_completion_notice(" in source
    assert "env=utf8_solver_process_env()" in source
    assert 'errors="strict"' in source
    assert "messagebox.showinfo(title, body, parent=self)" in source
