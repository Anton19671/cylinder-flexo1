"""Pure GUI feedback for saved variational runs and UTF-8 child stdout.

The child writes human-readable output to a pipe consumed as UTF-8.
On Windows a pipe from pythonw.exe can otherwise default to a legacy
code page, producing U+FFFD replacement glyphs in the live journal.
No text is silently discarded or post-processed to conceal data loss.
"""
from __future__ import annotations

import os


def utf8_solver_process_env(source=None):
    """Require UTF-8 output from a Python solver subprocess."""
    env = dict(os.environ if source is None else source)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def reference_completion_notice(completion, *, kind, method, rank=None):
    """Show verified completion only; OS exit code alone is never enough.

    Returns (success, dialog_title, dialog_text). A modal success must have
    exactly the requested positive number of accepted modes on disk.
    """
    record = completion if isinstance(completion, dict) else {}
    mode = str(kind or "").upper()
    solver = str(method or "").upper()
    method_label = (
        f"МВИ-{int(rank)}" if solver == "MVI" and
        isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= 8
        else "МКР/FDM" if solver in ("FDM", "MKR") else solver
    )
    accepted = record.get("accepted")
    requested = record.get("requested")
    valid_accepted = type(accepted) is int and accepted >= 0
    valid_requested = type(requested) is int and requested > 0
    success = record.get("success") is True and valid_accepted
    if mode == "MODAL":
        success = (success and valid_requested and accepted == requested
                   and record.get("status") == "MODAL_COMPLETED")
    elif valid_requested:
        success = success and accepted == requested
    if success:
        amount = (f"Принято собственных форм: {accepted} из {requested}."
                  if mode == "MODAL" else
                  f"Принято шагов: {accepted}" +
                  (f" из {requested}." if valid_requested else "."))
        return True, "Расчёт завершён — сошлось", (
            f"Расчёт завершён успешно.\nСошлось.\n"
            f"Метод: {method_label}.\n{amount}\n"
            "Результаты сохранены."
        )
    detail = str(record.get("detail") or record.get("status") or
                 "Итоговая сводка отсутствует или не подтверждена")
    amount = (f"Принято: {accepted} из {requested}.\n"
              if valid_accepted and valid_requested else "")
    return False, "Расчёт не завершён", (
        f"Сходимость не подтверждена.\nМетод: {method_label}.\n"
        f"{amount}Причина: {detail}\nСм. журнал расчёта."
    )
