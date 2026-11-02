"""Readable, concise human journal for cylinder modal MVI/FDM.

A pure presentation module. Full modal event dictionaries are retained in the
separate solver-side JSONL diagnostic file; no MVI physics or selection changes.
"""
from __future__ import annotations

VERBOSE_STAGES = frozenset({
    "MVI_CANDIDATE_START", "MVI_SWEEP_DONE",
    "MVI_CANDIDATE_CONVERGED", "MVI_CANDIDATE_REJECTED",
    "MVI_CANDIDATE_SELECTED", "MVI_MODE_DONE", "FDM_MODE_DONE",
})


def should_display_modal_event(event):
    return isinstance(event, dict) and str(event.get("stage", "")) in VERBOSE_STAGES


def _seed_description(seed):
    if not isinstance(seed, dict):
        return ""
    kind = str(seed.get("kind", ""))
    if kind == "random":
        return "случайное начальное поле"
    if kind == "quadrature_previous":
        return "поворот предыдущей формы"
    if kind == "harmonic_partner":
        n = seed.get("n", "?")
        phase = seed.get("phase", "cos")
        quadrature = seed.get("component_phase") == "circumferential_quadrature"
        return f"гармоника n={n}, {phase}" + (
            ", сдвиг фаз компонентов" if quadrature else "")
    return kind


def format_modal_event(event):
    """Return a short human line, or None for per-column/technical events.

    Never present unrefined million-hertz starting Rayleigh values as an
    accepted frequency. A converged candidate is NOT an accepted eigenmode.
    """
    if not should_display_modal_event(event):
        return None
    stage = str(event["stage"])
    mode, modes = event.get("mode", "?"), event.get("n_modes", "?")
    rank = event.get("rank")
    method = f"МВИ-{rank}" if rank is not None else "МКР/FDM"
    head = f"{method} | форма {mode}/{modes}"
    attempt = event.get("attempt")
    attempts = event.get("attempts")
    if attempt is not None:
        head += f" | попытка {attempt}/{attempts if attempts is not None else '?'}"
    seed = _seed_description(event.get("seed"))
    if seed and stage in ("MVI_CANDIDATE_START", "MVI_CANDIDATE_SELECTED"):
        head += f" | {seed}"
    frequency = event.get("frequency_hz")
    elapsed = event.get("elapsed_s")
    suffix = f" | t={float(elapsed):.1f} с" if elapsed is not None else ""
    if stage == "MVI_CANDIDATE_START":
        body = "поиск кандидата"
    elif stage == "MVI_SWEEP_DONE":
        sweep = event.get("sweep", "?")
        max_sweeps = event.get("max_sweeps", "?")
        body = f"цикл Y→X {sweep}/{max_sweeps}"
        if frequency is not None:
            body += f" | текущая f={float(frequency):.6f} Гц"
        change = event.get("shape_change_relative")
        if change is not None:
            body += f" | Δформы={float(change):.2e}"
    elif stage == "MVI_CANDIDATE_CONVERGED":
        body = "кандидат сошёлся (это ещё не принятая форма)"
        if frequency is not None:
            body += f" | f={float(frequency):.6f} Гц"
    elif stage == "MVI_CANDIDATE_REJECTED":
        body = "попытка отклонена"
        reason = str(event.get("reason", ""))
        if reason:
            body += f" | {reason[:150]}"
    elif stage == "MVI_CANDIDATE_SELECTED":
        body = f"выбран минимум из {event.get('converged','?')} сходящихся кандидатов"
        if frequency is not None:
            body += f" | f={float(frequency):.6f} Гц"
    else:
        body = "СОБСТВЕННАЯ ФОРМА ПРИНЯТА"
        if frequency is not None:
            body += f" | f={float(frequency):.6f} Гц"
    return head + " | " + body + suffix
