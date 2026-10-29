"""Child process adapter: solver success must determine the OS exit code."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cylinder_paths import activate
activate()
from cylinder_time_integrator_gui import plan_time_integrator_run, execute_time_integrator_plan


def result_succeeded(result, *, integrator="NEWMARK", series=False):
    """Fail closed for a single CASE, MVI rank series or experimental Gauss2."""
    if not isinstance(result, dict) or type(series) is not bool:
        return False
    if series:
        return result.get("all_converged") is True
    if integrator == "GAUSS2_IRK4":
        requested = result.get("n_steps_requested")
        accepted = result.get("n_steps_accepted")
        return (result.get("status") == "CONVERGED"
                and type(requested) is int and requested > 0
                and type(accepted) is int and accepted == requested)
    if integrator in ("NEWMARK", "STATIC", "MODAL"):
        return result.get("converged") is True
    return False


def reference_completion_on_disk(output_dir, *, series=False):
    """Inspect the durable solver summary, never infer success from OS rc alone.

    The outcome is deliberately fail-closed when the JSON is missing, partial,
    inconsistent or unreadable. It is a GUI classification, not a solver gate.
    """
    out = Path(output_dir)
    irk = not series and (out / 'GAUSS2_MVI_HISTORY.json').is_file()
    filename = ('MVI_1_8_SUMMARY.json' if series else
                'GAUSS2_MVI_HISTORY.json' if irk else 'FINAL_RUN_SUMMARY.json')
    try:
        data = json.loads((out / filename).read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('summary is not a JSON object')
    except (OSError, ValueError, TypeError) as exc:
        return {'success': False, 'accepted': 0, 'requested': None,
                'status': 'SUMMARY_MISSING_OR_INVALID', 'detail': str(exc)}
    mode = 'GAUSS2_IRK4' if irk else ('MODAL' if data.get("calculation_kind")=="MODAL" else 'NEWMARK')
    successful = result_succeeded(data, integrator=mode, series=series)
    if series:
        rows = data.get('results')
        rows = rows if isinstance(rows, list) else []
        accepted = sum(isinstance(row, dict) and row.get('converged') is True
                       for row in rows)
        ranks = data.get('ranks')
        requested = len(ranks) if isinstance(ranks, list) else None
        successful = successful and requested is not None and requested > 0 and accepted == requested
    elif irk:
        accepted = data.get('n_steps_accepted', 0)
        requested = data.get('n_steps_requested')
    else:
        accepted = data.get('accepted_steps', 0)
        requested = data.get("n_modes") if mode=="MODAL" else None
        if mode=="MODAL":
            successful = (successful and data.get("status")=="MODAL_COMPLETED"
                          and (out / "REFERENCE_MODAL_MODES.npz").is_file()
                          and accepted==requested)
    if type(accepted) is not int or accepted < 0:
        accepted, successful = 0, False
    if requested is not None and (type(requested) is not int or requested < 1):
        requested, successful = None, False
    if requested is not None and accepted > requested:
        successful = False
    return {'success': bool(successful), 'accepted': accepted,
            'requested': requested, 'status': str(data.get('status', 'UNKNOWN')),
            'detail': str(data.get('failed_step_code', data.get('failure_detail', '')))}


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit('Usage: restored_gui_runner.py CASE INTEGRATOR [--series]')
    if len(sys.argv) == 4 and sys.argv[3] != '--series':
        raise SystemExit('Unknown option')
    series = len(sys.argv) == 4
    plan = plan_time_integrator_run(sys.argv[1], sys.argv[2], series=series)
    result = execute_time_integrator_plan(plan)
    ok = result_succeeded(result, integrator=plan.integrator, series=series)
    print('RESULT: status=%s converged=%s accepted_steps=%s success=%s' % (
        result.get('status'), result.get('converged', result.get('all_converged')),
        result.get('accepted_steps', result.get('n_steps_accepted')), ok), flush=True)
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
