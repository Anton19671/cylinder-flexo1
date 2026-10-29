"""Normalize VARIATIONAL_REFERENCE events without inferring accepted states.

Only the durable FINAL_RUN_SUMMARY and accepted NPZ files count for progress.
Events can show currently running iterations; they cannot mark a step accepted.
"""
def normalize_reference_event(event):
    if not isinstance(event, dict):
        return None
    kind=str(event.get("event",""))
    if not kind.startswith("reference_"):
        return None
    step=event.get("step",event.get("step_index"))
    try:
        step=int(step)
    except (TypeError,ValueError):
        return None
    if step<1:
        return None
    inner=kind[len("reference_"):]
    iteration=event.get("iteration")
    try:
        iteration=int(iteration) if iteration is not None else None
    except (TypeError,ValueError):
        iteration=None
    value=event.get("projected_residual_scaled")
    if value is None:value=event.get("residual_scaled")
    return {"event":"reference_iteration","step_index":step,
            "iteration":iteration,"method":"MVI" if "sweep" in inner or "halfstep" in inner else "FDM",
            "solver_event":kind,"residual_value":value,
            "direction":event.get("direction"),
            "sweep":event.get("sweep"),
            "load_factor":event.get("load_factor"),
            "elapsed_s":event.get("elapsed_s"),
            "time_s":event.get("time_s")}


def accepted_reference_steps(output_dir):
    """Read accepted steps from durable summary; ignore failed iterate files."""
    import json
    from pathlib import Path
    root=Path(output_dir)
    try:
        summary=json.loads((root/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    except (OSError,ValueError,TypeError):
        return []
    rows=summary.get("steps",[])
    if not isinstance(rows,list):
        return []
    result=[]
    for i,entry in enumerate(rows,1):
        if not isinstance(entry,dict) or entry.get("converged") is not True:
            break
        if entry.get("index") != i:
            break
        state=entry.get("state_file")
        expected=("REFERENCE_DYNAMIC_STEP_" if summary.get("calculation_kind")=="DYNAMIC"
                  else "REFERENCE_STEP_")+f"{i:04d}.npz"
        if state!=expected or not (root/state).is_file():
            break
        result.append(entry)
    return result
