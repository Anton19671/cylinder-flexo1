"""Explicit physical CASE entrypoint for experimental rank-fixed Gauss2 MVI.

Reuses *exactly* the validated Newmark variational physical setup but does
not switch the established Newmark dispatcher. Never synthesizes fields.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from cylinder_reference_case import prepare_reference_case
from cylinder_gauss2_rk import Gauss2Settings, Gauss2State
from cylinder_gauss2_mvi import Gauss2MVISettings
from cylinder_gauss2_mvi_history import run_gauss2_variational_mvi_history


def prepare_gauss2_mvi_case(case, *, base_dir=None):
    """Validate experimental eligibility BEFORE writing any outputs."""
    if not isinstance(case,dict):
        raise ValueError("GAUSS2_CASE_REQUIRES_OBJECT")
    prepared=prepare_reference_case(case,base_dir=base_dir)
    if prepared.calculation_kind!="DYNAMIC" or prepared.method!="MVI":
        raise ValueError("GAUSS2_CASE_REQUIRES_DYNAMIC_MVI")
    if prepared.fdm_backend!="DENSE_REFERENCE":
        raise ValueError("GAUSS2_CASE_REQUIRES_DENSE_REFERENCE")
    if prepared.dynamics is None or prepared.mvi_settings is None:
        raise ValueError("GAUSS2_CASE_MISSING_PHYSICAL_DYNAMICS")
    dyn=prepared.dynamics
    dt=dyn.settings.dt_s
    settings=Gauss2Settings(dt_s=dt)
    mvi=Gauss2MVISettings(rank=prepared.mvi_settings.rank,
                          max_sweeps=prepared.mvi_settings.max_sweeps)
    initial=Gauss2State(dyn.settings.t_start_s,prepared.mechanical,
                        np.zeros_like(prepared.mechanical),prepared.electrical)
    # All physical loads are projected by prepare_reference_case; do not
    # replace face pressure with an ad hoc generalized force.
    def loads_at(t):
        factor=dyn.time_law.value(t)
        return (factor*prepared.dead_load,
                None if prepared.free_charge is None else
                factor*prepared.free_charge)
    # Time laws are checked by prepare_reference_case on Newmark nodes.
    # IRK also samples two *interior* Gauss times. Check their domains
    # explicitly before running; tabulated data must not extrapolate.
    from cylinder_dynamic_loads import normalize_time_law
    t0=dyn.settings.t_start_s
    for idx in range(dyn.settings.n_steps):
        for c in ((3.-np.sqrt(3.))/6.,(3.+np.sqrt(3.))/6.):
            t=t0+(idx+c)*dt
            val=dyn.time_law.value(t)
            if not np.isfinite(val):
                raise ValueError("GAUSS2_CASE_NONFINITE_STAGE_LOAD_FACTOR")
    return prepared,settings,mvi,initial,loads_at


def run_gauss2_mvi_case(case_path, *, output_dir=None):
    """Run opt-in IRK4 MVI from real physical JSON; Newmark stays default."""
    path=Path(case_path).expanduser().resolve()
    with path.open("r",encoding="utf-8") as f:
        case=json.load(f)
    prepared,settings,mvi,initial,loads_at=prepare_gauss2_mvi_case(
        case,base_dir=path.parent)
    if output_dir is None:
        raise ValueError("GAUSS2_CASE_REQUIRES_DISTINCT_EXPLICIT_OUTPUT_DIR")
    output=Path(output_dir).expanduser()
    if not output.is_absolute():
        output=(path.parent/output).resolve()
    newmark_out=Path(case["output_dir"]).expanduser()
    if not newmark_out.is_absolute():
        newmark_out=(path.parent/newmark_out).resolve()
    if output.resolve()==newmark_out.resolve():
        raise ValueError("GAUSS2_CASE_MUST_NOT_OVERWRITE_NEWMARK")
    # Avoid accepting new configuration keys into the established CASE.
    return run_gauss2_variational_mvi_history(
        prepared.problem,initial,settings,mvi_settings=mvi,
        n_steps=prepared.dynamics.settings.n_steps,
        density_kg_m3=prepared.dynamics.density_kg_m3,
        loads_at=loads_at,output_dir=output,
        newton_settings=prepared.settings,
        mass_damping_rate_per_s=prepared.dynamics.mass_damping_rate_per_s,
        eta_s=prepared.dynamics.eta_s,
        physical_case=case)


__all__=("prepare_gauss2_mvi_case","run_gauss2_mvi_case")
