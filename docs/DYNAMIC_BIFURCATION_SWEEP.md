# Dynamic cylindrical-shell parameter studies

## What is implemented

The main Tk window has a **Бифуркации** button. It opens
`interface/cylinder_dynamic_sweep_gui.py`, which validates a previously
saved DYNAMIC CASE and builds a study JSON with one varied scalar input.

`interface/cylinder_dynamic_parameter_sweep.py` executes independent
cold-start simulations with the existing nonlinear-elastic Newmark FDM or
fixed-rank MVI backend. It does not perform arc-length continuation, detect
equilibrium branch stability or adjust MVI rank. Each candidate is validated
before any result folder is created. An incomplete run is excluded from the
parameter diagram.

Outputs of each completed study:
- `CASES/CASE_XXXX.json` — exact inputs for each parameter value.
- `RUNS/RUN_XXXX/` — accepted physical field states and summaries.
- `SWEEP_STATUS.json` — atomic progress/error status, shown by the GUI.
- `BIFURCATION_RUNS.csv` — complete-run manifest.
- `BIFURCATION_ANALYTICS/06_BIFURCATION.xlsx` — peaks, forcing-period
  Poincare samples when available, and scalar Lyapunov estimates.
- `BIFURCATION_ANALYTICS/*.png` — plotted diagrams when data suffice.

Each dynamic run also produces six `DYNAMIC_ANALYTICS/*.xlsx` books and
PNG figures by default. Setting `export_each_run: false` suppresses only
per-run postprocessing, not saved physical states.

## Example study JSON

Place the study JSON in `config/`, next to the example base CASE:

```json
{
  "base_case": "CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json",
  "parameter_path": "load_definition.q",
  "values": [0.005, 0.01, 0.02],
  "output_dir": "../OUTPUT_DYNAMIC_PRESSURE_SWEEP",
  "parameter_name": "pressure_q",
  "probe_name": "P01_CENTER_0",
  "transient_fraction": 0.5,
  "export_each_run": true
}
```

Paths in the study file are resolved relative to that file. Run from the
project root:

```sh
python RUN_DYNAMIC_PARAMETER_SWEEP.py config/my_study.json --preflight
python RUN_DYNAMIC_PARAMETER_SWEEP.py config/my_study.json
```

Never use a nonempty output folder: the program intentionally refuses to
overwrite previous simulations. GUI operations run on a worker thread and
read atomic progress snapshots while preserving the main Tk event loop.

## Scientific limitations

A **bifurcation diagram from independent forced-response runs** is not
continuation of branches. The saved scalar **Rosenstein estimate** of the
leading Lyapunov exponent is not a verified spectrum of the coupled
PDE. The QR full-spectrum method needs actual constrained state-transition
matrices from the nonlinear time integrator; the solver does not yet produce
those matrices. Incomplete trajectories must not enter scientific plots.

The 20-step demonstration CASE is far too short for meaningful analysis of
1 Hz forcing: configure a physically sufficient number of periods, remove
transients and verify time-step, grid and rank sensitivity before reporting
quantitative conclusions. No material plasticity or unrelated constitutive
branch is enabled.

## Implementation references

- `interface/FINAL_GUI.py`
- `interface/cylinder_dynamic_sweep_gui.py`
- `interface/cylinder_dynamic_parameter_sweep.py`
- `numerics/cylinder_dynamic_analytics.py`
- `tests/test_dynamic_parameter_sweep.py`
- `tests/test_dynamic_sweep_gui.py`
