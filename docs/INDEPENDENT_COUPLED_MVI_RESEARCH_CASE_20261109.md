# Integrated independent coupled MVI research CASE — 2026-11-09

## What now reaches the official calculation dispatcher

`interface/cylinder_final_stage11a_dispatcher.py` ->
`interface/cylinder_independent_dispatch.py` ->
`independent_mvi/run_coupled_research.py`.

Explicit native schema: `INDEPENDENT_MVI_COUPLED_RESEARCH_V1`.
This is an opt-in **research** route, not a default production route.
It does not replace old native independent linear runners or reference
G3D/Gz cases, and does not import MKR numerical discretization.

Seven theories: KL, FSDT, Reddy, CUF TE1–TE4. Three geometry modes
(linear/VK/GL); material gradient modes G3D/Gz; electric modes OFF
or explicit grounded electrodes. Boundary subset is stationary and
homogeneous parallel mechanical ends. Static and native dynamic
Newmark, Bathe and Gauss2 have a single strict CASE entry.

A CASE requires `research_opt_in=true` and exactly these top-level
fields: schema, research_opt_in, physics, grid, boundary, solver,
rank, initial_state_npz, loads_npz, output_dir. Inputs are local
JSON and NumPy NPZ, not auto-converted or silently interpreted as
other CASE schemas.

Required physics fields:
theory, geometry_model, material, thickness_m, density_kg_m3, eta_s,
mass_damping_rate_per_s, n_gauss, potential_order.
Material: young_pa, poisson, ell_m, mu1_c_per_m, mu2_c_per_m,
kappa_f_per_m, gradient_model.
Grid: nx, ns, length_m, radius_m.
Boundary: mechanical, electric.
Solver: kind, integrator, dt_s, steps, field_scales, energy_scale_j,
max_sweeps, max_acceleration_coordinates, max_halfstep_coordinates.

Initial NPZ:
`x_profiles` shape (n_fields, rank, nx),
`y_profiles` shape (n_fields, rank, ns);
for dynamic cases also `velocity` shape (n_mechanical, nx, ns).
Load NPZ: `mechanical_load` shape (n_mechanical, nx, ns);
when electric ON, `free_charge` shape (n_electric, nx, ns).
The number of mechanical fields is KL=3, FSDT/Reddy=5 and
CUF_TE_k=3(k+1). The number of electric fields is potential_order+1
when electricity is active.

The runner verifies:
- independent surface/rank/field shape, no SVD or method fallback;
- explicit stationary essential mechanical end traces and supported
  grounded electrical end declarations;
- dimensional input validity and strict positive state/equation scales;
- actual rank*field*grid budget for each Y and X nonlinear halfstep;
- actual mechanical free-node budget for independent initial/endpoint
  acceleration inversion;
- output path availability before creating any output;
- explicit, matching static or dynamic integration method.

Validation is **non-solving** and creates NO output directory.

For explicit execution only, the runner creates an isolated directory
with CASE.json, RUN_SUMMARY.json, and atomic accepted-state NPZ files.
Static acceptance saves STATIC_ACCEPTED.npz. Dynamic acceptance stores
STEP_0000.npz (after coupled initial-state closure) and one file per
accepted time step. Failure records FAILED in RUN_SUMMARY.json and
never reports a failed time step as accepted. Outputs always record
scientific_accuracy_certified=false and
physical_boundary_closure_certified=false.

The current dynamic load input is a *constant-in-time* field from
loads_npz; the Bathe/Gauss stages correctly receive those fields at
their individual physical stage times. General time-varying load
histories are not yet exposed in this CASE schema.

## Invocation

Validate through the official final dispatcher or directly through:

    python -m independent_mvi.run_coupled_research --case CASE.json --validate-only

Execute only by explicit user action:

    python -m independent_mvi.run_coupled_research --case CASE.json --run

A normal `run_final_case(CASE.json)` also selects this new runner if
and only if this schema and research opt-in are present. GUI case creation
and GUI controls have not been changed or advertised as certified.

## Engineering status

Main branch now contains the method-specific spatial weak work,
fixed-rank X/Y nonlinear static solver, exact local inertial and viscous
work, Newmark/Bathe/Gauss2, rank-exact Gauss endpoint closure and
initial electric/acceleration closure, wired to one research CASE.

Tests for the 7x3 dry-run matrix, high-rank resource budgets and
no legacy/MKR dispatch were committed. They have **not been executed
in this conversation**. No loaded cylinder simulation was launched.

This is NOT proof of scientific readiness. The following remain
blocking: mathematically certified higher-order mechanical natural
conditions, microfree electric face/edge work, floating electric gauge,
source-model reciprocity and tangent validation, independent grid/rank/
time convergence, loaded physical benchmarks and comprehensive GUI
integration. Do not enable this path as an ordinary production default.
