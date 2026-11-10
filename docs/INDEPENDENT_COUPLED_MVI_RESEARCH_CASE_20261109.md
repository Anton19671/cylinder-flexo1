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

Optional nonlinear X/Y reduced tangent options (backward-compatible):
`linear_solver` = `AUTO`, `DENSE_LU` or `MATRIX_FREE_GMRES`;
`dense_coordinate_limit` (default 96);
`krylov_restart` (default 24);
`krylov_maxiter` (default 60);
`krylov_relative_tolerance` (default 1e-5).
The `AUTO` mode performs dense LU for small fixed-rank halfsteps,
then switches to the independent matrix-free GMRES tangent action above
the dense-coordinate limit. Krylov vectors live **only** in the X or Y
reduced factor coordinates. Two directional physical MVI weak-work
evaluations form each matrix-vector action; it never constructs or
factorizes the full nonlinear dense tangent. No SVD, pseudoinverse,
grid-field Newton or MKR fallback is permitted. The existing
`max_halfstep_coordinates` remains an explicit size safety ceiling,
not a dense allocation requirement.

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

## No-solve CI (2026-11-09 update)

`.github/workflows/independent-coupled-mvi-no-solve-contracts.yml`
now runs on changes to native coupled MVI source and tests on `main`,
as well as by explicit manual dispatch. It includes the
`tests/test_independent_coupled_mvi_matrix_free.py` synthetic
quadratic tests for X, Y, automatic method choice, Krylov failure and
forbidden numerical backends. The workflow is strictly algebraic,
syntax and method isolation only; it does **not** run a physical
cylinder time-history/loaded benchmark.

**Execution status:** do not claim success before an actual GitHub
Actions result has been observed. The independent MVI coupled research
route remains experimental pending source higher-order boundary and
electric face-condition validation.


## 2026-11-10: time-dependent CASE forcing and physical CUF thickness history

The **opt-in independent coupled MVI research runner** now accepts one
optional top-level `load_time_law` object for DYNAMIC cases:

    "load_time_law": {
      "kind": "TABULATED_LINEAR",
      "time_s": [0.0, 0.005, 0.01, 0.02],
      "mechanical_scale": [0.0, 0.3, 1.0, 0.0],
      "electric_scale": [1.0, 0.8, 0.2, 0.0]
    }

All entries are finite dimensionless multipliers, with strictly increasing
physical times in seconds, starting at zero. The tabulation must cover
`solver.steps * solver.dt_s`. The optional electric multiplier defaults
to the mechanical multiplier if electricity is ON; it is forbidden when
electricity is OFF. No extrapolation is performed. The tabulation is
linearly interpolated in **time only**; spatial nodal shape is preserved.
For STATIC, specifying `load_time_law` is an error.

The mechanical source from `loads_npz.mechanical_load` and the electric
volume source from `loads_npz.free_charge` are multiplied independently.
**Physical inner/outer face CHARGE is not scaled by the volume source law:**
the separate `electric_faces` contribution stays constant, respecting its
own prescribed boundary work. The floating gauge total free-charge test
covers every tabulation knot before any output directory is made.

- Initial electric/acceleration closure uses physical sources at `t=0`.
- Newmark uses the source at its physical END time.
- Bathe uses the source separately at `t+gamma*dt` and `t+dt`.
- Gauss IRK4 uses the source at both Gauss collocation nodes and for
  its separate END electric/acceleration closure.
- Omitted `load_time_law` preserves the previous constant-load behavior.

### Physical displacement output

Each accepted dynamic `STEP_NNNN.npz` for CUF TE1..TE4 now includes
`physical_z_stations_m` shape `(3,)` and
`physical_uvw_stations_m` shape `(3,3,nx,ns)`. Axes are:
`[inner/middle/outer, U/V/W, axial-node, circumferential-node]`.
The three thickness levels are `-h/2, 0, +h/2`.

Recovery uses the **actual active** weak-form thickness basis
`F_k(z)=(2*z/h)^k`, not the incompatible dimensional `z^k` legacy
provider. In particular, `W(outer)=sum_k W_k` and
`W(inner)=sum_k (-1)^k W_k`; neither is generally `W_0`.
Static accepted CUF NPZ now contains the same physical stations.

The dynamic runner exports `PHYSICAL_RADIAL_HISTORY.csv` containing
the time index and physical radial displacement at the axial midpoint
(`nx//2`) and angular node zero, plus the maximum absolute radial
value on each CUF face. For KL/FSDT/Reddy only the reference-surface
`W` is filled: no fictitious CUF thickness recovery is asserted.

The generated fields and histories are **diagnostic numeric outputs, not
external-reference verification**. Boundary microfree/electrical source
closure, rank sufficiency, mesh and time refinement, and scientific
comparison remain independently required. The tests added here mock
the time integrators; no loaded physical shell run was launched.
