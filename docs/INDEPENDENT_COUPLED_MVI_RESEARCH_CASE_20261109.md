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


## 2026-11-10: resumable long independent MVI time integration

The opt-in coupled MVI dynamic research runner now accepts up to
**100,000 explicitly specified time steps** (formerly 120); this is an
input capacity, **not** a claim of acceptable runtime or verified accuracy.
The physical integrators and rank-R X/Y solution remain unchanged.

Launch a NEW job with the same research schema:

    python -m independent_mvi.run_coupled_research --case CASE.json --run

If the job stops AFTER at least one committed state and BEFORE completion,
resume **with the unchanged CASE.json and original initial/load NPZ files**:

    python -m independent_mvi.run_coupled_research --case CASE.json --resume

A resume is never implicit. A new `--run` refuses an existing result folder.
`--validate-only` remains non-solving and never creates output; it cannot
be mixed with `--resume`. STATIC resume is unsupported.

The output folder contains `STEP_0000.npz` (the initial closed
mechanical/electric/acceleration state), `STEP_NNNN.npz` for each accepted
time step, `PHYSICAL_RADIAL_HISTORY.csv`, `RUN_SUMMARY.json`,
`CASE.json` and a writer-only `RUN_LOCK`.

The restart implementation:
- exclusively locks one output directory; a second writer fails rather
  than silently overwriting data, and the OS releases the lock on crash;
- stores SHA-256 of the **original CASE and NPZ bytes** in the first
  run summary; any input mutation makes restart fail;
- uses `RUN_SUMMARY.accepted_steps` as the commit marker;
- loads the exact accepted rank-R factors, velocity, acceleration and
  physical electric/mechanical states without solving earlier steps;
- verifies reconstructed physical CUF U/V/W surface fields at all
  three thickness stations and checks the physical time of the step;
- checks the CSV history already contains every committed step and
  truncates only records written beyond the last commit marker;
- deletes only specifically named, uncommitted `STEP_NNNN.npz` files
  above that marker, preserving all accepted results and unrelated files;
- appends new physical-history CSV rows in O(1) time per step, avoiding
  a full rewrite of the history at each iteration;
- refuses corrupted, incompatible, completed or non-dynamic checkpoints,
  without changing the saved scientific results or summary.

There is no automatic interpolation, no rank reduction, no MKR fallback,
no reinterpretation of modified source data and no fabricated acceptance.
The time integrator is restarted from the *full physical state* of the
previously accepted step.

The added no-solve tests simulate a failure at step 2, verify continued
acceptance of steps 2-3 without a repeated initial closure, exercise
changed input and corrupted NPZ rejection, and check orphan cleanup.
A 3,000-step case is **validated only**; no long loaded shell calculation
has been performed in this conversation.

Scientific accuracy, higher-order natural boundary physics, robust loaded
dynamics for all ranks, and independent references remain unverified.


### Checkpoint hardening (2026-11-10, subsequent commits)

Every accepted dynamic `STEP_NNNN.npz` is now fingerprinted by SHA-256
in `RUN_SUMMARY.last_state_sha256` **after** the whole NPZ and physical
CSV row are written. On resume the last accepted NPZ is rehashed before
reconstructing the field. This detects changes to mechanical/electric
coefficients, physical face fields, **velocity and acceleration**; a
changed checkpoint must never be silently accepted merely because its
displacement product still matches the stored factors.

The committed physical CSV history is also validated: column schema,
step indices, physical times and finite values must be consistent, and
its last accepted measured displacements must equal the reconstructed
checkpoint observables to floating-point precision. Uncommitted trailing
rows may be discarded only *after* all committed data pass validation.

For long trajectories, the checkpoint time comparison accommodates
floating-point accumulation from repeated `dt` additions, with a hard
bound below 1% of a time step. It does not silently relabel a missing
time step. Tests cover changed velocity/acceleration, corrupted physical
history, and additive drift from 100,000 increments. These tests remain
algebraic / mocked, without loaded physical shell calculations.
