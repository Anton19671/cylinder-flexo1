# Cylinder-flexo1 native strong dynamics: engineering completion gates

Date: 2026-11-09. This document is an accurate status inventory, not a
declaration that the full nonlinear solver has been certified.

## Newly committed, opt-in modules

- cylinder_native_mkr_newmark.py: independent native strong MKR Newmark
  (all seven theory kinematics), with parameter-stability preflight,
  simultaneous electrical field, stage-specific inertia and failure
  rejection.
- cylinder_native_mkr_bathe.py: two-stage Bathe with first trapezoidal
  stage, second three-point stage, loads at both physical stage times,
  original three-point kinematic audit and atomic full-step acceptance.
- cylinder_native_mkr_initial.py: initial acceleration and electrical
  potential determined simultaneously at specified displacement/velocity.
  Checks initial essential displacement/velocity traces and imposes zero
  acceleration on stationary essential traces. Moving supports are explicitly
  unsupported.
- tests/test_native_mkr_newmark.py, tests/test_native_mkr_bathe.py,
  tests/test_native_mkr_initial.py: mocked-callback algebraic checks only.

These modules intentionally DO NOT modify production GUI/CASE dispatch.
Importing them does not start a physical simulation. They are experimental
and require explicit caller-supplied physical end conditions, principal
half-orders, dimensional scales, acceleration/initial data and loads.

## Non-negotiable blockers for a complete scientific solution

1. FULL coupled G3D/Gz native MKR face gradients and conjugate fluxes,
   including even angular grids, source-consistent primal/dual placement,
   positive nodal inertia, and reciprocal local energy identities.
2. Independently certified axial natural/essential closure at orders two,
   four and six, including mechanical microfree traces and insulated
   electric boundaries. Do not infer PDE order from the highest jet.
3. Fully separate non-linear coupled MVI-1..8 assembly with its own
   weak surface discretization, own X/Y Newton and own stage solvers.
   No shared discretization or MVI-to-MKR fallback; no SVD.
4. Native, method-specific Gauss IRK4 route and complete initial-state
   integration with all time methods; independently check electrical
   equilibrium and mass/gyroscopic work at t0.
5. Real CASE and GUI wiring only AFTER (1)-(4), with reliable cancel/
   pause and incremental graphical monitoring, and atomic exported results.
6. Actual automated tests: import/compile, unit/interface tests,
   manufactured solutions, discrete power/symmetry tests, static/eigen/
   transient convergence, rank refinement, and benchmark comparisons.
   Source inspection or an unloaded exact state is NOT a loaded scientific
   calculation.

## Execution status

GitHub connector edits succeed. The container could not clone the GitHub
repository (external DNS unavailable), so the complete repository test
suite could not be run from that container. No GitHub Actions workflows
were manually triggered. No loaded shell calculation or time history was
performed. Therefore new files/changes are COMMITTED BUT NOT VERIFIED by
execution in this interaction, and their production readiness is unproven.

The older reference variational routes are retained for reproducibility.
They may not be advertised as independent coupled MKR/MVI algorithms.
