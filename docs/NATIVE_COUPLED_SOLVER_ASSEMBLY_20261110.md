# Native independent coupled cylinder solver assembly — 2026-11-10

This checkpoint records the **assembled executable research architecture**,
not a claim of completed scientific verification.

## Unified CASE dispatch

`interface/cylinder_final_stage11a_dispatcher.py` and
`interface/cylinder_independent_dispatch.py` now route two EXPLICIT
research schemas without silent fallback to legacy reference engines.

- `INDEPENDENT_MKR_COUPLED_RESEARCH_V1` ->
  `independent_mkr/run_coupled_research.py` ->
  MKR-owned `DirectStrongShell`, `direct_strong_jet_balance`,
  `assemble_physical_end_rows` and `solve_strong_nonlinear_equations`.
- `INDEPENDENT_MVI_COUPLED_RESEARCH_V1` ->
  `independent_mvi/run_coupled_research.py` ->
  MVI-owned SBP42/Fourier variational `NativeCoupledMVISurface`,
  exact X/Y frozen-partner iterations and rank-1..8 without SVD.

The two numerical routes share ONLY local physical G3D/Gz constitutive,
kinematic and inertial laws. They DO NOT share spatial differentiated
equations, nonlinear tangent, end rows or a Newton driver.

## Available physical research models

Each route supports KL, FSDT, Reddy, CUF TE1–TE4, geometries
LINEAR/VON_KARMAN/GREEN_LAGRANGE, G3D/Gz materials, electromechanical
coupling and electricity OFF. Native MKR accepts explicit per-field
axial differential half-orders and end equations; the order of stored
kinematic jets is NEVER used as a substitute for the certified
principal differential order.

Native MVI supports explicit fixed-rank 1–8 separated X/Y solutions.
For large reduced halfsteps the OWN matrix-free Krylov route is used;
small problems may use dense LU. Native strong MKR already has an
independent matrix-free GMRES strong Newton action.

## Native time integration and electric closure

For both methods the static branch is separate from dynamic branches.
Newmark, two-stage Bathe and Gauss–Legendre IRK4 use their native
stage equations and own force assembly. Dynamic initial velocity and
acceleration closure is method-owned. Both methods support explicit
electrical ends, grounded modes and a SINGLE floating potential gauge
for insulated electric axial ends, with total free charge compatibility.

Both methods accept curvature-weighted physical electrical charge on
the real inner/outer radial surfaces and save physical face induction
diagnostics. These diagnostics do NOT enforce a pointwise radial face
condition that the thickness-reduced physical shell basis may be
unable to satisfy.

Mechanical immobile parallel ends are explicitly represented.
Native MVI supports generalized first/second normal derivative end
work through exact variational SBP transpose; native strong MKR
supports explicit generalized natural end rows, with independent
source-appropriate outward normal conventions. Nonzero traction
conjugate to the already-fixed displacement is rejected.
No physically unsupported source microfree closure is silently added.

## Executable GUI / results

Existing GUI tabs, legacy run and output controls remain in place.
The main GUI has an opt-in **Native solver** menu opening the
new `interface/cylinder_native_research_gui.py` window. It can
open/edit/save JSON CASEs, choose output directories, preflight,
launch the selected native method in a separate process, request
pause between accepted dynamic steps, stop, view logs and plot
the accepted central deflection online from atomic NPZ files.

Research runner outputs use `CASE.json`, `RUN_SUMMARY.json`,
`STATIC_ACCEPTED.npz` or `STEP_####.npz`, and record that physical
scientific accuracy and source natural-boundary closure are uncertified.
An unsuccessful solve cannot create a falsely accepted step. A
failure marks the summary as FAILED when Python control is returned
(the OS may terminate an active process without completing that
summary).

## Explicit independent CASE fields

Native strong MKR requires top-level schema, research_opt_in, physics,
grid, boundary, solver, initial_state_npz, loads_npz, output_dir.
Optional electric_faces and mechanical_end_work retain exactly their
declared physical meaning. Its boundary includes mechanical,
electric and **axial_orders for every mechanical field**.
The strong solver includes dimensional equation and state scales
for mechanical/electric fields and a declared matrix-free DoF budget.

Native fixed-rank MVI requires its own independently defined schema,
X/Y factor NPZs, rank, physical scales and reduced halfstep limits.
The two schemas cannot be silently interconverted.

## Important remaining scientific work

This is an **assembled experimental execution system**, not a
scientifically certified solver. In particular, original-source
G3D/Gz mechanical face and microfree edge conditions, compatibility
of physical principal orders with every source parameter limit,
pointwise electric face enforcement where appropriate, production
grade GUI interaction and larger mesh/rank/time/physical benchmark
convergence remain open. Full testing and controlled physical-case
validation follow the assembly phase, not vice versa.

No loaded cylinder physical run was launched during assembly.
