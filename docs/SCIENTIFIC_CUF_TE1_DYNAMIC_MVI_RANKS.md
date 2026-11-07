# Real coupled Newmark dynamics — FDM and MVI-1…8

Theory CUF TE1, MICRO Green–Lagrange, gradient Gz with flexoelectricity ON
and two electric thickness coefficients. Non-axisymmetric fixed
reference-face patch pressure, matching electrode/support conditions.

The actual full-nodal FDM and rank-fixed separated MVI solutions share
one exactly hashed continuum physics passport. There is no electric
inertia. The chosen experiment uses the same 8×8 discretization and
NEWmark constant-average-acceleration (beta=1/4, gamma=1/2) over
**two steps**, dt=0.01 s, t_end=0.02 s.

A bounded push pilot executes four real cases: FDM and MVI 1, 2, 3.
The optional workflow_dispatch scope=full executes nine real cases:
FDM and MVI 1…8. Results are not claimed until individual physical
GitHub Actions jobs finish and are inspected.

Each NPZ from each time level, plus initial state and CASE, is retained.
The checker requires both accepted steps, consistent time, no electric
inertia, physical motion/electric/stress/induction arrays, and stored
SHA256 for every accepted state. For each rank it computes L2 norms
against FDM **at the same instant**, including motion (velocity and
acceleration) and electrical fields.

Rank-limited dynamic acceptance is **not** independent continuum
accuracy or a substitute for temporal step refinement, spatial
grid refinement, or verification of Bathe and Gauss IRK4. This
8×8 study is deliberately bounded; periodic even-grid behavior needs
separate spatial accuracy control. The coupled FDM and MVI currently
share the same variational physical assembly, so this is comparison
of representation/algorithm, not independent physics implementations.
