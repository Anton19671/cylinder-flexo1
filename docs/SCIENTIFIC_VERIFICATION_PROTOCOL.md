# Mandatory scientific verification protocol (cylinder-flexo1)

This is a **verification checklist**, not a certificate of validated results.
It records the two separate audit tracks requested by the project owner:
the five-field Reddy Green–Lagrange mathematical audit (8 October 2026)
and the genuinely independent MKR/FDM–MVI spatial/dynamic audit
(4 November 2026). No production solver should be silently changed merely
to make an algebraic or physical test pass.

## A. Physical and mathematical model: five-field Reddy + GL

1. **Geometry and kinematics.** Derive the displacement map, cylindrical
   metric, and deformation gradient `F` from the chosen five generalized
   fields. Derive *symbolically and independently* all six Green–Lagrange
   components `E_XX, E_YY, E_XY, E_XZ, E_YZ, E_ZZ`. Compare coefficient
   identities at each thickness order to implementation. For `E_ZZ`,
   distinguish the kinematic full-GL value from whether and why this
   component enters the *reduced* shell energy; it must not be omitted by
   an unproved assumption.
2. **Constitutive reduction and energy.** Derive the reduced transverse
   plane-stress shell energy from the intended constitutive law. Verify
   its relation to a full 3D GL formulation and any eliminated thickness
   variable. Check the reduced-energy quadratic form is nonnegative
   within its stated assumptions. Do **not** require the tangent stiffness
   to be positive definite in compression/buckling regimes.
3. **Variations and equilibrium.** Compute `delta Pi` and the appropriate
   stress transformations, including `P = F S` when the stated 3D
   hyperelastic formulation applies. Independently derive all five
   generalized equilibrium/variational equations and their coupling.
   Symbolically cross-check first variations and the consistency/symmetry
   of the tangent in the range where a potential exists.
4. **Natural boundary conditions.** Derive the complete integration-by-parts
   boundary terms for the five-field theory, including higher-order
   edge terms and **corner contributions**, not only essential clamps.
   Verify all implemented boundary-condition types against the derived
   work-conjugate traces.
5. **Thickness and forcing.** Verify layerwise Gauss thickness integration,
   metric factors, interfaces, integration bounds, and surface-pressure
   work: reference pressure vs follower load must be distinguished.
6. **Consistent limiting theories.** Independently verify the LINEAR,
   von Karman and full Green–Lagrange reductions from the *same*
   displacement kinematics. For von Karman, identify the retained and
   omitted terms explicitly; do not mix truncation orders.
7. **Physical checks.** Verify shear and other free-surface traction
   conditions **without overwriting computed stress profiles** to force
   artificial zeros. Verify geometric objectivity/rigid motions, the
   zero-curvature limit where applicable, and analytic/manufactured
   benchmarks. Compare local stress and displacement fields, not just a
   single center deflection.
8. **No premature edits.** Complete the symbolic/reference comparison
   before altering sensitive physics formulas. Every correction requires
   its own failing-before/fixed-after regression with physical scope and
   applicable theory clearly identified.

## B. Numerical independence and dynamics: MKR versus MVI

1. Trace actual entry points, operators, assembly routes and every
   integrator (Bathe, Newmark, two-stage Gauss IRK4); there must be no
   hidden invocation of the opposite solver or a shared spatial
   variational solve. Separate CASE, logs, and output folders.
2. Verify MVI `X/Y` factor initialization, frozen-partner
   `Y -> X` sweeps, updates and field-preserving normalization, while
   retaining the *separated-function MVI algorithm* (never substitute
   the full-grid MKR solution).
3. Rank oracle: use a manufactured nonseparable **rank-two field**.
   MKR/full-grid space and MVI rank 2 or 8 can represent its spatial
   rank (subject to the admissible boundary subspace); MVI rank 1
   cannot reproduce it exactly. Rank-capacity claims must be checked
   with the actual clamped basis dimensions.
4. Repeat algebraic checks across KL, FSDT, Reddy and CUF TE1–TE4,
   and check stage/intermediate and endpoint factors independently
   for Bathe, Newmark and Gauss IRK4.
5. Validate spatial convergence, temporal order, rank sensitivity,
   reference frequencies, static deflections and time histories using
   **independent external benchmarks**. MKR-vs-MVI agreement alone
   is not an external accuracy proof.
6. Exercise stop/pause, durable accepted-step files, crash recovery
   and restart; guarantee metadata reflects the exact accepted
   solution. Never claim successful restart without an actual test.

## C. Coupled flexoelectric research scope

Independent `independent_mkr` and `independent_mvi` seven-theory
routes are currently experimental **linear, non-electrical** mechanics.
Do not infer that they validate the original coupled G3D/Gz/flexoelectric
code. Its electromechanical energy, gradients, reciprocity, electric
boundary conditions, units, and exact physical benchmarks require a
separate staged audit.

## Status and rules

- **Protocol recorded:** yes.
- **Every physical equation symbolically checked:** not yet established.
- **Full flexoelectric solver scientifically certified:** no.
- Preserve existing GUI layout, tabs and controls unless a specific
  interface defect is separately confirmed.
- Prefer small, non-physical algebraic tests first; submit the smallest
  necessary GitHub Actions physical verification only when justified.
- Do not report doubtful, unverified numerical reference values as facts.
