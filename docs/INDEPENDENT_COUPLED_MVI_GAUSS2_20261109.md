# Native independent MVI Gauss-Legendre fourth-order collocation
## Research implementation checkpoint — 2026-11-09

Committed to `main`:
- `independent_mvi/coupled_gauss2.py`
- `independent_mvi/coupled_rank_endpoint.py`
- `independent_mvi/coupled_initial.py`
- changes to `coupled_halfstep.py` / `coupled_static.py` allowing only an explicitly selected field subset to vary
- no-solve tests: `test_independent_coupled_mvi_gauss2.py`,
  `test_independent_coupled_mvi_endpoint.py`,
  `test_independent_coupled_mvi_electric_only.py`
- the manual no-solve CI workflow now includes these modules and tests

## Mathematical route

For the two Gauss nodes `c_i=(3 +/- sqrt(3))/6`, the MVI method
owns a distinct fixed-rank factor pair at each node. The two stage
mechanical accelerations are recovered by inverting the 2x2
`A_Gauss^2` collocation matrix, and velocities use the full
`A_Gauss` matrix. Both stages therefore remain mechanically coupled.
The algorithm alternates stage-specific Y/X nonlinear Newton solves,
evaluates BOTH full admissible nodal first-variation equations, and
requires the two stages to be jointly stationary before proceeding.
Each stage has its own quasistatic electric potential, source loads and
electric charge evaluated at the appropriate physical time.

The endpoint `q_{n+1},v_{n+1}` follows the Gauss weights `b_i=1/2`.
It is not guaranteed to lie on the rank-R manifold. The new
rank-preserving endpoint fitter solves the axial/circumferential
least-squares equations by Gram linear solves and QR-like subspace
updates; it uses NO SVD, no pseudoinverse and no increase of R.
An exact-rank tolerance is mandatory. An insufficient rank rejects
the step instead of silently changing the computed physical endpoint.

After a representable endpoint is obtained, the terminal electric
equilibrium is re-solved at FIXED mechanical factors using the
electric-only Y/X directions. The physical endpoint acceleration is
then obtained by a separate independent MVI mass-operator solve on
unconstrained mechanical nodes. For KL this retains the nonlinear
director acceleration terms; for CUF TE1-TE4 it uses the exact shell
mass moments, and FSDT/Reddy retain their director kinetic law.
The acceleration solve is LINEAR in a at fixed q,v; it is not a
full-displacement-field Newton shortcut. Every failure to close
electricity, acceleration or essential traces rejects the step.
The maximum dense acceleration DOF count is explicit.

## Method isolation

The new Gauss2 module does not invoke the MKR time integrator, MKR
spatial derivatives, legacy variational time solvers, full-node
displacement Newton, SVD or SVD-based rank truncation. Local continuum
constitutive/kinematic laws are shared physics only. The new path
retains the classical fixed-rank X/Y MVI approach.

## IMPORTANT: experimental limitations

1. The physical high-order G3D/Gz end conditions and face flux closure
   are still NOT mathematically/numerically certified. Floating
   electrical gauge is not connected; electrodes must be explicitly
   grounded as supported by the existing boundary subset.
2. The Gauss2 scheme has a formal fourth-order collocation tableau,
   but **fourth-order accuracy on loaded shell fields is not proven**.
   Factor-stage alternating convergence and rank-constrained endpoint
   may fail. Failures must not become accepted physical states.
3. Full initial acceleration and electric equilibrium must be obtained
   via the new native MVI initial/endpoint closure, not guessed from a
   static initial displacement or assumed to vanish.
4. Kelvin-Voigt and mass-rate damping are handled by independent MVI
   weak work. Validation of nonlinear power and temporal stability
   is outstanding.
5. No GUI/CASE/export production dispatch is enabled. No physical
   cylinder calculation or time history was run.
6. New contract tests have been committed, but **have not been
   executed or demonstrated passing** in this conversation. GitHub
   clone from the current compute container failed (DNS unavailable).
   A manually triggered GitHub Actions run of the dedicated algebra
   workflow is needed to establish syntax and import correctness.

Scientific readiness remains BLOCKED until independent loaded-case,
mesh/rank/time convergence, source boundary conditions, and benchmark
comparisons pass. Source presence is not numerical certification.
