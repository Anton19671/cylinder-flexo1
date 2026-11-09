# Cylinder-flexo1 independent nonlinear MVI dynamic integration checkpoint

Date: 2026-11-09. Repository branch: main.

## Actual source additions

The independent coupled MVI now has its own fixed-rank X/Y spatial weak work,
Newton halfsteps and Y->X static iteration, plus the independent nodal
stationarity gate. New dynamic pieces:

- independent_mvi/coupled_inertia.py: mechanical inertial virtual work for
  KL, FSDT, Reddy and CUF TE1..TE4, assembled by the MVI-owned SBP/Fourier
  derivatives and quadrature. CUF uses exact odd/even curved-shell mass
  moments. Reddy uses gradient-dependent director velocities; KL uses the
  finite-rotation inertial director coefficients.
- independent_mvi/coupled_viscosity.py: Kelvin-Voigt thickness-integrated
  physical weak law with its own MVI spatial adjoints. This is distinct
  from optional mass-proportional damping and affects MECHANICAL components
  only.
- independent_mvi/coupled_newmark.py: own implicit fixed-rank Newmark step.
  Instantaneous q/v/a are reconstructed from the accepted MVI X/Y factors,
  physical inertia and Kelvin-Voigt work are added at t[n+1], and electrical
  equilibrium stays quasi-static in the same nonlinear stationarity system.
  Failed or rank-insufficient Y/X sweeps do not advance physical time.
- independent_mvi/coupled_bathe.py: own two-stage trapezoidal and three-point
  Bathe method, with nonlinear Y/X stationarity at each stage; independent
  time formulas and no call to Newmark.advance. Time-varying mechanical and
  electric loads are sampled at both stage times. Failure of either stage
  rejects the physical step.

These paths do NOT import MKR spatial operators or shared variational
dynamic/Newton solvers. They do NOT use SVD or a full-grid Newton fallback.
Only continuum constitutive/kinematic source kernels are shared.

## Automated checks added

- tests/test_independent_coupled_mvi_newmark.py
- tests/test_independent_coupled_mvi_bathe.py
- tests/test_independent_coupled_mvi_viscosity.py
- .github/workflows/independent-coupled-mvi-no-solve-contracts.yml

The workflow is MANUAL, does not start automatically, and only performs
syntactic and small algebraic/mocked-contract checks, not loaded cylinder
simulations. The module source writes succeeded, but these tests have NOT
been executed or shown passing in this conversation; GitHub cannot be cloned
into the current execution container (DNS resolution fails).

## Explicit unsatisfied completion gates

1. Full high-order natural/microfree mechanical conditions and coupled
   electric insulation/face fluxes, including the single floating potential
   gauge of the source boundary problem, remain not certified.
2. Initial mechanical acceleration and electric equilibrium are not yet
   determined from a standalone native MVI initial-state solve.
3. Native coupled MVI Gauss-Legendre IRK4 stages have not been implemented
   in this new branch. Existing older Gauss drivers are NOT substituted.
4. Grid, rank and temporal convergence and independent benchmarks are
   unverified. The full nodal first-variation test detects a rank-insufficient
   candidate but is not a proof of continuous-model correctness.
5. GUI/CASE/export activation is intentionally disabled until complete
   mathematics, validation and reliable cancellation/monitoring are proven.

**Status:** experimental method-specific nonlinear static + Newmark/Bathe
modules, NOT production-ready and NOT scientifically certified.
No physical cylinder calculation was launched by the assistant.
