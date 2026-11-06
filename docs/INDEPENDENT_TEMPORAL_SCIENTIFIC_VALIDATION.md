# Manufactured temporal verification (independent MKR/MVI)

Source reference: Krysko-Zhong corrected CUF TE1-TE4 equations (202)-(210),
with Bathe and two-stage Gauss-IRK4 used as explicit numerical alternatives.

`tests/test_independent_temporal_scientific.py` exercises the **real**
time stepping methods of each independent route, with only their spatial
mass and stiffness actions replaced by a prescribed scalar harmonic oscillator
so that an analytic, non-solver baseline exists. It tests:

- Exact known forced-oscillator end displacement/velocity at successively
  halved time steps; Newmark and Bathe should show second order and
  two-stage Gauss-IRK4 fourth order.
- Newmark/Gauss conservation of scalar undamped oscillator energy.
- Numerical damping of Bathe on free high-frequency vibration.
- Explicit rejection of non-finite or non-positive time increments.
- End-step acceleration satisfying the same known oscillator equation.

Passing these tests establishes **temporal algorithm algebra only**.
It does **not** validate the cylinder spatial discretizations, higher
microfree boundaries, real-field time-step convergence, electrical
quasistatic coupling, or comparison with an external WQEM solution.
In particular MVI rank dependence and standalone Gauss endpoint rank
admissibility require real shell-specific tests; those are separate gates.

The source case uses nominal outer-face radial loading and a single
global electric gauge for the coupled full model. No electric dynamics
is introduced in these standalone linear, non-electrical oscillators.
