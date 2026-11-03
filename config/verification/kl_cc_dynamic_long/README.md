# Physical forced KL shell: 25 Hz, 20 ms, Newmark versus Runge-Kutta Gauss IRK4

Numerical **physical** dynamic test on a homogeneous cylindrical shell; not
an externally published load-vibration benchmark.

Common inputs:
- KL linear elastic shell, L=2 m, R=1 m, h=0.05 m;
  E=210 GPa, nu=0.3, rho=7800 kg/m3.
- Both axial ends CLAMPED (C-C), fixed for every time step.
- Surface grid 9x9, SBP42_FOURIER_ODD.
- Uniform inward DEAD_REFERENCE midsurface pressure:
  p(t)=100000 sin(2*pi*25*t) Pa.
- Zero initial displacement and velocity, no damping.
- Physical interval 0..0.020 s = half the 25 Hz forcing period.

Six independent physical trajectories:
| Key | Spatial solver | Time integrator | dt [s] | steps |
| --- | --- | --- | --- | --- |
| FDM_COARSE | FDM/MKR | Newmark beta=.25 gamma=.5 | .002 | 10 |
| MVI1_COARSE | fixed-rank MVI-1 | Newmark | .002 | 10 |
| MVI4_COARSE | fixed-rank MVI-4 | Newmark | .002 | 10 |
| FDM_FINE | FDM/MKR | Newmark | .001 | 20 |
| MVI1_FINE | fixed-rank MVI-1 | Newmark | .001 | 20 |
| GAUSS2_MVI1_COARSE | fixed-rank MVI-1 | Gauss-Legendre two-stage implicit Runge-Kutta, formal order 4 | .002 | 10 |

The Gauss2 runner uses **exactly MVI1_COARSE.json** as the physical CASE,
with a different output directory; the Newmark-Gauss comparison verifies
the matching declared physics SHA256 digest, time grid, initial conditions
and actual stage equations and rank representation.

The primary artifact is a CSV of physically accepted node-wise fields
and comparisons at shared times, including w(t) and u(t)/v(t)/a(t)
discrepancies between Newmark MVI and FDM as well as Newmark temporal
refinement and independent Gauss2-IRK4 differences.

This work is NOT independent 3D-model validation; only Newmark temporal
comparisons between two step sizes are included (insufficient for a
measured order of convergence). A spatial-grid study and a three-step-size
Richardson check remain necessary before claiming convergence or fourth
order for constrained MVI/IRK4.

Entry point: `.github/workflows/kl-cc-long-dynamics-rk4.yml`.
