# Follow-up: third Newmark timestep and fixed-rank Gauss IRK4 real shell steps

Physical case is the previous homogeneous KL C-C cylinder (L=2m,
R=1m, h=0.05m, E=210GPa, nu=0.3, rho=7800kg/m3) on the
9x9 SBP42_FOURIER_ODD grid, uniform inward dead-reference midsurface
pressure p(t)=100000 sin(2*pi*25*t) Pa, zero initial conditions.

* MVI1_ULTRAFINE_40: Newmark beta=0.25 gamma=0.5, dt=0.0005 s,
  **40 independent real physical steps to 0.020 s**. Third temporal
  grid, complementing the completed MVI1 10x0.002 and 20x0.001 runs.
* MVI2_RK_STEP1: independent two-stage Gauss-Legendre IRK4,
  **fixed MVI rank 2**, one physical step to t=0.002 s.
* MVI4_RK_STEP1: independent two-stage Gauss-Legendre IRK4,
  **fixed MVI rank 4**, one physical step to t=0.002 s.

IRK4 ranks 2/4 are deliberately started as one-step diagnostics
because rank 1 **failed** at endpoint rank certification on the
same physical case. The stage and endpoint acceptance rules, rank,
field representation, mechanical BCs and equations must be preserved.
These one-step tests are not claims of full-interval fourth-order
accuracy. Any failed physical step must cause a failed CI job while
retaining the logs and saved states for diagnosis.

The complete initial Newmark 10-step and 20-step records are available
as GitHub Actions artifacts from run 41600941990; the 20-step FDM
fine run exceeded its allotted runtime and was cancelled, NOT accepted.

One-step Gauss and 40-step Newmark are launched independently by
.github/workflows/kl-cc-rk4-followup.yml.
