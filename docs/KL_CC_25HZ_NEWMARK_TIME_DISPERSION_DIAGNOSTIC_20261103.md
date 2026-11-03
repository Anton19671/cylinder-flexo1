# KL C-C 25 Hz forced dynamics: numerical time-dispersion diagnosis

Date: 2026-11-03. **Observed physical time trajectories; interpretation
is a hypothesis pending independent KL modal K/M verification.**

Physical CASE: linear isotropic KL cylinder L=2 m, R=1 m, h=0.05 m,
E=210 GPa, nu=0.3, density=7800 kg/m3, clamped axial ends C-C,
9x9 SBP42_FOURIER_ODD. Uniform inward dead-reference midsurface pressure
p(t)=100000 sin(2*pi*25*t) Pa. Zero initial displacement/velocity,
undamped. Newmark average-acceleration beta=0.25, gamma=0.5.

## Three independent accepted fixed-rank MVI-1 trajectories

| Time step | Steps | End time | GitHub artifact |
|---|---:|---:|---|
| .002 s | 10 | .020 s | [Run artifact](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41601663882/artifacts/13215576269) |
| .001 s | 20 | .020 s | [Run artifact](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41601663882/artifacts/13215841344) |
| .0005 s | 40 | .020 s | [Run artifact](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41620265914/artifacts/13224192523) |

All have physical accepted-step status CONVERGED; the MVI-1 fields
were individually reconstructed from the saved arrays. These runs
are **not** evidence of convergence in time.

At t=.020 s, the circumferential mean KL midspan displacement W is:

| dt, s | W(mean, midspan), micrometers |
|---|---:|
| .002 | +0.058247490 |
| .001 | -0.136024749 |
| .0005 | +0.123355274 |

Full-field differences are likewise non-monotone. At t=.020 s
their magnitudes are 1.100638067e-6 m and 1.541137640e-6 m in
[U,V,W] Euclidean grid norm for dt .002/.001 and .001/.0005,
respectively, so normal Richardson second-order extrapolation is
**not yet justified**.

## Newmark's exact undamped single-mode numerical dispersion

For a free linear mode of circular natural frequency omega and
average-acceleration Newmark, the numerical angular frequency is

    omega_num = 2/dt * arctan(omega*dt/2).

This is *phase dispersion*, even though average-acceleration Newmark
is unconditionally stable on a positive definite undamped linear
oscillator.

Subtracting the quasistatic 25-Hz sinusoidal response
W_static sin(2*pi*25*t), with W_static approximately -7.598499770 um
from the **separate** accepted KL static solve, exposes a strong
high-frequency transient of amplitude roughly 0.2 um.

The dominant apparent discrete frequency fitted from the three
independently accepted time histories is:

| dt, s | apparent frequency, Hz | inverted physical natural frequency, Hz |
|---|---:|---:|
| .002 | 222.741 | 920.145 |
| .001 | 394.002 | 920.283 |
| .0005 | 613.894 | 917.698 |

The inversion uses omega = 2/dt tan(omega_num*dt/2).
All three estimates agree within about 0.3%, strong evidence that
**one physical near-920 Hz mode** is responsible for the alternating
time-grid response. This is a frequency inferred from displacement
history **not** an independently verified eigenvalue.

At f~920 Hz, the nominal frequency ratio f_load/f_mode=25/920~0.027,
but sudden application of a zero-to-sine external force at t=0
induces *free transient* modes even under very low forced frequency.
At dt=.002 s one period of this mode is 0.00109 s, shorter
than one time step; at dt=.0005 s there are only ~2.17
steps per modal period. Neither is enough for accurate phase.
The high-frequency component's amplitude is small compared with
peak forced displacement but dominates the displacement near
zero crossings (such as t=.020 s). Therefore an individual signed
displacement at this time is a poor stand-alone convergence metric.

This does NOT justify adjusting the displacement, suppressing the
mode, adding damping, changing the forcing history, relaxing the
solver tolerances, or asserting convergence from identical ranks.
No such change has been made.

## Independent checks required

1. Calculate the real eigenpairs from independent full-grid KL M and K
   assembled on **the same** 9x9 SBP42 grid and C-C supports.
2. Independently reconstruct the exact-in-time linear semi-discrete
   response to sin(2*pi*25*t) pressure by modal sine convolution.
   This is the same spatial approximation but a different temporal
   solution, and tests the Newmark time discretization directly.
3. Compare full fields and phase at several common times (not only
   zero-load time) against this exact-in-time reference.
4. Verify that the excited near-920 Hz mode has nonzero generalized
   forcing, and perform dt refinement well below its 1/920 s period.
5. Repeat in MVI1 and FDM at same dt and time to separate *spatial rank
   approximation* from temporal integration errors.

The independent eigenvalue and exact-in-time temporal reconstruction
test is implemented in `ci/kl_cc_modal_time_oracle.py`, with its own
GitHub Action. Until that run succeeds, this is a clearly labelled
inference rather than a verified shell eigenfrequency.

**Important:** These results are limited to classical linear KL and do
not yet validate G3D/Gz, electric coupling, or nonlinear GL dynamics.
