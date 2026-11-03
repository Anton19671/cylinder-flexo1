# KL C-C 25 Hz real forced dynamic run (20 ms), status snapshot 2026-11-03

**Partial physical results, not a completed comparison.** Real accepted step
states have been checked from the GitHub Actions artifact.

Physical data:
- Isotropic KL cylindrical shell, L=2 m, R=1 m, h=.05 m.
- E=210 GPa, nu=.3, density=7800 kg/m^3.
- Clamped both axial ends (C-C), 9x9 SBP42_FOURIER_ODD.
- Uniform inward radial *dead-reference midsurface* load
  p(t)=100000 sin(2*pi*25*t) Pa, zero initial displacement/velocity.
- Newmark beta=.25, gamma=.5, dt=.002 s, 10 accepted steps to .020 s.
- The load and frequency are **chosen control values**, not published
  results from Rawat or Cammalleri.

## Accepted physical displacement history: rank-1 MVI, Newmark

GitHub run:
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41600941990
Saved physical data artifact (10 actual NPZ fields and JSON summaries):
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41600941990/artifacts/13215200119

| t, s | max abs W, um | mean W at axial midspan, um |
|---:|---:|---:|
| .002 | 2.427815219 | -2.279917448 |
| .004 | 4.877114692 | -4.601316662 |
| .006 | 6.358316428 | -5.973814467 |
| .008 | 7.887136136 | -7.435841162 |
| .010 | 7.865916198 | -7.398402970 |
| .012 | 7.877811519 | -7.415876718 |
| .014 | 6.376010674 | -6.011384755 |
| .016 | 4.852864041 | -4.550578793 |
| .018 | 2.456134647 | -2.337829801 |
| .020 | 0.059708344 | 0.058247490 |

All ten steps were accepted by the physical Newmark solver, with
final status CONVERGED, accepted_steps=10, not inferred from FDM.
These tabulated maxima are at the **ten sampled instants**, not
continuous-time absolute maxima. No time-grid convergence is claimed.

## IRK4 diagnostic and fix

In the **first** real Gauss-Legendre two-stage IRK4 physical attempt,
the independent rank-one Gauss2 MVI branch failed on step 1:
`GAUSS2_MVI_COMPLETED_SWEEP_TRACE_DRIFT`. There are no accepted IRK4
fields in that run. GitHub:
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41600941990

The Gauss2 code uses an independent implementation of constrained
frozen-partner X/Y subspaces. The same machine-roundoff-only trace
rank pathology previously corrected in Newmark remained in Gauss2.
Commit `af92113f06a9e4746b47548ba7599b8a3debb9b9` changes
the trace rank decision to include an absolute, direction-scaled floor;
prescribed physical trace defects remain enforced and no solver fallback
to FDM is permitted.

Three targeted tests were added for roundoff-only traces, a genuine
trace and inconsistent prescribed traces. Regression run:
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41601789883
(completed successfully).

The **new physical Gauss2 IRK4 rerun** remains a separate pending
proof and must not be labeled successful on test results alone:
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41601663882

## Required independent checks

Numerical FDM/MKR Newmark 10- and 20-step histories, MVI1 Newmark
20-step, MVI4 Newmark 10-step, and physical Gauss2 IRK4 10-step
must be accepted before claiming any complete comparison.
The time-step convergence comparison and 3D/literature verification
are NOT yet established.
