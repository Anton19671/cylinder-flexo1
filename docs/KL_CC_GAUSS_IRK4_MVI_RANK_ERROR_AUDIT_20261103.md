# KL forced cylinder: physical Newmark and Gauss IRK4 MVI rank audit

Status snapshot: 2026-11-03. These conclusions concern the homogeneous linear
KL 9x9 SBP42 Fourier cylinder under 25 Hz uniform radial dead-reference
pressure (100 kPa amplitude), clamped C-C and density 7800 kg/m3.

## Confirmed independent physical runs

### Newmark

- At 2 accepted physical steps, MVI-1..8 and full-grid FDM had already
  matched full U,V,W and velocities/accelerations to numerical accuracy.
- A separate nine-job **5 physical step** matrix was completed
  successfully for MVI1..8 and FDM/MKR, without method fallback:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41624708013
- An independent full-period MVI1 40-step run completed successfully:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41626140632
  - dt=.001s, T=.040s, 40/40 accepted
  - signed circumferential mean W at x=L/2 at t=.040s:
    -0.168755796063 micrometers.
- The independent modal K/M sine-forcing exact-in-time reference,
  using the SAME discrete KL 9x9 physics, has W=+0.252752900 micrometers
  at t=.040s, confirming substantial phase error at this timestep.

### Verification of temporal phase dispersion, not MVI error

The independent physical K/M eigensolve, using the existing model
stiffness and exact inertia, completed:
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41627560336

The main axially symmetric forced modes are 906.878345 Hz,
920.869888 Hz and 949.836501 Hz, so the 25 Hz load also excites
fast *free transient* oscillations. The observed Newmark numerical
frequency is strongly distorted at dt=.002/.001/.0005s.

Crucially the **independent modal Newmark recursion**, which uses no
MVI spatial sweeps and no full nodal Newton, reproduced the accepted
MVI1 midspan W at t=.020s at all three dt:

| dt, s | Real accepted MVI1 W, micrometers | Independent K/M modal Newmark W, micrometers | difference in m |
|---:|---:|---:|---:|
| .002 | +0.058247490 | +0.058247489604 | -3.960e-16 |
| .001 | -0.136024749 | -0.136024748962 | +3.788e-17 |
| .0005 | +0.123355274 | +0.123355274422 | +4.220e-16 |

Recorded in successful run:
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41629149007

Thus the non-monotone displacement at .020s is **NOT** evidence that
MVI's state update is defective in this linear KL case. The chosen dt
values undersample the fastest dynamically forced natural modes.

### Gauss IRK4 rank-fixed MVI (separate issue)

Physical original and rerun matrix:
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41624955865

| MVI rank | Reason for not accepting first physical step |
|---:|---|
| 1, 2, 3, 4, 5 | GAUSS2_MVI_ENDPOINT: GAUSS2_MVI_FIELD_OUTSIDE_FIXED_RANK |
| 6, 7, 8 | GAUSS2_MVI_ACTIVE_DOF_LIMIT |

For ranks 1..5 the two collocation stage fields were each represented
within fixed rank r, but the standard RK update constructs an endpoint
by **adding** contributions from both stages and the previous velocity.

An algebraic fact: if rank(Q1)<=r and rank(Q2)<=r,
then rank(b1*Q1+b2*Q2) can be up to 2r, not necessarily <=r.
The velocity and prior-position terms can increase rank further.
Therefore fixed-rank stage representation alone gives NO theorem that
the exact standard Gauss collocation endpoint is of rank r.
This was confirmed by dedicated QR-based tests:
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41629720944

Ranks 6..8 fail on the current explicit 256-free-coordinate **safety
limit**, not an eigenfrequency problem or a license to raise it without
memory/time-cost analysis.

A physically insignificant near-zero field component can also cause
a large relative rank-factorization error when the acceptance rule
normalizes each component against its own amplitude. This deserves
its own symmetry/scale consistency audit, not an unconditional change
to the required electric/mechanical component accuracy.

## Correct next algorithmic action

1. Keep existing Newmark branch unchanged for the verified linear KL
   physical case; validate it at adequately refined dt.
2. Keep Gauss physical stage equations, boundary constraints and rank
   rejection unchanged until an endpoint-closed low-rank time
   discretization has been derived. Candidate constructions must
   explicitly constrain the endpoint to the low-rank manifold and
   demonstrate projected variational stationarity.
3. Independently verify the ordinary (unrestricted spatial rank)
   two-stage Gauss **time integrator** on the same K/M system by modal
   equations and successive dt refinement. This separates a faulty
   time integrator from an incompatible rank-R stage representation.
   See ci/kl_cc_modal_time_oracle.py.
4. Never silently truncate the endpoint, increase rank, or substitute
   the full FDM solution for an advertised fixed-rank MVI method.
5. Investigate rank dependence under a nonseparable spatial load
   separately: this homogeneous axisymmetric case is not a meaningful
   convergence-in-rank benchmark.

All frequencies quoted above are the exact **9x9 discrete** KL
frequencies, not a comparison against 3D elasticity or an independent
literature benchmark. This memo does not claim G3D or flexoelectric
dynamic validation.

## Independently verified Gauss IRK4 time convergence (completed)

The seven-grid Gauss–Legendre two-stage Runge–Kutta scheme was integrated
on the *actual* 171-free-DOF KL 9x9 K/M spectrum (all modes), with the same
unloaded initial state and the same 25 Hz pressure modal forcing.
This test did not invoke any MVI fixed-rank spatial method.

GitHub Actions result: **SUCCESS**
https://github.com/Anton19671/cylinder-flexo1/actions/runs/41630345427

Exact-in-time reference at t=0.020s:
W(mean midspan)=+0.106970306779 micrometers.

| Gauss dt, s | IRK4 mean W at .020s, micrometers | absolute error, micrometers |
|---:|---:|---:|
| .002 | +0.166890768252 | 0.059920461473 |
| .001 | +0.003102550761 | 0.103867756019 |
| .0005 | +0.024851154251 | 0.082119152528 |
| .00025 | -0.074466809943 | 0.181437116723 |
| .000125 | +0.094636423124 | 0.012333883655 |
| .0000625 | +0.106243434125 | 0.000726872654 |
| .00003125 | +0.106922425630 | 0.000047881150 |

The finest successive errors have reduction factors approximately
16.97 and 15.18. The measured local orders are approximately 4.08
and 3.92, confirming fourth-order asymptotic time convergence on
this discrete linear KL system.

Coarse-grid errors are *not* monotonically improving: high-frequency
transient phases (physical modes near 906.88, 920.87, and 949.84 Hz)
are aliased and can interfere destructively at a specific snapshot.
An earlier diagnostic run incorrectly required monotonically
decreasing errors already at dt=.0005/.00025s. That condition was
invalid outside the asymptotic range; it was replaced by a seven-grid
test and **not** by weakening the physical equations or the fixed
MVI-rank acceptance.

**Conclusion:** the conventional Gauss IRK4 time discretization is
correctly implemented when applied to the independent K/M modal
system and sufficiently temporally resolved. This DOES NOT validate
the separate spatial rank-constrained Gauss MVI implementation:
its stage rank spaces are not generally closed under the endpoint
linear combination. Ranks 1..5 still fail strict endpoint rank
acceptance, and ranks 6..8 still hit the explicit 256-coordinate
safety limit on the physical shell case. A genuinely endpoint-closed
variational low-rank integrator must be derived and validated before
claiming Gauss IRK4+MVI production support.

The full seven-grid physical modal IRK4 and Newmark replay CSVs
are archived as artifacts of the successful run above.
