# CUF TE1 non-axisymmetric Gz flexoelectricity: MVI 1-8 vs FDM

This is the next required physical rank study after the 5x5 rank-one
experiment (initial quantitative observations in
`docs/SCIENTIFIC_COUPLED_INITIAL_NUMERICAL_EVIDENCE.md`).

## Fixed physical passport

- Theory CUF TE1, MICRO Green–Lagrange, Gz gradient elasticity, flexo ON
- Electric potential has order 1 = **two independent thickness fields**
- Same physical cylinder `L=1.2 m, R=2.0 m, h=0.2 m`
- Same nonlinear 3D Hooke and electroelastic material, electrode and clamped ends
- Same non-axisymmetric physical patch on the outer reference face
- Static, one load factor 1, same pressure amplitude and face-area convention
- Same `LEGACY_REFERENCE` surface family and **8 × 8** grid for every run
- Full FDM reference and fixed-rank MVI-1, 2, 3, 4, 5, 6, 7, 8

**Why 8×8 rather than 5×5?** Rank 8 is impossible on 5×5, and the
experimental dense halfstep safety bound is 512 unknowns. With six
mechanical and two electric TE1 fields, 8×8 preserves this bound even
for the largest rank. The 8×8 grid is not a claim of spatial
convergence: because the periodic angular grid is even, its Nyquist
aliasing behavior requires separate tests. The highest rank is also
the algebraic full-rank limit on this grid, not evidence of rank-8
accuracy on a large physical mesh.

The GitHub workflow
`.github/workflows/scientific-coupled-rank-1-8.yml`
runs CASE preflight first, then nine **real** physical jobs with
parallelism capped at two. Each saves exact CASE, code SHA, physical
NPZ, a scientific evidence certificate and failed-iterate journals.
The compare job checks all source hashes, identical physics, source
revision and spatial coordinates. It compares entire arrays of
mechanical coefficients, electric potential coefficients,
outer-face displacement, normal electric induction, stresses,
hyperstresses and volume induction. Each reported relative L2
difference is relative to the **same-grid FDM solution**, not
an independent analytic/WQEM/FEM reference.

`CONVERGED_PROJECTED` indicates only fixed-rank accepted
stationarity and is never labelled full-space convergence. An
uncompleted rank, shape mismatch, missing physical tensor or corrupt
file makes the comparison non-certified. Rank-dependent differences
need not decrease monotonically. Higher-order spatial and independent
physical benchmarks are still required before publication.
