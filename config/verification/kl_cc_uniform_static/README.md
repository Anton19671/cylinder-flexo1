# KL C-C, linear static pressure: first verification

**Status: prepared CASEs, no static PDE solutions asserted.**

This is a *new static verification*, **not** a static table from Rawat
or Cammalleri. The homogeneous isotropic cylinder's geometry/material are
the Rawat 2016 modal benchmark parameters, reused here for convenience:
L=2 m, R=1 m, h=0.05 m, E=210 GPa, nu=0.3. The chosen verification pressure
100 kPa (and 50 kPa control) is **not** attributed to Rawat.

- KL, LINEAR, HOOKE; macro model without strain-gradient/electric coupling.
- C-C axial supports; 9x9 SBP42_FOURIER_ODD for BOTH methods.
- Uniform radial **inward** dead-reference load **on the midsurface**,
  p=100000 Pa. This is not a physical follower pressure on a deformed face.
- Nine primary independent CASEs: MVI-1…8 plus MKR/FDM.
- Two additional 50000 Pa controls: MVI-1 and MKR/FDM. In true
  linear elasticity at fixed mesh, w(100 kPa)/w(50 kPa) should be 2.
- MVI uses its actual frozen-partner Y/X branch. MKR/FDM uses the
  separate full-node variational finite-difference branch.

## Commands (repository root)

Preflight only (default; **does not run a PDE**):

    python ci/kl_cc_uniform_static.py

To solve **locally, by explicit choice**:

    python ci/kl_cc_uniform_static.py --run

To read existing solutions and export comparative CSV and JSON:

    python ci/kl_cc_uniform_static.py --report

Each CASE saves to a unique folder beneath this directory's "output".
The runner does not overwrite existing nonempty result directories.

## Required comparison

For each accepted result compare full fields U,V,W (component,x,s),
midspan circumferential-average radial displacement W, max |W|,
circumferential spread at midspan (uniform load => axisymmetric response),
and the displacement on both clamped edges. Relative differences are
reported against *on-grid* MKR/FDM, **not** an analytical ground truth.

A result that did not complete successfully remains NOT_RUN / failed,
with blank numeric fields: no MVI rank is replaced by MKR values.
MVI projected acceptance remains valid in its own sense, not
automatically full-field convergence.

**9x9 is a first-grid smoke/consistency benchmark, not evidence of
mesh convergence.** Next stages: identical physical CASE at finer grids
with supported MVI and FDM memory budgets, analytical/independent
3D comparison, then GL (with the same explicitly fixed dead load),
and only after that a proper follower-pressure formulation.
