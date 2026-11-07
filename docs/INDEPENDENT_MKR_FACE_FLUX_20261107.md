# Strong MKR finite differences: paired face gradients, no 3D grid
Date: 2026-11-07. Algorithm: Krysko–Zhong corrected 2026-10-24,
§§13, 36, 40, 41. Status: experimental LINEAR MACRO HOOKE only.

## Corrected discrepancy

Previously the **independent** strong-form MKR family
`independent_mkr/linear_models.py` used a centered angular first
difference twice for first-order divergence-of-gradient. This does
not satisfy the stated face-gradient / paired-divergence instruction:
on even periodic grids it has a false checkerboard null mode.
Rejecting even grids masks but does not fix that structural problem.

The new MKR-owned 2D discretization evaluates all six derivative-order
channels (0, x, s, xx, xs, ss) at the **same surface cell centers**:
`(x_{i+1/2},s_{j+1/2})`. It uses two-point center interpolation,
one-cell face gradients, direct three-point nodal second derivatives
with one-sided axial endpoint closures, and compatible mixed gradients.
The angular face is periodic. The strong nodal operator accumulates
the method's cell fluxes with the respective transposed difference
maps (the discrete conservative divergence). The uniform physical
cell/dual-volume factors cancel in the interior strong equations.
This is the discrete first variation of the MKR-owned
thickness-integrated cell enthalpy, not a call to MVI.

The thickness model is STILL CUF TE1–TE4 with 12-point Gauss
quadrature. The only spatial grid is axial–circumferential; no z mesh
and no 3D finite elements have been introduced.

## Mathematical and engineering scope

The independent MVI operator is unchanged and remains its own
SBP42/Fourier weighted weak form with genuine X/Y factor iterations.
No MKR numerical function imports MVI or the reference backend.
Bathe/Newmark/Gauss IRK4 call their own matching method kernels.

The new algebra-only guards in
`tests/test_independent_mkr_compatible_faces.py` exercise:
- the even-grid alternating harmonic through a one-cell angular face;
- direct second differences rather than squared central first differences;
- seven theory/order placements on 2D cells; 
- the discrete cell-energy/strong-nodal-work identity and symmetry.

This corrects **one** important part of source §40. It is not a
claim of numerical research readiness: exact higher-order natural
end traces, curvature terms in all boundary fluxes, full static
support, nonlinear Green–Lagrange, 18 G3D gradients, electromechanical
unknowns, reciprocal flexoelectric blocks and mesh/time/rank convergence
remain unvalidated or unimplemented in these isolated linear models.
The old reference coupled solver may still support these models but
is not accepted as an independent MKR/MVI verification route.

The calculations prohibited by the user were not launched.
A standalone small-matrix algebra prototype of the face/adjoint
construction was inspected only; **the repository test suite has not
been run in this edit**, and no physical field result is certified.
