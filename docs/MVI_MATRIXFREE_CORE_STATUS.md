# Matrix-free fixed-rank MVI — experimental operator core

The dense reference MVI currently has a small-DOF limit. New modules
cylinder_mvi_direction_space.py and cylinder_mvi_krylov_core.py create a
scalable alternative for *only* the separated X/Y algebra.

SeparatedMVIDirections:
- physical array order (component, x, s);
- fixed rank 1..8 without automatic rank growth;
- exact T(v) and T.T(v) without creating full dense nodal basis;
- sparse T only when projecting physical constraint rows.

SparseHalfstepTraceProjector:
- imposes homogeneous variations C T da=0;
- pivoted QR of independent small constraint blocks and optional global
  electric gauge row; rejects blocks that exceed explicit size limits;
- does not use SVD, a pseudo-inverse or a physical-tangent penalty.

separated_krylov_sweeps:
- receives a caller-provided full admissible scaled residual;
- performs rank-fixed X/Y Newton--Krylov half-steps, finite-difference
  Jacobian-vector actions, and residual line search;
- cold-starts inactive partners from the admissible full residual;
- accepts only the full nodal admissible residual, not projected-only
  convergence, and never changes the selected rank.

This is an ALGEBRAIC ENGINE ONLY; it is NOT wired into CASE or GUI yet.
Specifically, the 81x81 G3D/Gz physics and Newmark wrapper still need a
sparse affine lift/trace path, a zero-copy full residual adapter, consistent
physical block scales, and an acceptance certificate that does not build a
dense full nodal basis. Until that integration is implemented, the dense
MVI remains the active solver and large-grid calculations are not supported.

Local regression of this algorithmic core passed 9 tests. Full coupled
seven-theory shell results, boundary work and numerical convergence have
not yet been tested. Do not mark this backend production ready.
