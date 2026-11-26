# Gao–Park: independent native-method convergence matrix

The benchmark uses linear plane strain, axisymmetry, CUF TE1–TE4, full G3D
and four source length-scale values. No electrical field. The physical
parameters are identical on every surface grid. Existing production boundary
programs are unchanged.

| Surface grid (axial × circumferential) | Methods |
|---|---|
| 21 × 9 | FDM; MVI-1 through MVI-8 |
| 31 × 13 | FDM; MVI-1 through MVI-8 |
| 41 × 17 | FDM; MVI-1 through MVI-8 |

The matrix contains 4 thickness orders × 4 length scales × 3 surface grids
× 9 methods = 432 separately executed cases. Each accepted output must identify
its method, rank, grid, quadrature, geometry and material branch.

## Reporting

1. Spatial convergence: compare each method at the same TE order and length
   scale on the three grids. Report inner, middle and outer displacement and
   maximum relative profile differences (normalization: maximum absolute
   reference profile value). Retain the actual grid in each table row.
2. Rank convergence: compare MVI ranks 1–8 on each fixed grid, versus the
   native FDM and independently evaluated same-order TE reference. Never
   substitute FDM output for MVI or assign a number to a failed case.
3. Thickness approximation: compare TE1–TE4 versus the Bessel reference.
   The independent radial FDM study (41, 81, 161, 321 radial nodes) is distinct
   from surface-grid convergence; it is not a native shell FDM calculation.
4. Exact axisymmetry and axial invariance admit a separated rank-one field.
   Thus a plateau at rank one is expected if the constraints and solvers
   are correct; this is a rank-consistency test, not evidence for rank
   convergence on general non-axisymmetric loads. Grid refinement should
   not change the exact surface-constant solution. Report any artificial
   spatial variation explicitly.
5. A failed, timed-out or unexecuted case has no accepted physical result.
   Preserve status and logs. No values from failed runs should be used
   in comparisons or graphs.

The workflow is a research verification matrix, not a completed benchmark.
