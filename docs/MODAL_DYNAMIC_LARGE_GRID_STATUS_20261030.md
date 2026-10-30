# MODAL / DYNAMIC: dense mesh status (2026-10-30)

## Separate numerical backends

- Existing DYNAMIC FDM on large grids: `MATRIX_FREE_GMRES` with `SBP21_PERIODIC_ODD_CSR`. The matrix-free Newmark path existed before this change; it is not an eigenproblem.
- New MODAL FDM on large grids: `MATRIX_FREE_GMRES` + `SBP21_PERIODIC_ODD_CSR`, implemented in `numerics/cylinder_modal_matrixfree.py` and dispatched from `numerics/cylinder_modal_reference.py`. It uses projected tangent and consistent mass actions, with LOBPCG on the constrained homogeneous mechanical space.
- Existing small-grid MODAL FDM: `DENSE_REFERENCE`, unchanged.
- MODAL MVI 1..8: unchanged separate Kryśko–Kirichenko fixed-rank Y/X method; **the new full-nodal FDM is not MVI**.
- DYNAMIC MVI 1..8: existing separated Newmark method, not replaced.

## Allowed scope

Linear unloaded mechanical cylinders: `scale_model=MACRO`, `mode=LINEAR_ELASTIC`, `material_model=HOOKE`; `theory` KL/FSDT/REDDY or CUF TE1/TE2/TE3/TE4; `geometry_model` LINEAR/VON_KARMAN/GREEN_LAGRANGE linearized at zero displacement. Electric massless coupling remains supported only in the earlier dense modal path.

The new large-grid mode is intentionally opt-in and fails rather than silently applying a different physical model. It uses homogeneous variational end constraints and no mass on constrained/electric degrees of freedom.

## Mesh planning

Use `nx,ns = (17,17), (25,25), (33,33), (41,41)` with **odd** circumferential count. The CSR matrix-free CASE cap defaults to 8192 surface nodes and 100000 total nodal DOFs; these are safety ceilings, not a guarantee of numerical convergence.

For the Rawat et al. 2016 comparison set `L=2`, `R=1`, `h=.05`, `E0=210e9`, `nu0=.3`, `modal.density_kg_m3=7850`, C-C supports. The table concerns m=1 and n=1..10, *not* the first ten eigenvectors sorted globally. A 6-eigenpair output can show repeated sin/cos doublets; compare by harmonic numbers and shapes.

## Validation limitations

The large-grid FDM modal LOBPCG solver has been added but **not independently numerically validated on 17x17..41x41**. A failed convergence check aborts; no result should be claimed verified yet. Existing DYNAMIC sparse FDM also requires per-theory/load/mesh convergence checks and a separate MVI comparison. Do not infer all theories or all MVI ranks work simply from CASE preflight.

Do not change eigenproblem to time stepping; do not conflate frequency outputs with all m,n branches.
