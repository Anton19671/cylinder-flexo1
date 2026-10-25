# Optional Newton-Krylov matrix-free FDM for nonlinear G3D + flexoelectricity

**Status: implemented as a separate experimental FDM route, NOT verified
or enabled automatically.** The original dense Newton and all existing
manual MVI ranks 1–8 are unchanged.

The new `fdm_backend: "MATRIX_FREE_GMRES"` CASE field explicitly opts
in to `numerics/cylinder_matrixfree_fdm.py`. Default remains
`"DENSE_REFERENCE"` and needs no SciPy. Matrix-free requires
`scipy>=1.11`.

The mathematical problem remains the **full original nodal stationarity**
of Green–Lagrange G3D/flexo, with director/CUF inertia, Kelvin–Voigt
viscosity, mass damping, all existing external generalized forces and
electrical saddle coefficients. Nothing is replaced by a linear model.

Let scaled unknowns be `z`, scaled residual `g(z)`, and
sparse normalized essential constraint matrix `C z=d`.
Set the orthogonal null projector
`P=I-C.T (C C.T)^(-1) C` and minimal lift
`z0=C.T (C C.T)^(-1) d`.
Then solve the fully coupled Newton correction

`[P J(z) P + (I-P)] delta = -P g(z)`

using restarted SciPy GMRES. A centered finite-difference
Jacobian-vector product invokes the **same full nonlinear physical weak
residual**, not a simplified shell stiffness. It does not assemble a
dense physical Jacobian. The sparse constraint Gram factorization is
separate and does not regularize the physical tangent.

Backtracking accepts a trial only when the full admissible residual
decreases. The solver returns exactly the result shape and diagnostic
fields expected by the existing dynamic Newmark acceptance workflow.

Caveats and remaining blockers:

- The opt-in reference CASE preflight now uses sparse affine traces,
  avoids dense SVD, and bypasses the old 256-node/4096-nodal-DOF
  guard. A distinct safety cap of 8192 surface nodes and
  `max_total_dofs` remains; **this is not a demonstrated
  production engineering mesh size**.
- Dynamic Newmark trace preparation likewise bypasses
  `BoundarySpatialProblem.assemble` for this backend, avoiding its
  dense null-space construction. The existing trace monitor itself
  still uses a dense mechanical constraint matrix and needs further
  refactoring for large grids.
- Dense eigen/gauge checks in SurfaceOperators and dense physical
  spatial derivative matrices are still present.
- Strict sparse Gram full-row-rank checks currently reject dependent
  boundary rows rather than coalescing mathematically identical
  electrode/edge constraints. This is intentional fail-closed behavior.
- Full GMRES has no physics-specific preconditioner, so speed and
  convergence are **not established**, especially with tiny electric
  saddle scales or high-order G3D differential blocks.
- MVI retains the existing dense halfstep route; we must not claim
  the full MVI method is matrix-free.
- `tests/test_matrixfree_fdm.py` provides staged regression checks
  but tests have not been run, per the mathematics-first request.

User-visible CASE choice:

```json
{
  "method": "FDM",
  "fdm_backend": "MATRIX_FREE_GMRES",
  "reference_newton": {
    "relative_tolerance": 1e-7,
    "absolute_tolerance_scaled": 1e-12,
    "tangent_step_scaled": 2e-6,
    "max_iterations": 15,
    "krylov_rtol": 0.03,
    "krylov_restart": 40,
    "krylov_max_matvecs": 180
  }
}
```

The example is an extract, not a full runnable reference CASE.
