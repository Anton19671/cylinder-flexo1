# Complete CSR SBP spatial route for coupled G3D/Flexo Newton--Krylov

**Status:** implemented in `main` as an opt-in *experimental* physical
operator path; tests are staged but not yet executed. This is not a
claim of production engineering-grid verification.

## Single physical mathematics, two matrix representations

The older `LEGACY_REFERENCE` and `SBP21_PERIODIC_ODD` remain dense,
unchanged, for reference FDM/MVI. The new explicit
`SBP21_PERIODIC_ODD_CSR` grid scheme is valid **only** with
`fdm_backend=MATRIX_FREE_GMRES`.

The derivative matrices are now O(nx+ns) in memory, with sparse
SBP(2,1) axial derivatives and centered periodic differences on odd
circumferential grids. No full dense SVD is performed to validate
the periodic derivative: the CSR mode accepts only the exact known
SBP21/odd-periodic stencil and checks both discrete Green identities.

Surface jets for KL/FSDT/Reddy/CUF TE1--TE4 apply the same
`D_x^a U (D_s^b)^T` in both representations. Every weighted first
variation uses the EXACT transpose adjoints. Powers of the one-dimensional
operators remain CSR instead of creating dense matrices.

Connected end-to-end:
- all Green--Lagrange covariant G3D and flexoelectric jets/variations;
- Kelvin--Voigt viscosity and consistent Reddy/FSDT inertia;
- nonlinear KL director acceleration (including gyroscopic terms);
- exact end/face virtual work and affine supports/electrodes;
- diagnostic strong/natural SBP Green decomposition and charge fluxes;
- Newton--Krylov finite-difference Jv from the same full FDM weak residual.

The sparse physical derivative operator is NOT a sparse physical
Jacobian. GMRES acts matrix-free on the coupled nonlinear saddle
system, using the existing sparse boundary trace projector.

### Case selection

```json
{
  "method":"FDM",
  "fdm_backend":"MATRIX_FREE_GMRES",
  "fdm_grid":{
    "nx":17, "ns":21,
    "operator_scheme":"SBP21_PERIODIC_ODD_CSR"
  }
}
```

An odd circumference node count is required; no MVI rank or AUTO
rank selection is changed. All other material, load, Newmark and
boundary settings remain those of the ordinary variational CASE.
SciPy >=1.11 is needed only for the explicit CSR/FDM option.

### Limits and next gates

CSR operators address *matrix storage*, not the expense of the
per-node nonlinear thickness quadrature (especially CUF TE4),
the sparse constraint Gram solve, or missing physics-specific GMRES
preconditioning. The numerical strong boundary flux on a coarse
SBP grid is not proof of a correct physical microfree stress.

The suite `tests/test_sparse_surface_backend.py` has been prepared
to compare dense versus CSR energy, coupled residual, viscosity,
KL/director/CUF inertia, trace work, and exact SBP Green identities.
No such result is claimed until the tests and full nonlinear FDM
load/time-step benchmarks have been executed.
