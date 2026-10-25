# Optional block-ILU Newton--Krylov preconditioning

The full coupled nonlinear system is unchanged. The positive auxiliary
matrix constructed in `numerics/cylinder_krylov_block_preconditioner.py`
uses:
- mechanical surface gradient and G3D curvature stencils;
- director/CUF thickness polynomial scales;
- Newton--Newmark mass and Kelvin--Voigt rate contributions;
- dielectric potential gradient and thickness coefficients;
- dimensionless unknown scaling and a sparse affine constraint projector.

Separate sparse ILU factors are applied per mechanical/electrical
coefficient. This is a LEFT preconditioner of the *actual* indefinite
GMRES tangent; it is not a replacement for direct/inverse flexoelectric
off-diagonal Jacobian blocks. All corrected fields must still satisfy
the original **unpreconditioned** fully coupled residual.

A second linear stop check now explicitly evaluates
`norm(Jv+g)` from the original matrix-free action after GMRES, so a
small *preconditioned* residual alone cannot be accepted as accuracy.

In `reference_newton` opt in with:
```json
{
  "krylov_preconditioner": "PHYSICS_BLOCK_ILU",
  "ilu_drop_tolerance": 0.001,
  "ilu_fill_factor": 8.0,
  "ilu_max_surface_nodes": 8192
}
```

Requires `fdm_backend=MATRIX_FREE_GMRES`, and for large-grid
sparsity the `SBP21_PERIODIC_ODD_CSR` surface scheme. Default is
`krylov_preconditioner=NONE`, preserving legacy results. Manual
MVI ranks are unaffected.

The shifts in the auxiliary ILU are PRECONDITIONER-ONLY. They do not
add foundations, viscosity, electrical penalties, or extra energy to
the original equations. If ILU cannot be built, the selected route
fails closed rather than silently changing the physics.

**Not numerically tested yet**, including difficult flexoelectric
saddle conditioning and scalable runtime. Tests are staged in
`tests/test_krylov_block_ilu.py`. Full MVI/FDM convergence studies
and true engineering benchmarks remain outstanding.
