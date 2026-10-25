# Unified nonlinear assembly throughput — implementation record

**State:** staged source changes only. No benchmark or end-to-end
numerical qualification has yet been performed.

The seven-model coupled variational solver continues to use the exact
Green–Lagrange/CUF/Director/KL kinematics and common covariant G3D
material/electric enthalpy. This change targets computational
duplication without simplifying the constitutive equations.

- Gauss-Legendre dimensionless abscissas and weights are cached by
  quadrature order; dimensional thickness integration still scales by h.
- CUF TE1–TE4 exact unit virtual gradients F,F_x,F_s,F_z are precomputed
  once per shell geometry, thickness station and TE degree and reused
  across surface nodes and Newton/GMRES evaluations.
- FSDT and Reddy exact virtual director gradients and inertial
  displacement directions are independently cached by geometry.
  Reddy's curvature-corrected warp is also memoized.
- KL complex-step unit geometry remains deformation-dependent,
  so it is NOT cached across changing configurations. Only its complex
  scratch arrays are reused within one thickness integration.
- Model thickness kernels are bound once per CylinderFunctional, avoiding
  repeated model branching in physical surface loops.

Caches have finite capacities and store immutable arrays. This does
not cache current mechanical strain, electric fields or time-dependent
state and does not change the nonlinear Jacobian-vector product.

The matrix-free full-FDM route now supports configurable
`reference_newton.max_surface_nodes` and a separately enforced
`max_total_dofs`, both checked before allocating reference surface
operators. Defaults remain 8192 nodes and 100000 DOFs. Enlarging a
limit does NOT demonstrate large-grid performance or convergence.
The dense FDM and all manual MVI ranks remain unchanged.

Test specifications in `tests/test_cached_thickness_kernels.py`
compare the cached local tensors to fresh exact unit variations
for CUF TE1–TE4, FSDT and Reddy, Gauss quadrature and capacity guards.

Remaining gates:
1. Execute source regression and exact Gateaux derivative checks.
2. Compare coupled MVI/FDM full residuals, boundaries and reactions.
3. Benchmark CSR memory and wall time across representative meshes.
4. Compare thickness quadrature and dynamic time-step convergence.
5. Validate a physical nonlinear G3D/flexoelectric loaded case for
   every shell model before declaring it engineering-ready.
