# Independent nonlinear coupled MVI surface assembly — research milestone

Date: 2026-11-09; module `independent_mvi/coupled_surface.py`.

## Implemented

- Dedicated MVI SBP42 axial and odd-grid Fourier circumferential operators
  from the independent MVI package, with their own one-dimensional surface
  quadrature. No MKR strong operator, nodal finite-difference field jet,
  CylinderFunctional or reference variational spatial assembler is invoked.
- Exactly fixed rank 1..8 per surface field, with a sum of X(x)Y(s)
  products. No SVD, pseudoinverse, rank growth or full-grid solve.
- Supported local physical kinematics: KL, FSDT, Reddy, CUF TE1--TE4;
  geometry LINEAR, VON_KARMAN, GREEN_LAGRANGE; material G3D and Gz;
  electric potential modes P=0..4 or electricity disabled.
- Shared constitutive laws are local physics ONLY:
  `cuf_g3d_weak`, `cylinder_director_g3d`,
  `cylinder_kl_g3d_weak`, `flexo_gradient_g3d`. This preserves
  source enthalpy, reciprocal direct/inverse coupling and the
  dimensional CUF thickness basis without copying physics.
- Local mechanical variation indices are exactly q,qx,qs,qxx,qxs,qss
  for CUF or additionally qxxx,qxxs,qxss,qsss for KL/FSDT/Reddy.
  Mixed derivatives appear ONCE; no duplicated qxs coefficient.
- The local first-variation coefficients are quadrature-integrated over
  the independent MVI surface. Each axial/circumferential derivative of
  the X/Y factors is varied directly, so gx and gy contain the
  independent rank-fixed frozen-partner virtual work. Mechanical loading
  is generalized force density; electrical loading is physical free
  charge with the enthalpy sign declared by the local material model.

## Exact numerical meaning

For U_c=sum_r X_cr(x)Y_cr(s), a local coefficient A_{c,alpha}
contributes to the discrete first variation as

  dH = sum_{x,s,alpha,c} wx[x]*ws[s] *
       A_{c,alpha}(x,s) *
       (D_x^alpha_x dX_cr(x))*(D_s^alpha_s Y_cr(s))

on the X halfstep, and equivalently with X frozen and dY on the Y
halfstep. Thus the separately assembled X and Y gradients are the
Gateaux derivatives of THIS MVI's own surface discretization, not
a projection of an MKR matrix or a reference FDM output.

A nonzero electric coupling is represented through the shared local
physical enthalpy. The full physical electric gauge, insulation on
inner/outer faces and natural end work remain separate obligations.

## Tests committed (not executed here)

`tests/test_independent_coupled_mvi_surface.py` covers:
- seven kinematic theories and both G3D/Gz material selectors;
- rank/factor/grid shape validation;
- independent centered-difference checks of X/Y gradients of a
  synthetic coupled functional with qxs term and reciprocal
  mechanical/electric cross-term;
- electrical-off charge rejection and forbidden import audit.

No actual physical shell, time history, benchmark or GitHub Actions run
has been performed. Green tests must NOT be claimed without execution.

## Not implemented / strict fail-closed boundary

This is a method-specific SPATIAL WEAK WORK ASSEMBLER, not yet a complete
independent coupled nonlinear MVI solver. It deliberately provides no
auto-dispatch and claims no convergence. The following are still required:

1. Physical essential and natural edge traces, high-order microfree
   conditions, electric insulation and global electric gauge.
2. Own X/Y frozen-partner nonlinear Newton subproblems, stable
   gauge/saddle solve with condition check (no SVD), iteration history,
   rank-sufficiency acceptance and cancellation checkpoints.
3. Dynamic MVI inertia for all seven theories, standalone Newmark/Bathe/
   Gauss stages, independent initial acceleration and whole-field work
   balance.
4. Compatible high-order discrete boundary/face closure for MKR and MVI,
   independently manufactured tests, mesh/rank/time convergence and
   verified external scientific comparisons.
5. CASE/GUI activation only after complete physical validation.

The legacy coupled reference MVI remains a reproducible reference branch,
NOT a fully independent spatial discretization. Do not relabel it as one.
