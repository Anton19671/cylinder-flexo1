# Global coupled Newton and boundary trace verification

**Primary reference**: Krysko-Zhong CUF TE1–TE4 corrected manuscript,
24 October 2026. This supplements, but does not replace, the local 18-gradient,
coupled enthalpy and manufactured end-traction identities.

## Algebraic global checks

- Seven kinematic theories `KL`, `FSDT`, `REDDY`, `CUF_TE1`–`CUF_TE4`
  and G3D/Gz: compare the total *global assembled* enthalpy directional
  derivative with mechanical+electrical generalized work. The test includes
  prescribed mechanical end tractions/springs, one grounded face electrode
  and positive outward free charge.
- With the identical nonlinear shell state, compare independently
  perturbed `dq dot (dR_q/dphi) dphi` and
  `dphi dot (dR_phi/dq) dq`. No numerical symmetrization is allowed.
  Unlike matching MKR and MVI displacement fields, this tests conjugacy.
- For KL, Reddy, CUF TE4: use actual affine boundary constraints, with
  *fully insulated electrical boundary and exactly one global potential
  gauge*. Check mechanical-electrical tangent symmetry on two independent
  admissible directions; eliminated electrode/support directions are not
  incorrectly treated as free.
- For the same three theories, the dynamic effective Newmark operator must
  leave the electrical block quasi-static and maintain reciprocity between
  mechanical and electrical Jacobian blocks under acceleration and velocity
  predictors. No artificial electric inertia is allowed.
- CUF TE4 is checked with LINEAR, VON_KARMAN and GREEN_LAGRANGE geometry
  at the global discrete work level, separately for G3D and Gz.

**Scope:** these tests use 4x5 SBP reference surfaces and deliberately three
thickness Gauss points to test *the discrete mathematical identity*, not
the converged TE4 through-thickness solution. They are non-physical local
algebra tests, not nonlinear shell solutions, not proof of a valid full
micro-free boundary closure, and not grid, TE-order or timestep convergence.
No change to GUI or production CASE dispatch follows from them.
