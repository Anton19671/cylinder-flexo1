# 2026-11-07: mathematical rejection of an overbroad MKR cell scheme

Source: `Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx`,
section 40; independent shell implementation
`independent_mkr/linear_models.py`.

## Decision

**REVERT commit 958737e's attempted collocated-cell MKR operator.**
It is not a verified implementation of the source equations. The
earlier independent linear MKR/MVI split and the corrected CUF outer
radial export (commit d2cb34a) must stay untouched.

The source requires compatible face gradients with paired
divergences **for the gradient and flux terms**, and an independently
consistent inertial contribution. The rejected code placed *every*
zeroth-order mechanical field, including the inertia, through
a circumferential pairwise cell-center interpolation

    (C_s q)_{j+1/2} = (q_j + q_{j+1})/2.

For even N_s, take the checkerboard field q_j=(-1)^j.
Then C_s q=0 identically, although q is not zero. In CUF
kinematics, mechanical kinetic energy has a nonzero pointwise
coefficient mass for the radial thickness fields; the attempted
`mass_action` based on cell-averaged velocities therefore gave
**zero kinetic energy to a nonzero physical velocity mode**.
Paired face gradients may remove the checkerboard defect of a
gradient/divergence pair, but this does *not* justify giving an
independent velocity field zero inertial mass.

The 958737e prototype also treated direct second derivatives
with one-sided end formulas and did not prove the full TE higher-order
natural boundary conditions. Its energy identity test was
insufficient: positive semidefinite mass can satisfy an energy
identity and still have the unacceptable checkerboard nullspace.

## Current verified scope (after rollback)

- The two linear solver packages still have independent spatial,
  time integration, CASE, GUI and result routes.
- The native experimental MKR linear model reverts to its previous
  centered nodal strong difference, while MVI retains its independent
  SBP/Fourier weak formulation.
- The reverted MKR is **not certified compliant** with the
  source's §40 compatible mechanical/electric face-flux requirement.
  On even periodic grids, a doubly applied centered derivative
  is also unacceptable; the current independent CASE restricts N_s
  to odd values, which does not constitute proof of correct physics.
- Neither isolated model implements full nonlinear, G3D/Gz,
  electric/flexoelectric coupling; legacy shared discretization
  cannot be used to claim independent verification.

## Required replacement, not yet implemented

The new MKR spatial scheme must separately place (1) primary
surface displacements and consistent positive-definite inertia,
(2) first/mixed/second mechanical strain derivatives and all
electric fields, and (3) their conjugate face fluxes. It must
derive its nodal action by first variation of **the full**
thickness-integrated enthalpy, preserving the work of the axial
natural/higher-order edge terms. Candidate placement and boundary
closure must be verified with manufactured constant/linear/high
Fourier modes and a positive kinetic-energy matrix before any
physical solve. Reusing the MVI discrete operator is forbidden.

No physical simulation or time stepping was performed for this
audit. The rejected version was not accepted as validated.

## Follow-up: independent nodal MKR end-adjoint correction

Commit dated 2026-11-07 introduces a **separate, limited change**:
`independent_mkr/linear_models.py` now obtains its nodal action by
exact variation of **its own** centered-FD, trapezoidally integrated
thickness-reduced energy. For higher axial derivatives the previous
formal `+ Dxx(flux)` did not equal the discrete energy adjoint near
the clamped ends; the correction uses `Dxx.T W_x flux`, with all
cross derivative-order terms assembled by the same dual pairing.
The physical kinetic-energy and nodal mass terms remain at the
primary nodes, not at circumferential cell averages. The independent
MVI kernels are not imported or changed.

**Still unproven**: this is NOT the complete compatible face-gradient
MKR of source §40. The centered angular difference is still present
and may underresolve high-frequency circumferential modes; odd angular
node counts are still required. No loaded shell result or end-closure
convergence test has been run. See the follow-up proof document
`docs/INDEPENDENT_MKR_NODAL_ENERGY_20261107.md`.
