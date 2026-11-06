# Scientific-readiness gates for the independent cylinder shell solvers

The *independent* `independent_mkr/` and `independent_mvi/` routes cover
KL, FSDT, Reddy, and CUF TE1–TE4 in **linear isotropic, homogeneous,
non-electrical** mechanics, with Bathe, Newmark, and Gauss IRK4 dynamics.

## Constitutive assumptions verified in code

- KL, FSDT and Reddy use the usual reduced **transverse-normal plane-stress
  shell law** (`sigma_zz = 0`). This is the Schur complement of 3D
  Hooke elasticity after statically eliminating thickness strain, *not*
  the full 3D law with `epsilon_zz = 0`. In-plane modulus `C_xxxx =
  E/(1-nu^2)`; transverse engineering shear moduli are `G = E/(2(1+nu))`.
- CUF TE1–TE4 retain `z`-dependent displacement and thickness-normal strain;
  they retain isotropic 3D Hooke elasticity and are quasi-3D cylinder
  *shell kinematics*. This is not a claim that free-surface normal traction
  is already enforced or that a high-order CUF stress solution is validated.
- Classical-shell material reduction applies **only to these independent
  experimental routes**. The original flexoelectric / G3D/Gz implementation
  is not modified by this change.
- New analytic tests compare the reduced constitutive matrix to the 3D
  Schur complement at several Poisson ratios, plus exact axial membrane
  stiffness thickness moments for every theory and each separate method.

## Required evidence before publishing physical results

1. Confirm the adopted FSDT shear correction/calibration and the natural
   boundary traces of Reddy and high-order CUF. The independent experimental
   routes currently use uncorrected shear moduli.
2. Compare MKR and MVI **separately** against an external analytic or FEM
   benchmark for static deflection, modes, and time histories. Agreement
   between MKR and MVI alone is not proof of physical accuracy.
3. Demonstrate grid convergence and time-step convergence, plus MVI rank
   sensitivity and full 3D stress profiles through thickness for CUF.
4. Check free-surface tractions, curvature limits, work-conjugate pressure
   loading and boundary conditions under each stated theory.
5. For genuinely flexoelectric research, independently verify G3D/Gz
   electro-mechanical coupling, reciprocal tangent blocks, electric boundary
   data, and dimensional units: **the independent routes do not implement
   these effects**. Use the separately audited full coupled solver only
   after benchmarking it.
6. Verify stop/pause and recovery behavior and preserve the exact
   input JSON, code revision, units, numerical method, and benchmark in
   every published run. Passing import/algebra tests is only a prerequisite.

**Status:** material reduction is analytically testable; seven-model
physical and coupled flexoelectric validation remains open. Do not report
the independent solver as experimentally verified merely because CI passes.
