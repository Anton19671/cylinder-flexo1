# CUF TE4: independent MKR/FDM and MVI solver architecture

Status: EXPERIMENTAL, 2026-11-04. These modules do **not** replace existing
production/reference solvers. Do not treat agreement as physical validation.

## Why a split is required

Legacy Bathe MKR/FDM and rank-fixed MVI both called
`BoundarySpatialProblem.evaluate_effective_residual`, which constructs
the same spatially discretized variational equations. Independent
Newton/Krylov versus alternating X/Y methods **did not** provide an
independent check of the underlying spatial discretization.

## New completely isolated computational paths

### MKR: `independent_mkr/linear_te4.py`

- Physical inputs: independent `PhysicalCase`.
- Independent axial centered FD / one-sided end differences and periodic
  circumferential FD (`_axial_fd`, `_periodic_fd`).
- Independent linear cylindrical CUF TE4 strain-displacement matrices,
  Hooke law, thickness-integrated mass and stiffness.
- Strong-form discrete divergence assembled directly; no transpose of a
  surface quadrature derivative is used to form equilibrium.
- Own pressure field, clamping, and direct two-substep Bathe kinematics.
- Linear isotropic elastic, electrical OFF only.

### MVI: `independent_mvi/linear_te4.py`

- Physical inputs: independent `VariationalCase`.
- Independent axial SBP(4,2) and angular Fourier differentiation.
- Independently assembled cylindrical CUF TE4 **weak** form, weighted
  adjoints, thickness integration, mass and applied load.
- Independent alternating frozen-partner X/Y subproblems with fixed
  separation rank; no FDM fallback.
- Own two-stage Bathe implementation.
- Linear isotropic elastic, electrical OFF only.

Only external numerical libraries (NumPy, SciPy) and Python standard
library are imported by each. No imports between MKR and MVI, or from
the legacy functional/mass/loading/Bathe implementations.

The same *physical continuous governing equations* and Hooke material
law are appropriate for both; separate implementations do not mean
inventing distinct physics.

## Verified and limitations

[CI #41890636903](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41890636903)
passed the algebra-only tests in
`tests/test_independent_te4_solver_isolation.py`. The tests verify
source-import isolation, action of circumferential derivative in its
correct orientation, differences of independent spatial operators,
symmetry of the MVI weak stiffness bilinear form, and inability of
a one-term ansatz to represent a manufactured rank-two field.

During inspection, wrong contraction indices on the angular derivative
were found in BOTH newly created prototypes and corrected in
`independent_mkr/linear_te4.py` and
`independent_mvi/linear_te4.py`. The MVI quadrature adjoint was also
corrected. These were defects in the new independent prototypes,
**not** evidence that the existing production solvers had the same bug.

Remaining: separate numerical convergence studies, physical boundary
closure verification (including higher-order natural TE4 traces),
independent mass/energy benchmarks, nonaxisymmetric forced response,
and GUI/CASE integration. No time-marching physics simulations have
been run to validate these new paths. In particular, the independent
strong-form MKR cannot yet be claimed to enforce all CUF TE4 higher
natural boundary terms. A complete production replacement must not
happen before these checks.

## Rank limitation

For homogeneous clamping at both ends and a discrete field with
`nx` axial nodes, every axial profile is constrained to vanish at the
two end points. Thus a **single CUF component** has at most
`min(nx-2, ns)` algebraically independent spatial ranks; at
`nx=ns=9`, exact rank eight of one component cannot be active.
For rank eight tests, use at least `nx>=10` and sufficient angular
resolution. This does not prevent requesting eight terms but limits
their independent information content on a 9x9 clamp grid.

## No regressions to legacy paths

Nothing in existing `interface/cylinder_reference_case.py`,
`numerics/cylinder_bathe_mvi.py`,
`numerics/cylinder_variational_dynamic.py`, or
`numerics/cylinder_matrixfree_fdm.py` has been rerouted to either
experimental independent module. The previous cases remain unchanged.
