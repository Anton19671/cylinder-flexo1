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


## 2026-11-04 continuation: six FULLY ISOLATED TEMPORAL PROTOTYPES

**Independent MKR (strong finite-difference, not the common variational
functional):**
- `independent_mkr/linear_te4.py` implements standalone **Bathe**,
  **Newmark average acceleration** (β=1/4, γ=1/2), and coupled
  two-stage **Gauss–Legendre implicit Runge–Kutta of order four (IRK4)**.
- `independent_mkr/run_case.py` performs MKR-only CASE validation,
  dispatch, independent history export and distinct named files.
- `config/verification/independent_te4/MKR_{BATHE,NEWMARK,GAUSS_IRK4}.json`.

**Independent MVI (rank-fixed frozen partner X/Y, own weak form):**
- `independent_mvi/linear_te4.py` implements standalone **Bathe**,
  **Newmark average acceleration**, and coupled two-stage **Gauss–Legendre
  IRK4** with rank-fixed X/Y variations on both stages.
- **IRK4 endpoint qualification:** a linear combination of rank-R stage
  fields need not have spatial rank R. This implementation does not
  project a nonrepresentable physical endpoint back onto rank R. Instead
  `MVI_IRK4_ENDPOINT_EXCEEDS_FIXED_RANK` aborts that step. This is a
  mathematical restriction of the experimental formulation, not a
  physical solver success.
- `independent_mvi/run_case.py` implements MVI-only CASE parsing,
  own Bathe/Newmark/IRK4 output and x/y factor archives.
- `config/verification/independent_te4/MVI_{BATHE,NEWMARK,GAUSS_IRK4}.json`.

The two trees import neither each other nor shared legacy
`cylinder_functional_assembly`, `cylinder_variational_dynamic`,
`cylinder_bathe_mvi`, `cylinder_variational_fdm` or
`cylinder_variational_mvi`. In the two new modules, **even the
mass integration, radial load, Hooke law, spatial derivatives, support
handling, and each time integrator are coded independently**. Equal
underlying continuous physics is necessary; shared discrete numerical
code is forbidden in these experimental routes.

CI [#41894005812](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41894005812)
confirmed **14 passing tests**, with separate six CASE validations.
The tests cover imports, independent operators, angular derivative
orientation, exact rank-2 exclusion for rank-1, Newmark predictors,
and a homogeneous manufactured Gauss two-stage problem in both
implementations. The job performs no forced shell time-marching run.

**Important limits:**
- As of this commit, both paths support **linear, isotropic, non-electrical
  CUF TE4 only**, homogeneous end displacement, zero start and prescribed
  harmonic uniform or patch pressure. They are not certified for KL,
  FSDT, Reddy, TE1–TE3, gradient/flexoelectric effects, nonlinear
  geometry, plasticity, or all extended boundary conditions.
- Time integration is implemented, but no nonzero physical results
  from these new independent solvers have been verified, so neither
  is production-ready and no quantitative agreement is claimed.
- `MKR` strong interior difference + displacement clamp still
  requires independent verification of TE4 natural higher-order
  boundary conditions. The experimental MVI weak operator likewise
  needs the full physical boundary classification and an external
  independent reference.
- The new CASE schema is intentionally **different** for MKR/MVI;
  a case for one solver cannot be given to the other. No common
  dispatcher was introduced.
- Existing GUI/legacy calculation cases have **not** been redirected
  to these prototypes. Only explicit method-specific commands with
  `--run` can calculate; `--validate-only` does not run physics.
