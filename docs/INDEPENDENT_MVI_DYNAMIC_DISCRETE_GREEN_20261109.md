# Independent coupled MVI: full dynamic discrete Green work

Date: 2026-11-09. Method: classical fixed-rank MVI (X/Y) for
KL, FSDT, Reddy and CUF TE1-TE4, with G3D/Gz electromechanics.
This checkpoint is **experimental**, not the certified source
microfree/electric-face boundary-value model.

## What was assembled

Three method-owned contributions are represented by THE SAME unweighted
local virtual-work jet coefficients that build the equations of motion:

1. Conservative elastic/G3D/flexoelectric enthalpy and generalized
   mechanical/electric loads: existing
   `NativeCoupledMVISurface.weak_work(...).coefficient_jets`.
2. Inertia and mass-proportional damping: new
   `NativeCoupledMVIInertia.local_weak_coefficients(q,v,a)`.
   CUF TE1-4 use exact curved-shell mass moments and no acceleration
   derivatives; FSDT/Reddy retain local q, q_x, q_s inertial
   coefficients and gradient-dependent Reddy inertia; KL retains the
   nonlinear finite-rotation director acceleration, including
   velocity-squared terms where applicable. The physical `apply()`
   path uses this same coefficient map and the independent MVI
   SBP42/Fourier weighted adjoint.
3. Kelvin-Voigt rate constitutive virtual work: new
   `NativeCoupledMVIViscosity.local_weak_coefficients(q,v)`.
   It is the original thickness-integrated nonlinear KV material
   law, including model-specific higher jets, without new physical
   approximations. The mechanical `apply()` uses this same map.

Inertia and viscosity have **zero electrical variation**. The
electrical flux derives only from the quasistatic electric enthalpy
and prescribed free charge.

`independent_mvi/coupled_dynamic_green.py` constructs separate exact
MVI discrete Green decompositions for conservative, inertial,
Kelvin-Voigt and total instantaneous virtual work. Each uses
the owned SBP42/Fourier identity with outward-normal end trace orders.
The module independently compares the inertial and viscous nodal forces
from the physical `apply()` methods to the direct weighted transpose
of their local jet coefficients, and checks that the total equals the
conservative nodal work plus these two physical forces.

## Execution and storage

`independent_mvi/run_coupled_research.py` now computes
`initial_discrete_green` and `last_discrete_green` from
the COMPLETE instantaneous force system before writing accepted
dynamic states. STATIC acceptance continues to use conservative
Green work, as appropriate.

The dynamic run summary distinguishes:
- `conservative_identity_error`
- `inertia_identity_error`
- `viscosity_identity_error`
- `total_identity_error`
- `mechanical_force_consistency_error`
- `electric_mass_viscosity_error`
- maximum magnitude of the discrete lifted dynamic end work.

This is not an additional boundary traction or a replacement of
the prescribed axial end program. A failed internal consistency
check rejects the step before writing its accepted NPZ state.

## Independent CI

`tests/test_independent_coupled_mvi_dynamic_green.py` includes
synthetic coefficient and force tests for all seven theories with
electricity ON/OFF, plus rejection of a corrupted Kelvin-Voigt force,
wrong operator instance, and invalid rates. These tests do not run a
physical cylinder or a time history.

The dedicated no-physical-solve GitHub workflow checks the complete
dynamic boundary identity separately from the earlier conservative
third-order Green identity and the general fixed-rank MVI algebra.

## Remaining scientific blockers

This proves the precise discrete integration-by-parts decomposition
of *the coded instantaneous virtual work*, NOT the source physical
G3D/Gz higher-gradient boundary model. Work remains to derive and
enforce source-consistent mechanical microfree/hypertraction traces
and inner/outer curved-surface electric flux conditions, validate
a floating potential gauge where appropriate, and test loaded
physical solutions, mesh/rank/time convergence and external benchmarks.
Dynamic acceleration/velocity work can enter natural END actions for
gradient-dependent kinematics; it is not generally negligible.
No unverified physical end/face conditions are silently activated,
and no MKR/FDM or SVD fallback is used. No physical cylinder simulation
was launched in producing these changes.
