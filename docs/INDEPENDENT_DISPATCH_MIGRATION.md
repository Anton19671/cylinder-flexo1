# Independent-only dispatch migration — preserved legacy until certified

The official `run_final_case` and `validate_final_case` dispatcher
now accepts **native** independent method CASE schemas:
`INDEPENDENT_MKR_SEVEN_MODELS_V1` and
`INDEPENDENT_MVI_SEVEN_MODELS_V1`.
The source `schema` is authoritative; MKR calls only
`independent_mkr.run_models`, MVI calls only
`independent_mvi.run_models`. No shared VARIATIONAL_REFERENCE
fallback is permitted from an independent CASE.

These native routes support seven theories (KL, FSDT, Reddy, CUF
TE1–TE4), linear MACRO isotropic HOOKE, **no electric or gradient**
physics, and Bathe/Newmark/Gauss IRK4. The associated real solver
uses its own schema, own saved results and own method-specific physics.
A full MICRO nonlinear flexoelectric case is rejected rather than
silently given a linear independent solution.

The older `VARIATIONAL_REFERENCE` path is **not removed yet**.
It remains explicitly selectable and intact for ongoing physics
verification and prior reproducibility. Its automatic deactivation
requires a new fully coupled independent MKR/MVI model that supports
all corresponding physics and boundaries, plus real benchmark
validation. Until then, hiding it would destroy currently supported
physical functionality and scientific traceability.

The existing reference-to-independent bridges are not invoked
automatically. Their allowed `pressure_face` is preserved (MID
or OUTER as provided) and constitutive mismatch is rejected.
For pressure-face comparison see dedicated seven-theory weak-work
tests. Native JSON is required for the independent solver parser;
no dangerous ad hoc YAML conversion is introduced.

`tests/test_independent_native_final_dispatch.py` verifies all
42 distinct theory × method × time algorithms, dry-run only;
for selected calls, mocked runners assert no opposite-method call,
no reference solver and rejection of unknown schemas.

## Explicit 2D CUF contract and corrected outer-face export

Source: `Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx`,
sections 13, 36, 39, 40 and 46. The G3D designation describes the **three
physical directions of strain gradients** in the constitutive law. It does
**not** require a 3D discretization: spatial fields are coefficients on
`(x,theta)`; through-thickness dependence is analytical (CUF or classical
kinematics) with 12-point Gauss integration. A 3D field can be reconstructed
at selected `z` stations for output, without a spatial `nz` grid.

The **two independent linear runners** now record
`spatial_grid=AXIAL_CIRCUMFERENTIAL_2D`,
`thickness_scheme=ANALYTIC_KINEMATICS_GAUSS12_NO_Z_GRID`, and an explicit
linear macro Hooke/electric-off/gradient-off scope in validation and run
summaries. These attributes are duplicated by design, not dispatched via
another method's module.

For CUF TE1–TE4 the physical radial displacement on the outer surface
`z=+h/2` is `w_out(x,theta)=sum_k w_k(x,theta)`, whereas `w_0` is the
middle-surface radial coefficient. Both MKR and MVI independently export
`central_outer_w_m` and `maximum_abs_outer_w_m` **in addition** to the
historical `central_w0_m` and `maximum_abs_w0_m` CSV columns. For KL,
FSDT and Reddy the two fields coincide. The separate reference-face
pressure convention and all existing solver equations are unchanged.

This is an **export and architectural safety correction, not a claim of
full Krysko–Zhong implementation**. In particular, the native linear MKR
still needs the compatible face-gradient/flux construction and complete
higher-order natural-boundary certification from source section 40; it is
not a verified G3D strong solver. G3D/Gz, nonlinear Green–Lagrange and
fully coupled flexoelectric research remain in legacy reference routes,
which **are not two independent spatial discretizations**. Do not silently
fall back to a legacy solver when an independent CASE requests unsupported
physics. The AST isolation tests scan all native method modules (including
GUI and input adapters) for cross-method and shared numerical imports.

No physical shell calculation was run to prepare these changes.

## 2026-11-07 MKR end-adjoint correction (strictly limited)

The independent MKR linear nodal strong balance now uses the discrete
first variation of its OWN centered finite-difference surface energy.
At axial endpoints, derivative transpose plus trapezoid weights
ensures that the linear bending contributions are self-adjoint on the
prescribed displacement/slope spaces; the former unpaired second
axial difference was not. Native MVI SBP/Fourier operators and the
entire coupled legacy physics are unchanged. This step does not
claim that centered angular differences are a full implementation of
source §40's compatible face gradients. No physical case was run.

## 2026-11-07: exact-zero trap in independent fixed-rank MVI

Both seven-theory MVI and separate CUF TE4 MVI historically used
arbitrary nonzero axial partners and zero angular partners. A localized
load orthogonal to these initial partners can make both projected
X/Y subproblems inactive despite a nonzero full two-dimensional
loading. The normalization of an all-zero partner can then destroy
the previously nonzero axial partner, creating a falsely accepted
zero field. This case is demonstrated by a small manufactured
matrix example independent of any shell time stepping.

The new method-owned activation constructs leading axial SVD directions
from the current admissible two-dimensional driving work, with fixed
rank and with angular partners initially zero. A zero previous field is
reactivated for a nonzero new load; both Newmark/Bathe and individual
Gauss stages use their own MVI operator. An unchanged zero field is
explicitly rejected when it does not satisfy all free equations.
No MKR assembler or physics subroutine is imported, and no 3D grid
is introduced. Algebraic seed tests for all seven theories and
ranks 1/2/4/8 (plus TE4) are in
`tests/test_independent_mvi_load_activation.py`.

This removes a proven false-zero path. It is not evidence of fixed-rank
sufficiency, complete nonlinear coupled G3D physics, or validated
physical dynamic behavior. No real shell calculation was launched.

## 2026-11-07: independent STATIC MKR and MVI, complete routing and inventory

For experimental LINEAR MACRO HOOKE, electric and gradient physics OFF,
seven theories now have their own stationary (non-time-stepping) solvers:

- `independent_mkr/run_static.py` drives only its method's own
  `IndependentStrongModel.solve_static`: nodal strong finite differences.
- `independent_mvi/run_static.py` drives only its rank-fixed
  `IndependentVariationalModel.solve_static`: SBP/Fourier weak form,
  alternating X/Y subproblems.
- Unique `INDEPENDENT_*_STATIC_SEVEN_MODELS_V1` JSON schemas and
  explicit own-method dispatch reject accidentally reused DYNAMIC
  time parameters, opposite-method files and nonlinear MICRO physics.
- Separate output files include original CASE, status, physical
  2D coefficient field, radial middle/outer fields, and in MVI the
  original rank-fixed X/Y factors.
- Both the dynamic and static native runners mark successful
  completed work with `converged=true`. The public console exit
  contract now recognizes these independently completed cases.

The scientific CASE-only catalog contains **532 distinct routes**:
the former 469 plus 63 static combinations (7 theories × MKR or MVI
ranks 1 through 8). This is a **catalog of possible preflights**,
not a numerical accuracy or physical convergence certificate.

Physical G3D/Gz, nonlinear geometry, reciprocal flexoelectric
coupling and complete higher-order boundary operators remain
**unimplemented in these independent packages**. Existing coupled
reference methods must not be relabeled as independent.

The new `tests/test_independent_static_dispatch.py` uses mock
stationary model instances to verify public routing, result
commit, method exclusivity and rejection of wrong SCHEMA without
a single actual physical calculation. New CI only compiles,
preflights and runs such algebraic/source tests; no loaded
shell simulation was executed in this work.

## Full STATIC CASE preflight coverage (2026-11-07)

The explicit `config/verification/independent_static/` directory
contains **63** ready-to-validate independent stationary cases:
seven theories × one MKR or eight MVI ranks. The 56 MVI inputs
use rank-specific output directories to prevent name collisions.

The independent algebra CI workflow now invokes **all 63 cases**
with the matching `--validate-only` CLI and tests the same
input files through the public `validate_final_case` dispatcher.
No `--run` invocation appears in this workflow.

This verification checks CASE schemas, method selection, unsupported
physics rejection, spatial grid shape and output routing. It is not
a physical solution or numerical accuracy demonstration.

## GUI: independent STATIC is available from each own method panel

The independent MKR/MVI dynamic windows now expose a `СТАТИКА`
button that opens the **same method's** separate static window.
`independent_mkr/static_gui.py` dispatches solely to
`independent_mkr.run_static`; `independent_mvi/static_gui.py`
dispatches solely to `independent_mvi.run_static`. Neither calls
the other method or the shared legacy variational backend.

Each static UI validates/edit/saves a distinct stationary JSON
schema, supports seven theories (MVI ranks 1–8), standalone output
folder selection, result viewing and STOP. PAUSE is deliberately not
shown for stationary solves because their Krylov/X-Y loops do not
implement safe pause checkpoints. Dynamic interfaces remain unchanged.
These UI controls **never start the physical solver automatically**.
Actual execution requires the explicit `РАССЧИТАТЬ` button.
The static UI smoke test is headless and algebra-only.

## 2026-11-07: full 2D equation verification for fixed-rank MVI

The source §39 explicitly requires that stopping a rank-fixed X/Y
alternation does not by itself certify the original 2D problem.
The previous MVI code accepted the numerical field whenever its change
between two sweeps became sufficiently small. This condition alone
can accept a stationary rank-R approximation that does not satisfy
all admissible surface equations.

The **own variational** weak-form action is now checked against the
applied load over the **entire constrained two-dimensional DOF space**
using the same already-present Krylov accuracy parameter. This test
is performed before acceptance in standalone independent seven-theory
STATIC, Newmark/Bathe effective solves, and both Gauss-IRK4 stages.
The independent CUF TE4 engine has its own check for all 15 free
mechanical fields and both Gauss stages. Prescribed axial DOFs do not
count as free equations. The check neither calls an MKR operator nor
uses a full-grid inverse as a substitute for X/Y iteration.

**Fixed rank remains fixed.** If the approximation cannot meet the
full free-space equations, the method raises an explicit insufficient-
rank diagnostic instead of saving a misleading CONVERGED physical
field. Increase R in a distinct user-controlled CASE rather than
silently changing a requested R or interpolating another solver.
This is an algorithmic acceptance safeguard, not proof of scientific
accuracy, spatial resolution, correct G3D boundary terms, or
coupled electromechanical physics. The original §40 angular face
scheme is still unfinished in independent MKR.

The new checks in
`tests/test_independent_mvi_global_surface_check.py` use tiny
manufactured arrays and bypass all physical time marching or shell
solves; they verify the rank-insufficient rejection and the exact
free-space acceptance contract independently of a spatial benchmark.

## 2026-11-07 MKR periodic primal/dual update

Independent linear MKR stiffness now samples one original
thickness-integrated elastic energy on both nodal and periodic
angular-face grids and uses paired transpose maps for every mixed
mechanical derivative. The direct node curvature and face gradient
remove the two distinct even-grid alternating blind spots.
The inertia remains explicitly nodal; no accidental zero-inertia
face averaging can reappear. This is a method-owned strong/energy
finite difference, not the MVI SBP/Fourier weak assembly. The input
validation now allows even circumferential node counts in MKR only.

This remains experimental linear **electric/gradient OFF** physics;
complete source §40 G3D/Gz and edge boundary accuracy remain open.
Algebraic tests are included but no physical calculation was run.

## 2026-11-07 coupled reference MVI: static and dynamic Newton isolation

The **coupled reference** X/Y branch had an architectural intersection
with the old FDM Newton: static MVI imported the entire FDM subspace
solver, and dynamic MVI called a function also used by FDM.
Both are now separated at the **numerical halfstep** level:

- `numerics/cylinder_mvi_subspace_newton.py` owns the static
  mechanical/electric X/Y-constrained Newton operator and settings.
- `numerics/cylinder_mvi_dynamic_subspace.py` owns the dynamic
  constrained Newmark tangent and reduced solve. Both new modules
  explicitly reject missing X/Y trial directions. Neither imports
  the FDM spatial solver or its Newton code.
- The legacy dynamic FDM continues through its existing own
  `solve_dynamic_subspace`, while coupled MVI calls
  `mvi_solve_dynamic_subspace` on its frozen separated directions.
- Old JSON/parameter settings are converted to method-owned MVI
  settings by values only; no FDM numerical routine is called.

**Important limitation:** the coupled `CylinderFunctional` and
continuum boundary/physical assembly remain common across the
reference routes. They have not been reimplemented as two
independent full nonlinear G3D/Gz spatial assemblies. The method
isolation above is numerical at X/Y halfstep level, not proof of
full physical independence. The old FDM/MVI-dispatch code in
higher time-integration adapters needs separate auditing.

No physical shell simulations or time steps were launched.
Tests inspect source and reject full-grid method fallback without
assembling any numerical shell state.
