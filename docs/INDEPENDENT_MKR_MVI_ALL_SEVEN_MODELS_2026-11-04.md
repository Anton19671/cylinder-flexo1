# Independent MKR and MVI for all seven cylinder theories

Date: 2026-11-04. Repository `Anton19671/cylinder-flexo1`, `main`.

## User-required architecture

**NO numerical solver intersections between MKR and MVI.**

`independent_mkr/linear_models.py` implements its own:
- KL (3 components); FSDT/Mindlin and curvature-corrected Reddy (5);
  CUF TE1/TE2/TE3/TE4 (6/9/12/15).
- Continuous linear cylindrical strain-to-coefficient maps,
  12-point thickness Hooke stiffness and consistent velocity-gradient mass.
- Centered and one-sided axial finite differences, periodic
  circumferential finite differences, strong-form divergence.
- Own radial force and homogeneous clamping; for KL, radial slope is
  constrained on both axial ends as well as all displacement traces.
- Own Bathe, Newmark β=1/4 γ=1/2 and coupled Gauss-Legendre IRK4.
- `independent_mkr/run_models.py`, own MKR-only CASE parser and outputs.

`independent_mvi/linear_models.py` implements separately:
- All seven theory-specific linear kinematics, stiffness, thickness
  integration, inertia, pressure, CLAMPED constraints and KL slope.
- Own SBP42 axial and Fourier angular derivative operators.
- Own **quadrature weighted weak-form operator**, not strong FD.
- Rank-fixed separated X/Y fields, alternating 1D variations and
  projected frozen-partner Krylov steps; rank 1..8 subject to boundary
  capacity.
- Own Bathe, Newmark and fully coupled two-stage Gauss-Legendre IRK4;
  Gauss endpoint exceeding prescribed spatial rank is rejected rather
  than silently truncated.
- `independent_mvi/run_models.py`, own MVI-only CASE parser, state
  factor archives and distinct result paths.

Neither model module nor runner imports the other method, legacy
`CylinderFunctional`, `BoundarySpatialProblem`, the legacy variational
solvers or legacy time integrators. Both only share standard Python,
NumPy and SciPy libraries. Both *must* describe the same continuous
mechanical physics; duplicating this law is not proof of its correctness.

## Input and output matrix

42 separate test inputs:
`config/verification/independent_all_models/{MKR,MVI}_{KL,FSDT,REDDY,CUF_TE1,CUF_TE2,CUF_TE3,CUF_TE4}_{BATHE,NEWMARK,GAUSS_IRK4}.json`

Each is independently parsed by:
- `python -m independent_mkr.run_models --case PATH --validate-only`
- `python -m independent_mvi.run_models --case PATH --validate-only`

The physical `--run` switch is separate and was **not used**.
Incorrect-method CASE schemas are rejected.

## Evidence

[Actions run #41899426957](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41899426957):
**92 passing algebraic CASE/operator tests**, including all 42
method-theory-time configurations. Preflight alone does not create
result directories or perform physical time marching.

Independent cylindrical Reddy shear surface identities were checked.
An initial *test expression* omitted the curvature contribution
`z/(R+z)`; the test was corrected without altering physical equations.
The symmetric weak form, own finite-difference strong form, and
homogeneous KL radial slope restrictions were also checked.

## EXACT status — what cannot yet be called complete

These are **independent experimental linear-elastic calculation
routes**, not yet accepted production solvers. They have not
been validated under nonzero physical transient loading or against
a third-party analytical/FEM benchmark.

In particular:
1. **KL:** higher-order natural edge terms and consistency of strong
   finite differences at the clamped boundary remain to be proved.
2. **Reddy:** curvature-corrected shear face behavior is checked,
   but complete kinetic and natural boundary validation remains.
3. **CUF TE1–TE4:** independent thickness-polynomial mass/energy and
   convergence across mesh and thickness quadrature require studies.
4. **Gauss IRK4 MVI:** endpoint of two rank-R internal stages can have
   higher spatial rank; strict rank containment is enforced, but its
   behavior on actual forced time histories is not yet established.
5. **All theories:** only homogeneous isotropic, linear, non-electrical
   dynamics. Geometrically nonlinear VK/Green–Lagrange, gradient/flexo,
   constitutive nonlinearities and all mechanical BC sets have **not**
   been independently reimplemented.
6. **GUI/production:** old reference/legacy routes have not been
   redirected, to avoid silently changing already existing calculations.

Therefore, the architecture/code **is separated for all seven linear
models**, but *full-model physical verification and complete separation
of every nonlinear/electrical capability remain unfinished*. It would
be incorrect to claim otherwise.


## 2026-11-05 GUI integration — two EXCLUSIVE method panels

The actual user-facing entrypoint is `interface/FINAL_GUI.py` (loaded
through the project-root `FINAL_GUI.py`). Its **last** `_build_ui`
implementation in `FinalSolverGUI` now exposes two explicit buttons:
- **МКР — свой решатель…** → `independent_mkr.gui.open_window`
- **МВИ — свой решатель…** → `independent_mvi.gui.open_window`

Each window lives in its OWN package. Its editor validates only that
method's CASE schema and uses only its own
`independent_mkr.run_models` or `independent_mvi.run_models` executable.
Bathe, Newmark and two-stage Gauss IRK4 are chosen independently;
all KL/FSDT/Reddy/CUF TE1..4 options are offered. The MVI window alone
supports ranks 1–8, constrained by physically admissible axial fields.

Every run creates a unique `CASES/METHOD_THEORY_TIME_...json` and a
distinct `RUNS/METHOD_THEORY_TIME_...` directory under the user-selected
result root. Existing CASE files, old numerical output and GUI workflows
are not overwritten. Input validation and file saving NEVER execute
physical calculations; **only a user's explicit «РАССЧИТАТЬ» button**
starts the subprocess. The child writes own method-specific stage NPZ,
history CSV, summary JSON and (for MVI) factor NPZ. Stop terminates
only the window's own child process. Exit-code zero alone does NOT
imply success: the GUI checks the method's own completed summary and
fully accepted step count.

The historical shared-functional GUI dispatch is deliberately NOT
redirected into these prototype solvers. The legacy CASE contains
nonlinear, micro and electric fields that must NOT be discarded or
silently replaced by a linear/electric-OFF independent model.

[CI run #42045606914](https://github.com/Anton19671/cylinder-flexo1/actions/runs/42045606914):
**148 passed**. All 42 own CASE combinations are checked headlessly;
both independent GUI modules and the main GUI compile; import checks
fail upon cross-method or legacy-variational backend imports. No GUI
window or forced physical shell simulation is started in CI.

IMPORTANT: this demonstrates that the two prototype GUI routes are
present and static/semantic preflight works. The GUI has not yet been
visually verified on the user's Windows display; and no nonzero
independent physical transient has been run or externally validated.
**Only linear MACRO HOOKE, with no flexoelectric, no GL/VK, no plasticity
and no modal analysis, is available in these isolated windows.**
