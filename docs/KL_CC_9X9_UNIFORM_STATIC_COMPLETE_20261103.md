# KL C-C: complete physical static verification of MVI-1..8 versus FDM/MKR
**Date:** 2026-11-03  
**Status:** ALL 11 PHYSICAL SOLVES ACCEPTED; ONE GRID ONLY.

## Scope

This is a **new static numerical verification**, **not** a published
static pressure table from Rawat or Cammalleri. Only the homogeneous
cylinder geometry and material were reused from the Rawat benchmark.

- Theory: Kirchhoff–Love (KL), linear elasticity, MACRO, HOOKE.
- Cylinder: L=2 m, R=1 m, h=0.05 m.
- Material: E=210 GPa, nu=0.3; no flexoelectric or strain-gradient effects.
- Supports: C-C on the two axial ends, including the KL zero radial slope.
- Grid: 9 x 9, axial SBP42 + odd periodic Fourier, the SAME operator for
  independently computed full-grid variational FDM and frozen-partner MVI.
- Load: uniform inward *dead-reference* radial midsurface traction
  100,000 Pa. It is NOT a follower pressure on the current physical face.
- MVI: all fixed ranks 1–8 (distinct jobs and saved X,Y profiles);
  FDM/MKR: independent full nodal solve, a **numerical control**.

## Full physical results at 100 kPa

All 9 solver runs below were accepted. The MVI profiles were saved and
their reconstruction was independently checked against each solver's
mechanical field; no FDM field was substituted into an MVI rank.
The full-field error compares all mechanical components U,V,W in the
same nodal ordering via
100 * || u_MVI - u_FDM ||_2 / || u_FDM ||_2.

| Method | Saved physical status | Mean W at x=L/2, um | Max abs(W), um | Full-field difference vs FDM, % |
|---|---|---:|---:|---:|
| FDM / MKR | CONVERGED | -7.598499770229 | 8.071950353839 | 0 (self) |
| MVI-1 | CONVERGED_FULL | -7.598499770008 | 8.071950349487 | 4.360362e-8 |
| MVI-2 | CONVERGED_FULL | -7.598499770058 | 8.071950349650 | 4.333773e-8 |
| MVI-3 | CONVERGED_FULL | -7.598499769220 | 8.071950364491 | 1.239880e-7 |
| MVI-4 | CONVERGED_FULL | -7.598499770706 | 8.071950350930 | 4.701693e-8 |
| MVI-5 | CONVERGED_FULL | -7.598499770051 | 8.071950350043 | 4.302284e-8 |
| MVI-6 | CONVERGED_FULL | -7.598499770217 | 8.071950350033 | 4.360284e-8 |
| MVI-7 | CONVERGED_FULL | -7.598499770012 | 8.071950349861 | 4.347293e-8 |
| MVI-8 | CONVERGED_FULL | -7.598499784226 | 8.071950346282 | 1.572219e-7 |

The highest full-field difference is approximately
1.6 x 10^-7 **percent**, not 1.6 x 10^-7 as a dimensionless fraction.

## Independent pressure-halving controls

| Method | Mean W at 50 kPa, um | Max abs(W) at 50 kPa, um | Ratio W(100 kPa)/W(50 kPa) |
|---|---:|---:|---:|
| MKR/FDM | -3.799249885115 | 4.035975176920 | 1.9999999999999363 |
| MVI-1 | -3.799249885047 | 4.035975174720 | 1.9999999999770137 |

Both comply with linear scaling to floating-point precision.

## Boundary / kinematic observations

- All accepted fixed-rank profiles rebuild the saved mechanical field to
  the precision of stored floating-point arithmetic.
- Maximum nodal displacement at either clamped end is zero (FDM) or
  of order 10^-37 to 10^-38 m (MVI), which is numerical zero.
- Circumferential variation at the midspan is negligible.
- Axisymmetric pressure on a homogeneous axisymmetric cylinder can admit a
  rank-one separated solution. Agreement across ranks **does not** by
  itself validate rank refinement on non-separable loadings.
- FDM is a same-discretization control. It is not an independent solution
  of the full 3D elasticity equations.

## Diagnosed and corrected MVI-1 bug

**Before the fix** MVI-1 reached the FDM-like stationary state in the
first Y->X sweep but drifted away on the second sweep, and finally
returned MAX_SWEEPS_PROJECTED_NOT_CONVERGED.

The isolated cause was roundoff-only entries in C*T for frozen-partner
directions that already satisfy C-C constraints. A relative QR rank
decision could count numerical noise as independent boundary traces,
create spurious affine corrections, and destroy the converged field.

The corrected static half-step applies an additional absolute,
direction-scaled cutoff to the rank of those trace equations while
**retaining the strict physical boundary-defect check**. No physics,
rank, prescribed support or load was altered; no FDM fallback or
alternative Ritz solver was introduced.

- Fix: commit `8832591c`.
- Real rank-1 regression after fix, KL C-C 100 kPa:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41593147749
  (2 tests passed).
- Additional physical rank-2 and rank-3 regression:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41594310549
  (2 tests passed).
- **Complete 11-case physical matrix**:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41594274172
  (11 physical jobs and aggregate job all successful).
- Aggregate artifact (CSV, JSON, logs with physical run status):
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41594274172/artifacts/13211404146
- Input CASE definitions:
  https://github.com/Anton19671/cylinder-flexo1/tree/main/config/verification/kl_cc_uniform_static

The workflow was additionally corrected to return the **actual solver
exit code** on non-convergence, while still uploading diagnostic files.

## Verification boundaries and next independent checks

**Passed:** correct KL C-C support constraints on this mesh; uniform
load sign and loading work; independent MVI ranks 1–8 all accepted;
on-grid agreement with FDM; pressure linearity at 50/100 kPa;
stationary-state preservation across successive Y/X sweeps.

**Not yet passed and NOT claimed:** grid refinement (only 9x9);
analytical/literature/3D numerical reference for this static pressure
case; non-separable loading rank-convergence; GL nonlinearity;
true follower-pressure or current-area mechanics; and TE1–TE4
static/dynamic verification. Identical results on this axisymmetric
case are expected and cannot alone establish general MVI correctness.

Next physical tests should be a 13x13 mesh of the same KL C-C
problem (within the configured dense memory budget) and a supported
non-separable patch load, with the same independent MVI1..8 plus FDM
matrix. Do not treat the original 9x9 result as a converged grid limit.
