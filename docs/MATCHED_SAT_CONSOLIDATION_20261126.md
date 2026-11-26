# Matched SAT consolidation, 2026-11-26

Verified software changes: explicit strong FDM numerical-scheme and
axisymmetry controls in the native research GUI, and the pressure campaign
with common physical parameters and consistent through-thickness sampling.

- Core + axisymmetric Newton: [304 passed, 3 subtests](https://github.com/Anton19671/cylinder-flexo1/actions/runs/45826424053).
- GUI and dispatch: [188 passed](https://github.com/Anton19671/cylinder-flexo1/actions/runs/45828505005).
- Seven-theory matrix preflight: [54 passed](https://github.com/Anton19671/cylinder-flexo1/actions/runs/45827131361).
- Four initial physical cases, using four Gauss points: KL/G3D/static,
  FSDT/G3D/Bathe, Reddy/Gz/Newmark and CUF TE1/Gz/Gauss IRK4 passed.
- Expanded 12-Gauss-point matrix: 4 successful physical results captured at consolidation.
  The remaining jobs are tracked by run 45827131361; this file does not
  certify results not yet available.

Machine-readable evidence: verification/MATCHED_SAT_VERIFIED_RESULTS_20261126.json
contains the exact JSON emitted by the successful jobs and their run/job IDs.

The numerical mode remains explicitly opt-in, not a silent change of the
legacy FDM or MVI implementation. All original workflows are preserved.
The new matrix workflow is manual-only; there is no new automated
pressure-campaign launch from this commit.

Important limits: 21x4 is a first axisymmetric grid, not a mesh convergence
certificate. The dynamic pilot covers only one time step; the expanded
Newmark series covers two. Physical higher-order boundary closure and
independent article/experimental accuracy remain uncertified. The GUI
tests are automated lifecycle and dispatch regressions, not a full
interactive desktop session. No full-suite rerun after the combined commit
is claimed.
