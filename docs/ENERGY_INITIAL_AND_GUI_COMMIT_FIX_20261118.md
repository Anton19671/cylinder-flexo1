# Energy-FDM initial fields and committed GUI history — 18 November 2026

Reviewed base: `8268d0dd1356d2fa1e42420150d33c9621a4e8c4`.

## Reproduced defects

1. Native energy dynamics accepted and silently projected away an imposed
   support velocity of `1e-12` because its compatibility gate used an
   absolute `1e-9` threshold. The same defect affected displacement,
   acceleration and accepted electrical states. A larger unrelated field
   or angular profile must not mask a small field's incompatible support.
2. Energy-FDM CASE preflight reused the strong-route support gate and did
   not validate the initial acceleration guess in its own constrained
   space. A CASE could pass validation and fail before initialization.
3. The native live GUI treated an NPZ as accepted before the corresponding
   `RUN_SUMMARY.json` accepted counter was committed. A stop/crash between
   those writes could show an uncommitted point and false completed progress.
4. The four-character step glob ignored `STEP_10000.npz` and later files.
   Nonfinite or negative stored times could also reach the live graph.

The added regression cases failed on the reviewed base: nine initial-field
cases and eight GUI cases. Large incompatible fields already rejected by
the old gate were retained as controls.

## Changes

Energy-FDM compares round-trip constraint compatibility separately for each
component and circumferential column, with a relative tolerance of `1e-11`
and no unit-valued absolute floor. Fields are normalized before subtraction
to avoid overflow/underflow. Displacement, velocity, acceleration and electric
potential keep separate physical units. Valid tiny fields and homogeneous
normal-derivative constraints through second order retain projection
roundoff tolerance.

`validate_initial_fields` performs the same non-solving compatibility checks
in CASE preflight and initialization. Electrical initial guesses retain their
existing projection semantics; already accepted electrical states must obey
their imposed potential/gauge. The strong route retains its existing explicit
absolute support tolerance and no longer supplies this method-specific gate
to energy-FDM.

The native GUI now reads a valid nonnegative integer accepted counter before
reading state files. It retries files on later polls when the counter has
not yet committed them, sorts arbitrary-length step numbers numerically,
rejects invalid times, and takes progress from the committed counter.
Static results use the same commit gate. Initial state zero does not count
as a completed physical time step.

## Verification

**225 passed, 26 subtests passed, no failures.** One warning comes from the
installed external pytest/anyio plugin. Python 3.12, NumPy 2.5.5, SciPy
1.18.1, pytest 9.2.0; one OpenBLAS thread.

Checked groups: native energy work/end spaces and all three time drivers,
energy and strong CASE dispatch, strong initial-state contracts, physical
normal-trace polynomial/Green operator identities, native GUI lifecycle,
independent GUI stop/history contracts, repository import integrity.

All driver tests in this pass use artificial/manufactured work or mocked
solvers. No loaded physical cylinder campaign or physical time simulation
was launched. The commit uses `[skip ci]` because pushing these source paths
would otherwise automatically launch loaded energy-FDM physical workflows.

## Scope

These fixes do not establish physical mesh/time convergence or certify
higher-order mechanical/electrical boundary closure. The documented strong
axial Green mismatch is not repaired by this initial-field/GUI change.
