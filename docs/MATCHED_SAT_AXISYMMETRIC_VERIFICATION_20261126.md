# Matched SAT axisymmetric strong-FDM Newton verification

Verification branch commit 722b6f34f588e9c7a9d8048c82bac4466703eedf.
GitHub Actions https://github.com/Anton19671/cylinder-flexo1/actions/runs/45826424053
job 139931637103: 304 passed, 3 subtests passed in 53.67 s.

The optional axisymmetric setting restricts the unknowns to angularly
constant fields. It rejects angularly varying initial fields, prescribed
loads, boundary data, scales, and inconsistent full residuals. All original
full-grid equations are evaluated, including at final acceptance. This
is not a certificate of stability to nonaxisymmetric perturbations.
The general full-grid route remains the default; MVI is unchanged.

The homogeneous local constitutive evaluation reuses identical angular
jets without approximation. The 14 real-kernel regression tests compare
the resulting coefficients to the ordinary full angular loop.

A verified manufactured nonlinear coupled test compares the full and
restricted Newton solution and counts residual calls. This is a reduction
in the number of unknowns, not a measured speedup of real pressure
calculations. The four pressure jobs were in progress when this record
was written. No physical convergence, timing, mesh or external
benchmark certification is inferred from the software tests.
