# Gao–Park: accelerated native solver campaign

The pilot at TE1, ell=0.5 micrometre, grid 21×9, confirmed local constitutive
work and energy equivalence (relative errors < 2e-11 for twelve independent
jet probes). Native FDM and MVI ranks 1–7 passed in the verified pilot logs.
The full campaign uses the same fail-closed local audit for EACH material and
thickness order. Only the constant linear constitutive map is reused across
grids and methods. The native FDM strong balance, SAT closure, and native MVI
fixed-rank halfstep iterations remain unchanged. No nonlinear production code,
GUI or main branch changes.

Sixteen groups × 27 independent solves = 432 cases:
TE1–TE4, ell/a = 0, 0.1, sqrt(0.05), 0.5, grids 21×9, 31×13, 41×17,
FDM and MVI1–8. In each MVI solve the original zero displacement and Fourier
rank seed are independently created; the FDM result is never supplied to MVI.
The earlier redundant direct-Gauss campaign is superseded by the new run
through workflow concurrency. All failures and timeouts are retained.
Compare accepted native profiles with the independent same-order reference,
and separately with the analytical Bessel solution; do not confuse TE-order
error with solver or surface-grid convergence error.
