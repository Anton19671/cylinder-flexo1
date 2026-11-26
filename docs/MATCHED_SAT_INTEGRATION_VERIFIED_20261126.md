# Matched SAT targeted integration verification, 2026-11-26

Source baseline: 5ed979657f4720b91c130f7ae63f33472d1b5e93.
Verification branch commit: bb2037830d922d76207aa7cad3eba4290da8c082.
GitHub Actions run: https://github.com/Anton19671/cylinder-flexo1/actions/runs/45822602362
Job 139918429117: **235 passed in 5.08 s** (Python 3.11, NumPy 2.4.6,
SciPy 1.17.1, pytest 9.2.0).

The first run 45819278446 passed 234 tests and failed only the
manufactured initialization/Newmark test at the explicit 512-unknown
guard. The test has (6+2)*21*4 = 672 unknowns. The verified correction
raises its local test budget to 1024, not the production default or
any tolerance. The new test's manufactured data and exact guess
exercise the actual native SAT initialization and Newmark route;
the test is not a pressure-driven convergence or performance result.

All 14 real nonlinear coupled work-identity tests passed for seven
theories and G3D/Gz. Existing initial-state/Newmark/Bathe/Gauss
regressions also passed. This does not yet demonstrate physical
nonlinear pressure-driven convergence of all three integrators with
the SAT option. No 252-case matrix or full-suite rerun was performed.
