# All seven independent models and all time methods

42 opt-in CASE files: KL, FSDT, REDDY, CUF_TE1–TE4 × MKR/MVI × BATHE/NEWMARK/GAUSS_IRK4.

The MKR runner is `python -m independent_mkr.run_models --case PATH --validate-only`.
The MVI runner is `python -m independent_mvi.run_models --case PATH --validate-only`.

Only replace `--validate-only` with `--run` when a physical calculation is explicitly authorized. Calculations have not been run as part of the new architecture audit.

The two routers do not import one another; each has a distinct CASE schema and export directory. These new seven-theory solvers support **linear homogeneous isotropic shells with electrical physics disabled only**. Natural high-order boundary traces and external physical convergence references are still pending. No old solver routes were changed.

Grid 15×9 is intentionally used to permit all MVI ranks 1–8 for CLAMPED KL as well. All initial templates request MVI rank 1; edit the MVI-only `rank` field to request ranks 2–8 after confirming their mathematical and numerical readiness.
