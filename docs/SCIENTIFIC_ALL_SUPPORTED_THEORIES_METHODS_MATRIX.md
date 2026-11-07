# Comprehensive cylinder model/method combinations (no physical solves)

This test inventory proves CASE schema validation and dispatch only. It
does not certify a physical shell solution, convergence, or an externally
independent model.

| Branch | Physics | Theories | Methods and ranks | Time | Count |
|---|---|---|---|---|---:|
| MICRO reference | GREEN_LAGRANGE, G3D/Gz, flexo ON/OFF | Seven | FDM and MVI-1 | STATIC, DYNAMIC Newmark | 112 |
| MACRO reference | LINEAR, VON_KARMAN, GREEN_LAGRANGE; elastic | Seven | FDM and MVI-1 | STATIC, DYNAMIC Newmark | 84 |
| Independent linear experimental | LINEAR MACRO HOOKE | Seven | own MKR and own MVI ranks 1..8 | BATHE, NEWMARK, GAUSS_IRK4 | 189 |
| Total | | | | | 385 |

The earlier MICRO branch test used TE1 electrical thickness order for
higher CUF models. Now it imports exactly the physical pilot CASE
generator: TE2 has 3 electric coefficient fields, TE3 has 4, TE4 has 5.
KL is tested with a nondegenerate 5x4 grid and mid-surface reference
pressure; other models use outer reference pressure.

The 28 direct coupled first-variation/inertia tests also now use TE-N
electrical coefficient counts. The independent linear experimental
routes use their own distinct case readers, and wrong-method CASE
schemas fail instead of silently selecting the other method. Static
and dynamic MACRO LINEAR/VK/GL are tested only through the original
variational reference backend.

385 is NOT every imaginable material, load, boundary, damping,
geometry or time scheme. Coupled MICRO Bathe and Gauss IRK4 are not
part of the reference Newmark CASE route and are not claimed complete.
Neither the independent linear MKR/MVI packages nor these checks
validate nonlinear flexoelectric dynamics in independent methods.

The workflow only validates inputs and dispatch, never advances a
physical time or load step. The scientific physical evidence campaigns
and external reference comparisons remain separate.


The seven-theory independent MVI runner now rejects a saved `X/Y`
factor pair unless reconstructing the accepted END displacement field
agrees to a tight, scale-aware norm tolerance. This closes a gap where
correct-shaped, finite factors could disagree with the stored physical
field yet still have the solver status `CONVERGED`. The associated
189-case mocked stage-dispatch matrix checks Bathe's two stages,
Newmark's end stage, Gauss IRK4's two internal stages plus endpoint,
method-specific NPZ, time and case metadata, and MVI factor reconstruction.
All numerical step outputs in these tests are manufactured stubs, not
physical solutions or validation against external benchmarks.


## Full-versus-rank-constrained MVI classification (112-case pilot)

The real physical 112-case verifier now uses the same solver-certificate
classification as the rank/refinement campaign. For static MVI it
accepts `CONVERGED_PROJECTED` only with a fully consistent projected
certificate, and explicitly persists `RANK_CONSTRAINED_ONLY` and
`full_space_stationarity_confirmed=false`. It accepts
`CONVERGED_FULL` only when the matching physical full-space
certificate is true. Static FDM and dynamic schemes preserve their
own required `CONVERGED` acceptance rules. A successful fixed-rank
calculation is not automatically full-space accuracy.

No physical calculations are launched by this patch or the added tests.
