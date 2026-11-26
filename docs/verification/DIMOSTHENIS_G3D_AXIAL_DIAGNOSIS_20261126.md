# G3D exact axial limit: independent initial-acceleration diagnosis (2026-11-26)

Scope: LINEAR CUF TE1, Poisson ratio zero, electric OFF, stationary parallel ends. E=1e9 Pa, rho=2000 kg/m3, L=1 m, R=2 m, h=0.08 m, ell=0.2 m. Reference: DOI 10.5937/jaes0-51432, axial strain-gradient bar equation. The static sine body-force case is an independently derived extension, not a published table. No claim about flexoelectricity, radial modes, other theories, or general nonlinear dynamics.

Full shell FDM results from Actions run 45838033928, commit 49f8de49c30c51ee821a058d9ba08b244d565279:
- 21x4 static field relative error 3.8063239555652575e-5; PASS.
- 41x4 static field relative error 9.344025062402571e-7; PASS.
- 21x4 Newmark: initial acceleration relative error 0.5399206546362255, unweighted acceleration projection frequency 2759.17328693313 rad/s; after 2 steps displacement relative error 0.00020970200975756857; original benchmark FAIL.
- 41x4 Newmark: initial acceleration relative error 0.21554159384211063, unweighted projection frequency 2643.7980560638584 rad/s; after 2 steps displacement relative error 0.00020890644766234352; original benchmark FAIL.
- New 21x9 static FDM, run 45838908452, commit a1cf266210b7abba33c7072beacee8a20ee28621: relative field error 3.8063239538262704e-5; PASS.

Independent scalar diagnosis: for the pure axial mode, use the exact matched quartic operator coefficients from numerics/cylinder_matched_axial_quartic.py at main 6db1a20b9c7a06e65e47597c5dbbd3b99354cd0f. Let H be the dimensionful positive diagonal axial quadrature, P the backward-paired derivative, M its partner, D2=M P. Restrict to the free interior coordinates. The scalar stiffness is K=P^T H P + ell^2 D2^T H D2 and mass is H (the common area cancels). Acceleration is -(E/rho) H^{-1} K u on the free coordinates. This scalar calculation was carried out independently in JavaScript from the stored rational coefficients, not from the full shell solution or time integration.

For sampled u=sin(pi*x/L), the independently reproduced unweighted projection frequencies are 2759.173286933284 and 2643.79805606843 rad/s (21 and 41 nodes); reproduced acceleration errors are 0.5399206546362133 and 0.21554159387639704. These agree with actual Newmark initial states to roundoff/solver tolerance. The error therefore exists in the spatially discrete initial state before any time integration. For n=41, 99.9999516% of the squared acceleration error lies within the first/last 12 nodes. The sampled continuum mode is not an exact discrete eigenvector and has nonzero discrete D2 at the ends.

Correct mass-weighted Rayleigh estimates of the sampled sine are 2623.5497653053058 (21), 2623.5442257305635 (41), 2623.544172212794 (81) rad/s. These are estimates, not eigenvalues.

Independent generalized eigenmode calculation using Cholesky and inverse iteration on H^{-1/2} K H^{-1/2}, E/rho factor restored:
| axial nodes | omega rad/s | relative eigen-equation defect | relative mode-shape error (midpoint normalized) |
|---|---:|---:|---:|
| 21 | 2623.544949542085 | 2.7064112439065642e-12 | 3.8325658679302433e-5 |
| 41 | 2623.5441787756313 | 1.6979323644791074e-11 | 9.349596223757418e-7 |
| 81 | 2623.544171693017 | 2.635854542670477e-10 | 2.1189443801943756e-8 |

Exact continuum omega=2623.544171769313 rad/s. Finite-precision cancellation limits the finest-grid estimate. No claim of globally certified modal spectrum or full-shell modal solver validation. No production source or tolerance changed. The original acceleration criterion is retained; the failed tests are not retroactively marked passed. Separate discrete-mode time integration could isolate integrator accuracy; it must not replace the original continuum initial-data test.

All-rank campaign: run 45838908452. Preflight validated 72 input cases; first-grid matrix has 36 physical jobs (FDM and MVI1..8, STATIC/NEWMARK/BATHE/GAUSS_IRK4). At documentation time only FDM STATIC completed successfully; other jobs in progress/queued. Rank-one physical solution is not a general test of higher-rank approximation capacity. 
