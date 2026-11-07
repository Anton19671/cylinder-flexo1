# CUF TE1 non-axisymmetric Gz flexoelectricity: MVI 1-8 vs FDM

This is the next required physical rank study after the 5x5 rank-one
experiment (initial quantitative observations in
`docs/SCIENTIFIC_COUPLED_INITIAL_NUMERICAL_EVIDENCE.md`).

## Fixed physical passport

- Theory CUF TE1, MICRO Green–Lagrange, Gz gradient elasticity, flexo ON
- Electric potential has order 1 = **two independent thickness fields**
- Same physical cylinder `L=1.2 m, R=2.0 m, h=0.2 m`
- Same nonlinear 3D Hooke and electroelastic material, electrode and clamped ends
- Same non-axisymmetric physical patch on the outer reference face
- Static, one load factor 1, same pressure amplitude and face-area convention
- Same `LEGACY_REFERENCE` surface family and **8 × 8** grid for every run
- Full FDM reference and fixed-rank MVI-1, 2, 3, 4, 5, 6, 7, 8

**Why 8×8 rather than 5×5?** Rank 8 is impossible on 5×5, and the
experimental dense halfstep safety bound is 512 unknowns. With six
mechanical and two electric TE1 fields, 8×8 preserves this bound even
for the largest rank. The 8×8 grid is not a claim of spatial
convergence: because the periodic angular grid is even, its Nyquist
aliasing behavior requires separate tests. The highest rank is also
the algebraic full-rank limit on this grid, not evidence of rank-8
accuracy on a large physical mesh.

The GitHub workflow
`.github/workflows/scientific-coupled-rank-1-8.yml`
runs CASE preflight first, then nine **real** physical jobs with
parallelism capped at two. Each saves exact CASE, code SHA, physical
NPZ, a scientific evidence certificate and failed-iterate journals.
The compare job checks all source hashes, identical physics, source
revision and spatial coordinates. It compares entire arrays of
mechanical coefficients, electric potential coefficients,
outer-face displacement, normal electric induction, stresses,
hyperstresses and volume induction. Each reported relative L2
difference is relative to the **same-grid FDM solution**, not
an independent analytic/WQEM/FEM reference.

`CONVERGED_PROJECTED` indicates only fixed-rank accepted
stationarity and is never labelled full-space convergence. An
uncompleted rank, shape mismatch, missing physical tensor or corrupt
file makes the comparison non-certified. Rank-dependent differences
need not decrease monotonically. Higher-order spatial and independent
physical benchmarks are still required before publication.

## Verified first actual physical comparison — MVI-1 versus FDM, 8×8

Evidence source: [real rank-run 42360094375](https://github.com/Anton19671/cylinder-flexo1/actions/runs/42360094375).
The confirmed completed result archives are
`SCI_RANK8_TE1_NONAX_8X8_FDM` and
`SCI_RANK8_TE1_NONAX_8X8_MVI_R1`.
Both use exact Git commit
`923172e35649558ae8615159ad573816c6edfc00`
and continuum-physics passport SHA256
`e9ed508a761b06ad9b0319437ed57aec90e19cc2433a759550e14028c18a856f`.

| Entire stored physical array | Relative L2 difference, 100×norm(MVI-1 − FDM)/norm(FDM) |
|---|---:|
| Mechanical coefficient fields | 0.13987306 % |
| Outer-face radial displacement | 0.13731956 % |
| Through-thickness stress tensor (6 components) | 0.84006619 % |
| Through-thickness hyperstress tensor | 0.44584773 % |
| Electric potential coefficient fields | 3.71260192 % |
| Through-thickness electric induction (3 components) | 7.20114967 % |
| Outer-face normal electric induction | 13.41158673 % |

FDM's central radial displacement is `0.00042674609500213147 m`
and the largest radial displacement is `0.0004283413879903528 m`.
MVI-1's central radial displacement is
`0.00042657559391599874 m` and its largest radial displacement is
`0.0004286183294354862 m`.

**Actual physical state integrity**:
- FDM NPZ SHA256:
  `ce3045b4c6ab2b9c268e1ab6733f7885829ba40ddc5d261dfa1fc1b5f74391ab`
- MVI-1 NPZ SHA256:
  `1dd030ac09107222c95fda19be2e9bd75ea25e1b746a622dd66859bde0c19f7e`

The MVI-1 record carries the solver's
`RANK_CONSTRAINED_ONLY` qualification:
`full_space_stationarity_confirmed=false`. Its difference from FDM,
particularly for the outer-face electric induction, is **not** an
externally established model error. The 8×8 grid is not spatially
converged, and surface flux/second-gradient natural boundary
conditions are not publication-certified merely because this small
reference solve succeeded.

**Status of this report:** partial physical comparison for rank 1,
not proof for ranks 2–8. Do not quote numerical values for a rank until
its own accepted NPZ has been independently inspected.

## Verified MVI-2 and MVI-3 on the same 8×8 real physical case

Source archives: `SCI_RANK8_TE1_NONAX_8X8_MVI_R2` and
`SCI_RANK8_TE1_NONAX_8X8_MVI_R3` in
[GitHub Actions 42360094375](https://github.com/Anton19671/cylinder-flexo1/actions/runs/42360094375).
They share exactly the FDM physical signature
`e9ed508a761b06ad9b0319437ed57aec90e19cc2433a759550e14028c18a856f`
and revision `923172e35649558ae8615159ad573816c6edfc00`.
Both final NPZ SHA256 values match the stored scientific evidence.
Both results are **CONVERGED_PROJECTED**, not fully stationary.

All columns below are **relative L2 field differences vs the same
FDM NPZ, percent**; none is an external exact-solution error.

| Physical array | MVI-1 | MVI-2 | MVI-3 |
|---|---:|---:|---:|
| Mechanical thickness coefficients | 0.1398730558 | 0.01466070782 | 1.038736811e-7 |
| Outer radial displacement | 0.1373195584 | 0.008035276977 | 1.062119682e-7 |
| Thickness stress (six components) | 0.8400661871 | 0.1077488355 | 1.494626916e-7 |
| Thickness hyperstress | 0.4458477324 | 0.0361898534 | 1.45493112e-7 |
| Electric potential coefficients | 3.712601918 | 3.70886584 | 0.07050071832 |
| Volume electric induction (three components) | 7.201149671 | 7.077566731 | 0.1915453721 |
| Outer normal electric induction | 13.41158673 | 13.34791122 | 0.2539926578 |

The mechanical fields at rank 3 agree exceptionally closely with FDM,
but the electric induction still differs by 0.254% on the outer face.
At rank 2 mechanical fields improve much faster than electric
induction. The latter cannot be certified by monitoring displacement
alone. These are finite-rank, identical-weak-kernel and identical-grid
**comparisons**, not independent discretization or continuum accuracy
validation. Rank 4–8 numerical findings must be appended only after
actual GitHub artifacts are available and inspected.

## Verified MVI-4: increasing rank is not monotonic

The real MVI-4 archive from run
[42360094375](https://github.com/Anton19671/cylinder-flexo1/actions/runs/42360094375)
matches the original FDM state by physics SHA256, Git SHA and saved NPZ
hash. MVI-4 reports `CONVERGED_PROJECTED` and
`RANK_CONSTRAINED_ONLY`, not full-space certification.

| Complete physical field | MVI-3 vs FDM (%) | MVI-4 vs FDM (%) |
|---|---:|---:|
| Mechanical coefficients | 1.038736811e-7 | 3.759195996e-7 |
| Outer radial displacement | 1.062119682e-7 | 3.094113350e-7 |
| Stress tensor | 1.494626916e-7 | 1.254757885e-6 |
| Hyperstress tensor | 1.454931120e-7 | 7.296485422e-7 |
| Electric potential coefficients | 0.07050071832 | 0.1443538263 |
| Volume electric induction | 0.1915453721 | 0.4436617305 |
| Outer-face electric induction | 0.2539926578 | 0.5200632908 |

These are relative L2 differences to **the same 8x8 FDM computed
field**, not reference accuracy. Rank-4 electrical differences are
larger than rank-3 ones despite successful rank-constrained stationarity.
Do not impose monotonicity or extrapolate convergence from only ranks
1–4. Real MVI-5..8 results are separate and must be inspected before
making any claim about them.
