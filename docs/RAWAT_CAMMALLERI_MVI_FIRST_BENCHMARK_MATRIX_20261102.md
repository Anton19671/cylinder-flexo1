# Rawat 2016 + Cammalleri 2023: MVI-first modal benchmark

## Rule fixed by project owner (2026-11-02)

**The main numerical method is the classical Kryško–Kirichenko variational iteration method (MVI-1 through MVI-8).**
The full variational FDM grid solver (called MKR in the GUI) is a **control**, not the main method. Do not publish MKR-only comparisons as an MVI benchmark.

Each physical case requires nine independently computed results:
**MVI-1, MVI-2, MVI-3, MVI-4, MVI-5, MVI-6, MVI-7, MVI-8, MKR/FDM**.
Do not replace a failed MVI rank with FDM, Ritz, a harmonic-restricted surrogate, another rank or an interpolated/extrapolated number. Do not fabricate missing frequencies.

## Planned primary comparison matrix

| ID | Publication / geometry | Shell theory | Mesh | MVI ranks | MKR/FDM |
|---|---|---|---|---|---|
| R1 | Rawat 2016 | CUF TE1 | 13x13 | 1-8 | 1 |
| R2 | Rawat 2016 | CUF TE1 | 17x17 | 1-8 | 1 |
| R3 | Rawat 2016 | CUF TE4 | 13x13 | 1-8 | 1 |
| R4 | Rawat 2016 | CUF TE4 | 17x17 | 1-8 | 1 |
| R5 | Rawat 2016 | CUF TE4 | 21x21 | 1-8 | 1 |
| C1 | Cammalleri 2023 | KL | 13x21 | 1-8 | 1 |
| C2 | Cammalleri 2023 | KL | 17x31 | 1-8 | 1 |
| C3 | Cammalleri 2023 | KL | 21x41 | 1-8 | 1 |
| R6 | Rawat 2016 supplementary | KL | 13x13 | 1-8 | 1 |
| C4 | Cammalleri 2023 supplementary | CUF TE1 | 13x21 | 1-8 | 1 |

**TOTAL: 10 case definitions x (8 MVI + 1 MKR) = 90 requested individual solves**:
80 MVI, 10 MKR. This is the *planned* matrix, **not a claim that 90 solves completed**.

### Known implementation limits (not successful runs)

Current unrestricted-MVI modal preflight in `numerics/cylinder_modal_reference.py`
limits projected variation directions to `MAX_PROJECTED_DIRECTIONS=1600`
(`numerics/cylinder_modal_mvi.py`). The number of directions is
`n_components * rank * max(nx, ns)`.

Thus **five** planned entries require a memory-bounded genuine MVI implementation:
R4 MVI-7, R4 MVI-8, R5 MVI-6, R5 MVI-7, R5 MVI-8.
At present **85 / 90** are within this particular dimension limit;
this is **not** a guarantee of completion or convergence. Do not bypass
the hard cap by silently changing discretization, reducing rank or
substituting FDM. A legitimate scalable projected Y/X solve, constraints
and K/M eigenspace comparisons must be developed and tested first.

### Runtime jobs and artifacts

- [MVI-first additional grid matrix](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41478725265) from
  `ci/modal_primary_mvi_benchmarks.py` and
  `.github/workflows/modal-primary-mvi-benchmarks.yml`.
  It covers R2, R4, R5, C1, C2, C3 (six cases), and outputs one independent
  result JSON and modal NPZ per method, including failed and unsupported status.
  Consolidated CSV contains MVI-1..8 *before* MKR/FDM; missing entries remain blank.
- [All-rank 13x13 modal audit](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41476553532):
  covers R1, R3, R6 (and other seven-theory combinations) against FDM,
  including the physical-mass modal subspace comparison.
- [User's Cammalleri CUF TE1 13x21 matrix](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41476553973):
  covers C4, MVI-1..8 and FDM, using the actual GUI-child solver.
- The statuses shown above must be inspected at read time; an Actions run
  started or queued is not a successful physical calculation.
- Reproduce the same 12 positive finite eigenfrequencies, record 3D modal
  displacements, wavenumbers, job status and compute times. Only acceptance
  checks supported by the actual results may be marked PASS.

## Literature comparison coordinates

Rawat, Matsagar & Nagpal (2016):
`doi:10.16943/ptinsa/2016/48426`, Table 2, page 354, C-C,
`L=2 m, R=1 m, h=0.05 m, E=210 GPa, nu=0.3, rho=7850 kg/m3`,
axial mode `m=1`, circumferential `n=2,3,4,5`.

Cammalleri & Castellano (2023):
`doi:10.3390/designs7060122`, Table 6, PDF page 11, C-C,
`L=0.305 m, R=0.076 m, h=0.000254 m, E=207 GPa, nu=0.3,
rho=7833 kg/m3`, axial `m=1`, `n=5,6,7`.
Published `Present` = authors' Donnell–Mushtari dynamic matrix;
published `FEM` = their ANSYS SHELL181.
KL production uses 3D Lamé under zero normal strain.
The separate plane-stress Schur diagnostic is NOT the same production
material law and should not be described as a normal CASE.

For both studies: first identify the physical circumferential `n` and
axial `m` from the computed mode fields. Never choose the closest
frequency without that check. At repeated eigenvalues validate the
mass-MAC modal **subspace**, not arbitrary individual sin/cos eigenvectors.

## Reporting contract

For **every** theory / mesh row, the report needs the source value and
nine method columns, in this order:

`n | paper_Hz | MVI1_Hz | MVI2_Hz | MVI3_Hz | MVI4_Hz | MVI5_Hz |
MVI6_Hz | MVI7_Hz | MVI8_Hz | MKR_FDM_Hz`.

No numeric MVI value may be copied from the MKR column just because a
homogeneous cylinder is separable; independently compute the rank.
If any case remains queued, failed or excluded by an actual limit, print
that state explicitly and leave its numerical entry **empty**.

Final Word manuscript sent to colleagues must omit GitHub and CI
implementation details. It must state the two problem definitions and cite
the article page/table locations, compare physical modes, frequencies,
rank and grid convergence, and show MVI (not MKR) as the principal method.
