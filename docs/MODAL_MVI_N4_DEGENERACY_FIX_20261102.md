# Modal MVI: correction of the missing n=4 doublet

Date: 2026-11-02

## Control case

KL, linear modal problem; C-S axial supports; 9x9 SBP42_FOURIER_ODD grid.

Full FDM reference:

| Modes | Frequency, Hz | Circumferential number |
|---|---:|---:|
| 1-2 | 275.266096 | 3 |
| 3-4 | 293.138249 | 4 |
| 5-6 | 347.495011 | 2 |
| 7-8 | 398.603192 | 4 |

## Confirmed cause

The rank-one KL seed plan supplied the same circumferential Fourier phase to
all displacement components.  The physical n=4 pair needs an independent
quarter-period phase for the circumferential displacement component.  With
only the common-phase starts, the rank-one separated space omitted that
direction; the unrestricted alternating search could then settle on a higher
n=2/n=3 branch.

This was an initialization-space defect.  Stiffness K, mass M, SBP42
operators, essential traces, mass deflation and the classical Y-to-X
Krysko-Kirichenko sweeps were not changed.

## Implemented correction

In `numerics/cylinder_modal_mvi.py`:

- `_modal_seed_plan` adds cosine and sine harmonic seeds with an independent
  phase for the KL rank-one circumferential displacement component.
- `admissible_start` preserves that componentwise phase while constructing
  a BC-admissible start.
- A single X-frozen variational conditioning halfstep is used only for a
  pure rank-one harmonic seed before the normal Y-to-X sweeps.  It uses the
  existing constrained modal halfstep, K/M, traces and completed-mode
  mass orthogonalization.
- Diagnostic events record frequency, dominant circumferential harmonic,
  its energy fraction and overlaps with prior modes after each halfstep.

## Evidence before full all-ranks audit

The component-phase probe gave 398.603192 Hz for both n=4 phase starts after
the X halfstep and retained 398.603192 Hz after the following Y halfstep.

The seed-sensitivity forensic run on the corrected implementation completed
the eight-mode spectrum for:

| Method / start | Modes 7-8, Hz |
|---|---:|
| MVI-1, seed offset 31 | 398.603192, 398.603192 |
| MVI-1, seed offset 101 | 398.603192, 398.603192 |
| MVI-2 | 398.603192, 398.603192 |
| MVI-4, offsets 0, 31, 101 | 398.603192, 398.603192 |

## Final KL acceptance audit

The full FDM versus MVI-1..8 audit completed successfully
(run 41423315137, 2026-11-02).  All nine solves (FDM plus MVI ranks 1..8)
returned the requested eight modes.

| Modes | FDM, Hz | MVI-1 before | MVI-2 before | MVI-1 after | MVI-2 after | n after |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 275.266096 | 275.266096 | 275.266096 | 275.266096 | 275.266096 | 3 |
| 3-4 | 293.138249 | 293.138249 | 293.138249 | 293.138249 | 293.138249 | 4 |
| 5-6 | 347.495011 | 347.495011 | 347.495011 | 347.495011 | 347.495011 | 2 |
| 7-8 | 398.603192 | 521.701083 | 406.912967 | 398.603192 | 398.603192 | 4 |

MVI-3..8 retained the same eight frequencies as FDM to the displayed
precision.  The maximum frequency difference in the final run was below
5.3e-8 percent.

For the degenerate 7-8 pair, individual-vector MAC is phase-basis dependent,
so the audit compares two-dimensional mass-metric subspaces.  The squared
principal mass overlaps were [1.0, 0.9999999999999997] for both MVI-1 and
MVI-2; their two forms are mutually mass-orthogonal.  Thus the accepted
object is the same n=4 two-dimensional eigenspace, not merely two equal
frequencies.

## Rawat CUF TE1 C-C 13x13 follow-up

The standalone Rawat audit completed successfully (run 41423596503).  It is
kept as a separate observation, because the KL-only correction deliberately
does not change CUF operators or rank-one CUF starts.

| Branch observation | FDM, Hz | MVI-1, Hz |
|---|---:|---:|
| Near published 342 Hz branch | 340.806 (n=4, doublet) | 346.059 (n=2, doublet) |
| Near published 352 Hz branch | 353.577 (n=3, doublet) | 346.059 (same nearest MVI-1 branch) |

This exposes an existing CUF-TE1 rank-one approximation discrepancy; it is
not hidden, and it is outside the scope of the KL component-phase correction.

## Rollback points

- `baf6f101`: initial harmonic-branch instrumentation and conditioning.
- `f36598cf`: spectrum audit captures branch diagnostics.
- `9f67ff78`: independent KL rank-one component-phase seeds.
- `b69a8a7e`: limits conditioning to the affected rank-one branch.
