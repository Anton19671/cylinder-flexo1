# Modal MVI: physical rank and frequency families

The modal algorithm continues to use the fixed-rank Kryshko-Kirichenko
Y/X variational halfsteps. This change does not alter its stiffness,
mass, constraints, eigenvalue selection, initialization or stopping rules.

For an MVI modal CASE, the optional `modal.mvi_spectrum_method` specifies:
- `AUTO` (default): use the unrestricted fixed-rank Y/X route; never
  select Fourier families silently.
- `FOURIER_FAMILIES`: explicitly require that Fourier family scan;
  refuse unsupported boundary, material or discretization settings.
- `UNRESTRICTED`: solve the unseparated circumferential fixed-rank MVI
  stationary mode candidates. This is a mode search, not proof that the
  entire global spectrum has been obtained.

`modal.eigen_tolerance` and `modal.mode_tolerance` are forwarded to
the MVI eigenmode algorithm (rather than only validated).

The output `FINAL_RUN_SUMMARY.json` records the actual selected
spectrum route and QR-estimated effective separation rank of each
displacement component in each computed mode. A selected MVI rank of 8
is *capacity*, not a requirement that all eight terms be nonzero. Pure
Fourier harmonics of an axisymmetric shell can be exactly representable
with MVI-1, so identical frequencies at ranks 1..8 are not by
themselves evidence of a software error.

The workbook contains:
- `MVI_RANK_AUDIT`: selected rank versus effective ranks of computed forms.
- `FOURIER_FAMILIES`: candidate circumferential wave number, axial
  candidate index, frequency and rotation-pair multiplicity.
- `MODE_FAMILY_IDS`: mode index versus its physical harmonic identity.

Rotational sine/cosine partners have the same frequency but are distinct
mass-orthogonal eigenvectors. A Fourier-family candidate list may miss
axial overtones; the summary continues to report
`global_spectrum_completeness_verified=false`. Compare individual
circumferential (n) / axial (m) families with published tables, not
the indices of an arbitrarily truncated doubled spectrum.

No solver computations are run as part of this source-only change.

## Confirmed source-level limitations (2026-10-31)

- Unrestricted MVI uses the dense reference grid with a maximum of 256
  surface nodes, 4096 mechanical DOFs, and 1600 nominal projected directions.
  These limits are checked **before** allocating the prepared problem.
- 21x63 is NOT currently supported by unrestricted modal MVI. The explicit
  Fourier-family route may support that grid but is not an independent
  unrestricted MVI benchmark against full-sector FDM.
- The original `Anton19671/cylinder` internal MVI is a *static equilibrium*
  reduced-direction correction algorithm. Its separate Y/X structure can
  be inspected, but it is not itself a published or coded modal eigenvalue
  algorithm; no complete equivalence to the modal generalized eigenproblem
  is claimed without a corresponding variational derivation.
- Neither this documentation nor source-only regression coverage certifies
  numerical eigenfrequencies. No physical calculations were run here.
