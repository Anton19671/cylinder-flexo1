# Quantitative first evidence: coupled cylindrical shell physical refinements

Reference run: [GitHub Actions 42355615451](https://github.com/Anton19671/cylinder-flexo1/actions/runs/42355615451)
at code revision `d0db6f02ffbbf084b84707a51943f8d966559f4e`.
Source records: downloaded `SCI_CONVERGENCE_*` GitHub artifact archives,
`CONVERGENCE_EVIDENCE.json` files. All quantities below are measured
from accepted physical NPZ states, **not literature reference values**.

## Spatial refinement: CUF TE1, Gz, flexoelectric ON, STATIC FDM

Both cases have the same continuum-physics SHA256
`31bfedf6702e10904358a85ecf115db36c6084f835cd0bbef8e7437d14d6c94f`,
the same `LEGACY_REFERENCE` spatial operator family and load, and use
the same point `(x=L/2, theta=pi)`.

| Grid | Radial displacement at fixed point (m) | Outer radial displacement (m) | Potential coefficient 0 (V) | Mid-thickness sigma_xx (Pa) |
|---|---:|---:|---:|---:|
| 4 x 4 | 0.00015580325744415325 | 0.00015753364195446892 | 2.2292300504873936e-9 | 0.003502397915726707 |
| 5 x 5 | 0.0005969328883371084 | 0.0005918064499295353 | 1.0460835707760134e-7 | 0.007554202993151904 |

**Interpretation:** the fixed-point radial response changes markedly.
There is no spatial-convergence certificate on these two grids. The
larger-grid value must not be presented as a converged or exact
reference; no Richardson order is justified from two points.

## Temporal refinement: CUF TE1, Gz, flexoelectric ON, DYNAMIC MVI-1

Both cases use physical horizon 0.02 s and the same continuum-physics
SHA256 `dcd5efc34faae1c54e5a6f9379d6cd2a4237cff9b6a2957406579141a82bf903`.

| Steps | Delta t (s) | Radial displacement at final time (m) | Outer radial displacement (m) | Radial velocity at final time (m/s) |
|---|---:|---:|---:|---:|
| 2 | 0.010 | 3.614215358664661e-10 | 1.4823565221681538e-9 | 4.817396395159185e-8 |
| 4 | 0.005 | 3.314281588288154e-10 | 1.3598261538998156e-9 | 4.8185429467413306e-8 |

**Interpretation:** numerical sensitivity is observable, and a third
refinement level (eight steps) is needed for an order-of-accuracy
assessment. The small absolute displacement should not be confused
with verified temporal accuracy.

## Independent FDM vs rank-constrained MVI benchmark

The non-axisymmetric static TE1 rank-one case in the same run uses
continuum-physics SHA256
`e9ed508a761b06ad9b0319437ed57aec90e19cc2433a759550e14028c18a856f`.
The FDM solution was accepted. The MVI-1 solver accepted the
**rank-constrained** field with `CONVERGED_PROJECTED`, and explicitly
did **not** certify `CONVERGED_FULL`. This is a capacity limitation
under the nonseparable patch; the numerical evidence must remain
visible rather than be passed off as full equilibrium.

These findings apply solely to the stated micro-GL, G3D/Gz and
electrical choices, geometry, boundary conditions and grids; they
are not universal error figures for the seven shell theories.

## Remaining scientific release gates

- At least three comparable grids, time increments and MVI ranks,
  with independent benchmark verification.
- Full coupled FDM/MVI spatial independence; current full reference
  solvers share physical weak assembly.
- Fully certified free-face higher-order tractions, electrical boundary
  fields, thickness stresses, dimensional calibration, and external
  WQEM/FEM/analytic comparisons.
