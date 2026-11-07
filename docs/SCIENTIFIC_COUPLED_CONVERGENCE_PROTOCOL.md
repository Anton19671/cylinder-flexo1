# Coupled shell numerical verification: real field refinement campaign

Reference: corrected Krysko–Zhong CUF TE1–TE4 dated 24 October 2026.

This is a genuine solver campaign in `ci/scientific_coupled_convergence.py`,
running on GitHub Actions via `scientific-coupled-convergence.yml`.
It does not reclassify CASE validation as physical accuracy.

Two actual physical cases of each group are started by the bounded push
pilot (8 jobs), while the explicit `workflow_dispatch: scope=full`
contains 13 cases. No local numerical run is required.

| Study | Pilot | Full | Same physical continuum conditions |
|---|---|---|---|
| KL, G3D, flexo ON, static FDM grid | 5x5, 7x7 | 5x5, 7x7, 9x9 | Mid-surface radial dead reference pressure (KL does not support offset face) |
| CUF TE1, Gz, flexo ON, static FDM grid | 4x4, 5x5 | 4x4, 5x5, 6x6 | Outer radial dead-reference pressure |
| CUF TE1, Gz, flexo ON, dynamic MVI-1 time increment | 2, 4 steps | 2, 4, 8 steps | Fixed final physical time 0.02 s |
| CUF TE1, Gz, flexo ON, non-axisymmetric static patch, fixed 5x5 | FDM, MVI-1 | FDM, MVI-1/2/3 | Identical patch geometry; MVI is rank-limited |

The same SBP family `LEGACY_REFERENCE`, continuum pressure, material and
boundary conditions are retained for each comparison. Changing resolution,
time increment or separation rank does not alter the physical passport.
A SHA256 of that passport is persisted and mismatch fails the comparison.
This prevents spurious observed accuracy from pairing different physics.

The numerical CSV reports physical displacement at (L/2, theta=pi),
maximum radial response, outer-face radial response, potential coefficient
and electric induction when available, mid-thickness sigma_xx and
dynamic mid-point velocity when relevant. All results are extracted from
accepted NPZ physical fields (never fabricated, extrapolated or inferred
from the solver status). Each file includes its CASE, code SHA, and
physical state digest; failed attempts retain a separate journal.

**What comparisons mean**: differences between two mesh/time/rank
approximations are *not* certified discretization errors or rates of
convergence. At least three resolutions and an independent reference are
needed for a robust order study. The full 13-case series adds three
levels but still cannot certify all seven theories or all coupled modes.
A numerical agreement between coupled FDM and MVI does not establish
their spatial independence: the reference variants still share the same
variational source and assembly.

**Important validity gate**: CUF TE-N flexoelectric ON must include
N+1 electric coefficient fields. The earlier real pilot incorrectly
carried `potential_order=1` from the TE1 example into TE2/TE4 cases;
corrected CASE generator and tests must pass before the previous TE2/4
pass result can be used as evidence of full CUF electrical order.
