# Physical science verification: coupled G3D/Gz real shell cases

Reference: Krysko–Zhong corrected CUF TE1–TE4 manuscript, 24 October 2026.

The workflow scientific-real-coupled-physical.yml performs **real** physical
equilibrium or time-integration solves in the coupled reference solver. It
is not a synthetic algebra check, CASE-only preflight, or external benchmark.

On push, an eight-case PILOT spans seven shell theories (KL, FSDT, Reddy,
CUF TE1–TE4), G3D/Gz, flexoelectric ON/OFF, STATIC/DYNAMIC and FDM/MVI.
Each pilot job saves one physically accepted static load step or one
Newmark dynamic step. In GitHub Actions, selecting workflow_dispatch with
scope=full executes 112 variants (7 × 2 × 2 × 2 × 2) at significant cost.
The full sweep does not launch automatically on every commit.

Every folder contains the input CASE, exact Git SHA, input SHA256,
full solver NPZ/JSON result, accepted-state SHA256, and log.
A failed solve or missing accepted step makes the job fail and preserves
the failed journal; no numerical fields are fabricated or substituted.
In MVI the saved X/Y factors must reproduce the accepted physical state.

The MICRO material uses GL geometry, explicit outer reference face
DEAD_REFERENCE pressure (KL on MIDDLE face; FSDT, Reddy, CUF on OUTER face),
SI units, positive gradient length, correct
G3D (18 gradients) or Gz (6) and the requested electric switch.
The pilot uses 4x4 except KL, which uses 5x4: with four axial points,
clamping both ends also fixes their axial slopes and may remove every KL
radial bending degree of freedom. One accepted status at zero displacement
is NOT a valid pressure-response check. A campaign-specific nonzero radial
response gate now guards all pilot cases at the stated E and load.
It is not a general numerical threshold or a substitute for convergence.
One load step or one time step cannot
establish mesh convergence, time accuracy or rank convergence.

**Critical limitation:** Coupled FDM and coupled MVI reference modes
still share variational physical assembly. This campaign validates
individual executable branches and persistence, not mathematical
independence of the two spatial discretizations. External analytic,
WQEM or FE benchmarks, higher-order edges, free-surface traction, and
real 3D field convergence remain publication-release gates.

## Pilot discovery and scientific caveats

The initial all-green eight-job physical campaign (run `42299220539`)
was **not** yet a fully meaningful mechanical smoke result: inspecting
its archived NPZ shows the 4x4 clamped KL case had identically zero
mechanical displacement, electric potential and all recovered stresses
under nonzero pressure. The case was therefore under-resolved in its
admissible radial trial space, although the solver reported an accepted
step. The successor pilot explicitly rejects this degenerate state.

The small-grid GL results may carry the solver's diagnostic
`thickness_locking: YES` for the classical/director theories and TE1.
This is a physical-model caution, not a number to silently correct.
Full higher-order free-face strong traction testing remains incomplete;
a completed small-grid pilot cannot certify higher-order boundary work.
