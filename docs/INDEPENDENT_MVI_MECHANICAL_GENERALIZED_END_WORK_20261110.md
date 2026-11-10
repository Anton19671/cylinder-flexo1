# Native coupled MVI generalized mechanical end work — 2026-11-10

The independent seven-theory MVI research CASE now accepts an OPTIONAL
`mechanical_end_work` top-level object with exact keys `left` and
`right`. Each end may be null, empty, or specify generalized mechanical
tractions with the following independent names:

- `traction`: conjugate to a prescribed mechanical displacement
  `q`. Nonzero values are rejected for the current stationary
  fixed-displacement ends; this is a support reaction, not an
  admissible load in the current CASE subset.
- `double_traction`: conjugate to axial-coordinate `d_x q`.
- `normal_double_traction`: conjugate to outward `d_n q=n*d_x q`,
  with `n=-1` at the left and `n=+1` at the right.
- `triple_traction`: conjugate to `d_xx q`.
- `normal_triple_traction`: conjugate to `d_nn q=d_xx q`.

The coordinate/normal forms for a given derivative order cannot
coexist on the same end. A field is a finite scalar (broadcast to all
generalized mechanical coefficients and circumferential nodes) or a
finite `(n_mechanical,ns)` array. Explicit zero or omitted keys mean
no applied generalized traction. KL/FSDT/Reddy support second-normal
traces in this discrete jet model; CUF TE1–4 have only second-order
surface jets and reject third-order generalized end traces.

The external potential added to the own MVI surface energy is

`-sum_ends int_end (t1*d_x(q) + t2*d_xx(q)) ds`

with the explicitly converted outward-normal signs. The exact own
SBP derivative transpose supplies the nodal gradient, and each
X/Y halfstep receives the contraction of this same nodal gradient
with the frozen complementary factor. The time integrators reuse the
same material/kinematic spatial work without any MKR/FDM fallback,
SVD, or rank adjustment.

The discrete Green audit separates physical volumetric weak jets
from the already-applied external end work. This prevents applying
user-prescribed generalized end tractions twice. The dynamic audit
retains the same separation for conservative, inertial and
Kelvin–Voigt physical contributions.

The optional CASE key is validated before any output folder is
created. No physical cylinder simulation was run. A dedicated
synthetic algebra CI set checks seven-model end orientation, exact
X/Y directional energy derivatives, CUF third-order rejection,
ambiguous traction rejection and Green material/end separation.

**Important:** an omitted higher traction is an unprescribed natural
trace in this Ritz/weak discretization. This is NOT a proof that the
reduced model contains every physical gradient-microstress face/edge
condition from the source continuum theory. Complete physical
microfree boundary certification, high-order strong MKR conditions,
and numerical convergence/benchmark validation remain separate tasks.
