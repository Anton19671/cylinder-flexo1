# Nodal MKR: rigorous discrete energy and axial higher-order end action

Date: 2026-11-07. Source: Krysko-Zhong CUF TE1–TE4 corrected
2026-10-24, §§22, 30–31, 39–41.

## Proven mathematical defect in the previous independent MKR

In the standalone **linear** MKR implementation,
`D_x` was a centered difference in the interior with independent
one-sided endpoint rows, and `D_{xx}=D_x D_x`. The former strong
equilibrium applied another `D_{xx}` to the flux. When the axial
endpoint displacement is fixed, `B^T D_{xx} D_{xx} B` is generally
not self-adjoint; its correct discrete-energy counterpart is
`B^T D_{xx}^T W_x D_{xx} B`, where `B` is the MKR-owned
essential-boundary basis and `W_x` uses trapezoid endpoint weights.
A separately performed small-matrix check for 9/11/15 axial nodes
showed the non-self-adjoint effect even on KL slope-constrained
spaces; this is an algebraic fact, **not a shell simulation**.

## Narrow correction made

The existing own MKR derivative matrices and own thickness-integrated
6-by-6 jet stiffness / 3-by-3 inertial tensors are retained.
For each independent mechanical derivative channel `D_i` the nodal
action uses `D_i^T W_x sum_j C_ij D_j q`. This is the exact
discrete first variation of MKR's **own** nodal quadrature energy
and produces the axial natural work on unprescribed higher traces
without inserting guessed extra end rows.

The free interior row is a strong nodal equilibrium per reference
surface dual area: common `dx*ds` cancels from its force, stiffness
and inertia contributions. The exterior displacement DOFs remain
fixed. This does not invoke SBP/Fourier quadrature, the MVI solver,
the common variational reference backend or any third spatial mesh.

Crucially, the mass is STILL assembled directly from nodal
coefficients. The previously rejected experiment that averaged
all kinetic fields to angular cell centers is not resurrected.

## Scope and limits

- Supported: experimental linear MACRO HOOKE, no electric/flexo
  and no gradient-material physics; seven shell kinematics.
- New algebra-only tests check energy-first-variation identity,
  symmetry on the essential subspaces and positive kinetic energy,
  including an alternating angular field.
- Still OPEN: a certified compatible angular *face* gradient and
  flux divergence, the corresponding linear/nonlinear material
  operator including all 18 G3D components and electric coupling,
  exact gradient natural boundary terms, as well as grid/time/rank
  convergence. The code still restricts the independent angular
  grid to odd size; that restriction masks but does not solve the
  high-frequency mode problem of centered derivatives.
- No physical transient, static solve, or benchmark was launched.
  Actual pytest execution has not been claimed for this patch.

This is a **limited proven algebraic correction**, not certification
of the complete scientific algorithm.
