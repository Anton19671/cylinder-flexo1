# Experimental covariant generalized face traction reconstruction

This file documents a newly implemented *recovery* operator for the
mechanical face terms required by the Krysko-Zhong algorithm, section 47.

Constitutive arrays at reference-grid thickness points:
- F: deformation gradient in rotating orthonormal (x,s,z) frame;
- S: work-conjugate second Piola stress;
- M_ijk: hyperstress dH/d(E_ij|k) from the same G3D or Gz enthalpy.

First integrate M by parts in the 3D covariant derivative:
  S_effective = S - Div M,  P_effective = F S_effective.
With n = +/- e_z, Q_j = (F M_z)_j,
  t = n_z (P_effective)_z - Div_face(n_z Q_t),
  g = Q_z,
  edge_left = -n_z Q_x,  edge_right = +n_z Q_x.

In the rotating orthonormal reference basis, at r=R+z:
  Div M = partial_x M_x + partial_s M_s/H + partial_z M_z
       + [Omega M_s + M_s Omega.T + M_z]/r
where H=r/R and Omega rotates the s/z basis with
Omega_(s,z)=+1, Omega_(z,s)=-1.

For the cylindrical z=constant face,
  Div_face(B_x,B_s) = partial_x B_x + partial_s B_s/H + Omega B_s/r.
The edge line forces use co-normal +/- e_x. Periodic s creates no
physical circumferential edge.

The implementation differentiates recovered thickness values with
second-order one-sided differences at the faces; surface derivatives use
the existing Dx/Ds (dense or CSR) with exactly the same (nx,ns) axis order.
Consequently, this is an APPROXIMATE recovered strong-form diagnostic:
its independent continuum derivation, integration against independent
face variations, and full mesh/thickness convergence have NOT been
certified. The solver must not use it as an automatic PASS/FAIL gate or
apply it as an extra nodal condition to an already variationally reduced
CUF shell; finite-order CUF enforces projected weak face equilibrium.

This output is added to every MVI/FDM state NPZ, for G3D and Gz, static
and Newmark, and the absolute-max / physical area-weighted RMS values
are recorded in face_conditions. It does not change rank 1..8, X/Y
half-steps, external loads or Newton time integration.

Five local manufactured component tests passed (not a full solver test).
