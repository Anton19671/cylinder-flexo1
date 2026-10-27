# Compatible modal electric face flux for G3D/Gz (experimental)

The new `numerics/cylinder_electric_modal_faces.py` is a **separate electric
subsystem**, not yet a separate *fully coupled* mechanical/electric strong FDM.
It does not replace the established variational `CylinderFunctional`, MVI,
Newmark, or any of their boundary and rank choices.

## Physics and exact cylindrical thickness factor

On the reference circular cylindrical midsurface `(x,s=R*theta)`, write

    phi(x,s,z) = sum_m psi_m(x,s) * G_m(z), G_m=(2z/h)^m, 0 <= m <= 4.
    H(z) = 1 + z/R, E = (-phi_x, -phi_s/H, -phi_z).
    P_flexo = mu1*t + mu2*b, D = kappa*E + P_flexo.

Here `t`, `b` are taken from the **active** G3D (18 gradients) or Gz (six
through-thickness gradients) constitutive law; no replacement material or
kinematic assumption is introduced. The electric part of enthalpy is

    H_electric = -kappa*(E dot E)/2 - P_flexo dot E.

Each potential mode has face fluxes and a local z-action:

    F^x_m = int(-h/2,h/2) H(z) G_m(z) D_x dz   (at axial x-faces)
    F^s_m = int(-h/2,h/2) G_m(z) D_s dz        (at periodic s-faces)
    B_m   = int(-h/2,h/2) H(z) G'_m(z) D_z dz (at surface nodes)

The electric coefficient's nodal action in coulombs is

    r_m = Gx.T Wx_face F^x_m + Gs.T Ws_face F^s_m + Wx Ws B_m.

Both face gradients and their transposed actions use exactly the **same**
physical finite differences. There are no duplicated periodic nodes. The
even-grid alternating circumferential mode is therefore detected, while the
TE0 constant-voltage gauge remains a physical null direction. Higher
thickness electric modes are *not* gauges; their z-gradients are included.

A shell-to-face helper evaluates the **same** Green-Lagrange/CUF/Director/KL
material state at Gauss thickness stations and exact inner/outer faces. It
uses covariant G3D/Gz flexopolarization `P_flexo`. When interpolating nodal
polarization to circumferential face positions, the `(s,z)` vector is
rotated into the face frame before averaging. This preserves spatially
constant Cartesian vector polarization on the curved shell.

For each face, the physical outward normal induction is

    D_out_inner = -D_z(z=-h/2), D_out_outer = +D_z(z=+h/2).

At insulated surfaces it is a physical natural flux; at a prescribed
potential it represents an electrode charge reaction and must not be
independently prescribed as a vanishing normal flux.

## Scope and verification limits

`modal_electric_face_weak` accepts optional frozen constitutive
`FacePolarizationSamples` sampled at compatible spatial locations. Its
return value is the **exact discrete derivative with respect to electric
potential coefficients** of the electric enthalpy of this *face scheme*.
It **does not** include a derivative of P_flexo with respect to mechanical
unknowns. Therefore directly attaching only this electric equation to the
existing mechanics would destroy a consistent two-way coupled Jacobian.
A future independent strong solver must use a common face enthalpy for
both mechanical and electrical fields, impose its physical boundary fluxes
and establish independent mesh/thickness and manufactured-solution checks.

The current isolated unit tests cover TE0--TE4, energy differentiation,
polarized x/s/z fluxes, constant-voltage gauge, exact cylindrical TE0
reference, face induction, even/odd periodic grids and vector-aware
collocation. Integration tests have also been prepared for all seven shell
theories in both G3D and Gz; they are not proof of production accuracy.

Status: `EXPERIMENTAL_ELECTRIC_MODAL_FACE_SUBPROBLEM`.
