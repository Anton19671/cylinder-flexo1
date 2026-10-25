# Dynamic natural end actions from one seven-model weak functional

Development stage only. **Do not enforce these sampled fluxes as strong
boundary conditions on the present non-SBP reference grid.**

For each mechanical surface jet q_(x^a s^b), define the local full
dynamic virtual-work coefficient

  A_ab^eff = A_ab^elastic/G3D/flexo
             + A_ab^KV(q,qdot)
             + A_ab^mass(q,qdot,qddot)
             + alpha_M A_ab^mass(q,0,qdot).

The Kelvin-Voigt constitutive term uses Green-Lagrange strain RATE
with eta_s [seconds] and six engineering strain components. It does not
create electrical inertia or electric viscosity.

The director inertial contributions are:
- CUF TE1..4: only q-jet mass coefficient; no inertial normal derivative
  boundary traction in this reference basis.
- FSDT/Reddy: q, q_x, q_s inertial coefficients from the SAME
  thickness integration as the existing mass residual.
- KL: q, q_x, q_s coefficients from the exact finite-rotation
  geometric normal including D2n(q)[qdot,qdot]. Its mass-proportional
  damping contribution uses the *linear mass metric* M(q)qdot without
  the quadratic gyroscopic term.

Axial natural fluxes of A^eff are given by the standard model-independent
independent-jet Green identity (no doubling of mixed jets):

  B0 = A10 - d_x A20 - d_s A11
       + d_xx A30 + d_xs A21 + d_ss A12
  B1 = A20 - d_x A30 - d_s A21
  B2 = A30.

Outward-normal physical end actions:
  traction=n_x B0, normal_double=B1, normal_triple=n_x B2,
  electric_flux=n_x (d L / d phi_x), n_x=-1/+1.

The electric contribution remains from quasi-static enthalpy and is
unchanged by mechanical viscosity or inertia. The local constitutive
coefficients are assembled from the SAME model-specific weak form
as the full dynamical nodal residual (except that the KL mass coefficient
implementation has been factored out for shared use).

Code connections:
- `mechanics/cylinder_kl_exact_inertia.py` exports
  `kl_inertial_jet_coefficients`, reused in `kl_inertial_weak_action`.
- `CylinderFunctional.natural_end_fluxes(q, phi, velocity=v,
  acceleration=a, density_kg_m3=rho, eta_s=eta,
  mass_damping_rate_per_s=alpha)` combines conservative, KV and
  inertial coefficients. Called without dynamic args, retains
  conservative-only behavior.

**Verification and enforcement are still outstanding.** The existing
FD Dx/Ds and trapezoidal quadrature lack an independently verified SBP
Green identity. These sampled continuum-form fluxes are not the same
as the exact discrete boundary reactions and must not be used as a
strong boundary-closure switch or a convergence criterion. Coupled
MVI/FDM numerical tests, discrete power checks and end reactions
are needed after full assembly.
