# G3D/Flexoelectric axial natural boundary terms: mathematics and status

This document derives the *continuum-form* axial natural boundary
quantities directly from the SAME thickness-integrated variational
coefficients used by the seven-model full G3D/flexoelectric assembler.

Let q_a be each generalized mechanical component (KL:3, FSDT/Reddy:5,
CUF TE1-4:6/9/12/15), and write
`A[a,b]=partial L / partial(q_{,x^a s^b})`, where L is the
thickness-integrated electric enthalpy per reference midsurface area.
The machinery uses only listed independent jets, so mixed q_xs,
q_xxs and q_xss are not double-counted.

For a periodic circumferential coordinate s, integration by parts
on the open axial interval yields the end virtual work

`[int_{0}^{2piR}( B0*delta q + B1*delta(q_x)
                   + B2*delta(q_xx) + P_x*delta(phi)) ds]_(left)^(right)`.

The exact analytic continuum expressions are:

- `B0 = A10 - dx A20 - ds A11 + dxx A30 + dxds A21 + dss A12`
- `B1 = A20 - dx A30 - ds A21`
- `B2 = A30`
- `P_x = partial L / partial(phi_x)`

Only derivatives present in each model contribute. The left outward
normal is n_x=-1 and the right outward normal is n_x=+1. Thus the
physical outward-normal-conjugate end actions are:

- `traction = n_x B0`
- `normal_double_traction = B1`
- `normal_triple_traction = n_x B2`
- `electric_charge_flux = n_x P_x`

These equal the generalized first, second and third end-action
quantities that may be paired with the explicitly named outward-normal
prescribed tractions in `cylinder_variational_boundary.py`. Their
formulas include classical, gradient, dielectric and flexoelectric
effects by construction; no second constitutive approximation is made.

### Status of natural/microfree conditions

`higher=NATURAL` leaves the higher-order traces free in the discrete
weak variational solve. For a legitimate variational discretization,
this is the weak meaning of unprescribed natural boundary work. But
the current axial and circumferential finite-difference matrices are
NOT a certified SBP pair with the chosen trapezoidal quadrature.
Consequently, sampled continuum fluxes above are **diagnostics**, not
exact boundary reaction operators for this grid. Do not impose them
as extra strong conditions and do not claim that small sampled flux
is equivalent to small fully admissible discrete residual.

The mathematics of dynamic natural boundaries can additionally include
velocity-dependent Kelvin-Voigt stress and inertia from
gradient-dependent director kinematics. The present module reports
*conservative enthalpy* terms only. Those dynamic contributions must
be assembled explicitly before calling its output a **total dynamic
boundary force**.

Implemented:
- `mechanics/cylinder_g3d_natural_edges.py`: analytic continuum-form
  end-flux extraction from independent surface weak jets
- `CylinderFunctional.natural_end_fluxes(...)`: same seven-model G3D
  enthalpy coefficients (electric ON/OFF, CUF TE1–TE4)
- `tests/test_g3d_natural_edges.py`: deferred diagnostics tests

Outstanding before scientifically verified natural BC:
- independently derive and assemble dynamic/dissipative normal fluxes,
  especially Reddy and finite-rotation KL;
- implement compatible SBP quadrature/derivatives or a conforming weak
  discretization with exact discrete Green identities;
- reproduce manufactured boundary problems and mesh-converged
  electrical fluxes;
- compare MVI ranks 1–8 and FDM with FULL admissible residuals.
