# Independent nonlinear MVI: one floating global electric gauge

2026-11-10. Research-only integration for all seven theories, G3D/Gz,
geometries LINEAR/VK/GL, rank 1..8 and all three own dynamic methods.
No physical cylinder model or time-history calculation was executed.

## Physics and discretization

The electrostatic potential polynomial is
`phi(x,s,z)=sum_m G_m(z)*psi_m(x,s)`, with
`G_m(z)=(2z/h)^m`. For completely insulated electric ends and
inner/outer faces there is exactly ONE arbitrary constant potential,
not one arbitrary constant for each polynomial mode.

The source CUF thickness moments are
`I_m = int_{-h/2}^{h/2} (1+z/R)G_m(z) dz`.
With the MVI-owned SBP/Fourier surface quadrature `W_ij`, the
zero-mean physical gauge is

    sum_m I_m * sum_ij W_ij * psi_m(x_i,s_j) = 0.

The new MVI surface carries ONE scalar offset added only to the
physical psi_0 field: `offset = -sum_m I_m int psi_m/(I_0 int 1)`.
The rank-R separated raw X/Y fields and their local differentiation
do not gain a fictitious electric mode or extra rank. The offset is
computed again for each new X/Y state; physical gradients and the
electromechanical constitutive law remain unchanged. The source
global charge compatibility condition is checked against the total
generalized zeroth-mode free charge:
`sum_ij W_ij * f_0(x_i,s_j)=0`.
Both the charge from loads.npz and curvature-weighted inner/outer
physical electrode charges contribute. A nonzero relative net charge
is a hard CASE preflight error; no multiplier conceals it.

## Numerics

The explicit `boundary.electric="INSULATED_FLOATING"` sets every
electric axial coordinate free at both ends while mechanical supports
remain the declared immobile parallel ends. The old grounded options
remain supported with the unchanged default surface constructor.

Frozen-partner X/Y solves detect a constant mode **only when the
constant-1 vector lies in the span of the frozen complementary
factor**, within the tight declared algebraic tolerance. Only this
true gauge-null direction is projected away from the physical gradient
and removed from the reduced Jacobian by a rank-one positive
regularization. The same correction applies to the own MVI dense LU
and matrix-free GMRES paths. No SVD, pseudoinverse, rank expansion or
MKR spatial operator is used. A physical virtual-work gradient that
changes along the exact gauge direction is rejected as an invariance
violation rather than silently masked.

The same surface/gauge program is reused in native static, Newmark,
Bathe, initial electric closure and Gauss2 stages/endpoint. The
separate rank-exact endpoint fitter still fails closed if the
mechanical endpoint cannot be represented with the selected rank.

## Explicit limitations

This implements a scalar electrostatic datum, not full physical
face/edge boundary certification. Pointwise flux on inner and outer
cylindrical surfaces remains a diagnostic. Natural high-order
mechanical microfree and generalized boundary tractions and
independent loaded physical numerical convergence remain unverified.
The gauge is only enabled by the explicit research opt-in CASE route,
not ordinary GUI/CASE production defaults.

The no-physical-shell contract workflow checks exact volume-mean
elimination, preservation of the unshifted higher electric
coefficients, free electric axial traces, net charge compatibility,
constant-null representability, and synthetic gauge-invariant
nonlinear X Newton with both LU and GMRES.
