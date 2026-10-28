# Exact discrete SBP Green decomposition for the coupled G3D/Flexo weak form

This is an **assembly-stage mathematical operator**, not an assertion that
all seven nonlinear shell calculations have passed numerical verification.

Let q be the component-major vector of a KL, FSDT, Reddy or CUF TE1–TE4
surface model. For the thickness-integrated enthalpy L, let
A_ab=partial L /partial(Dx^a Ds^b q) with independent mixed jets.
Its already implemented discrete weak gradient is

R_direct = sum_(a,b) (Dx^a)^T [ W A_ab ] (Ds^b),

where W is the tensor product of axial/circumferential quadrature
and matrix products act on the appropriate spatial index. The
electrical gradient uses potential, Dx potential and Ds potential jets.

For the explicit SBP21_PERIODIC_ODD scheme,
  Dx^T Wx + Wx Dx = Bx;  Ds^T Ws+Ws Ds=0.
Repeated exact weighted Green identities yield

R_direct = R_bulk + R_boundary.

R_bulk = W sum_(a,b) (-1)^(a+b) Dx^a Ds^b A_ab.

The end virtual work is

 int_end [t delta q + m_n delta(D_n q)
                   + h_nn delta(D_nn q) + Q_n delta(phi)] ds,

with B0=A10-Dx A20-Ds A11+Dx^2 A30+Dx Ds A21+Ds^2 A12,
B1=A20-Dx A30-Ds A21, B2=A30, outward normal n_x=-1/+1,
t=n_x B0, m_n=B1, h_nn=n_x B2, Q_n=n_x A_electric,x.
For a CHARGE input sigma_free on an electrode, the natural boundary
condition is Q_n + sigma_free = 0; Q_n alone denotes outward induction.
The matrices transpose-contribute at the actual boundary rows,
including d_n q=n_x Dx q and d_nn q=Dx^2 q.
The mixed derivative appears exactly once in the independent-jet
weak form. Electric natural charge flux follows the same SBP identity.

Modules:
- mechanics/cylinder_sbp_green_reduction.py: exact direct/strong/boundary
  array decomposition with fail-closed non-SBP and reconstruction defects.
- CylinderFunctional.sbp_green_reduction(q, phi): obtains the SAME
  constitutive coefficients as the original seven-model enthalpy weak
  residual and returns separate nodal body and edge contributions.
- tests/test_sbp_green_reduction.py: staged 6-/10-jet and seven-model
  electric/mechanical Gateaux consistency checks.

Limits:
- The operation is a **decomposition** of the existing conservative
  discrete residual, not a second set of constraints or a new
  strong-boundary solve.
- Inertia and nonlinear viscosity are still included in the full
  dynamic variational residual; the diagnostic decomposition above
  is conservative-only. Their analogous discrete decomposition
  remains an engineering task.
- It does not prove local boundary stress accuracy, dynamic energy
  stability, MVI convergence or mesh convergence. These require
  independent tests and an engineering-scale sparse solver.
- The SBP scheme currently requires odd periodic ns>=5 and
  reference dense Newton is limited by its free-DOF cap.
- No AUTO rank logic was added; manual MVI ranks 1–8 remain unchanged.
