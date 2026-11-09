# Independent coupled MVI: formal axial end action audit (2026-11-09)

New research-only files:
- independent_mvi/coupled_end_actions.py
- tests/test_independent_coupled_mvi_end_actions.py

Updated independent_mvi/coupled_surface.py to retain a read-only snapshot of
the through-thickness integrated local weak coefficients on the MVI-owned
surface grid. No separate physical law, MKR stencils or reference
CylinderFunctional is imported or invoked.

For each generalized field c and local derivative (a,b), let A[c,a,b]
be the coefficient multiplying variation d_x^a d_s^b(delta U_c).
On a periodic circumference, after formal integration by parts in s,
B[c,a] = sum_b (-D_s)^b A[c,a,b].
At an axial end, with n_x=-1 (left) or +1 (right), the formal action
conjugate to the k-th axial normal *coordinate* derivative is

  T[c,k] = n_x sum_(a=k+1)^m (-D_x)^(a-k-1) B[c,a].

This is a continuum-style integration-by-parts diagnostic, evaluated
with MVI's own SBP42 / Fourier operators. The highest axial jet is
third order for KL/FSDT/Reddy, second order for CUF, first order for
electric potential; no jet is duplicated. The returned tuples
have one component per formal axial trace derivative.

The signs and simple polynomial/trigonometric examples are encoded in
no-solve tests covering all seven kinematics and electric ON/OFF.
Tests were created but **not executed in this session**. No physical
cylinder calculation, no solver convergence test, and no GitHub Actions
workflow were launched.

IMPORTANT SCIENTIFIC LIMITATION

A formal continuum end-action expression is not, by itself, a discrete
SBP-consistent boundary closure for high-order mixed derivatives. It
does not establish source-correct microfree terms, angular edge/corner
work, radial/inner/outer electric insulation, floating electrical gauge
compatibility, physical principal mixed order or consistency of the
nonlinear tangent. The module is deliberately diagnostic only and
is NOT used as an automatically enforced end-program for static or
dynamic MVI. The independent coupled MVI nonlinear branch remains
experimental, with the previously implemented fixed-rank Y/X iteration
and full admissible 2D stationarity gate.

Before activating additional boundary conditions, separately prove and
test the correct surface Green identities for every highest-order term,
with declared essential and natural traces of the coupled system.
