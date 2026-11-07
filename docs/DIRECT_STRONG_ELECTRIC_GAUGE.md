# Direct strong electrical gauge implementation

The experimental strong Newton driver now accepts `StrongElectricGauge`.
It appends one multiplier and one volume-mean potential equation, retaining
every electrical field equation. This implements the single global
constraint required by sections 24 and 33 of
`Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx`.

`cuf_mean_potential_gauge` constructs the mean functional for the normalized
CUF basis G_k=(2z/h)^k and H=1+z/R. Dividing the thickness integral by h gives
1/(k+1) for even k and h/(2R(k+2)) for odd k. The caller supplies the actual
surface quadrature weights. This helper must not be used with an unnormalized
z^k basis. Dimensional state/equation scales remain explicit inputs.

The border is built in the scaled algebraic coordinates. Its multiplier is
a numerical compatibility variable, not a physical potential mode. Acceptance
also checks the original equations, the constraint and the multiplier
separately, so an incompatible electrical source cannot be accepted merely
because the bordered equations have a solution.

The previous `grounded_electric_elsewhere=True` shortcut is rejected by the
strong solver: this driver never assembled the purported external constraint.
Supply actual potential end conditions or an explicit global gauge instead.

Local verification: 21 tests in `test_direct_strong_electric_gauge.py`,
`test_direct_strong_newton.py`, `test_direct_strong_boundary.py`, and
`test_direct_strong_fdm.py` passed. These include quadrature checks for
potential orders 0--4, a singular algebraic electric system with one constant
mode, and rejection of incompatible source data. No loaded shell or time
history was solved. This does not certify strong boundary closure, spatial
convergence or GUI dispatch; the strong route remains experimental.
