# Exact discrete SBP/Fourier Green identity for native coupled MVI

Date: 2026-11-09. Implemented in
`independent_mvi/coupled_discrete_green.py`, with algebraic and
runner-contract tests in
`tests/test_independent_coupled_mvi_discrete_green.py`.

## Exact identity on the method-owned grid

Let `D` be the MVI axial SBP42 first derivative, `H` its diagonal
positive quadrature, and `B=diag(-1,0,...,0,+1)`. The exact discrete
Green identity is

`H D + D.T H = B`.

Let `S` be the odd Fourier differentiation on the periodic circumferential
grid, with constant positive norm `Hs`. Then

`Hs S + S.T Hs = 0`.

For the *one and same* thickness-integrated virtual-work coefficients
`A[c,a,b]` multiplying `delta(D**a q_c S**(b).T)`,
first reduce the periodic coordinate:

`C[c,a] = sum_b (-S)**b A[c,a,b]`.

Define the discrete axial interior expression

`V[c] = H Hs sum_a (-D)**a C[c,a]`

and each boundary-momentum field

`P[c,k] = sum_(a=k+1)^m (-D)**(a-k-1) C[c,a]`.

The exact nodal first variation from the original MVI weak assembler
equals

`G[c] = V[c] + sum_(k=0)^(m-1) (D**k).T B Hs P[c,k]`.

The left and right coefficients conjugate to the AXIAL-coordinate
trace `delta(D**k q)` are respectively `-P[c,k](x_left)` and
`+P[c,k](x_right)`. For the physical OUTWARD-normal coordinate
`d_n**k q = n**k d_x**k q`, where `n=-1,+1`, the conjugate
action is `n**(k+1)*P[c,k]`. In particular, left normal
double traction does not have the same sign as left axial-coordinate
double traction. Triple traction regains the normal sign.

No interpolation from a sampled continuum flux to an SBP boundary node
is needed. No strong boundary row is invented. This is an exact finite
dimensional summation-by-parts decomposition of the actual chosen MVI
discrete weak work, including mixed x-s derivatives and electrical
potential modes.

## Implementation and acceptance

The module verifies BOTH the axial SBP identity and Fourier skew
adjointness before assembling the Green decomposition. It compares
the volume plus lifted axial boundary action to the exact complete
nodal gradient produced by the same physical MVI virtual work and
fails closed on any inconsistency. It returns immutable snapshots of
the volume action, boundary lift, axial-coordinate and outward-normal
end actions, and an explicit algebraic identity error.

The opt-in research CASE runner now invokes this audit on every
accepted static field, every accepted dynamic field, and the initial
dynamic field. It saves the identity status, maximum absolute/relative
error, maximum discrete end-work coefficient and field trace orders
in the run summary; it does not claim this is total dynamic end work.
A failed audit rejects the field BEFORE it is stored as an accepted
state.

## Verified and NOT verified

The independent no-shell MVI CI for commit `03a9fcac8e` completed:
**208 passed, 3 skipped**. It included seven theories, electric ON/OFF,
synthetic mixed third-order jets, random virtual-work directional
checks and both normal orientations.

This verifies the *DISCRETE algebra* on the current numerical grid.
It **does not** certify the continuum boundary-value problem, source
Kryško–Zhong higher-gradient microfree conditions, radial inner/outer
face fluxes, floating electric gauge, or physical high-order dynamic
natural conditions (inertia/Kelvin–Voigt). It does not establish
nonlinear solutions, temporal accuracy, spatial convergence, or
scientific benchmark agreement. The weak end work is diagnostic only;
do not force any unverified natural end traction into the coupled
MVI or independent MKR solver.

No physical shell calculation was started during this work.
