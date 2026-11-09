# Independent coupled nonlinear MVI: native Y/X iteration

Date: 2026-11-09. Added to main:

- `independent_mvi/coupled_halfstep.py`: method-owned Newton halfstep.
- `independent_mvi/coupled_static.py`: repeated Y->X static sweep.
- `tests/test_independent_coupled_mvi_halfstep.py`: artificial quadratic work checks.
- `tests/test_independent_coupled_mvi_static.py`: control-flow checks.

## Mathematical algorithm and methodological separation

The state of every generalized mechanical and electrical mode is fixed rank:

    U_c(x,s) = sum(r=1..R) X[c,r](x) Y[c,r](s), 1<=R<=8.

The Y halfstep holds X fixed and solves exclusively for Y. The X halfstep
holds Y fixed and solves exclusively for X. Each halfstep differentiates
the MVI-owned surface enthalpy built in coupled_surface.py. It uses central
finite differences in the REDUCED frozen-partner coordinates to assemble an
indefinite Newton tangent. Linear corrections are obtained by LU with an
explicit reciprocal-condition check; no SVD, pseudoinverse, eigenvalue
clipping, rank growth, full-node linear correction or spatial MKR fallback.

Each change of profile coordinates by QR preserves the represented field.
When orthogonalizing the axial partner, only the admissible FREE axial
coordinates participate; the resulting zero essential coefficients cannot
be made nonzero by completing an inactive QR column. Hence fixed endpoint
values are not reintroduced through normalization.

After Y and X both complete, the driver checks stationary projected
work along both frozen-partner directions and the relative change of the
reconstructed physical field. It accepts a full Y->X sweep atomically;
failure during X keeps the previous fully accepted sweep, not a Y-only
physical field. An accepted status deliberately says
MVI_FIXED_RANK_PROJECTED_STATIONARY_ONLY.

## Explicitly supported reduced boundary subset

The boundary declaration is mandatory, not guessed:

- Mechanical: IMMOBILE_PARALLEL_ENDS: every q coefficient has zero value
  at both axial endpoints.
- Electrical OFF: program OFF, with no electric field.
- Electrical ON: GROUNDED_LEFT or GROUNDED_BOTH for each potential mode
  to eliminate the additive potential constant explicitly.

An insulated floating electric system with the source's one global volume
gauge is NOT connected. Nonhomogeneous or moving mechanical supports, all
other end types, and extra physical microfree traces are NOT connected.
The current strong-form differential order analysis remains unverified.
Do not interpret the constrained weak discretization as a certificate for
natural G3D/Gz higher-gradient boundary conditions.

## Status, limitations and verification

The new solver is a **research-only fixed-rank weak-work stationarity
algorithm**, not a complete publication-grade cylinder solver. In particular:

1. MVI halfstep tests are only manufactured quadratic factor equations,
   rank immutability, QR invariance, explicit boundary rejection, failure
   atomicity and cancellation. They are not loaded cylinder calculations.
2. A new mandatory full-admissible-nodal first-variation gate has now
   been implemented in the independent coupled MVI surface and Y/X sweep
   (2026-11-09). A fixed-rank projected-stationary field is NOT accepted
   if any admissible two-dimensional nodal direction fails this gate.
   This catches rank insufficiency but does NOT validate physical high-order
   edge fluxes or prove mesh convergence.
3. Consistent physical higher-order natural end conditions, electric
   gauge/insulation, reciprocal dynamic inertia and source boundary work
   require their own mathematical and numerical checks.
4. Dynamic MVI-1..8 Newmark/Bathe/Gauss remain to be built over THIS
   independent spatial assembly, with separate time stage equations.
5. No production CASE, GUI or export dispatch has been enabled. Existing
   reference MVI, independent linear MVI, and independent MKR remain intact.
6. The new tests have been COMMITTED but NOT EXECUTED here. No physical
   cylinder simulation has been launched by this development step.

Only local continuum constitutive kernels are shared between methods;
the MVI 2D spatial operators, factor gradients, and nonlinear Newton
algorithms are owned by independent_mvi. MKR code is not imported.


## 2026-11-09: full admissible 2D nodal certification

The method-owned spatial assembler now exposes `nodal_gradient`, the
weighted first variation with respect to every component and every
independent surface nodal coefficient, assembled by the SAME MVI
SBP42/Fourier derivative powers and thickness-integrated local virtual
work. It does not import MKR, perform a full-node Newton solve or change
rank. It confirms internally that contractions with the frozen opposite
factors reproduce the independently assembled X/Y derivatives.

The stationary Y/X driver checks the full gradient on each field's
explicitly admissible axial coordinates (mechanical ends immobile;
grounded electric endpoint(s) excluded separately). Acceptance requires
both frozen-partner conditions, small change of reconstructed field AND
all admissible nodal tests. Where projected stationarity holds but
the full nodal criterion does not, status is
`MVI_FIXED_RANK_INSUFFICIENT_FULL_NODAL_WORK` with `converged=False`.
Acceptance status is now `MVI_FIXED_RANK_FULL_NODAL_STATIONARY`.
The full norm is carried in the result and sweep history.

Test code covers a manufactured mixed X/S local functional, mechanical/
electric factor-to-node chain rule, exclusion of essential endpoints,
and deliberate rank-insufficient cases. Tests were committed but not
executed in this interaction. No physical cylinder calculation was run.

This does not settle higher-order microfree and electric face
boundary work, the mixed principal PDE order, the floating electric
gauge problem, or independent dynamic MVI. The module remains opt-in.
