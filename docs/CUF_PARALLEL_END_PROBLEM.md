# Stationary CUF end problem: code integration and limits

Source: `Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx`,
sections 24 and 30--33. Implementation date: 2026-11-08.

`numerics/cylinder_cuf_parallel_ends.py` now supplies a reusable
`CUFParallelEndProblem` with `evaluate`, `equations`, and explicit `solve`
methods. Constructing the problem never starts a calculation.

## End program

- All 3(N+1) displacement coefficients are zero at both end sections,
  for TE1--TE4. This fixes the physical displacement for every thickness z.
- Each caller-declared fourth-order field receives a zero double-traction
  equation. Its axial slope is free. No zero-slope constraint is added.
- A declared second-order field receives only its displacement equation.
  Per-field axial half-orders (1 or 2) are mandatory; they are not inferred
  from the highest stored kinematic jet. The constructor validates counts,
  not the physical differential order or ellipticity of the coupled model.
- Every active potential mode has zero generalized axial flux at both ends.
  The existing variational thickness projection represents the free faces;
  this implementation does not additionally impose pointwise face equations.
- Exactly one volume-mean potential gauge is passed into the bordered Newton
  solve. All original electric equations, including end equations, remain
  present. The previously implemented multiplier/original-equation check
  remains active. Electrical OFF has no potential unknown or gauge.
- Nonzero net free charge in the constant potential mode is rejected for
  this fully insulated problem. There is no separate charge-integral or
  gauge constraint on every higher thickness mode.

## Basis, grid and load conventions

This adapter now accepts the native `DirectStrongShell` CUF implementation:
F_i=G_i=(2z/h)^i. Its own grid supplies the physical length, periodic
circumference and trapezoid quadrature for the cylindrical volume gauge.
An optional `length_m` must agree with that grid. The constructor copies
generalized loads, so a subsequent caller edit cannot silently change the
problem. The prototype constructor that accepted a `CylinderFunctional`
has been replaced; build a `DirectStrongShell` from physical parameters.
The separate legacy `solve_direct_strong_reference` API remains available.

`mechanical_load` and `electrical_charge` are generalized densities per
reference midsurface area, not raw face pressure or unintegrated volume
charge. Mechanical loading contributes -integral(f*q); physical free charge
contributes +integral(rho*phi) to the stationary electric enthalpy. Thus the
electric Euler equation is E0-d_x(E1)-d_s(E2)+rho=0, consistent with div(D)=rho.
Both `formal_coupled_balance` and `direct_strong_jet_balance` previously
subtracted `electrical_charge`; this sign is now corrected. External callers
that used that argument as a signed equation right-hand side must instead
supply physical generalized free charge. Zero-charge results are unaffected.

State and equation scales must be supplied to `solve`. Boundary rows and
bulk rows have different dimensions; array-valued scales support this.
No guessed physical scale or automatic displacement/charge initialization
is introduced.

## Verification scope

The local tests use the real CUF kernels for TE1--TE4 at an unloaded exact
state with a constant electric potential. A separate tiny manufactured
coupled fourth/second-order problem checks the full Newton/end/gauge chain:

    W = q_xx**2/2 - p_x**2/2 + c*q_xx*p_x
    q = x**4 - 2*x**3 + x
    p = x**2*(1-x)**2 + constant

At x=0 and x=1, q=q_xx=p_x=0, but q_x is +1 and -1. The test recovers these
nonzero slopes, zero higher end work and the prescribed mean potential.
Charge-sign tests compare the real reference enthalpy variation with both
strong operators and independently check div(D)=rho for a linear flux.

Local verification completed: 48 tests and 22 subtests passed in 5.24 s
across `test_cuf_parallel_ends`, `test_formal_strong_balance`, the four
`test_direct_strong_*` modules, `test_cuf_g3d_weak`, and
`test_functional_assembly`. These are algebraic/manufactured and zero-state
tests, not physical loaded-shell benchmarks.

## Remaining work

This is an experimental stationary strong-equation adapter, not a completed
production shell solver. Native MKR now owns field preparation, constitutive
coefficient assembly and the call to direct bulk/end differentiation; it
does not call `CylinderFunctional` or MVI. The MVI coupled spatial route
still needs its own implementation. Full compatible G3D/Gz face-flux
placement, coupled differential-order certification, physical grid tests,
dynamic integration of this end program and GUI dispatch remain open.
See `NATIVE_MKR_FIELD_ASSEMBLY.md` for the preserved stencil convention.
No loaded-shell calculation, transient series or GitHub Actions run is invoked.
