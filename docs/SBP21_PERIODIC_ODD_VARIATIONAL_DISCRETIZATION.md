# SBP21 + odd periodic spatial discretization for seven-model G3D/Flexo

Status: **assembled as an optional reference spatial discretization, not
numerically tested or production certified**.

## Why the old nodal operators were not a Green pair

The original `surface_trapezoid` uses an endpoint second-order
one-sided axial derivative and an upwind-like forward periodic
derivative. Those are valid verification finite differences but do not
satisfy the skew-adjoint / boundary Green identity for their
trapezoidal surface weights. Strong natural fluxes sampled from
continuous integration by parts are therefore not identical to the
boundary terms of the fully discrete adjoint.

## New exact semidiscrete identities

The explicit `grid.operator_scheme=SBP21_PERIODIC_ODD` selects

- Axial first-derivative SBP(2,1): centered in interior, 1st-order
  one-sided at ends, diagonal trapezoid mass weights Wx.
- Circumferential centered periodic derivative with uniform Ws.

For all finite nodal vectors u,v:

`u.T Wx Dx v + (Dx u).T Wx v = u_R v_R - u_L v_L`

and

`u.T Ws Ds v + (Ds u).T Ws v = 0`.

These are *identities to roundoff*, not convergence-rate estimates.
`surface_sbp21` calls `require_green_identity` at construction time.
The periodic node count **must be odd and at least 5**, as on even
counts a real skew-adjoint derivative with constant null vector also
has a spurious zero eigenvalue (checkerboard). This differs from the
old forward difference, which permits even node counts.

The same discrete operators enter:
- Green–Lagrange deformation and all surface derivative jets;
- G3D hyperstress and direct/inverse flexoelectric coefficients;
- nonlinear Kelvin–Voigt rates and viscous forces;
- KL / FSDT / Reddy / CUF mass and director terms;
- exact weak residuals with transpose adjoints;
- affine higher trace constraints for MVI and FDM.

Powers of Dx and Ds are deliberately used consistently in all these
parts. Repeated applications of the discrete SBP law supply the
algebraic Green identities of mixed and higher derivatives, with
**one** q_xs or q_xxs jet and no fictitious factor of two. They do not
automatically establish stability of every high-derivative boundary
closure or spectral accuracy.

## CASE example

```json
"mvi_grid": {
  "nx": 7,
  "ns": 9,
  "operator_scheme": "SBP21_PERIODIC_ODD"
}
```

For an FDM CASE replace `mvi_grid` by `fdm_grid`. The default
`LEGACY_REFERENCE` remains unchanged and supports previous CASEs.

## Important limitations

1. SBP exactness does NOT mean the sampled continuum natural fluxes in
   `CylinderFunctional.natural_end_fluxes` equal an independently
   verified physical microfree boundary solution.
2. The diagonal norm is second-order-accurate in the interior and
   first-order at axial boundaries. Third-order director jets and G3D
   boundary constraints need convergence/energy studies.
3. The existing nonlinear coupled Newton/alternating MVI solver is a
   dense small-grid reference path, not an efficient sparse production
   solver.
4. The new scheme and staged tests are **not yet executed**, per the
   user-requested mathematics-first assembly order.
