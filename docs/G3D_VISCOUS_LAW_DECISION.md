# Constitutive decision: eta_s in the nonlinear G3D/flexoelectric solver

Status: mathematical specification for implementation, **not numerically verified**.

## What the user's Krysko–Zhong theory actually specifies

The source manuscript "Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024"
sections 12 and 21 chooses **classical bulk Kelvin–Voigt viscous
stress based on the six Green–Lagrange strain rates**. Section 21
explicitly states that the base material law has **no strain-gradient
viscosity**. Thus `micro.eta_s` must not be interpreted as Rayleigh
damping and must not multiply the G3D hyperstress without a separately
postulated constitutive law.

## Objective finite-strain Kelvin–Voigt model

Reference Green–Lagrange strain E = (F^T F - I)/2.
The time derivative is E_dot = sym(F^T F_dot).

Elastic: S_el = lambda tr(E) I + 2 G E.
Viscous: S_v = eta_s [lambda tr(E_dot) I + 2 G E_dot].

eta_s has SI units seconds (s), and eta_s >= 0.
The viscous Rayleigh density is

  R_v = eta_s/2 * [lambda (tr E_dot)^2 + 2 G E_dot:E_dot].

S_v = partial R_v / partial E_dot, and local dissipation

  D = S_v:E_dot = 2 R_v >= 0

when the isotropic elasticity C is positive definite. In the
engineering-shear six-vector (e11,e22,e33,2e12,2e13,2e23), strain
energy and viscous power must be evaluated with the correct shear
normalization. Do not simply multiply six-vector components without
accounting for the engineering shear convention.

The **mechanical** viscous first variation is

  delta R_v / delta q_dot
      = integral_V0 [S_v : delta E(q)] dV0
      = integral_V0 [eta_s C:E_dot(q,q_dot)] :
                      DE(q)[delta q] dV0

at the current configuration. In CUF, KL, FSDT and Reddy use the
respective analytic nonlinear strain/virtual-strain maps and the SAME
thickness H(z) and surface adjoints as G3D enthalpy.

For current q and v, form velocity jets by applying the same spatial
operators to v; obtain E_dot via the analytic directional derivative
of E(q) along v. This is **not** the small-strain velocity tensor
unless the small-strain option is selected explicitly. Newmark's
effective tangent includes both dependencies q and v(q) and must not
treat Kelvin–Voigt forces as fixed loads.

## Relationship to strain-gradient elasticity

G3D elastic hyperstress M and flexoelectric direct/inverse coupling
remain given by the common enthalpy. A bulk Kelvin–Voigt term does
not damp strain-gradient rates by itself; it is a valid, separately
chosen dissipative extension of an elastic G3D material.

If the intended physical model requires **gradient viscosity**, it
needs a *different and explicit parameter* eta_g >= 0 and a potential,
for example (only after scientific agreement)

  R_g = eta_g * ell^2 / 2 * [lambda |grad tr E_dot|^2
                            + 2 G |covgrad E_dot|^2].

The corresponding viscous *hyperstress*, higher-order dissipative
boundary work and surface adjoints must then be derived. This is NOT
automatically implied by a G3D elastic length scale.

## Not interchangeable

- `micro.eta_s`: nonlinear material Kelvin–Voigt rate factor (s).
- `dynamics.mass_damping_rate_per_s`: Rayleigh C=alpha_M M, (1/s).
- hypothetical `eta_g`: additional independently specified strain-gradient
  viscous coefficient; not present in the baseline manuscript.

**Integration gate:** nonzero `micro.eta_s` is currently rejected by
the reference dynamic CASE until its six-component weak viscous action
is consistently connected to all seven theories. Do NOT remove that
safety guard merely by changing the CASE validator.

## Verification required after assembling the whole solver

For each of KL/FSDT/Reddy/CUF TE1–TE4:
1. D >= 0 for random finite admissible q and v.
2. D = 0 at v = 0 and eta_s=0.
3. Analytical Gateaux virtual-power residual equals finite-difference
   derivative of the Rayleigh potential with respect to v.
4. Exact rigid-body motion yields no spurious dissipation.
5. Work/power identity and temporal energy balance with Newmark, same BC.
6. Independent full-residual FDM versus fixed-rank MVI under same CASE.
