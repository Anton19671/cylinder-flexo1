# Scientific branch matrix: STATIC and DYNAMIC, G3D/Gz, flexo ON/OFF

Primary source: `Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx`.

This report distinguishes different levels of verification; **a green
algebraic or CASE-preflight CI is not proof of a physically correct shell
solution**.

## Intended model families

| Dimension | Confirmed test variants |
|---|---|
| Kinematics | KL, FSDT, Reddy, CUF TE1–TE4 (7) |
| Material gradient | G3D (18 covariant components), Gz (6 through-thickness) |
| Material switch | MICRO gradient ON with flexoelectric ON or OFF |
| Equation | Static assembled variational work; dynamic Newmark effective operator |
| Direct/indirect coupling | Global cross-tangent symmetry tests (other scientific workflow) |
| Spatial CASE method | FDM and rank-one MVI preflight; same constitutive variational source |

## Independent evidence recorded

- `test_scientific_static_dynamic_branches.py` exercises 28 separate
  seven-theory/material configurations and tests the **static global energy
  first variation** plus **dynamic mechanical inertia, mass damping and
  Kelvin–Voigt action**. For electric ON, the dynamic electrical equations
  must equal their quasi-static equations at the same mechanical/electric
  state (no spurious electric inertia).
- The same file performs 112 **real CASE preflights**: 7 theories × 2
  material gradients × 2 switches × 2 calculation kinds × 2 spatial
  selections. Preparation computes initial acceleration in DYNAMIC and
  ensures no result directory is created. Passing preflight does **not**
  imply any subsequent solver step converges or that spatial methods are
  mathematically independent.
- Production `MICRO` CASE intentionally accepts **GREEN_LAGRANGE only**;
  LINEAR and VON_KARMAN research-kernel identities are not direct
  publication-ready MICRO CASE routes. Tests require these to fail closed.
- Flexoelectric OFF means the electric unknowns and both coupling
  coefficients are physically **absent**, while the G3D or Gz strain
  gradient remains active. This does not simulate the `MACRO` limit
  unless the gradient length also vanishes.

## Still OPEN for scientific release

1. Full static equilibrium **solutions**, force–displacement response,
   thickness stresses and surface fluxes for each physical material/mode.
2. Complete dynamics (accepted steps, time integration accuracy and energy
   histories) **for every** seven-theory × G3D/Gz × flexo switch,
   not just representative example cases.
3. Physically independent coupled MKR versus MVI spatial discretizations.
   Currently the `VARIATIONAL_REFERENCE` methods share one weak
   constitutive/assembly source and differ by trial subspace; this is
   not independent physical discretization validation.
4. External WQEM/FEM/analytic benchmarks matching supports, load face,
   end micro-free conditions, electrostatic conditions and units.
5. Spatial/time-step/rank/TE-order convergence and robust STOP/PAUSE/restart.

This is a **scientific readiness gate**, not a certification statement.
No GUI layout or production physical operator is changed by this report.
