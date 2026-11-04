# CUF TE4 linear forced Bathe dynamics: independently archived FDM versus MVI-1

Date: 4 November 2026. The figures below compare **actual stored physical
solver states**, not an illustrative waveform, fit, or rerun.

## Provenance

- Independent FDM, **40/40 steps completed**:
  [GitHub Actions 41808812214](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41808812214),
  physical archive `CUF_TE4_LONG_LINEAR_DYNAMIC_PHYSICAL_HISTORY_AND_GRAPHS`,
  artifact ID `13327642600`.
- Classical fixed-rank **MVI-1**, first **10/120 steps accepted**:
  [GitHub Actions 41850750858](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41850750858),
  physical archive `CUF_TE4_MVI_1_STAGE_01_ACCEPTED`,
  artifact ID `13336468445`.
- Time interval compared: **0 to 0.020 s**.
  The total requested number of steps differs (FDM 40, MVI 120), but all
  physical inputs and the time law throughout this initial interval are
  identical. The MVI restart flag is execution-only.

## Identical physical setup through 20 ms

- Isotropic homogeneous cylinder: length **2 m**, reference radius **1 m**,
  wall thickness **0.05 m**.
- **CUF TE4** displacement kinematics: 15 generalized mechanical
  coefficients, ordered `[U0..U4,V0..V4,W0..W4]`.
- Linear isotropic Hooke material: **E = 210 GPa**, **nu = 0.30**,
  **density = 7800 kg/m^3**. No damping.
- Clamped left and right axial end sections; higher-order end terms NATURAL.
  Electrical coupling off in the MACRO material.
- Uniform radial inward reference **dead** pressure on the middle face,
  `q(t) = 100 kPa * sin(2*pi*25 Hz*t)`.
- Both methods: **9 x 9** grid, BATHE time integrator,
  `dt = 0.002 s`, initially zero displacement and velocity.
- Spatial comparison: independent full-nodal matrix-free FDM versus
  classical spatial rank-one X/Y MVI.
- For physical CUF fields, through-thickness basis is **(2z/h)^k**,
  not the dimensional monomials `z^k`.
  The midsurface radial displacement equals `W0`.

## Measured central midsurface radial displacement

Values extracted directly from `mechanical[10,4,4]` in the accepted
`REFERENCE_DYNAMIC_STEP_NNNN.npz` files, in **mm**.

| t (s) | Independent FDM (mm) | MVI-1 (mm) |
| ---: | ---: | ---: |
| 0.002 | -0.002877944989 | -0.002877936954 |
| 0.004 | -0.005283159505 | -0.005283179158 |
| 0.006 | -0.007354369984 | -0.007354388643 |
| 0.008 | -0.008624035149 | -0.008624042529 |
| 0.010 | -0.009068408572 | -0.009068411409 |
| 0.012 | -0.008627753277 | -0.008627747803 |
| 0.014 | -0.007337506623 | -0.007337550827 |
| 0.016 | -0.005331726460 | -0.005331618936 |
| 0.018 | -0.002803232527 | -0.002803181899 |
| 0.020 | -0.000000360838 | -0.000000328680 |

At **all 21 accepted time samples** (initial plus both Bathe stages
for ten full steps):

- Peak absolute FDM midsurface radial displacement:
  **0.009068408572 mm**.
- Maximum absolute MVI-1/FDM central midsurface difference:
  **1.265483e-10 m**, equal to **0.0013955%** of that peak.
- Across ten final full-step samples only, the maximum difference
  is **1.075248e-10 m**, equal to **0.0011857%** of the same peak.

These percentages are normalized to the **peak physical amplitude**,
not the instantaneous value. At `t = 0.020 s`, the displacement
approaches zero due to the load phase; pointwise relative percentages
then become misleading.

## Scope of this validation

This establishes close agreement of the **dominant central radial
displacement** on the first 0.020 s. It is **not** a certificate of
fieldwise identity in every generalized CUF component, or of accuracy
over all 120 requested MVI steps. Larger relative differences persist
in some very small generalized components.

Do not describe MVI-8 as completed or verified until its separate
physical rank-eight dynamic solve completes successfully.
