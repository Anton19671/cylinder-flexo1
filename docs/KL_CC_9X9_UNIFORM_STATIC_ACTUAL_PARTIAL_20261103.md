# KL C-C uniform inward 100 kPa: independently run static MVI vs MKR/FDM
Date: 2026-11-03. **PARTIAL PHYSICAL RESULTS; NOT A COMPLETED VERIFICATION**.

## Physical CASE
- Homogeneous isotropic cylindrical shell, KL / linear / HOOKE / MACRO.
- L=2 m, R=1 m, h=0.05 m, E=210 GPa, nu=0.3.
- C-C, including zero KL radial slope at both axial ends.
- 9x9, the same SBP42_FOURIER_ODD operator for separate MVI and FDM branches.
- Uniform **inward radial dead-reference midsurface** traction q=100,000 Pa.
  It is **not** a follower pressure on the actual inner/outer surface.
- Parameters reuse Rawat (2016) homogeneous cylindrical geometry/material;
  **no static pressure reference is attributed to the Rawat paper**.
- GitHub Actions physical run (original code at commit `1e1e4097`):
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022

## Physical results read from each accepted REFERENCE_STEP_0001.npz
Each MVI entry below was an independent actual solver run, not copied
from FDM. Differences use the complete [U,V,W] nodal field on 9x9.

| Method | Status | W at midspan, circumferential mean [um] | max abs(W) [um] | Full-field L2 difference vs FDM [%] | Sweeps |
| --- | --- | ---: | ---: | ---: | ---: |
| MKR / FDM | CONVERGED | -7.598499770229 | 8.071950353839 | 0 (self) | N/A |
| MVI-1 | FAILED: MAX_SWEEPS_PROJECTED_NOT_CONVERGED; accepted_steps=0 | | | | 30 |
| MVI-2 | RUNNING / NOT VERIFIED | | | | |
| MVI-3 | RUNNING / NOT VERIFIED | | | | |
| MVI-4 | CONVERGED_FULL | -7.598499770135 | 8.071950350117 | 4.468104e-8 | 2 |
| MVI-5 | CONVERGED_FULL | -7.598499770051 | 8.071950350043 | 4.302284e-8 | 2 |
| MVI-6 | CONVERGED_FULL | -7.598499770217 | 8.071950350033 | 4.360285e-8 | 2 |
| MVI-7 | CONVERGED_FULL | -7.598499770012 | 8.071950349861 | 4.347294e-8 | 2 |
| MVI-8 | CONVERGED_FULL | -7.598499784461 | 8.071950346447 | 1.592763e-7 | 2 |

The 50 kPa FDM control is now independently CONVERGED; its midspan W is -3.799249885115 um and maximum |W| is 4.035975176920 um. The FDM midspan deflection ratio W(100 kPa)/W(50 kPa) is 1.999999999999936. The 50 kPa MVI-1 control is not yet independently verified.

## Additional physical-consistency checks
- FDM exactly satisfies both clamped displacement edges in the stored
  displacement field (the nodal-edge max is exactly 0.0).
- The KL radial slope at both clamped ends was independently calculated
  from the SBP42 boundary differentiation row: max ~3.812e-21.
- FDM and all verified higher ranks have negligible circumferential
  variation of W and negligible V, consistent with axisymmetry.
- FDM max |U| ~0.64721555 um, max |V| ~6.64e-10 um.
- Linear static internal energy/work identity for the MVI-4 state:
  external virtual-work integral W_ext = 6.58910536805596 J;
  total potential Pi = -3.29455268405667 J,
  so |W_ext + 2 Pi|/W_ext ~ 8.7e-12.
- The full-field FDM solution U,W is numerically separable by x,s to
  relative error ~1e-9, explaining why ranks 4..8 may agree. This is
  **not** an independent check that ranks 2..8 improve general
  non-separable problems.

## Restrictions: what has NOT been verified
- MVI-1 at 100 kPa **actually finished without convergence** after 30 sweeps and 60 inner iterations. The saved state is named REFERENCE_FAILED_ITERATE.npz and must NOT be used for any numerical comparison. Its GitHub Actions job displayed technical SUCCESS solely because the CI workflow intentionally captures nonzero solver exit codes to preserve results and continues; the physical solve status is authoritative. MVI-2/3 and the 50 kPa MVI-1 were not yet verified at this snapshot.
- This is one coarse grid: mesh convergence is NOT VERIFIED.
- Comparison to FDM is on the **same variational discretization**.
  It does **not** independently certify the model or the true
  three-dimensional elasticity solution.
- Face stress postprocessing is explicitly labelled approximate/
  not independently verified in the saved solver outputs.
- No GL, true follower pressure, flexoelectricity or dynamic analysis is
  established by these static results.
- No solver algorithm was changed to force agreement.

## Artifacts for independent review
- MKR/FDM 100 kPa:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022/artifacts/13209378982
- MVI-4:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022/artifacts/13208994077
- MVI-5:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022/artifacts/13209542647
- MVI-6:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022/artifacts/13209234447
- MVI-7:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022/artifacts/13210455208
- MVI-8:
  https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022/artifacts/13209224842

- MVI-1 failed-state artifact (for diagnosis only): https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022/artifacts/13209529507
- FDM 50 kPa successful artifact: https://github.com/Anton19671/cylinder-flexo1/actions/runs/41589938022/artifacts/13210560774
