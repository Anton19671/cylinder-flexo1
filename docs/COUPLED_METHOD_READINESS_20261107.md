# Full algorithm readiness: independent method separation status

Date: 2026-11-07.
Authoritative mathematics: Krysko–Zhong corrected CUF TE1–TE4
2026-10-24, §§13–15, 22–25, 30–33, 36–41, 46.
The operational requirement is TWO independent numerical spatial
solvers, not merely two labels around one common discrete system.

## Demonstrated repository architecture at this commit

| Requirement | MKR/FDM | MVI | Status |
|---|---|---|---|
| Seven linear mechanical kinematics | Independent linear_models | Independent linear_models | Code implemented; physics benchmarks open |
| Linear static | Own run_static + own strong equations | Own run_static + own rank-fixed X/Y | Preflight covered; real fields not certified |
| Bathe/Newmark/Gauss linear dynamics | Own run_models | Own run_models | Own dispatch; real dynamics not certified |
| Linear angular derivatives | Own primal/dual face energy | Own SBP/Fourier weak | Algebraic method isolation; spatial convergence open |
| CUF outer radial face output | Own radial sum | Own radial sum | Algebraic export corrected |
| Coupled G3D/Gz local material | Shared physical enthalpy | Shared physical enthalpy | Local physical formula source only |
| Coupled G3D/Gz STATIC Newton | Old reference FDM route | Own MVI static X/Y Newton | Numerical halfsteps now split; **physical assembly still common** |
| Coupled G3D/Gz dynamic Newmark/Bathe | Old FDM subspace | Own MVI dynamic X/Y Newton | Numerical halfsteps now split; same continuum physics |
| Coupled G3D/Gz Gauss | Own variational FDM adapter | Own MVI Gauss adapter | MVI parameter/scaling import from FDM removed |
| Full higher-order mechanical/electric edge closure | Not certified | Not certified | **BLOCKER** |
| Fully independent nonlinear spatial assembly from CUF coefficients | Not yet complete | Not yet complete | **BLOCKER** |
| Direct/inverse flexoelectric tangent globally | Not independently certified | Not independently certified | **BLOCKER** |

## Independent MVI numerical modules

- `numerics/cylinder_mvi_subspace_newton.py`: stationary coupled
  Newton halfsteps constrained to explicit separated X/Y directions.
- `numerics/cylinder_mvi_dynamic_subspace.py`: dynamic coupled
  Newmark X/Y halfstep Newton, its OWN tangent, scaling and linear
  solve; missing X/Y directions is rejected.
- `numerics/cylinder_gauss2_variational_mvi.py`: own MVI
  Gauss2 spatial adapter, now using MVI-owned scaling and settings.
- `numerics/cylinder_variational_mvi.py`: own rank-fixed static
  alternation; public reference CASE parameters remain compatible.
- The shared `CylinderFunctional` still computes physical 3D
  strains/gradients at **through-thickness quadrature points**; there
  is no justification for introducing a separate 3D spatial mesh.

## Deliverable and certification boundaries

The above is a **partial architectural separation of the legacy
coupled solver**, and a complete two-method separation ONLY for the
experimental linear electric-off subset. It is *not* scientifically
valid to label all reference G3D/Gz numerical routes as independent.

Physical computation is deliberately not carried out by this audit.
The manually selectable structural workflow
`manual-coupled-mvi-method-isolation.yml` runs only source/API
and method-isolation tests; no loaded-shell benchmark or time series
is launched.

To certify the entire original algorithm, further implementation is
needed of (1) independent strong full G3D/Gz operators and ALL
associated natural/essential/electrical end traces, (2) both reciprocal
flexoelectric cross blocks from the same energy, (3) own MVI weak
spatial high-order discretization and rank control across all TE,
and (4) independent manufactured solutions and the user's subsequent
local physical calculation suites. Never infer this certification
from a green source-only preflight.
