# Kelvin–Voigt eta_s: integrated dynamic implementation (assembly stage)

Reference: Krysko–Zhong corrected manuscript, section 12 (six Kelvin–Voigt
viscous stresses) and section 21 (no gradient viscosity in baseline).

The **three different physical mechanisms** must never be conflated:

1. Elastic covariant strain-gradient theory G3D, with length ell [m].
2. Bulk nonlinear Kelvin–Voigt viscosity eta_s [s].
3. Optional mass-proportional damping alpha_M [1/s].

Strain: E = 0.5(F(q).T F(q)-I)
Rate: E_dot = sym(F(q).T DF(q)[q_dot])
S_v = eta_s C:E_dot, with engineering six-component shear convention.
Rayleigh dissipation density:
  R_v = 0.5 eta_s E_dot:C:E_dot.
Power density:
  P_v = eta_s E_dot:C:E_dot = 2 R_v >= 0.
Mechanical virtual viscous work:
  V_v[q,q_dot;delta q] = int_{A0} int_{-h/2}^{h/2}
       H(z) * S_v:E'(q)[delta q] dz dA0.

The same three-dimensional kinematics as the conservative functional are
used independently for KL (finite-rotation director), FSDT, Reddy and all
CUF TE1..TE4; mechanical residuals use the same surface quadrature and
matrix-transpose derivative operators. No electric viscosity is postulated.
The Newton effective residual depends on trial q and the Newmark-generated
trial q_dot (not a fixed external force).

## Implementation connectivity

- mechanics/cylinder_kelvin_voigt.py: local finite-deformation viscous
  first variations, 6 viscous stresses and positive viscous power.
- mechanics/cylinder_functional_assembly.py: surface integration and weak
  adjoint assembly; returns KelvinVoigtGlobal(mechanical_force, power_w).
- numerics/cylinder_boundary_spatial.py: adds bulk viscous force to full
  dynamic mechanical residual; electric residual unchanged.
- numerics/cylinder_variational_dynamic.py: propagates eta_s to the Newton
  corrector, full FDM and alternating MVI X/Y half-steps.
- interface/cylinder_reference_case.py: accepts micro.eta_s >= 0 in DYNAMIC
  and stores physical viscosity alongside distinct Rayleigh damping.
- tests/test_kelvin_voigt_all_theories.py: prepared verification of
  positive power and velocity Gateaux derivatives for all seven theories.

**State of verification: not yet numerically tested.** A successful code
commit is not proof of convergence, dissipation positivity or correct
global traces. Run the staged checks and independent coupled FDM/MVI CASE
experiments before claiming engineering verification. The existing legacy
classical dispatcher is NOT a substitute for the coupled reference solver.

Actual *strain-gradient viscosity* would require a separate eta_g and
higher-order dissipative stresses and traces. It is intentionally absent
from the baseline manuscript and this implementation.
