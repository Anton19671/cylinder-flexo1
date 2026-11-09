# Native MKR Gauss-Legendre IRK4 — implementation and strict scope

Date: 2026-11-09. New opt-in module:
`numerics/cylinder_native_mkr_gauss2.py`.

This is a distinct **native MKR** temporal implementation. It uses the
existing DirectStrongShell constitutive/strong field equations, independently
of MVI, shared variational FDM, and legacy Gauss time drivers.

## Mechanics

Two physical collocation stage positions Q_i and accelerations A_i satisfy

    Q_i = q_n + h*c_i*v_n + h*h*sum_j (A_Gauss^2)[i,j]*A_j

The 2x2 Gauss matrix is invertible. The implementation derives accelerations
from both simultaneous stage positions, then velocities from

    V_i = v_n + h*sum_j A_Gauss[i,j]*A_j

Both mechanical and quasistatic electric bulk/end equations are assembled
at the two stage times, with their own load and free charge. The stage
unknowns are solved as ONE coupled Newton system, never in a sequential
explicit stage approximation. End-of-step displacement and velocity use

    q_next = q_n + h*sum_i b_i*V_i
    v_next = v_n + h*sum_i b_i*A_i

The physical acceleration at t+h is NOT set equal to a Gauss internal
stage acceleration. Instead, the pre-existing independent initial-state
equilibrium helper is reused at the endpoint to solve for the actual
acceleration and electric potential at q_next, v_next. Failure rejects
the whole step. This implementation uses Gauss2's formal fourth-order
tableau, but no real-field time convergence has been demonstrated.

## Floating electric potential

An insulated electric model may have one additive potential constant.
The Gauss stage driver imposes the supplied volume-mean gauge separately
at both times, replacing one constant-mode electric equation per stage.
The omitted original equation is checked explicitly after convergence,
along with the physical mean constraint. Incompatible electric data reject
the step. This method of gauge handling still requires independent
physical boundary and electroneutrality validation.

## Validation and limitations

The new `tests/test_native_mkr_gauss2.py` contains only algebraic/
mock-callback tests (constant-acceleration identities, two electric stage
unknowns, unsuccessful nonlinear solve, endpoint failure, import isolation).
The GitHub repository could not be cloned in the current container because
DNS to github.com was unavailable. Tests have **not been executed** here.
No actual loaded-shell calculations or time histories were run.

Existing limitations are unchanged: complete source-consistent G3D/Gz
face fluxes, high-order end conditions and principal order certification,
an independent full nonlinear MVI, method/mesh/rank/time convergence,
benchmark comparisons, and GUI/CASE integration are outstanding.

Do NOT silently enable this experimental module in production GUI.
