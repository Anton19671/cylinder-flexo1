# Native MKR dynamics: independent Newmark and Bathe adapters

Date: 2026-11-09. Research-only implementations:
- `numerics/cylinder_native_mkr_newmark.py`
- `numerics/cylinder_native_mkr_bathe.py`

Both drivers use `DirectStrongShell.balance` for the seven shell theories,
`assemble_physical_end_rows` for explicit field-specific physical conditions,
and `solve_strong_nonlinear_equations` for the simultaneous mechanical and
quasi-static electric unknowns. Neither uses the legacy variational spatial
assembly, MVI frozen-partner equations, the old Bathe driver or its time solver.

## Bathe stage definitions

Let `h` be a physical time step and `g=2-sqrt(2)`.
At `t+g*h`, the first stage is the implicit trapezoidal rule:

    q_pred1=q_n+g*h*v_n+(g*h)^2*a_n/4
    v_pred1=v_n+g*h*a_n/2
    a_1=4*(q_1-q_pred1)/(g*h)^2
    v_1=v_pred1+(g*h)*a_1/2

For the second stage at `t+h`, define:

    c1=(1-g)/(g*h)
    c2=-1/((1-g)*g*h)
    c3=(2-g)/((1-g)*h)

Then the three-point Bathe formula is:

    v_2=c1*q_n+c2*q_1+c3*q_2
    a_2=c1*v_n+c2*v_1+c3*v_2

The actual second-stage Newton equations use the algebraically equivalent
predictors:

    q_pred2=-(c3*(c1*q_n+c2*q_1)+(c1*v_n+c2*v_1))/c3^2
    v_pred2=-(c1*v_n+c2*v_1)/c3
    a_2=c3^2*(q_2-q_pred2)
    v_2=v_pred2+a_2/c3

A separate final kinematic audit checks the original identities.

Mechanical inertia, Kelvin--Voigt and mass damping use the native
continuum constitutive coefficients with the stage's v and a. Each stage
also solves all electrical modes as quasi-static unknowns, under the
caller-provided end conditions and optional single global potential gauge.
For time-dependent excitation the user supplies `loads_at(time_s)` returning
generalized mechanical force and electric free-charge densities evaluated
at both stage times. There is no conversion of pascals into generalized
surface loads inside the integrator.

The first or second nonlinear failure yields no accepted full-step state.
An unsuccessful second stage exposes the first-stage result only for
diagnostics; the initial state remains unchanged.

## Demonstrated scope and remaining blocks

Two new test files provide algebra-only checks and controlled substitutes
for Newton and constitutive callbacks. They do **not** validate physical
solutions. They have been committed but not executed in this session. No
loaded shell, GitHub Actions, or time history was run.

Before scientific use, the experimental spatial route still needs full
compatible nonlinear G3D/Gz face-flux discretization, independent
physical higher-order end closure and principal-order certification.
Initial accelerations must be obtained from physically consistent initial
conditions. A separately implemented coupled MVI, grid/rank/time
convergence, and the user's benchmark calculations remain outstanding.

**Do not auto-connect this code to GUI or production CASE routes.**
