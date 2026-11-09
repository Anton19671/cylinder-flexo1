# Native MKR implicit Newmark step (research only)

Date: 2026-11-09. Implemented in `numerics/cylinder_native_mkr_newmark.py`.

This is an **opt-in** adapter for the native `DirectStrongShell` of all seven
models (KL, FSDT, Reddy, CUF TE1--TE4). It makes no change to GUI, CASE or
the legacy separate MVI/FDM routes. No simulation is launched by importing it.

## Full physical equations

For mechanical displacement q, velocity v and acceleration a, use
the existing native MKR instantaneous constitutive jet assembly and
strong bulk/end equations, with elastic, gradient G3D/Gz, reciprocal
flexoelectric, Kelvin--Voigt and inertial contributions. Electric
potential is quasi-static and is solved together with the mechanical
unknowns at the new time point. No electric inertia is introduced.

For the Newmark parameters beta and gamma, and time increment dt:

    q_predictor = q_n + dt*v_n + dt**2*(1/2-beta)*a_n
    v_predictor = v_n + dt*(1-gamma)*a_n
    a_next = (q_next-q_predictor)/(beta*dt**2)
    v_next = v_predictor + gamma*dt*a_next

Default beta=1/4 and gamma=1/2 (average acceleration). These update
identities are kinematics only; a valid physical time step requires
successful convergence of the FULL strong coupled field equations.

End conditions and axial differential half-orders are mandatory caller
inputs, never guessed from jet counts. The optional electric gauge is the
one full-volume potential constraint; source-specific CUF parallel-end
programs may be reused with
`NativeStrongNewmark.from_cuf_parallel_problem(problem, density_kg_m3=...)`.
Generalized mechanical loads are **not** face pressures in pascals.

The implementation copies input states and load arrays. On nonlinear failure,
`NativeNewmarkStep.state` is `None`; the old state is not mutated and no
failed step is accepted. Actual initial acceleration must be supplied by
the caller and physically checked.

## Tests and limitations

`tests/test_native_mkr_newmark.py` contains only source/contract and
algebraic checks with controlled callbacks (Newmark predictor, updated
velocity/acceleration, electric participation, immutable state, failure
atomicity, absence of imported MVI and reference spatial solvers). The
tests are added but **have not been executed** by the assistant. No loaded
shell, convergence study, transient simulation or GitHub Actions run was
initiated.

**Not yet certified:** compatible full nonlinear G3D/Gz face fluxes,
field-specific high-order end closures, coupled differential-order
classification, time convergence and scientific benchmarks. The adapter
does not make the overall solver production-ready.
