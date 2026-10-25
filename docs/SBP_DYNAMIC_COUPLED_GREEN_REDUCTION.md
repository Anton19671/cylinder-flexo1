# Complete SBP decomposition of dynamic G3D + flexoelectric weak residual

**Development assembly, not numerical qualification.**

The local mechanical weak coefficients for every independent surface jet
q_(x^a s^b) are now constructed from one routine:
`CylinderFunctional._dynamic_g3d_weak_jet_coefficients`.

They contain (1) Green-Lagrange material elastic/strain-gradient and
direct/inverse flexoelectric stationarity, (2) finite-strain bulk
Kelvin-Voigt viscosity, (3) correct shell kinetic inertial virtual work
and (4) optional mass-proportional damping.

For KL the nonlinear unit director contributes q, q_x and q_s inertial
coefficients, including second time derivative of the director.
For FSDT/Reddy the gradient-dependent thickness velocity ansatz
contributes q, q_x and q_s inertia. CUF TE1–TE4 contributes only
local q mass moments, with full thickness coupling.

The electric weak coefficients come *only* from the G3D/flexoelectric
enthalpy; there is no fictitious electric mass or viscous response.

With SBP21_PERIODIC_ODD the same coefficients yield an exact
discrete weighted Green identity (up to floating-point roundoff):

  R_weak_internal = W * R_strong_bulk + R_axial_edges

with the full mechanical and electric end virtual work already defined
by G3D_NATURAL_BOUNDARY_DERIVATION.md. All differentiated mixed jets
are treated once. No second natural-traction law is introduced.

**Important:** External dead loads, end work, prescribed supports,
electrode voltage and free charges are separately included by
`BoundarySpatialProblem.evaluate_effective_residual`; the split is
an INTERNAL operator. Do not compare it to a forced or constrained
residual without explicitly including those external corrections.
This implementation is a decomposition/diagnostic; it does not
automatically impose stronger physical microfree BC.

Usage:
```python
split = functional.sbp_green_reduction(
    q, phi, velocity=v, acceleration=a, density_kg_m3=rho,
    eta_s=eta_s, mass_damping_rate_per_s=alpha_M)
mechanical_internal = split.mechanical.reconstructed
electrical_internal = split.electrical.reconstructed
```

The `natural_end_fluxes` method now consumes the identical
coefficient assembly; its fields are sampled continuum-like face
densities, while the SBP boundary array includes discrete virtual
normal-derivative contributions at multiple near-edge nodes.

The deferred seven-model test
`tests/test_sbp_dynamic_green.py` compares these fields against the
existing full effective residual with no external loading, including
finite q, v, acceleration and positive eta_s.

**Not yet achieved:** execution of the tests and independent
convergence studies; external-load-inclusive work balance and
manufactured natural BC solutions; production sparse matrix-free
linearization, Newton preconditioning and engineering grids. Manual
MVI rank 1–8 selection remains unchanged.
