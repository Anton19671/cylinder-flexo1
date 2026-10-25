# Full loaded dynamic SBP work ledger (assembly stage)

This is an auditable decomposition of the original G3D/flexoelectric
variational residual for KL, FSDT, Reddy and CUF TE1–TE4. It does NOT
alter Newton, the alternating MVI X/Y algorithm, or the chosen rank.

At a fixed physical state (q, phi, qdot, qddot, t):

R_m = R_G3D+flexo_bulk + R_G3D+flexo_natural_edges
      + R_KV + R_inertia + R_alphaM_damping
      - W*f_dead + R_prescribed_end_mechanical_work

R_phi = R_electric_bulk + R_electric_natural_edges
        - W*f_charge + R_electrode_charge_work

The code groups all internally derived mechanical terms into the exact
SBP bulk and natural-edge parts without double-counting. No artificial
electrical mass or electrical viscosity is introduced.

`numerics/cylinder_sbp_full_balance.py` obtains the internal split
from `CylinderFunctional.sbp_green_reduction` and independently
constructs the reference using
`BoundarySpatialProblem.evaluate_effective_residual`.
It checks the pointwise nodal identity, in force/charge units, with
explicit relative and absolute tolerances. This compares two assembly
routes; it does not evaluate whether the equilibrium residual is small.

The internal natural edge contributions are *derived from the weak
form*, whereas externally prescribed tractions, springs and charges
are supplied by `compile_boundary_program`. Essential supports,
higher trace constraints and voltage electrodes remain affine
constraints, never fabricated external loads. Consequently a
constraint reaction is computed by the separate admissible
projection and is not silently interpreted as an imposed traction.

API:
```python
from cylinder_sbp_full_balance import audit_full_sbp_work
ledger = audit_full_sbp_work(
    problem,q,phi,time_s=t,dead_load=f,
    free_charge=charge,velocity=v,acceleration=a,
    density_kg_m3=rho,eta_s=eta_s,
    mass_damping_rate_per_s=alpha_M)
report = ledger.summary()
```

The opt-in diagnostic rejects legacy non-SBP spatial operators
and preserves the separate MVI rank selection. Its dimensional
absolute tolerance for mechanics is expressed in newtons and for
electricity in coulombs.

The suite `tests/test_sbp_full_balance.py` has been staged for the
subsequent integrated-test phase. The complete physical shell
solver is **not verified**: no actual MVI/FDM convergence,
manufactured solutions, grid/time refinement or engineering-scale
sparse backend has yet been demonstrated.
