# Physical multi-step fixed-rank MVI + Gauss IRK4

`numerics/cylinder_gauss2_mvi_history.py` runs **sequential** two-stage
Gauss–Legendre IRK4 steps with **both physical stages solved by fixed-rank
X/Y MVI**, using `solve_gauss2_variational_mvi_step`.

The method is opt-in, small-grid and **experimental**. Default Newmark
CASE/GUI routing is unchanged.

At every step the driver:
1. Supplies the previous accepted physical displacement, full velocity,
   electrical potential, and fixed-rank MVI profiles to the next step.
2. Assembles coupled shell stage equations at actual Gauss stage times.
3. Independently checks stage positions and velocities against Gauss
   collocation, endpoint time, rank-R profile reconstruction and fields.
4. Writes **only** an audited accepted step as
   `GAUSS2_MVI_STEP_XXXX.npz`, with stage mechanical/electric states,
   accelerations, full endpoint velocity, profile factors and field scales.
5. Updates `GAUSS2_MVI_HISTORY.json` atomically. When a step fails,
   earlier accepted steps remain intact; the history is marked FAILED.

Mechanical supports that move inside a step remain unsupported in this
experimental branch and are explicitly rejected. Electrode constraints
are evaluated at actual stage times and at the endpoint. No SVD-based
rank truncation, full-nodal correction or method fallback is permitted.

**Important limitation:** fixed-rank stage solutions generally do not
guarantee a fixed-rank endpoint under Gauss collocation. Such an endpoint
is **rejected**, not projected. As a result this restricted algorithm is
not guaranteed to produce a full trajectory for arbitrary physical
initial data, and global fourth-order accuracy is not asserted for the
rank-restricted nonlinear DAE.

The driver does not by itself complete physical verification across the
seven cylinder kinematic theories, G3D/Gz, electric boundary conditions,
all MVI ranks or time-grid refinement. The manufactured regression tests
are `tests/test_cylinder_gauss2_mvi_history.py`; they never establish
the full physical shell verification.

Example entrypoint (with a separately prepared physical problem):

```python
from cylinder_gauss2_mvi_history import run_gauss2_variational_mvi_history
result = run_gauss2_variational_mvi_history(
    problem, initial_state, Gauss2Settings(0.01),
    mvi_settings=Gauss2MVISettings(rank=1),
    n_steps=20, density_kg_m3=1300,
    loads_at=lambda t: (mechanical_load(t), free_charge(t)),
    output_dir="GAUSS2_EXPERIMENT")
```

All values and boundary programs above must refer to the **same**
physical shell model as a Newmark comparison. Results should not be
presented as independently validated until physical and refinement
comparisons are completed.
