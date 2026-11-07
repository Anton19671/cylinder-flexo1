# Independent-only dispatch migration — preserved legacy until certified

The official `run_final_case` and `validate_final_case` dispatcher
now accepts **native** independent method CASE schemas:
`INDEPENDENT_MKR_SEVEN_MODELS_V1` and
`INDEPENDENT_MVI_SEVEN_MODELS_V1`.
The source `schema` is authoritative; MKR calls only
`independent_mkr.run_models`, MVI calls only
`independent_mvi.run_models`. No shared VARIATIONAL_REFERENCE
fallback is permitted from an independent CASE.

These native routes support seven theories (KL, FSDT, Reddy, CUF
TE1–TE4), linear MACRO isotropic HOOKE, **no electric or gradient**
physics, and Bathe/Newmark/Gauss IRK4. The associated real solver
uses its own schema, own saved results and own method-specific physics.
A full MICRO nonlinear flexoelectric case is rejected rather than
silently given a linear independent solution.

The older `VARIATIONAL_REFERENCE` path is **not removed yet**.
It remains explicitly selectable and intact for ongoing physics
verification and prior reproducibility. Its automatic deactivation
requires a new fully coupled independent MKR/MVI model that supports
all corresponding physics and boundaries, plus real benchmark
validation. Until then, hiding it would destroy currently supported
physical functionality and scientific traceability.

The existing reference-to-independent bridges are not invoked
automatically. Their allowed `pressure_face` is preserved (MID
or OUTER as provided) and constitutive mismatch is rejected.
For pressure-face comparison see dedicated seven-theory weak-work
tests. Native JSON is required for the independent solver parser;
no dangerous ad hoc YAML conversion is introduced.

`tests/test_independent_native_final_dispatch.py` verifies all
42 distinct theory × method × time algorithms, dry-run only;
for selected calls, mocked runners assert no opposite-method call,
no reference solver and rejection of unknown schemas.

## Explicit 2D CUF contract and corrected outer-face export

Source: `Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx`,
sections 13, 36, 39, 40 and 46. The G3D designation describes the **three
physical directions of strain gradients** in the constitutive law. It does
**not** require a 3D discretization: spatial fields are coefficients on
`(x,theta)`; through-thickness dependence is analytical (CUF or classical
kinematics) with 12-point Gauss integration. A 3D field can be reconstructed
at selected `z` stations for output, without a spatial `nz` grid.

The **two independent linear runners** now record
`spatial_grid=AXIAL_CIRCUMFERENTIAL_2D`,
`thickness_scheme=ANALYTIC_KINEMATICS_GAUSS12_NO_Z_GRID`, and an explicit
linear macro Hooke/electric-off/gradient-off scope in validation and run
summaries. These attributes are duplicated by design, not dispatched via
another method's module.

For CUF TE1–TE4 the physical radial displacement on the outer surface
`z=+h/2` is `w_out(x,theta)=sum_k w_k(x,theta)`, whereas `w_0` is the
middle-surface radial coefficient. Both MKR and MVI independently export
`central_outer_w_m` and `maximum_abs_outer_w_m` **in addition** to the
historical `central_w0_m` and `maximum_abs_w0_m` CSV columns. For KL,
FSDT and Reddy the two fields coincide. The separate reference-face
pressure convention and all existing solver equations are unchanged.

This is an **export and architectural safety correction, not a claim of
full Krysko–Zhong implementation**. In particular, the native linear MKR
still needs the compatible face-gradient/flux construction and complete
higher-order natural-boundary certification from source section 40; it is
not a verified G3D strong solver. G3D/Gz, nonlinear Green–Lagrange and
fully coupled flexoelectric research remain in legacy reference routes,
which **are not two independent spatial discretizations**. Do not silently
fall back to a legacy solver when an independent CASE requests unsupported
physics. The AST isolation tests scan all native method modules (including
GUI and input adapters) for cross-method and shared numerical imports.

No physical shell calculation was run to prepare these changes.

## 2026-11-07 MKR end-adjoint correction (strictly limited)

The independent MKR linear nodal strong balance now uses the discrete
first variation of its OWN centered finite-difference surface energy.
At axial endpoints, derivative transpose plus trapezoid weights
ensures that the linear bending contributions are self-adjoint on the
prescribed displacement/slope spaces; the former unpaired second
axial difference was not. Native MVI SBP/Fourier operators and the
entire coupled legacy physics are unchanged. This step does not
claim that centered angular differences are a full implementation of
source §40's compatible face gradients. No physical case was run.
