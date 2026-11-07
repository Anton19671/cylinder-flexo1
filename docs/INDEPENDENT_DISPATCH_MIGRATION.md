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
