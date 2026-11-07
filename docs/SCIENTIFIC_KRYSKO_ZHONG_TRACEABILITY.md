# Scientific traceability of Krysko-Zhong CUF TE1-TE4

**Reference:** `Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx`
(24 October 2026). This matrix describes code coverage, not a declaration
that the full physical boundary-value problem is solved or validated.

| Source equation group | Implemented evidence | Verification status |
|---|---|---|
| (1)-(18): cylinder metric, displacement gradients, six GL strains | `cuf_g3d_weak.py`, `point_state`, analytic finite rigid rotation tests | Local algebra under test; full-field verification remains |
| (19)-(33): all 18 covariant gradients | `flexo_gradient_g3d.py`, independent tensor-connection commutator tests | Local algebra under test |
| (34)-(56): material electric field, total enthalpy, direct/inverse flexoelectric coupling | `enthalpy_parts`, `enthalpy_derivatives`, finite-difference Gateaux and mixed tangent reciprocity across G3D/Gz | Local constitutive and weak derivatives under test |
| (61)-(82): TE1-TE4 dimensionless thickness basis, mass and odd moments | `te_basis`, `mass_matrix_te`, `gauge_moments`, existing analytic tests | Static algebra covered; complete modal/dynamic benchmark pending |
| (83)-(85): nominal outer-reference-face radial work | Corrected independent MKR/MVI `_loading`; `test_independent_outer_face_load.py` | Algebra covered; revised physical histories pending |
| (88)-(132): virtual deformation and thickness-integrated weak coefficients | `thickness_weak_coefficients`; 24-way test over 4 TE orders, 3 geometry models, 2 gradient models | Local weak Gateaux covered; global strong form and boundary terms pending |
| (135)-(136): global electric gauge | `gauge_moments` and constant-potential invariance tests | Local gauge invariance covered; global constrained solve pending |
| (192)-(220): essential, micro-free and electrical boundary conditions; consistent face operators | `cylinder_g3d_natural_edges.py`, `cylinder_boundary_spatial.py`, boundary/electric tests | **Not independently certified** for all seven models and higher-order edges |
| (202)-(210): coupled Newmark/Newton tangent | Dynamic operator tests and mixed local tangent symmetry | Local symmetry checked; full Newton/time order pending |
| (213)-(221): compatible face derivatives and checkerboard mode | `periodic_face_gradient`, existing even-grid tests | Algebra covered; full nonlinear convergence pending |
| (227)-(240): time histories, stress recovery, surface work | Various exporters, earlier reference test suites | **Not yet validated as a comprehensive publication-grade benchmark** |

The new `test_scientific_krysko_zhong_audit.py` tests *material and
local weak-form identities* only. Its numerical finite differences are
independent of the code's hand-derived first variations, but do not constitute
external experimental or 3D-FE validation. Publishing new coupled
flexoelectric results requires matching the exact source material and boundary
conditions of the external reference, documenting mesh/rank/TE-order/time-step
convergence, and verifying local stress/induction recovery.

**Current independent branches limitation:** `independent_mkr` and
`independent_mvi` implement experimental linear electrical-off shells only.
The full nonlinear G3D/Gz research implementation lives in the coupled
mechanics/electric modules; passing independent-branch architecture tests
must never be described as full flexoelectric verification.


## Manufactured axial natural boundary work (source §§22, 30–33, 40–41)

An independent continuum integration-by-parts manufactured test now
covers all second- and third-order surface derivative positions (including
`q_xs`, `q_xxs`, `q_xss`) one-by-one and in combination. The comparison
uses low Fourier modes, exact polynomial derivatives, independent Gaussian
quadrature, both axial end orientations, natural single/double/triple
tractions and the electric natural flux. This verifies the **continuum
boundary work algebra**, not convergence or stable imposition of
discrete microfree boundary conditions in the full nonlinear solver.

Because the circumferential direction is physically periodic for a
*closed cylinder*, there are no independent angular boundary corners in
the benchmark. Open shell patches require separate corner/edge checks.


## November 7, 2026: numerical MVI halfstep split

See `docs/COUPLED_METHOD_READINESS_20261107.md` for a precise
status matrix. Static X/Y Newton, dynamic Newmark/Bathe X/Y Newton,
and Gauss2 MVI scaling/settings no longer import FDM numerical
solvers. Shared continuum electromechanical physics still exists,
and independent full high-order G3D/Gz edge closure has not been
certified. This split must not be advertised as full method physics
independence. Only a manual, structural verification workflow is
provided; this change does not trigger physical computation.
