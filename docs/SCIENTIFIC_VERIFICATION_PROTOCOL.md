# Mandatory scientific verification protocol — cylinder-flexo1

**Primary mathematical specification:** `Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx` (24 October 2026), the owner's corrected reference derivation for **geometrically nonlinear flexoelectric cylindrical microshells**. This is a verification checklist, not proof that the implementation is already correct. The separate five-field Reddy–Green–Lagrange audit of 8 October and the MKR/MVI independence audit of 4 November remain supplementary and must never replace the primary CUF equations.

## A. Primary physics-to-code trace: corrected Krysko–Zhong CUF TE1–TE4

1. **Geometry and displacement:** check `H(z)=1+z/R`, signed reference-face area factors, nine cylindrical physical displacement derivatives, and CUF polynomial `F_k=(2z/h)^k` for TE1–TE4 (§§3–4, 13–14, Eqs. 1–9, 61–73).
2. **Green–Lagrange strain:** derive all six components from `F.T@F` at every quadrature point, including `E_zz`; verify zero strain for finite rigid rotations (§5, Eqs. 10–18, §48). Do not treat the reduced Reddy plane-stress law as if it were full CUF thickness kinematics.
3. **Covariant G3D/Gz strain gradients:** test all 18 covariant physical gradient components (and the precise six active in Gz), especially cylindrical rotation-basis terms; compare direct and batched implementations (§6, Eqs. 19–33).
4. **Electro-mechanical enthalpy:** independently verify 3D Hooke energy, gradient terms, material electric field, both direct/inverse flexoelectric couplings, three electric induction components, and six stress/18 hyperstress components by derivative tests (§§7–11, Eqs. 34–56). G3D and Gz are distinct constitutive models, not postprocessing switches.
5. **Damping and inertia:** verify Kelvin–Voigt dissipative work, the exact curved-cylinder CUF thickness moments including nonzero odd moments, and a mechanically constrained initial acceleration (§§12, 15, 34).
6. **Reference outer-face pressure:** enforce **every radial CUF coefficient** `Q_(w,k)=-pressure_pa*H_out*F_k(h/2)` for positive inward pressure; here `F_k(h/2)=1` and `H_out=1+h/(2R)`. This is the corrected document's Eqs. (83)–(85), not a mid-surface point load (§16). Check mechanical virtual work and the stated sign convention.
7. **Weak/strong equations:** derive the full variations of all mechanical and electrical fields, the mixed-derivative coefficients, the thickness-integrated six mechanical coefficient families, and the unique global electric gauge. Confirm 6+2, 9+3, 12+4, 15+5 coupled local equations for TE1–TE4 (§§17–29).
8. **Boundary closure:** verify the full essential torques/tractions and higher-order **micro-free** axial trace for the chosen end supports. Do not impose `q_x=0` alongside a micro-free natural higher-order condition without variational justification. Keep angular periodicity and electric boundary/gauge conditions (§§30–35, 41).
9. **Independent discretizations:** check that the MKR strong route employs compatible face gradients/divergences and resolves the even-grid Nyquist/checkerboard mode; MVI must preserve separated X/Y sweeps and genuine rank limitation. Both must act on the *same physical source* with independent spatial algebra (§§39–40.1).
10. **Time integration and publication gates:** verify consistent nonlinear Newton and Newmark, Bathe and Gauss stages, the complete coupled mechanical/electric tangent, physical manufactured solutions, frequencies, fields, stress recovery, rank/grid/time-step convergence, and independent WQEM/other external benchmarks after matching model passports (§§37–48).

**Status:** neither the full coupled flexoelectric model nor the independent seven-theory prototype is fully scientifically validated merely because CI passes. In particular, the independent `linear_models.py` branches implement only *linear, electrical-off* mechanics; full G3D/Gz claims must be audited on the original coupled modules. The 8 October five-field Reddy GL proof remains a separate supplementary track.

## B. Numerical independence and dynamics: MKR versus MVI

1. Trace actual entry points, operators, assembly routes and every
   integrator (Bathe, Newmark, two-stage Gauss IRK4); there must be no
   hidden invocation of the opposite solver or a shared spatial
   variational solve. Separate CASE, logs, and output folders.
2. Verify MVI `X/Y` factor initialization, frozen-partner
   `Y -> X` sweeps, updates and field-preserving normalization, while
   retaining the *separated-function MVI algorithm* (never substitute
   the full-grid MKR solution).
3. Rank oracle: use a manufactured nonseparable **rank-two field**.
   MKR/full-grid space and MVI rank 2 or 8 can represent its spatial
   rank (subject to the admissible boundary subspace); MVI rank 1
   cannot reproduce it exactly. Rank-capacity claims must be checked
   with the actual clamped basis dimensions.
4. Repeat algebraic checks across KL, FSDT, Reddy and CUF TE1–TE4,
   and check stage/intermediate and endpoint factors independently
   for Bathe, Newmark and Gauss IRK4.
5. Validate spatial convergence, temporal order, rank sensitivity,
   reference frequencies, static deflections and time histories using
   **independent external benchmarks**. MKR-vs-MVI agreement alone
   is not an external accuracy proof.
6. Exercise stop/pause, durable accepted-step files, crash recovery
   and restart; guarantee metadata reflects the exact accepted
   solution. Never claim successful restart without an actual test.

## C. Coupled flexoelectric research scope

Independent `independent_mkr` and `independent_mvi` seven-theory
routes are currently experimental **linear, non-electrical** mechanics.
Do not infer that they validate the original coupled G3D/Gz/flexoelectric
code. Its electromechanical energy, gradients, reciprocity, electric
boundary conditions, units, and exact physical benchmarks require a
separate staged audit.

## Status and rules

- **Protocol recorded:** yes.
- **Every physical equation symbolically checked:** not yet established.
- **Full flexoelectric solver scientifically certified:** no.
- Preserve existing GUI layout, tabs and controls unless a specific
  interface defect is separately confirmed.
- Prefer small, non-physical algebraic tests first; submit the smallest
  necessary GitHub Actions physical verification only when justified.
- Do not report doubtful, unverified numerical reference values as facts.
