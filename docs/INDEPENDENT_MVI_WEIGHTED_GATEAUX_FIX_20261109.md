# Independent MVI weighted Gateaux correction — 2026-11-09

## Confirmed source defect and correction

In `independent_mvi/coupled_surface.py`, the original directly
assembled frozen-partner gradients used the following contracted
quadrature weights:

- X factor variation: circumferential weights ws only;
- Y factor variation: axial weights wx only.

Both gradients were therefore missing the weight associated with their
OWN varying coordinate. In particular, the axial weights of the SBP42
diagonal norm are not constant at the boundary. This is a material
mathematical defect: the X and Y gradients could fail to represent the
first variation of the discrete surface energy. The independently
assembled full nodal first variation already used wx*ws, so the
internal factor-to-node chain-rule guard was capable of exposing it.

The correction supplies **wx*ws to both** directly assembled
frozen-partner gradients, before multiplication by the transpose of
the respective one-dimensional derivative matrix. The resulting
factor gradient contracts the same weighted nodal first variation
against the opposite fixed factor. No continuum constitutive formula,
geometry, MVI rank, or MKR code was changed.

## Regression prevention

`tests/test_independent_coupled_mvi_weighted_gateaux.py` adds:
- a constant physical local virtual coefficient whose exact X/Y
  gradients MUST contain the nonuniform SBP42 axial boundary weights;
- a mixed x-s derivative manufactured quadratic functional, checked
  by centered directional differentiation of the MVI energy;
- full factor-to-nodal discrete chain-rule consistency.

The new test is included in the existing manual no-physical-case
GitHub Actions algebra workflow.

## Verification status and scientific implication

The changes are committed to main. The tests have **NOT been executed**
in the current conversation. No loaded cylinder cases were launched.
Consequently, existing previously generated MVI nonlinear/dynamic
results based on the pre-correction native coupled surface assembler
must NOT be treated as verified without recalculation after code tests
and boundary-condition validation.

High-order natural/microfree boundary conditions and electric face
flux/gauge closure remain unverified. The production GUI/CASE dispatch
is deliberately untouched.
