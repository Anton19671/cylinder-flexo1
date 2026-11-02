# KL modal Cammalleri audit: numerical defect and model compatibility

## Confirmed numerical defect

The previous `_LocalMechanicalTangent.__init__` in
`numerics/cylinder_modal_mvi.py` sampled the linear KL weak force at
surface-jet perturbations of +/-2e-6. The KL geometry constructs
F = I + displacement_gradient; `strain_jets` subsequently subtracts I.
Cancellation in small diagonal strains contaminates the weak-force columns.

For the Cammalleri dimensions and nine thickness quadrature points, local
entries (20,25) and (25,20) were 65.87816337618163 and 65.88074308665811.
The exact linear variational Hessian gives 65.88074303761823 and
65.88074303762005. This is not an empirical stiffness correction.

The Fourier block before Hermitian symmetrization agreed with the full K
action to 3.264184e-12 for rejected mode 5 (n=7). Symmetrization of the
contaminated block introduced a normalized action difference of
1.135207e-6; the axial projection difference was 6.824700e-17.

Commit 2eb6e46d55c9c15906c08781a2f197e2824a4378 assembles the linear KL
local Hessian directly from virtual strains and the same material energy.
It does not change the constitutive law, boundary conditions or 1e-6
full-mode certificate. Nonlinear KL and other theories keep their paths.

GitHub run https://github.com/Anton19671/cylinder-flexo1/actions/runs/41451191077
passed 12 regression tests and the 12-mode Cammalleri case.
This does not certify all MVI ranks or mesh convergence.

## Remaining constitutive mismatch with the paper

Primary source: Cammalleri and Castellano (2023), Designs 7, 122,
https://doi.org/10.3390/designs7060122, equation (5) and its definitions:
membrane stiffness K = E h / (1 - nu^2), bending stiffness
D = E h^3 / [12 (1 - nu^2)].

These are the plane-stress shell coefficients. The current KL normal
kinematics impose epsilon_zz = 0. The constitutive implementation
`enthalpy_derivatives` in `flexoelectric/flexo_gradient_g3d.py`
uses the 3D Lamé law. Its restriction to the in-plane strains therefore
has C11 = lambda + 2G and C12 = lambda, rather than the paper's
Q11 = E/(1-nu^2) and Q12 = nu E/(1-nu^2).

For E = 207 GPa, nu = 0.3:

| Coefficient | Current 3D law restricted to epsilon_zz=0 | Paper plane stress |
| --- | ---: | ---: |
| 11 (GPa) | 278.653846153846 | 227.472527472527 |
| 12 (GPa) | 119.423076923077 | 68.241758241758 |

C11 is 22.5% greater. This is a coefficient comparison, NOT a frequency
correction or proof that it explains the whole observed frequency gap.
The frequency change depends on modal strain energy, coupling, curvature,
inertia and discretization. Do not divide frequencies by an empirical factor.

For classical macro elasticity, eliminating the transverse normal strain
under sigma_zz=0 gives the Schur complement
Qaa = Caa - Caz * Czz^(-1) * Cza.
That reduction has NOT been applied to the production solver. Extending it
to G3D/flexoelectricity or nonlinear geometry requires its own derivation;
a global replacement of Lamé coefficients would be unjustified.

## Validation status and next checks

- Numerical Fourier certification failure: reproduced; corrected case passed.
- Published ANSYS comparison: preliminary, +7.82%, +8.62%, +9.26% for n=5,6,7.
- Axial branch matching: still requires inspecting the saved mode fields.
- Full matrix versus Fourier: independent audit script and workflow added.
- Mesh convergence: existing Cammalleri and Rawat workflows must finish.
- Other theories and MVI ranks: report only completed regression outcomes.
- Do not treat a computation-only green job as agreement of physical models.
