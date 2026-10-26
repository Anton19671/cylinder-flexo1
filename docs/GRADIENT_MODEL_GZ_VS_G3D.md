# Selectable thickness-only Gz versus complete covariant G3D flexoelectric theory

## Model meaning

Both modes are real constitutive branches of the SAME nonlinear
Green–Lagrange shell functional and thickness Gauss quadrature:

| Mode | Active covariant strain gradients | Entries |
|---|---|---:|
| **G3D** (default) | `E_{ij,x}`, `D_s E_{ij}`, `E_{ij,z}` | 18 |
| **Gz** (reduced) | `E_{ij,z}` only | 6 |

The engineering strain basis is
`[E_xx,E_ss,E_zz,gamma_xs,gamma_xz,gamma_sz]`.
The full cylindrical rotating-basis corrections and
`H(z)=1+z/R` are retained in the geometric **G3D**
kinematic gradient for transparency. In **Gz**, constitutive
energy and its Gateaux derivative receive the projection
`Pz grad(E) = [0,0,grad(E)[:,z]]`.
Consequently **both** in-plane strain-gradient contributions and
their associated hyperstress vanish. No artificial in-plane
gradient elasticity or flexoelectric source is left behind.

The constitutive enthalpy is exactly
`H = W_classical(E) + W_gradient(P grad E) - 0.5 kappa |Efield|^2
     - mu1 (Efield dot trace(P grad E))
     - mu2 (Efield dot div(P grad E))`
with `P=I` (G3D) or `P=Pz` (Gz). In Gz the active invariants are
`t=(0,0,g_xx,z+g_ss,z+g_zz,z)` and
`b=(0.5*g_xz,z, 0.5*g_sz,z, g_zz,z)`,
with engineering shear convention. All three *electric-field*
components remain physical; Gz does **not** mean artificially
forcing `E_x=E_s=0`. A through-thickness shear strain gradient
can still induce tangential electric polarization through mu2.

Constitutive derivatives use the projection chain rule:
`M_Gz = Pz * (dH/d(Pz grad E))`.
In particular `M[:,x]=M[:,s]=0` for Gz.
The electric displacement and direct/converse flexoelectric
forces follow consistently from the same projected enthalpy.
No independent coupling law is postulated.

## Scope

The material `G3DMaterial.gradient_model` (default `"G3D"`)
is passed through CUF TE1--TE4, FSDT/Timoshenko, Reddy and
nonlinear KL without changing their geometry or their inertial
and Kelvin--Voigt constitutive maps. Static and Newmark dynamic
MVI/FDM paths, both dense and CSR where supported, use the same
variational residual and the same selection. Full G3D remains the
scientific reference; Gz is an optional simplified comparative model.

The CASE selector is `micro.gradient_model`:

```json
"micro": {
  "gradient_model": "Gz",
  "enable_flexoelectric": true,
  "ell_m": 0.08,
  "mu1_c_per_m": 0.001,
  "mu2_c_per_m": -0.002,
  "kappa_f_per_m": 1.5,
  "rho_kg_m3": 1300,
  "eta_s": 0,
  "potential_order": 1
}
```

The GUI exposes `G3D / Gz` and preserves the selection on
CASE editing. Summary outputs label both the model and the number
of active strain-gradient components (18 or 6). Example cases:
- `config/CASE_REFERENCE_CUF_GZ_STATIC.json`;
- `config/CASE_REFERENCE_CUF_GZ_MVI_DYNAMIC.json`.

There is no new material parameter for Gz. `ell_m`, `mu1`,
`mu2`, `kappa` are the same inputs, enabling fair model
comparison. `enable_flexoelectric=false` works for either model
without adding artificial electric variables.

## Verification status

Mathematical implementation and deferred tests are provided in
`tests/test_gradient_model_gz.py`:
- numerical verification of all 6 strain, 18 gradient and 3 electric
  enthalpy derivatives for both modes;
- exact disappearance of in-plane hyperstress in Gz;
- survival of direct/converse flexoelectric coupling from thickness
  gradients;
- static/dynamic consistency for all seven shell theories.

These tests have **not** been executed on the private GitHub
repository. They do not establish numerical convergence of MVI/FDM,
global boundary accuracy or engineering-scale stability.
