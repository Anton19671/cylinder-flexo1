# Unified G3D and flexoelectric mathematics — seven shell models

**Development state:** assembled in the small-grid `VARIATIONAL_REFERENCE`
path. Numerical end-to-end validation, grid/time convergence, and a
production sparse solver are **not yet complete**. Do not advertise full
engineering certification.

## 1. Common reference configuration and coupled enthalpy

Reference coordinates are (x,s=R\theta,z). The cylindrical volume element
is (dV_0 = H(z)\,dx\,ds\,dz), (H(z)=1+z/R).

For each shell model independently evaluate its exact displacement
gradient (F), Green–Lagrange strain (E=(F^TF-I)/2), six engineering
strain components (e), and all **three** covariant strain derivatives
(g_{i\alpha}=\nabla_\alpha e_i) with (α=x,s,z).

The shared material electric enthalpy (SI) is

[
\mathcal H(e,g,\mathcal E)=\frac{\lambda}{2}(\operatorname{tr}E)^2
  +G\,E:E
  +\ell^2\left[\frac{\lambda}{2}|t|^2+G\,g:g\right]
  -\frac{\kappa}{2}|\mathcal E|^2
  -\mu_1\,\mathcal E\cdot t
  -\mu_2\,\mathcal E\cdot b,
]
where (t_\alpha=\nabla_\alpha\operatorname{tr}E) and
(b_i=\nabla_j E_{ij}).
Use engineering-shear factors as implemented in
`flexoelectric/flexo_gradient_g3d.py`, not an independent duplicate
constitutive equation.

The reference electric field is
[
\mathcal E=-\left(\phi_{,x}, H^{-1}\phi_{,s},\phi_{,z}\right).
]
Thus (\mathcal H) generates **both** directions of flexoelectric
coupling: (\mu_1,\mu_2) contribute to electric displacement and
mechanical hyperstress. Its exact local derivatives are assembled into
the same discrete stationarity residual for **both** MVI and FDM.

The total quasistatic stationarity functional is
[
\Pi_h(q,\phi)=\int_{A_0}\int_{-h/2}^{h/2}
H\mathcal H\,dz\,dA_0-W_{\rm ext}(q,\phi).
]
The electric potential is a **saddle** variable. Stationarity is not
joint minimization in (q,\phi); global electric gauge and free-charge
compatibility must be enforced.

## 2. Kinematics and connected gradients by model

| Model | Mechanical fields | Surface jets for G3D | Thickness field |
| --- | --- | --- | --- |
| Kirchhoff–Love | (u,v,w) (3) | up to third spatial derivatives, nonlinear unit normal | (q+z(n-e_r)) |
| Timoshenko/Mindlin FSDT | (u,v,w,\theta_x,\theta_s) (5) | up to third spatial derivatives | independent linear directors |
| Reddy TSDT | same 5 | up to third spatial derivatives | curvature-corrected cubic warping |
| Carrera CUF TE1 | 6 | up to second derivatives | (\sum_{k=0}^1\xi^k(u_k,v_k,w_k)) |
| Carrera CUF TE2 | 9 | up to second derivatives | same, (k\le 2) |
| Carrera CUF TE3 | 12 | up to second derivatives | same, (k\le 3) |
| Carrera CUF TE4 | 15 | up to second derivatives | same, (k\le 4) |

Here (\xi=2z/h). Electric potential is independently approximated
(\phi(x,s,z)=\sum_{j=0}^P\xi^j\phi_j(x,s)), (P=0\ldots4) when the
electric switch is ON.

The mechanical coefficients and electric coefficients are *derivatives
of the same thickness-integrated enthalpy*, then assembled with the
transposes of the very same surface derivative matrices used in the
forward jets. This is required for the discrete Gateaux identity and
reciprocity of direct/inverse flexoelectric coupling.

Current implementations:
- `mechanics/cylinder_kl_g3d_weak.py`
- `mechanics/cylinder_director_g3d.py`
- `flexoelectric/cuf_g3d_weak.py`
- `mechanics/cylinder_functional_assembly.py`

## 3. Dynamic terms

CUF: thickness-integrated constant (in reference coordinates) mass
moments for the complete TE ansatz.

FSDT and Reddy: thickness-integrated mass action on both fields and
the director derivative contributions. Reddy must not be reduced to a
constant 5-by-5 lumped mass matrix.

KL finite rotations: (n=(A\times B)/|A\times B|), with
(A=e_x+q_{,x}), (B=(q_{0,s},1+q_{1,s}+q_2/R,q_{2,s}-q_1/R)).
For (U=q+z(n-e_r)):

[
\ddot U=\ddot q+z\left(Dn(q)[\ddot q]+D^2n(q)[\dot q,\dot q]\right).
]

The second term is the geometric/gyroscopic director acceleration.
The exact weak inertial action is

[
\delta\mathcal I =
\int_{A_0}\int_{-h/2}^{h/2}
\rho H(z)\ddot U\cdot(\delta q+zDn(q)[\delta q])\,dz\,dA_0.
]

The curved reference thickness moments are
[
m_0=\rho h,\quad m_1=\rho h^3/(12R),\quad m_2=\rho h^3/12.
]
The assembled discrete action uses surface adjoints of the director
variation and is implemented in
`mechanics/cylinder_kl_exact_inertia.py`.
The Newmark effective residual for KL uses the current `q, v, a`
and does not replace the nonlinear director by its undeformed value.

Electric inertia is zero in this quasi-electrostatic model. Mechanical
mass-proportional damping (C=\alpha_M M(q)) is a distinct optional
model and must **not** be called the material gradient viscosity
`micro.eta_s`.

## 4. Independent physical switches (no hidden substitutions)

- **MACRO:** \(\ell=\mu_1=\mu_2=\kappa=0\); purely classical elastic.
- **MICRO, gradient ON, flexo OFF:** \(\ell>0\);
  \(\mu_1=\mu_2=\kappa=0\); zero electric unknowns.
- **MICRO, gradient ON, flexo ON:** \(\ell>0,\kappa>0\);
  coefficients \(\mu_1,\mu_2\) independently configurable.
  Both direct and inverse flexoelectric derivatives appear.
- A MICRO case is NEVER dispatched through an ordinary MACRO solver.

## 5. Remaining engineering blockers

1. Full dynamic and coupled-shell numerical runs for all seven theories
   using MVI ranks 1–8 and independently assembled full FDM solutions.
2. Configuration-dependent KL inertial tangent and discrete energy-work
   audit over physical time steps; current reference Newton differentiates
   the **full effective residual**, so a separate hand-coded tangent has
   not been asserted as verified.
3. Separate thermodynamically admissible material gradient viscosity
   (nonzero `micro.eta_s` currently blocked) with a defined dissipative
   potential and consistent nonlinear variation.
4. Strong/weak higher-order natural traces, electrode conditions, and
   exhaustive manufactured solutions; load regularization studies.
5. Grid, quadrature and time-step convergence; sparse engineering scale;
   GUI choices consistently routed into this same functional.
