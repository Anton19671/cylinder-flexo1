# Native MKR field preparation and seven-theory assembly

Implementation: 2026-11-08. Source physics remains the Krysko--Zhong
corrected CUF algorithm and the repository's separate KL/FSDT/Reddy local
kinematic laws. This change removes reference spatial calls from the
experimental direct-strong MKR route; it does not replace the physical model.

## Implemented code

- `cylinder_mkr_field_jets.MKRFieldJets`: own mechanical/electric field
  differences, with cached mixed/higher jets and no shared spatial matrices.
- `cylinder_direct_strong_model.DirectStrongShell`: own grid/state validation,
  local-jet construction, thickness-coefficient collection and direct strong
  bulk/end assembly for KL, FSDT, Reddy and CUF TE1--TE4.
- `DirectStrongShell.solve`: explicit stationary Newton entry point for all
  seven theories, with caller-supplied orders, end equations, scales and gauge.
- `CUFParallelEndProblem`: now uses this native route and its own quadrature,
  including the microfree end program and single global electric gauge.
- `kl_inertial_coefficients_from_jets`: pure physical KL inertia from supplied
  displacement, velocity and acceleration jets. The reference matrix API
  delegates to the same physical law; native MKR supplies its own derivatives.

Instantaneous coefficient assembly includes conservative elasticity,
G3D/Gz gradients, reciprocal flexoelectric coupling, quasi-static electric
equations, Kelvin--Voigt mechanics, inertia and mass damping. For finite
rotation KL the director's velocity-square acceleration is retained.
Mass damping evaluates M(q)*v without including that velocity-square term.
CUF inertia remains on primary nodes, including even angular checkerboards;
it is never interpolated away at faces. No electric inertia is introduced.

## Numerical convention preserved during extraction

The former direct-strong reference adapter used `surface_trapezoid` to
prepare fields. Native MKR preserves that convention explicitly:

- x: centered second-order first difference, three-node one-sided end rows;
- s: forward periodic first difference;
- higher/mixed field jets: compositions of those first differences;
- coefficient differentiation: the existing direct-strong seven-point axial
  derivative and periodic backward/compatible-curvature operators.

These are two distinct stages: field preparation and coefficient divergence.
The backward electric divergence pairs with the forward potential gradient
and retains the even-grid alternating mode. This fact alone does **not**
prove complete coupled face placement, mixed-jet work identities or a
production high-order end closure. The field second derivative has not been
silently changed into a different direct-curvature stencil. This is an
architectural extraction with preserved numerical behavior, not a claim of
a newly certified finite-difference scheme.

There is no import/call of `CylinderFunctional`, `SurfaceOperators`, MVI
assembly, or MVI Newton in the native path. Shared local continuum physics
is intentional. The old `direct_strong_from_functional` and
`solve_direct_strong_reference` remain explicit legacy APIs; existing GUI
dispatch is not silently switched to the experimental native route.

## Minimal construction (does not run a calculation)

```python
from cylinder_direct_strong_fdm import DirectStrongGrid
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_cuf_parallel_ends import CUFParallelEndProblem

# material is an existing validated G3DMaterial; orders are supplied after
# checking the coupled model's physical differential order per field.
model = DirectStrongShell("CUF_TE2", material,
    grid=DirectStrongGrid(nx=11, ns=8, length_m=length, radius_m=radius),
    thickness_m=thickness, n_gauss=12, geometry_model="GREEN_LAGRANGE")
problem = CUFParallelEndProblem(model, axial_orders=orders, potential_order=2)
```

`model.coefficients`/`model.balance` evaluate a prescribed instantaneous
state, including optional mechanical velocity/acceleration. They do not
advance time. A real solve requires an explicit `solve` call and dimensional
state/equation scales. No physical run is started by construction or import.

## Verification and remaining work

The added tests compare prescribed nonzero field coefficients against the
old uniform-trapezoid route for every theory in G3D/Gz. That comparison checks
preservation during extraction; it is not independent physical validation.
Additional checks cover linear/von Karman geometry, analytical polynomial
and discrete Fourier symbols, coupled static/dynamic electric consistency,
positive CUF checkerboard inertial work, zero-state solver wiring, and execution
with reference/MVI imports or spatial derivative callbacks forbidden.

Local run completed: 81 tests and 48 subtests passed in 63.13 seconds,
including the new native module, CUF end integration, existing dynamic
Green splitting, Kelvin--Voigt and quasi-static electrical contracts,
reference functional checks, and the direct-strong Newton/gauge/boundary
tests. No loaded-shell solution or time series was computed.

Still required: the full source-compatible nonlinear face-flux discretization
and high-order end convergence; separate coupled MVI spatial assembly; native
time-integration adapters and initial acceleration handling; CASE/GUI dispatch;
and the user's physical convergence/comparison series. This commit does not
declare the whole solver ready for scientific production calculations.
