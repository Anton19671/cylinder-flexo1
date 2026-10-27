# Independent Gauss–Legendre IRK4 dynamics reference

## Mathematical contract

The existing **MVI X/Y + Newmark** path remains the primary production
reference. The new `numerics/cylinder_gauss2_rk.py` module implements an
independent **two-stage, fourth-order, implicit Gauss–Legendre** time
collocation for generic second-order electromechanical equations.

With two-stage matrix A, abscissae c and weights b,

```
c = ((3-sqrt(3))/6, (3+sqrt(3))/6)
A = [[1/4, 1/4-sqrt(3)/6],
     [1/4+sqrt(3)/6, 1/4]]
b = [1/2, 1/2]
```

The **unknown** mechanical accelerations a_i and electric potential
coefficients psi_i are solved **together** for both stages. Stage
kinematics for the full mechanical surface fields are

```
v_i = v_n + dt * sum_j A_ij * a_j
q_i = q_n + dt * c_i * v_n + dt^2 * sum_j (A^2)_ij * a_j
```

The user-supplied physical stage operator must enforce at every internal
time t_n+c_i*dt:

```
M(q_i)*a_i + physical_forces(q_i,v_i,psi_i,t_i) = 0
electrical_stationarity(q_i,psi_i,t_i) = 0
```

At step end, q and v are assembled with the Gauss weights. With electric
physics ON, the endpoint electric potential must be solved from its
independent **algebraic** electrical equations at t_(n+1): the stage-2
potential is not automatically a valid endpoint potential.

## Implemented and intentionally NOT implemented

**Implemented:** coupled Gauss-stage nonlinear equations, dense reference
Newton method with explicit scales, full-field velocity and acceleration,
electrical algebraic stages, independent end electric callback, and
manufactured tests for order four, coupled dynamics, and moving stage
boundary conditions.

**Not integrated:** a real cylinder-shell stage callback for all seven
theories, fixed-rank separated MVI stage parameterization, trace/gauge
elimination at both stage times, post-step rank-R endpoint compatibility,
GUI selection, results export, or entire physical runs.

In particular, a generic Gauss collocation step on unrestricted nodal
accelerations need **not** preserve rank R of q_(n+1) even when its stage
displacements have rank R. Calling the current module 'MVI + RK' would be
incorrect. A genuine rank-fixed MVI/IRK path must be derived and tested
as a two-stage coupled variational X/Y solve, with endpoint compatibility.
There is no silent projection of IRK results onto rank R.

The existing MVI/Newmark code and numerical acceptance behavior remain
unchanged. This module is an **opt-in autonomous temporal reference only**;
no CASE or GUI routes silently switch to it.

## Validation

Run isolated tests without shell simulations:

```bash
python -m unittest tests/test_cylinder_gauss2_rk.py
```

After a genuine strong-coupled shell stage adapter is implemented,
compare matching physics, time-dependent mechanical/electric BCs, and
complete q/v/psi histories against Newmark with independent temporal
refinement. Gauss IRK4 linear-order results do not by themselves certify
nonlinear DAE order, moving boundary enforcement, or low-rank spatial
accuracy.
