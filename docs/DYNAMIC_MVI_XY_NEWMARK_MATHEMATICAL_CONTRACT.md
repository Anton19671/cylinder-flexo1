# Krysko-Zhong dynamic MVI: spatial X/Y factors + stepwise Newmark

**Decision (2026-10-27):** keep the existing **rank-fixed spatial
variational iteration method** for the coupled shell on *each physical time
step*, with implicit Newmark temporal integration. Do not silently append
independent time factors T_r(t) to the existing X/Y solver. A global
space-time separated algorithm is a different, not-yet-implemented method.

## Unknowns and time integration

For each generalized mechanical component a and electrical potential
coefficient j (when the flexoelectric problem is ON):

    q_a(x,s,t_n)   = sum_{r=1}^R X^q_{ar}(x,t_n) Y^q_{ar}(s,t_n)
    psi_j(x,s,t_n) = sum_{r=1}^R X^e_{jr}(x,t_n) Y^e_{jr}(s,t_n)
    R in {1,2,3,4,5,6,7,8} is fixed throughout one run.

The factors for different components are independent. At t_n they encode
one surface configuration. They are not a tensor-product approximation to
the entire three-dimensional domain (x,s,t).

Mechanical Newmark update, for the FULL nodal fields:

    q_pred = q_n + dt*v_n + dt^2*(1/2-beta)*a_n
    v_pred = v_n + dt*(1-gamma)*a_n
    a_(n+1) = (q_(n+1)-q_pred)/(beta*dt^2)
    v_(n+1) = v_pred + gamma*dt*a_(n+1)

Solve the coupled stationarity equations at each t_(n+1), including
Green-Lagrange elasticity, G3D/Gz, inertia with the actual KL/FSDT/Reddy/
CUF kinematics, damping only when selected, and both direct/inverse
flexoelectric coupling. Electric potential is a quasi-electrostatic saddle
unknown: it has **no** Newmark velocity or acceleration.

The mechanical velocity, acceleration, AND q_pred are FULL nodal arrays
and **must not be forcibly projected onto rank R**. Even an exact rank-one
displacement q(t) = X(t) Y(t) has

    d/dt q = X_dot Y + X Y_dot
    d^2/dt^2 q = X_ddot Y + 2 X_dot Y_dot + X Y_ddot.

Their spatial ranks can exceed R. Projecting them to rank R would change
the inertial problem and could lose the cross term 2 X_dot Y_dot.
The implementation instead updates the complete physical fields by
Newmark, so no profile-by-profile temporal derivative is inserted.

## Alternating variational iterations

At a fixed t_(n+1), X and Y are updated by the same coupled effective
Newmark equations, with the partner profile family frozen in each
half-step. The existing order is Y then X. Orthogonalization is only
a coordinate change of the represented field; it cannot change rank,
supply an additive full-grid field, or change physical constitutive laws.

Only accepted step profiles may initialize the next step. A predictor
q_pred need not itself admit rank-R separation. Exact rank-R
representability of a prescribed *initial displacement/electrical field*
is checked separately before creating the results directory.

Boundary traces are imposed for the physical step time; moving
mechanical essential supports must obey the corresponding Newmark
kinematics. Initial conditions currently support the ZERO model only;
arbitrary nonzero velocity/displacement are not advertised.

## Newmark parameters and limitations

The default beta=1/4 and gamma=1/2 is average acceleration.
Linear unconditional stability statements do **not** certify stability
or accuracy of an arbitrary strongly nonlinear low-rank projected shell.
Independent spatial-rank, mesh, quadrature, and time-step studies,
including physical displacement, stress, electrical induction, face
conditions, and dynamic trajectories, are still required.

## Why not T_r(t) now?

A *global space-time* approximation, such as

    q(x,s,t) = sum_r X_r(x) Y_r(s) T_r(t),

requires new time basis/test functions, initial-condition treatment,
and coupled nonlinear space-time variations. It cannot be implemented
by simply multiplying the current per-step X/Y solution by T_r(t).
It may be investigated separately under a new, explicitly labelled
space-time method. It must not replace or be mislabeled as the current
Krysko-Zhong stepwise MVI.

Relevant literature:
- Kirichenko & Krysko (1982), variational iterations for plates/shells,
  https://www.mathnet.ru/rus/ivm6817
- Krysko et al. (2023), nonlinear plate computational iterative methods,
  https://doi.org/10.1016/j.ijnonlinmec.2022.104346
- Bamer et al. (2021), *A Newmark space-time formulation in structural
  dynamics*, https://doi.org/10.1007/s00466-021-01989-4
  (distinct global space-time methodology, not evidence that the
  Krysko-Zhong solver requires temporal factors).

**Status:** implemented mathematical choice, NOT a statement that all
seven nonlinear theories, MVI ranks 1..8, and time-step grids are
numerically certified. Do not run a shell case merely to record the
contract.
