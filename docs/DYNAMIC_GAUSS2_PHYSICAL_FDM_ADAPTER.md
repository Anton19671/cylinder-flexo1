# Gauss IRK4 physical variational FDM stage adapter

The opt-in `numerics/cylinder_gauss2_variational_fdm.py` connects the
standalone two-stage Gauss--Legendre IRK4 integrator to the same
`BoundarySpatialProblem` used by the Newmark variational cylinder
reference. It is NOT automatically selected by CASE/GUI and is NOT
an implementation of fixed-rank MVI + Gauss IRK.

For each stage it evaluates the physical seven-theory Green--Lagrange
G3D/Gz mechanics, mass action, selected damping, quasi-static electric
stationarity, and the physical time-dependent load and boundary program.
The spatial stage equations use the admissible basis to separate the
variational stationarity equations from the complementary essential-trace
constraints. The potential at the endpoint is solved separately at the
final physical time rather than copied from the second Gauss stage.

For this initial physical adapter, mechanical essential support matrices
AND prescribed values must be stationary throughout a step. The displacement
and velocity at the beginning of the step must satisfy them. If a support
moves, the adapter explicitly refuses the step: a valid IRK treatment
needs consistent moving-support position and derivative conditions at the
internal stages and final time. There is no unadvertised poststep
projection of displacement or velocity.

Electrical Dirichlet and gauge rows may change with stage time as defined
by the existing boundary program; the end potential must satisfy the
electrical equations with the mechanical endpoint field held fixed.

The physical adapter is dense/small-grid **FDM**. Mechanical and electric
stage fields, and in particular the final q field, are *not* projected to
a fixed separation rank. To implement genuine MVI + IRK4, derive a
two-stage coupled rank-fixed X/Y variational algorithm that handles
stage-dependent supports and guarantees the representability of endpoint
displacement/potential with unchanged rank. Do not rename the existing
nodal adapter 'MVI + IRK'.

The existing Newmark and MVI backend remains untouched.
