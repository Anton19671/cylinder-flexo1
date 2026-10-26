# Three-dimensional physical field recovery

All accepted and failed-iterate MVI/FDM states now include eleven actual
through-thickness stations, including z=-h/2 and z=+h/2.

The NPZ contains full Green--Lagrange strain (6 engineering components),
18 actual covariant strain gradients, work-conjugate second-Piola strain
stresses (6), 18 work-conjugate hyperstresses (Gz returns zero x/s columns),
electric field E and electric displacement D, det F, enthalpy density,
plus physical fields at both thickness faces.

Directly samples the existing KL/FSDT/Reddy/CUF TE1..4 kinematic kernels
and the existing common G3D/Gz enthalpy derivatives. MVI profile fields are
already reconstructed by the X/Y solver and not recomputed by an FDM solver.
The reconstruction is diagnostic only and does not enforce or claim a verified
3D traction-free face boundary law. Full mechanical face tractions still
need the appropriate second-gradient natural boundary operator.

This produces output; it does not change ranks, convergence, time integration,
grid operators or constitutive material.
