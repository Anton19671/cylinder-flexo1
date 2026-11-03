# KL C-C harmonic pressure: Newmark, independent FDM and MVI-1…8

First **physical dynamic** comparison after the accepted KL static matrix.

* Cylinder: KL linear isotropic; L=2 m, R=1 m, h=0.05 m, E=210 GPa, nu=0.3; rho=7800 kg/m3.
* Mechanical supports: C-C (including KL end slopes).
* Same 9x9 SBP42_FOURIER_ODD operator in all 9 CASEs.
* Applied radial inward **midsurface dead-reference** pressure: p(t)=100000 sin(2*pi*25*t) Pa, not follower pressure.
* All start from zero displacement and velocity. Consistent initial acceleration is computed from the constrained mass problem.
* Average acceleration Newmark (beta=0.25, gamma=0.5), dt=0.001 s, 2 physical time steps. No damping.
* Separate genuine MVI-1…8 fixed-rank X/Y solvers vs full nodal FDM/MKR branch; failure of any method must remain visible.
* Two time steps are a **minimal smoke** of the time integrator, not a time-step convergence result or an independent analytical validation.

The CI workflow `.github/workflows/kl-cc-newmark-physical.yml` first validates the CASEs, then runs all 9 real physical trajectories and compares full U,V,W, velocity and acceleration at every accepted step.

Next: temporal refinement 0.001 -> 0.0005 -> 0.00025 s over the SAME physical final time, and comparison with separate Gauss IRK4 integration. Do not label the current two-point history time converged.
