# Physical Bathe time integration, parallel to existing Newmark

Bathe (gamma = 2-sqrt(2)) is an **order-two** implicit two-substep method
with high-frequency algorithmic damping. Not Gauss IRK4 and not order four.

Common KL C-C physical CASE: linear isotropic cylinder L=2m, R=1m,
h=0.05m, E=210GPa, nu=.3, rho=7800kg/m3, 9x9
SBP42_FOURIER_ODD; uniform inward midsurface pressure
p(t)=100000 sin(2*pi*25*t) Pa, zero initial state.
Two full steps of dt=0.002s (four genuine physical variational
substep solves) for each MVI rank 1..8 and FDM control.

CASE choice: dynamics.integrator=BATHE, independent output folders.
This does not modify the Newmark nor Gauss IRK4 CASE settings.
The second substep uses the absolute time t_n+h (NOT t_n+2*gamma*h).
Stored physical velocities and accelerations are unrestricted fields,
while each mechanical displacement endpoint is represented by exactly
the selected MVI rank. There is no post-step truncation, SVD,
Gram, or hidden spatial substitution.
The CASE route always calls the original Y->X variational MVI solver
for each physical implicit substep.

Acceptance requires two successful stage solves, full admissible
kinematic and trace checks and a fixed-rank MVI certificate.
One-step failures are archived, not silently accepted or replaced.

This is a linear KL physical verification only, not validated yet for
nonlinear GL, moving supports, active damping or coupled electrical CASEs.
Rank-independent agreement here is not a nonseparable rank study.

Run using:
python RUN_FINAL_CASE.py config/verification/kl_cc_bathe/BATHE_MVI1.json
(through standard physical reference CASE dispatcher).
