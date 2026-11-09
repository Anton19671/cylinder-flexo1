# Native MKR time integrators: stationary-end state gate

Date: 2026-11-09. Files changed in main:

- numerics/cylinder_native_mkr_initial.py
- numerics/cylinder_native_mkr_newmark.py
- numerics/cylinder_native_mkr_bathe.py
- numerics/cylinder_native_mkr_gauss2.py
- tests/test_native_mkr_newmark.py
- tests/test_native_mkr_bathe.py
- tests/test_native_mkr_gauss2.py
- tests/test_native_mkr_support_traces.py

## Concrete defect corrected

The native Newmark and Bathe entry points previously accepted arbitrary
initial velocities and accelerations even when all axial essential support
traces were declared stationary. A supplied velocity or acceleration on a
clamped displacement trace is incompatible with a fixed support. A later
nonlinear solve cannot turn this input into a consistent transient initial
value problem.

All three native time integrators now use the same explicit physical-trace
preflight **before** their first physical nonlinear stage. For every
ESSENTIAL trace of axial normal order k=0,1,2, the existing MKR strong
spatial derivative gives physical outward-normal value at the left/right
end. The check requires:

1. Initial q normal traces equal the supplied, time-independent values.
2. Initial v normal traces equal zero.
3. Initial a normal traces equal zero.

No guessed default support or weak/MVI spatial derivative is introduced.
The initial-acceleration solver checks the first two before solving for
the unknown a0. Its accepted state is constrained by zero essential
acceleration trace equations; passing a known a0 into a time integrator
checks the third too.

Non-stationary or time-varying supports require a separate implementation
of prescribed boundary motion, its derivatives, and corresponding dynamic
end equations. This change refuses such data instead of treating them as
stationary.

## Verification scope

The previously added algebraic tests now use valid zero boundary v and a
where a stationary support is specified, and also deliberately inject
incompatible initial q/v/a to verify rejection before Newton. Independent
slope-trace tests check the physical normal sign on left and right ends
and rejection at the first derivative trace.

These test cases are present in the repository but have not been executed
in this conversation. This patch does NOT certify the coupled G3D/Gz
strong-equation spatial discretization or high-order boundary closure.
No physical shell calculation, time series, or Github Actions workflow was
started. The external container has no DNS access to clone github.com.
