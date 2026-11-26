# Independent strong FDM with explicit natural-trace SAT

## Problem and remedy

A discrete Green identity before boundary closure does not guarantee
stable collocation after replacing additional interior equations by
pointwise natural boundary rows. A separate axisymmetric LINEAR
electromechanical pilot on the exact matched-quartic axial matrices
found strong grid sensitivity for Reddy/Gz with this row replacement.
The pilot uses the source small-rotation thickness kinematics and
elastic/strain-gradient/dielectric/flexoelectric enthalpy. It is a
separate linear reduction, not a full native runner or GL verification.

Let r be the explicit strong bulk residual, H the positive axial norm,
T_k the outward normal slope or curvature trace, a_k its conjugate
action and g_k the prescribed natural action. With fixed displacement
at both ends add

    r_sat = r + H^(-1) sum_(k>=1) T_k^T (a_k - g_k).

Only displacement end equations are subsequently replaced. This
identity follows from the discrete integration-by-parts formula for
the actual matched kinematic and coefficient derivatives. The bulk
is still differentiated explicitly; no weighted-transpose interior
assembly, reference solve or MVI import is introduced. Algebraically,
on the fixed-displacement test space the result equals the discrete
first variation of the same enthalpy, so it is NOT an independent
comparison against a variational method using these identical jets and
norm. The natural conditions are imposed via work, NOT claimed to be
pointwise satisfied on a finite grid.

## Locally computed independent reduced physical pilot

L=1.2 m, R=2 m, h=.08 m, E=1e9 Pa, nu=.3, ell=.002 m,
mu1=.001 C/m, mu2=-.0005 C/m, kappa=1e-8 F/m, rho=2500 kg/m^3.
Twelve thickness Gauss points, potential order 1 for KL/FSDT/Reddy,
equal to TE order for CUF. Uniform outer radial traction -100 Pa,
zero prescribed displacement and electric modes at both axial ends,
natural higher traces. Axisymmetric LINEAR reduction, constant load
and one average-acceleration Newmark step of 1e-4 s from rest.

Reddy/Gz center radial displacement (micrometres):

| nx | pointwise row replacement | SAT |
|---:|---:|---:|
| 21 | -2.726143 | -3.448516 |
| 41 | -2.451114 | -3.477688 |
| 81 | -7.040082 | -3.512978 |
| 121 | -3.884593 | -3.528094 |
| 161 | -3.709219 | -3.534431 |
| 241 | -3.611546 | -3.538480 |

All seven theories and both G3D/Gz at nx=21,41,81 (42 combinations)
were evaluated in the independent pilot in static and one-step
dynamic form with SAT. The maximum relative matrix work defect
against an independently assembled weighted work matrix was 5.81e-16.
The center displacement grid trend improved; this does not establish
mesh convergence of every field, especially the electrical maxima.
Ten isolated NumPy/pytest SAT tests passed locally on the exact
committed quartic matrices (blob 41bec75d77e65f3860a4cbd4827e5876f54495a7).
The repository-wide source checkout and actual native integration tests
were not available for local execution.

## Explicit research option

Native coupled strong CASE grid field:
    "axial_scheme": "MATCHED_QUARTIC_SAT_V1"

20..257 axial nodes. Same matched-quartic derivative matrices and norm;
fixed mechanical displacement on both sections, all higher traces
natural, prescribed higher actions via the existing mechanical end
work program, omitted higher natural targets zero. Electric OFF or
GROUNDED_BOTH only. Floating/one-sided-grounded electric and higher
essential mechanical traces explicitly rejected. Energy FDM and MVI
are unchanged. The original legacy and matched pointwise modes remain.

Initial acceleration uses the same SAT-corrected mass/bulk equations:
no natural interior rows are overwritten, and no incompatible extra
pointwise-natural gate is imposed. Essential q,v,a traces are checked.
Newmark/Bathe/Gauss use the same boundary assembly, but their physical
nonlinear integration still needs a full checkout and testing.
No continuum microfree boundary or production accuracy certification.
