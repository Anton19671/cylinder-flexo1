# Quartic-exact axial strong-FDM research option

The cubic-exact candidate from the preceding commit satisfies the
discrete Green identities but does not provide a consistent sampled
fourth-derivative natural action for sixth-order equations. On the
manufactured microfree scalar BVP with u=exp(x)+sin(2x),
(-1)^m D^m(a D^m u)+pi^(2m)u=f, a=1, m=3, its maximum absolute
error was 0.916, 0.748, 0.683 at nx=41,81,161.

The new independent candidate uses an eighth-order centered interior,
quartic-exact left/right closures and a positive diagonal norm.
The dual correction uses the undivided fifth difference E and S=E^T E/512.
The Green identities and the distinct end fluxes of the previous derivation
are unchanged. The exact rational boundary entries are stored in
cylinder_matched_axial_quartic.py. The free closure parameters are numerical
stencil choices, not material parameters.

Local NumPy/SciPy verification of the exact quartic coefficient module,
including the scalar row-replacement boundary closure, gave the following
max-norm errors for a=1+0.3x and prescribed displacement plus natural
higher traces at both ends:

| nx | second order | fourth order | sixth order |
|---:|---:|---:|---:|
| 21 | 1.22084e-5 | 7.53974e-5 | 3.233997e-2 |
| 41 | 3.98067e-7 | 1.87151e-5 | 1.438419e-2 |
| 81 | 1.24708e-8 | 2.92802e-6 | 6.974410e-3 |
| 161 | 3.88398e-10 | 4.04803e-7 | 3.404474e-3 |

The scalar tests also examined constant a and essential higher traces.
The sixth-order essential result reaches a roundoff/conditioning plateau
on the 161-point grid. These are manufactured scalar tests, not a
cylindrical-shell accuracy or stability certificate.

The scheme is available **only as explicit opt-in** in native
strong research CASE JSON: add "axial_scheme":"MATCHED_QUARTIC_V1"
to grid. It requires 20..257 axial nodes. The previous default remains
LEGACY_TRAPEZOID; MVI and native energy FDM are unchanged. Both the
constitutive jets and explicit coefficient differences use the chosen
scheme, end fluxes use their correct paired derivatives, and the area
quadrature (including electric gauge/charge checks) uses the same norm.
All existing fieldwise differential-order and physical BC caveats
remain. No automatic switch, full-shell run or full-suite CI is claimed.
The repository tests additionally exercise actual strong assembly and
CASE preflight. These require verification in the complete checkout
before scientific use.
