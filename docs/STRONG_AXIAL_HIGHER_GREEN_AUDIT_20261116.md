# Flexo1 cylindrical native FDM: first-, second-, third-order axial Green audit

Scope: \`Anton19671/cylinder-flexo1\` main, the experimental native
**strong** FDM route. This is *not* a nonlinear shell verification.

## Derivation

Let \(Q_j\) denote the actual MKR kinematic derivative of axial order
\(j\), \(F_j\) the actual seven-point coefficient (flux) derivative,
\(H=\mathrm{diag}(w_x)\) the existing trapezoidal weights,
and \(B=\mathrm{diag}(-1,0,\ldots,0,+1)\).
\(Q_0=F_0=I\).

For one physical constitutive coefficient \(A_k\) multiplying
\(\delta(\partial_x^k q)\), the **discrete** kinematic weak work is

\[
\delta q^T Q_k^T H A_k.
\]

The current strong FDM bulk balance and continuum-sampled end actions
give the discrete nodal work

\[
\delta q^T\left[(-1)^k H F_k+
\sum_{j=0}^{k-1}(-1)^{k-j-1}Q_j^T B F_{k-j-1}\right]A_k.
\]

Here the \(j=1\) and \(j=2\) traces use the *same* MKR kinematic
operators as the material strain jets; the outward signs are
\(n_x B_0\), \(B_1\), \(n_x B_2\) for virtual
\(q,d_nq,d_{nn}q\). Mixed/circumferential jet reductions do not
make this missing axial identity disappear.

The exact matrix-level defect for each \(k=1,2,3\) is their
difference. The new \`audit_native_higher_axial_green\` uses **the
actual code paths** for \(Q_j\) and \(F_j\), not recoded stencils.
Regressions independently reconstruct local virtual work through
\`direct_strong_jet_balance\` and both physical torques/tractions.
No physical case, load history or time integration is run.

## Interpretation

Nonzero defects show that the present strong flux discretization with
sampled continuum boundary tractions is **not** the weighted adjoint
of its actual kinematic derivative family. This is a verified
*discrete operator mismatch*, not evidence of a wrong continuum
constitutive law or proof of physical instability.

The positive remedy requires a **single matched derivative/norm/trace
family**, including third-order jets and second-order normal traces.
Simply replacing \(Q_2\) by \(Q_1^2\) is not acceptable: it can
restore a formal SBP chain but create the axial checkerboard null
previously detected. A compatible narrow second derivative and a
third-order closure need independent derivation and stability/polynomial
tests. The independent MVI SBP/Fourier solver must not be imported or
used to solve the FDM equation.

Until that development is complete, strong physical natural end rows
and the G3D/Gz dynamic FDM solver remain **EXPERIMENTAL / NOT
BOUNDARY CERTIFIED**, even when their Newton residual is small.
