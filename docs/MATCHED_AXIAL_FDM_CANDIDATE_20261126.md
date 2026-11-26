# Matched axial strong-FDM candidate, 2026-11-26

The current native strong route remains unchanged. Its existing trapezoid
quadrature, sampled boundary rows, and unmatched derivatives are not
certified. Replacing only a flux derivative by its weighted transpose
would not establish consistency. In particular, a trapezoidal norm
cannot support exact differentiation of all quadratics by both members
of an endpoint Green pair: test q=x and A=x^2, for which the integrated
product derivative is 3*x^2 and trapezoidal quadrature is not exact.

## Independent explicit difference construction

All matrices below are dimensionless (physical derivatives divide by the
mesh increment raised to the derivative order). Let H be the positive
diagonal SBP(6,3) norm, B=diag(-1,0,...,+1). Construct C with
C+C^T=B, the standard centered sixth-order interior, and polynomial
conditions through cubic at the first six nodes. Use the one remaining
free upper triangular entry C[4,5]=139/200, near the least-norm choice.
The rational closure is listed in the implementation. This is a choice
of discrete stencil, not a physical constitutive parameter.

Let E be the undivided fourth forward difference, S=E^T E/128.
P=H^-1(C-S), M=H^-1(C+S) are explicit dual differences.
Since E annihilates cubics, P and M differentiate cubics exactly.
P^T H=B-H M and M^T H=B-H P.

Use kinematic matrices K1=P, K2=M P, K3=P M P and explicit
coefficient differences F1=M, F2=M P, F3=M P M.

Direct multiplication gives the exact identities:

- K1^T H = -H M + B.
- K2^T H = H M P - B P + P^T B.
- K3^T H = -H M P M + B P M - P^T B M + K2^T B.

Thus the second-order displacement end flux must use P (not M).
The third-order displacement end flux uses P M and the third-order
slope flux uses M. Existing common-F_k end formulas cannot simply be
reused. Physical scaling uses H_phys=h H, K_j, F_j scaled by h^-j,
and the indicated end flux derivatives scaled by their respective orders.

For the interior alternating sequence, P q=-2q, M q=2q,
K2 q=-4q, K3 q=8q in dimensionless units. This avoids a zero or
wrong-sign curvature response at Nyquist. The fourth-difference
coefficient 1/128 sets the magnitude of the interior first-difference
Nyquist response to 2/h.

## Local verification

Eleven isolated NumPy/pytest tests passed. The relative matrix defects
for orders one, two and three are about 9.35e-17, 2.24e-16 and 4.12e-16.
Polynomial reproduction, positive quadrature, correct derivative
nullity, even/odd-grid Nyquist response, and refinement are tested.
For exp(x) on [0,1], maximum kinematic/coefficient derivative errors
including endpoints decrease with approximately orders 3, 2 and 1
respectively. This is NOT a third-order boundary-accurate third derivative.

## Integration gate

This candidate is not dispatched or substituted into any solver.
Integration requires consistent quadrature, actual mixed-jet and
boundary trace implementation, independent manufactured second/fourth/
sixth-order boundary-value solutions, physical KL coupled orders,
electric compatibility and gauge weights, and static/dynamic shell
comparisons. In particular, exact Green work before row replacement
does not prove stability of a strong row-replacement BVP.
The dense prototype is limited to 15..257 nodes, and needs sparse
production assembly before large-grid deployment. No shell solve
or full-suite CI was performed for this change.
