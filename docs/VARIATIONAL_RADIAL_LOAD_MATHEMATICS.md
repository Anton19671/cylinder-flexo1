# Variational radial loading mathematics — implementation status

For the reference midsurface of a circular cylinder use coordinates
\(x\in[0,L]\), \(s=R\theta\), and \(z\in[-h/2,h/2]\).
The face metric is \(H(z)=1+z/R\). A fixed-reference radial surface traction
\(p(x,s)\) acting on a face \(z=z_f\) has external virtual work

\[
\delta W_{\rm ext}=\int_0^L\int_0^{2\pi R}
H(z_f)\,p(x,s)\,\delta u_r(x,s,z_f)\,ds\,dx.
\]

This is **DEAD_REFERENCE**, not follower hydrostatic pressure.
The total load functional is \(\Pi_h=\Pi_{\rm internal}-W_{\rm ext}\).
Hence its exact discrete load gradient on a surface grid with quadrature
\(W_{ij}=w^x_i w^s_j\) is \(-W_{ij} f_{k,ij}\).
The same load array enters both full-nodal FDM and alternating MVI and the
same load field is multiplied by the externally specified Newmark time law.

## Carrera TE-N
\[
u_r(x,s,z)=\sum_{k=0}^N\xi^k w_k(x,s),
\qquad \xi=2z/h.
\]
Thus \(f_{w_k}(x,s)=H(z_f)\xi_f^k p(x,s)\).
In the component-major field order (all u coefficients, all v, all w),
\(w_k\) is component \(2(N+1)+k\).
This includes correctly the sign alternation on INNER and the factor
\(H(z_f)\). All other generalized components receive zero force.

## FSDT and Reddy
The implemented director ansatz has \(u_r=w_0\) independent of \(z\),
so \(f_{w_0}=H(z_f)p\). Rotation variations are absent from this
radially oriented *dead* load, not from the internal bending/shear equations.

## Kirchhoff–Love
At the midsurface \(f_w=p\).
An off-midsurface **normal-transported** KL director depends on
\(u,v,w\) and their first surface derivatives. Its load work therefore
cannot be replaced by \(H(z_f)p\delta w\).
This route rejects INNER and OUTER for KL until those extra geometric
virtual-work terms (and any needed consistent tangent) are implemented.

## Spatial load types

* UNIFORM_RADIAL_FACE: constant face pressure \(q\) [Pa].
* HARMONIC_RADIAL_FACE: \(q\sin(m\pi x/L)\cos(n s/R+\phi)\).
* PATCH_RADIAL_FACE: pressure multiplied by a C1 finite-support window.
* LINE_RADIAL_GENERATOR: physical line force \(p_{\rm line}\) [N/m]
  distributed over an explicitly specified finite circumferential band.
* POINT_RADIAL_FORCE: resultant \(P\) [N] spread by finite-width C1
  compact kernels with explicitly specified physical radii in x and s.

For resultant-controlled loads we normalize using the **same** face metric
and the **same** numerical quadrature as the functional:
\[
p_{ij}=\frac{P\,K_{ij}}
{\sum_{a,b} W_{ab}\,H(z_f)\,K_{ab}},
\qquad \sum_{ij}W_{ij}H(z_f)p_{ij}=P.
\]
For generator loading substitute \(P=p_{\rm line}(x_2-x_1)\).
Kernel radii/band widths do not depend silently on the grid. Grid resolution
must be sufficient; otherwise CASE validation rejects the load.

## What this implementation does NOT claim

* A Dirac point load, a zero-width generator line or a grid-independent
  converged local singular stress field.
* Follower loads, updated surface normals, current-configuration pressure,
  external electromechanical forces, contact, thermal loading or foundation.
* Automatic verification of mesh convergence, full model agreement or
  the physical validity of arbitrary boundary constraints.

The new implementation is
`physics/cylinder_variational_loads.py`, connected to the common
`interface/cylinder_reference_case.py` in STATIC and DYNAMIC.
Independent tests belong in `tests/test_variational_loads.py`.
**Do not mark the entire solver as verified before the later full test stage.**
