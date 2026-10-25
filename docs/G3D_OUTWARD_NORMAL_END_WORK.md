# Outward-normal higher-order end work for G3D cylindrical shells

Development record; assembled in small-grid reference variational solver,
not yet validated by full engineering mesh or convergence studies.

## Geometric orientation

Use reference axial coordinate x in [0,L] and shell area measure ds
on both ends. The outward axial unit normal is n_x=-1 on x=0 (left),
n_x=+1 on x=L (right). Therefore for a field component q_k:

  d_n q_k = n_x*d_x q_k
  d_nn q_k = n_x^2*d_xx q_k = d_xx q_k.

A prescribed generalized end load has virtual work

  delta W_end = int_end [
       t_k delta q_k + m_nk delta(d_n q_k)
                       + h_nnk delta(d_nn q_k)
     ] ds.

The nodal external potential is -W_end; its exact first variation
contains transpose Dx/Dxx operators weighted by circumferential
quadrature. No penalty term is added.

## Explicit CASE keys

- `traction`: conjugate to q, independent of the end orientation.
- `double_traction`: **legacy** conjugate to d_x q, same axial coordinate
  at left and right. Preserved without silently changing existing CASEs.
- `normal_double_traction`: conjugate to d_n q; its contribution changes
  sign on the left end.
- `triple_traction`: **legacy** conjugate to d_xx q.
- `normal_triple_traction`: conjugate to d_nn q = d_xx q; equal
  orientation factor at the two ends.

Choose either the axial or the outward-normal convention for a derivative
order, not both. CUF TE1–TE4 have second-order surface jets, so nonzero
triple tractions remain unavailable. KL/FSDT/Reddy use third-order jets,
so the mathematical generalized second-normal trace is representable.

These are generalized traction *work inputs*, not an automatic physical
reconstruction of 3D Mindlin hypertractions. Deriving fully natural
microfree boundary closures needs the integration-by-parts surface
boundary terms of the specific G3D and model kinematics, including
boundary edge/corner contributions.

## Coupling

The same `compile_boundary_program` supplies STATIC and DYNAMIC,
MVI and FDM via `combine_functional_and_boundary`. In the dynamic route
the explicit time-law multiplier applies to the generalized load.
All flexible electrical potential/charge constraints are unchanged.

The regression suite `tests/test_variational_boundary.py` now includes
left/right orientation equivalences and failure on ambiguous mixed keys;
the full integrated tests remain to be executed once assembly is complete.
