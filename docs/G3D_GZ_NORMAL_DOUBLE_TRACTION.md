# Face mechanical field recovery: conjugate double traction, not full face traction

As part of §47, the physical G3D/Gz recovery now computes:
- Full 3x3 **classical** first-Piola tensor P_classical = F S.
  It intentionally omits the higher-gradient terms of the effective
  nominal traction.
- The normal-normal double traction vector g_a = (F M_z)_{a z}, where
  M_z is the work-conjugate symmetric tensor built from the engineering
  hyperstress derivatives M_{ij,z}. In physical units g has Pa*m = N/m.
- Both inner and outer faces carry the same n_z*n_z sign in g, while
  classical outward Piola tractions use n_z=-1 at z=-h/2 and n_z=+1 at
  z=+h/2. Engineering shear conjugates must not be halved a second time.

These quantities are calculated for the SAME MVI/FDM displacement,
gradient and electric fields as the enthalpy. Gz has no tangential
strain-gradient columns, but can have nonzero M_z and therefore g.

For the complete generalized mechanical strong-face condition, the
operator must still add divergence of tangential hypertractions,
covariant curvature effects and edge-line work. Until it is implemented,
the JSON field mechanical_face_strong_check remains explicitly
NOT_IMPLEMENTED_FULL_SECOND_GRADIENT_FACE_OPERATOR.
Only the partial normal-normal double traction is available.

Also corrected a fatal postprocessing mismatch: Carrera CUF PointState
exposes det_f while KL/FSDT/Reddy use determinant. Both are now read
with the proper attribute, preventing valid CUF solves from failing
while writing physical state fields.
