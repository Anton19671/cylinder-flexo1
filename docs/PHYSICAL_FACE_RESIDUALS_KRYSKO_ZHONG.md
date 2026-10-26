# Physical face field recovery and electric natural flux (26 October 2026)

The main variational MVI/FDM solver writes actual 3D face displacements
for KL (nonlinear midsurface normal), FSDT, Reddy and CUF TE1--TE4.
Unlike the midsurface coefficient w0, the outer radial displacement can
depend on the thickness polynomial and nonlinear KL normal.

For G3D and reduced Gz, physical induction D is evaluated at the two
reference faces z=-h/2 and z=+h/2 from the constitutive law. At each face:

    D_outward = n_z * D_z,  n_z=-1 (inner), +1 (outer).
    residual = D_outward - prescribed_outward_induction.

The variational boundary input CHARGE means prescribed outward induction
because its work is minus charge times potential on the physical face.
INSULATED is a zero-flux natural condition. POTENTIAL imposes an
essential voltage: its D_outward is a reaction and has no natural
electrical flux residual.

RMS residuals use the physical face area element (1+z/R)*dA_mid.
The user may define a fixed positive electric induction scale for
normalized face comparisons. If unspecified, no artificial scale
or normalized residual is invented.

Example optional CASE:
    "postprocess": {
      "thickness_stations": 11,
      "electric_induction_scale_c_per_m2": 0.001
    }

The choice 0.001 in this example is NOT a physical default.
Each static/dynamic step saves full fields to its NPZ and a
face_conditions diagnostic in FINAL_RUN_SUMMARY.json.
The mechanical strong face traction check is explicitly
NOT_IMPLEMENTED_FULL_SECOND_GRADIENT_FACE_OPERATOR:
classical stress alone is NOT the full G3D mechanical face traction,
which also requires higher tractions, curvature and edge terms.

This is postprocessing only. It never modifies X/Y factors,
acceptance, fixed ranks 1--8, electromechanical equations or
Newton/Newmark time integration.
