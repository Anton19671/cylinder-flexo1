# Native coupled MVI: curved-shell physical electric face charge and induction

Date: 2026-11-09. Research-only extension of the separated fixed-rank
X/Y G3D/Gz seven-theory shell solver.

## Physical work and sign convention

At the real cylindrical faces z=-h/2 (inner) and +h/2 (outer), the
reference shell metric factor is H(z)=1+z/R. Electric potential uses
the normalized expansion

    phi(x,s,z) = sum(k=0..P) G_k(z) psi_k(x,s),
    G_k(z)=(2z/h)^k, P=0..4.

The boundary contribution for physical prescribed free electrode charge
sigma [C/m² of actual physical face area] uses the existing model
enthalpy convention +sigma*phi dA_face. Thus, for each generalized
electric potential coefficient psi_k, its reference midsurface charge
load is

    f_k(x,s) = sum(faces) H(z_face)*G_k(z_face)*sigma_face(x,s).

The inner and outer sigma carry the SAME positive-charge work sign:
there is no extra outward-normal sign on prescribed sigma. This differs
from the normal induction check: outward D_n=n_z D_z, with n_z=-1
(inner) and +1 (outer), and the diagnostic natural condition is

    n_z D_z + sigma = 0.

Every term is derived from the same local G3D/Gz electric enthalpy
coefficient and normalized CUF electric basis; mechanical theories
KL/FSDT/Reddy/CUF TE1-TE4 differ only in the constitutive state.

## Implementation

- independent_mvi/coupled_electric_faces.py:
  `face_generalized_free_charge` converts scalar or
  (nx,ns) charge densities on the two physical faces into all P+1
  generalized electric source terms. INSULATED contributes zero.
  `recover_mvi_electric_faces` evaluates the same local physical
  kinematics and `enthalpy_derivatives(...)[2]` at z=+/-h/2,
  with full curvature, G3D or reduced Gz effects. It reports
  physical face areas, outward induction, sign-correct natural
  mismatch, weighted RMS and absolute maximum. The returned arrays
  are immutable.
- independent_mvi/run_coupled_research.py:
  Optional top-level `electric_faces` object with both keys
  `inner` and `outer`. Each may be null (insulated), or a
  strict object of shape
  `{"mode":"INSULATED","charge_c_per_m2":null}`, or
  `{"mode":"CHARGE","charge_c_per_m2":<scalar or (nx,ns) data>}`.
  Face charge is ADDED to the existing generalized `free_charge`
  array from loads_npz before any static or dynamic solve.
  The existing NPZ field is still required for an electric-on case
  and may contain the zero baseline. The input remains unchanged.
  The run summary gets `electric_faces`, `initial_electric_faces`
  or `last_electric_faces` diagnostics before writing any accepted
  state. Existing CASEs without the optional face section are
  backward-compatible with zero physical surface free charge.

## Deliberate scientific restrictions

Potential-controlled inner/outer faces are NOT yet implemented.
A request for `POTENTIAL` is rejected before a calculation: adding
potential constraints requires separate X/Y-compatible constraint
operators, correct treatment of face electrodes and charge reactions.
Floating gauge is NOT implemented in this independent MVI route;
the explicitly grounded axial electrical end mode still fixes datum.

A polynomial-through-thickness Ritz/weak shell does not in general
satisfy a natural pointwise face induction equation at each z face.
The recovered D_n+sigma is a DIAGNOSTIC only and is NOT enforced as
an independent strong face condition, an automatic solver acceptance
criterion or a claim of physical correctness.

Mechanical hypertraction/microfree conditions and radial face/edge
mechanics remain to be derived and validated. There is no MKR
numerical fallback, SVD or full-grid displacement Newton in this
new MVI feature.

No loaded physical cylinder solve or time history was executed by
the assistant. The no-physical-solve GitHub CI separately tests
all 7 theories in G3D/Gz, exact curvature-weighted charge work,
inner/outer normals, matching prescribed charge, invalid face
conditions and backward-compatible CASE preflight.
