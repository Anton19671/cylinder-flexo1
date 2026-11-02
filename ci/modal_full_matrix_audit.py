"""Independent full-matrix audit of the modal Fourier coordinate reduction.

Run only in GitHub Actions: explicit full projected K/M, an independent
full trace QR null space, and a dense generalized symmetric eigensolve.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, qr
from cylinder_paths import activate
activate()
from cylinder_modal_reference import _modal_spec
from cylinder_modal_matrixfree import _projected_actions
from cylinder_matrixfree_fdm import SparseTraceProjector
from cylinder_variational_fdm import state_scales
from cylinder_modal_fourier_fdm import solve_modal_fourier_fdm

def run(nx, ns, thickness):
    root = Path(__file__).resolve().parents[1]
    source = root / "config/verification/rawat2016/modal_linear/fdm_13x13/RAWAT2016_KL_13x13_MODAL_FDM.json"
    case = json.loads(source.read_text())
    case["geometry"] = {"L": .305, "R": .076, "h": thickness}
    case["material"] = {"E0": 207e9, "nu0": .3}
    case["modal"].update(density_kg_m3=7833, n_modes=12)
    case["fdm_grid"] = {"nx": nx, "ns": ns, "operator_scheme": "SBP42_FOURIER_ODD_CSR"}
    prepared, density, _ = _modal_spec(case)
    q = prepared.mechanical
    scales = state_scales(prepared.problem.functional, q, None, prepared.settings)
    p = SparseTraceProjector.from_problem(
        prepared.problem, q, None, scales, time_s=None,
        max_rows=prepared.settings.max_constraint_rows)
    k, m, scales, conversion, mass_scale = _projected_actions(prepared, density, p)
    # Coordinate columns of the FULL operators, no axial/Fourier reduction.
    identity = np.eye(q.size)
    K, M = k @ identity, m @ identity
    c = p.c.toarray()
    if len(c):
        orth, tri, _ = qr(c.T, mode="full", pivoting=True)
        diagonal = np.abs(np.diag(tri))
        rank = int(np.count_nonzero(diagonal > 1.e-11 * diagonal.max()))
        assert rank == p.count
        basis = orth[:, rank:]
    else:
        basis = identity
    trace_error = float(np.max(np.abs(c @ basis), initial=0.))
    kr, mr = basis.T @ K @ basis, basis.T @ M @ basis
    symmetry = float(np.linalg.norm(kr-kr.T)/np.linalg.norm(kr))
    assert symmetry < 1.e-10
    vals, vecs = eigh((kr+kr.T)/2, (mr+mr.T)/2,
                      subset_by_index=(0,min(15,kr.shape[0]-1)), driver="gvx")
    direct = basis @ vecs
    direct_freq = np.sqrt(vals*conversion)/(2*np.pi)
    result = solve_modal_fourier_fdm(
        prepared, density, 12, p, k, m, scales, conversion, mass_scale)
    freqs = np.asarray(result["frequencies_hz"])
    vectors = np.asarray(result["mechanical_modes"]).reshape(12,-1).T
    vectors = vectors * np.sqrt(mass_scale) / scales[:,None]
    def quality(vector, value):
        kv, mv = k@vector, m@vector
        return float(np.linalg.norm(kv-value*mv) /
            max(np.linalg.norm(kv)+abs(value)*np.linalg.norm(mv),1.e-30))
    full_quality = [quality(vectors[:,j],(2*np.pi*freqs[j])**2/conversion) for j in range(12)]
    dense_quality = [quality(direct[:,j],vals[j]) for j in range(12)]
    gram_error = float(np.max(np.abs(vectors.T@M@vectors-np.eye(12))))
    relative_frequency = float(np.max(np.abs(freqs-direct_freq[:12])/direct_freq[:12]))
    # Compare invariant eigenspaces, allowing rotations within doublets.
    captured_mass = []
    for j in range(12):
        matching = np.abs(direct_freq-freqs[j])/freqs[j] < 1.e-7
        overlaps = direct[:,matching].T @ M @ vectors[:,j]
        captured_mass.append(float(np.sum(overlaps**2)))
    report = dict(grid=[nx,ns],thickness_m=thickness,
        fourier_hz=freqs.tolist(),dense_hz=direct_freq[:12].tolist(),
        circumferential_n=result["circumferential_wavenumbers"],
        full_KM_quality=full_quality,dense_KM_quality=dense_quality,
        relative_frequency_difference=relative_frequency,
        captured_mass_in_dense_eigenspace=captured_mass,
        mass_gram_error=gram_error,trace_basis_error=trace_error,
        reduced_K_asymmetry=symmetry)
    passed = (max(full_quality+dense_quality)<1.e-6 and
        relative_frequency<1.e-7 and gram_error<1.e-6 and
        min(captured_mass)>1-1.e-6 and trace_error<1.e-9)
    report["passed"]=bool(passed)
    path=root / f"modal_full_matrix_{nx}x{ns}_h{thickness:.6g}.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print("[FULL_MATRIX] "+json.dumps(report),flush=True)
    return passed

if __name__=="__main__":
    nx,ns=int(sys.argv[1]),int(sys.argv[2])
    h=float(sys.argv[3]) if len(sys.argv)>3 else .000254
    raise SystemExit(0 if run(nx,ns,h) else 1)
