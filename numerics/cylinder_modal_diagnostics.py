"""Read-only physical diagnostics of saved shell eigenvectors.

This module does not solve an eigenproblem or change the variational model.
MAC here uses the Euclidean coefficient metric. A consistent-mass MAC needs
the solver's mechanical mass operator and is reported separately.
"""
from __future__ import annotations

import numpy as np


def _component_groups(modes, theory):
    name = str(theory).upper()
    count = modes.shape[1]
    if name.startswith("CUF_TE") and name[-1:] in ("1", "2", "3", "4"):
        degree = int(name[-1]) + 1
        if count != 3 * degree:
            raise ValueError("CUF modal component count does not match theory")
        return (modes[:, :degree], modes[:, degree:2*degree],
                modes[:, 2*degree:3*degree]), True
    if name in ("KL", "FSDT", "REDDY"):
        if count != (3 if name == "KL" else 5):
            raise ValueError("Modal component count does not match theory")
        return (modes[:, 0:1], modes[:, 1:2], modes[:, 2:3]), False
    raise ValueError("Unrecognized modal theory")


def inspect_modal_shapes(fields, frequencies_hz, theory, *, operator_scheme="UNKNOWN"):
    """Mode type, circumferential harmonic, Euclidean MAC and doublets.

    An exactly repeated eigenfrequency does not imply that the corresponding
    eigenvectors are duplicates. Degenerate shell modes must be tested by
    both their spatial phase and their independence.
    """
    q = np.asarray(fields, dtype=float)
    f = np.asarray(frequencies_hz, dtype=float)
    if (q.ndim != 4 or q.shape[0] != f.size or f.ndim != 1 or
            not q.size or min(q.shape[2:]) < 2 or
            not np.isfinite(q).all() or not np.isfinite(f).all()):
        raise ValueError("Invalid modal fields or frequencies")
    groups, _ = _component_groups(q, theory)
    flat = q.reshape(len(f), -1)
    norms = np.linalg.norm(flat, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Zero modal field")
    unit = flat / norms[:, None]
    mac = np.square(np.clip(unit @ unit.T, -1., 1.))
    modes = []
    for index in range(len(f)):
        energies = np.array([float(np.sum(g[index]**2)) for g in groups])
        fraction = energies / np.sum(energies)
        dominant = int(np.argmax(energies))
        combined = sum((np.abs(np.fft.rfft(g[index], axis=-1)) ** 2).sum(axis=(0, 1))
                       for g in groups)
        harmonic = int(np.argmax(combined))
        component = groups[dominant][index]
        spectrum = np.fft.rfft(component, axis=-1)[:, :, harmonic]
        where = np.unravel_index(int(np.argmax(np.abs(spectrum))), spectrum.shape)
        phase = (float(np.degrees(np.angle(spectrum[where])))
                 if harmonic else None)
        if harmonic == 0 and fraction[1] > .80:
            kind = "AXISYMMETRIC_TORSION"
        elif fraction[2] > .20:
            kind = "BENDING_OR_MIXED"
        elif fraction[0] > .80:
            kind = "LONGITUDINAL"
        else:
            kind = "CIRCUMFERENTIAL_OR_MIXED"
        modes.append({
            "mode": index + 1, "frequency_hz": float(f[index]),
            "harmonic": harmonic, "angular_phase_deg": phase,
            "dominant_component": ("U", "V", "W")[dominant],
            "mode_type": kind,
            "u_energy_fraction": float(fraction[0]),
            "v_energy_fraction": float(fraction[1]),
            "w_energy_fraction": float(fraction[2]),
        })
    pairs = []
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            gap = float(abs(f[i] - f[j]))
            if gap > 1e-8 * max(1., abs(f[i]), abs(f[j])):
                continue
            a, b = modes[i], modes[j]
            phase_difference = None
            if a["angular_phase_deg"] is not None and b["angular_phase_deg"] is not None:
                phase_difference = abs(
                    (b["angular_phase_deg"] - a["angular_phase_deg"] + 90.) % 180. - 90.)
            if mac[i, j] > .99:
                status = "POSSIBLE_DUPLICATED_EIGENVECTOR"
            elif (a["harmonic"] == b["harmonic"] and a["harmonic"] > 0 and
                  phase_difference is not None and abs(phase_difference - 90.) < 5. and
                  mac[i, j] < 1e-6):
                status = "CIRCUMFERENTIAL_SINE_COSINE_DOUBLET"
            elif (a["harmonic"] == b["harmonic"] == 0 and
                  a["mode_type"] == b["mode_type"] == "AXISYMMETRIC_TORSION" and
                  q.shape[2] == 5 and str(operator_scheme).upper() == "SBP21_PERIODIC_ODD"):
                status = "CHECK_SBP21_NX5_AXIAL_DISCRETE_DEGENERACY"
            else:
                status = "REPEATED_FREQUENCY_NEEDS_PHYSICAL_CHECK"
            pairs.append({
                "mode_i": i + 1, "mode_j": j + 1, "frequency_gap_hz": gap,
                "euclidean_mac": float(mac[i, j]),
                "phase_difference_deg": phase_difference, "classification": status,
            })
    return {"modes": modes, "pairs": pairs, "euclidean_mac": mac}


def recover_modal_displacements(fields, theory, thickness_m):
    """Physical U,V,W at inner/middle/outer thickness stations for CUF.

    Modal amplitudes are mass-normalized shapes, not absolute response.
    For KL/FSDT/Reddy only the saved midsurface U,V,W can be recovered
    without calling the original theory-specific kinematics.
    """
    q = np.asarray(fields, dtype=float)
    groups, is_cuf = _component_groups(q, theory)
    if not np.isfinite(thickness_m) or thickness_m <= 0:
        raise ValueError("Positive thickness required")
    xi = (-1., 0., 1.) if is_cuf else (0.,)
    z = np.asarray(xi, dtype=float) * (float(thickness_m) / 2.)
    result = np.zeros((q.shape[0], len(xi), 3, q.shape[2], q.shape[3]))
    for iz, value in enumerate(xi):
        for axis, coefficients in enumerate(groups):
            powers = value ** np.arange(coefficients.shape[1], dtype=float)
            result[:, iz, axis] = np.einsum("mcxs,c->mxs", coefficients, powers)
    return z, result


__all__ = ("inspect_modal_shapes", "recover_modal_displacements")
