"""Fourier-harmonic modal family search for homogeneous circular shells.

The X/Y modal solves remain fixed-rank variational iterations, not Ritz.
The real sine/cosine partner is generated only by exact rotational symmetry.
Do not apply these projections to circumferentially nonuniform materials/BC.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr

from cylinder_boundary_spatial import separated_profile_directions


def _harmonic_phase_basis(nodes, wave_number):
    """Orthonormal real Fourier harmonics on odd periodic shell nodes."""
    if nodes % 2 != 1 or wave_number < 0 or wave_number > nodes // 2:
        raise ValueError("MODAL_MVI_INVALID_CIRCUMFERENTIAL_HARMONIC")
    angle = 2. * np.pi * np.arange(nodes) / nodes
    if wave_number == 0:
        return np.ones((1, nodes)) / np.sqrt(nodes)
    return np.stack((np.cos(wave_number * angle),
                     np.sin(wave_number * angle))) * np.sqrt(2. / nodes)


def _harmonic_variation_directions(x, y, axis, wave_number):
    """Exact Y/X variations restricted to one circumferential Fourier sector.

    Restriction is justified for rotationally invariant circular shells only.
    Each variation still has the separated Kryshko--Kirichenko form X*Y;
    the stiffness, consistent mass, and variational halfsteps are unchanged.
    """
    nc, rank, nx = x.shape
    ns = y.shape[2]
    phase_basis = _harmonic_phase_basis(ns, wave_number)
    axis = axis.upper()
    if axis == "Y":
        directions = np.zeros((nc * nx * ns, nc * rank * len(phase_basis)))
        for c in range(nc):
            for r in range(rank):
                for j, phase in enumerate(phase_basis):
                    column = (c * rank + r) * len(phase_basis) + j
                    field = np.zeros((nc, nx, ns))
                    field[c] = x[c, r, :, None] * phase[None, :]
                    directions[:, column] = field.ravel()
    elif axis == "X":
        # QR partner completion can insert inactive off-sector vectors
        # for ranks > 2; work with the actual independent Fourier span.
        directions_list = []
        for c in range(nc):
            partner_coordinates = y[c] @ phase_basis.T
            q, r, _ = qr(partner_coordinates.T, mode="economic",
                         pivoting=True, check_finite=False)
            pivots = np.abs(np.diag(r))
            number = int(np.count_nonzero(
                pivots > 1.e-12 * np.max(pivots, initial=0.)))
            if not number:
                continue
            angular_frame = q[:, :number].T @ phase_basis
            for axial_node in range(nx):
                for angular_vector in angular_frame:
                    field = np.zeros((nc, nx, ns))
                    field[c, axial_node, :] = angular_vector
                    directions_list.append(field.ravel())
        if not directions_list:
            return np.empty((nc*nx*ns, 0))
        directions = np.column_stack(directions_list)
    else:
        raise ValueError("MVI half-step must be X or Y")
    if directions.shape[1]:
        _, r, piv = qr(directions, mode="economic", pivoting=True,
                       check_finite=False)
        magnitudes = np.abs(np.diag(r))
        maximum = float(np.max(magnitudes, initial=0.))
        keep = int(np.count_nonzero(magnitudes > 1.e-12 * maximum))
        directions = directions[:, piv[:keep]]
    return directions


def _phase_quadrature_partner(field, wave_number):
    """Quarter-period rotation of the computed nonaxisymmetric MVI mode."""
    field = np.asarray(field, dtype=float)
    ns = field.shape[-1]
    if not 1 <= wave_number <= ns // 2:
        raise ValueError("rotation needs nonzero circumferential harmonic")
    cos_phase, sin_phase = _harmonic_phase_basis(ns, wave_number)
    coefficient_cos = np.einsum("...s,s->...", field, cos_phase)
    coefficient_sin = np.einsum("...s,s->...", field, sin_phase)
    return (coefficient_sin[..., None] * cos_phase -
            coefficient_cos[..., None] * sin_phase)


def _pure_harmonic_fraction(field, wave_number):
    """Fraction of squared nodal amplitude in the requested Fourier sector."""
    field = np.asarray(field, float)
    basis = _harmonic_phase_basis(field.shape[-1], wave_number)
    projected = np.einsum("...s,bs,bt->...t", field, basis, basis)
    magnitude = float(np.linalg.norm(field))
    if magnitude <= 0.:
        return 0.
    return float(np.linalg.norm(projected)**2 / magnitude**2)


def _solve_modal_mvi_harmonic_spectrum(prepared, density, requested, *,
                                        progress_callback=None,
                                        detail_callback=None,
                                        eigen_tolerance=1.e-7,
                                        mode_tolerance=1.e-5):
    """Enumerate fundamental circumferential families using MVI X/Y.

    One lowest stationary state is found for each harmonic. For n > 0
    its degenerate partner is obtained by exact rotation, not by a Ritz
    substitution. Higher axial overtones are NOT exhaustively searched.
    """
    from cylinder_modal_mvi import _SeparatedModalProblem, solve_modal_mvi
    f = prepared.problem.functional
    nx, ns = prepared.mechanical.shape[1:]
    if (requested > 8 or requested < 1 or
            f.electrical_on or f.material.ell_m != 0 or ns % 2 != 1 or nx < 9):
        raise ValueError("MODAL_MVI_HARMONIC_SCAN_REQUIRES_MACRO_AND_UP_TO_8_MODES")
    operator = _SeparatedModalProblem(prepared,density,prepared.mvi_settings.rank,
                                      harmonic_scan=True)
    candidates = []
    family_summary = []
    for n in range(ns // 2 + 1):
        operator.circumferential_harmonic = n
        if detail_callback is not None:
            detail_callback(stage="MVI_HARMONIC_START", harmonic=n,
                            total_harmonics=ns // 2 + 1,
                            rank=prepared.mvi_settings.rank)
        partial = solve_modal_mvi(
            prepared, density, 1, detail_callback=None,
            eigen_tolerance=eigen_tolerance, mode_tolerance=mode_tolerance,
            _cached_operator=operator)
        q = partial["mechanical_modes"][0].reshape(-1)
        p = None if partial["electrical_modes"] is None else partial["electrical_modes"][0].reshape(-1)
        wave_fraction = _pure_harmonic_fraction(q.reshape(operator.shape), n)
        if wave_fraction < 1. - 1.e-9:
            raise ValueError(f"MODAL_MVI_HARMONIC_PURITY_LOST: n={n}, fraction={wave_fraction}")
        lam = float(partial["omega_squared_rad2_s2"][0])
        frequency = float(partial["frequencies_hz"][0])
        if detail_callback is not None:
            detail_callback(stage="MVI_HARMONIC_DONE", harmonic=n,
                            frequency_hz=frequency, sweeps=partial["mvi_sweeps"][0])
        family_summary.append({"n":n,"frequency_hz":frequency,
                               "degeneracy":1 if n==0 else 2})
        candidates.append((lam, n, q, p, partial["mvi_sweeps"][0], "MVI_YX"))
        if n:
            partner_q = _phase_quadrature_partner(q.reshape(operator.shape), n).ravel()
            mass = operator.mass(partner_q)
            partner_norm = float(np.sqrt(partner_q @ mass))
            if not np.isfinite(partner_norm) or partner_norm <= 0.:
                raise ValueError("MODAL_MVI_ROTATED_PARTNER_NONPOSITIVE_MASS")
            partner_q /= partner_norm
            overlap = abs(float(q @ operator.mass(partner_q)))
            if overlap > 1.e-6:
                raise ValueError(f"MODAL_MVI_ROTATED_PARTNER_NOT_ORTHOGONAL: n={n}")
            partner_p = None if p is None else _phase_quadrature_partner(
                p.reshape(operator.p0.shape), n).ravel() / partner_norm
            lam_partner = operator.eigenvalue(partner_q, partner_p)
            if abs(lam_partner-lam) > 1.e-7 * max(1., abs(lam)):
                raise ValueError(f"MODAL_MVI_ROTATED_PARTNER_FREQUENCY_MISMATCH: n={n}")
            candidates.append((lam_partner, n, partner_q, partner_p,
                               partial["mvi_sweeps"][0], "ROTATIONAL_PARTNER"))
    candidates.sort(key=lambda c:c[0])
    if len(candidates)<requested:
        raise ValueError("MODAL_MVI_INSUFFICIENT_HARMONIC_FAMILIES")
    best = candidates[:requested]
    for i,mode in enumerate(best):
        qi = mode[2]
        for previous in best[:i]:
            if abs(float(qi @ operator.mass(previous[2]))) > 1.e-6:
                raise ValueError("MODAL_MVI_HARMONIC_MODES_NOT_ORTHOGONAL")
    eigs=np.asarray([m[0] for m in best])
    qout=np.stack([m[2].reshape(operator.shape) for m in best])
    pout=(None if not operator.ne else np.stack([
        m[3].reshape(operator.p0.shape) for m in best]))
    if progress_callback is not None:
        for i in range(requested):
            progress_callback(i+1, requested)
    if detail_callback is not None:
        for i,m in enumerate(best):
            detail_callback(stage="MVI_MODE_DONE", mode=i+1, n_modes=requested,
                            rank=operator.rank, sweep=m[4],
                            circumferential_wavenumber=m[1], source=m[5],
                            frequency_hz=float(np.sqrt(eigs[i])/(2*np.pi)),
                            modal_amplitude_max=float(np.max(abs(m[2]))))
    return {
        "theory":operator.f.theory, "method":"MVI", "rank":operator.rank,
        "density_kg_m3":density,
        "frequencies_hz":np.sqrt(np.maximum(eigs,0.))/(2*np.pi),
        "omega_rad_s":np.sqrt(np.maximum(eigs,0.)),
        "omega_squared_rad2_s2":eigs,
        "mechanical_modes":qout,"electrical_modes":pout,
        "n_mechanical_free_dofs":operator.n_mechanical_free_dofs,
        "n_electric_free_dofs":operator.n_electric_free_dofs,
        "mvi_sweeps":[m[4] for m in best],
        "circumferential_wavenumbers":[m[1] for m in best],
        "harmonic_family_scan":family_summary,
        "mode_generation":[m[5] for m in best],
        "mvi_modal_algorithm":"fixed_rank_constrained_variational_YX_stationarity",
        "modal_spectrum_selection":"circumferential_harmonic_fundamentals_rotational_pairs",
        # This scan finds one axial fundamental per Fourier sector; it does
        # not enumerate higher axial overtones or certify the global spectrum.
        "axial_overtone_search_complete":False,
        "global_spectrum_completeness_verified":False,
    }
