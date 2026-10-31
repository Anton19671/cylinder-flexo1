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
    if ns > 21 and axis == "X":
        # Independent Fourier phases on disjoint component/axial supports.
        return directions
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
                                        mode_tolerance=1.e-5,
                                        axial_modes_per_harmonic=1):
    """Compute fixed-rank X/Y modes per Fourier sector, including axial overtones.

    For nonzero n, the eigenspace is doubly degenerate. Both members
    must be independently mass-orthogonal before advancing the axial
    index. An incomplete or mismatched pair is an error, not a mode.
    This is a finite candidate search, not a certified global spectrum.
    """
    from cylinder_modal_mvi import _SeparatedModalProblem, solve_modal_mvi
    f = prepared.problem.functional
    nx, ns = prepared.mechanical.shape[1:]
    if (requested > 8 or requested < 1 or
            f.electrical_on or f.material.ell_m != 0 or ns % 2 != 1 or nx < 9):
        raise ValueError("MODAL_MVI_HARMONIC_SCAN_REQUIRES_MACRO_AND_UP_TO_8_MODES")
    if (type(axial_modes_per_harmonic) is not int or
            not 1 <= axial_modes_per_harmonic <= 8):
        raise ValueError("MODAL_MVI_AXIAL_MODES_PER_HARMONIC_MUST_BE_1_TO_8")
    operator = _SeparatedModalProblem(prepared, density, prepared.mvi_settings.rank,
                                      harmonic_scan=True)
    candidates, family_summary = [], []
    for n in range(ns // 2 + 1):
        operator.circumferential_harmonic = n
        if detail_callback is not None:
            detail_callback(stage="MVI_HARMONIC_START", harmonic=n,
                            total_harmonics=ns // 2 + 1,
                            rank=prepared.mvi_settings.rank)
        if axial_modes_per_harmonic == 1:
            # Retain the verified fundamental path and exact rotational partner.
            partial = solve_modal_mvi(
                prepared, density, 1, detail_callback=None,
                eigen_tolerance=eigen_tolerance, mode_tolerance=mode_tolerance,
                _cached_operator=operator)
            q = partial["mechanical_modes"][0].ravel()
            p = (None if partial["electrical_modes"] is None else
                 partial["electrical_modes"][0].ravel())
            lam = float(partial["omega_squared_rad2_s2"][0])
            sweep = partial["mvi_sweeps"][0]
            members = [(lam, q, p, sweep, "MVI_YX")]
            if n:
                rotated = _phase_quadrature_partner(q.reshape(operator.shape), n).ravel()
                norm = np.sqrt(float(rotated @ operator.mass(rotated)))
                if not np.isfinite(norm) or norm <= 0.:
                    raise ValueError("MODAL_MVI_ROTATED_PARTNER_NONPOSITIVE_MASS")
                rotated /= norm
                if abs(float(q @ operator.mass(rotated))) > 1.e-6:
                    raise ValueError("MODAL_MVI_ROTATED_PARTNER_NOT_ORTHOGONAL")
                rp = (None if p is None else _phase_quadrature_partner(
                    p.reshape(operator.p0.shape), n).ravel() / norm)
                lp = operator.eigenvalue(rotated, rp)
                if abs(lp-lam) > 1.e-7*max(1., abs(lam)):
                    raise ValueError("MODAL_MVI_ROTATED_PARTNER_FREQUENCY_MISMATCH")
                members.append((lp, rotated, rp, sweep, "ROTATIONAL_PARTNER"))
            groups = [members]
        else:
            # The core MVI independently enforces mass orthogonality for
            # successive modes; request both phases for every axial family.
            count = axial_modes_per_harmonic * (2 if n else 1)
            partial = solve_modal_mvi(
                prepared, density, count, detail_callback=None,
                eigen_tolerance=eigen_tolerance, mode_tolerance=mode_tolerance,
                _cached_operator=operator)
            groups = []
            for m in range(axial_modes_per_harmonic):
                members = []
                for j in range(m * (2 if n else 1),
                               (m + 1) * (2 if n else 1)):
                    q = partial["mechanical_modes"][j].ravel()
                    p = (None if partial["electrical_modes"] is None else
                         partial["electrical_modes"][j].ravel())
                    members.append((float(partial["omega_squared_rad2_s2"][j]),
                                    q, p, partial["mvi_sweeps"][j], "MVI_YX"))
                if n:
                    a, b = members
                    if abs(a[0]-b[0]) > 1.e-6*max(1.,abs(a[0]),abs(b[0])):
                        raise ValueError(
                            f"MODAL_MVI_INCOMPLETE_ROTATIONAL_PAIR: n={n}, axial={m+1}")
                    rotated = _phase_quadrature_partner(
                        a[1].reshape(operator.shape), n).ravel()
                    ma = np.sqrt(float(rotated @ operator.mass(rotated)))
                    mb = np.sqrt(float(b[1] @ operator.mass(b[1])))
                    if ma <= 0. or mb <= 0. or not np.isfinite(ma*mb):
                        raise ValueError("MODAL_MVI_INVALID_PAIR_MASS")
                    match = abs(float(rotated @ operator.mass(b[1]))) / (ma*mb)
                    if match < 1. - 1.e-5:
                        raise ValueError(
                            f"MODAL_MVI_ROTATIONAL_PAIR_SHAPE_MISMATCH: n={n}, axial={m+1}")
                groups.append(members)
        for m, members in enumerate(groups, start=1):
            for lam, q, p, sweep, source in members:
                fraction = _pure_harmonic_fraction(q.reshape(operator.shape), n)
                if fraction < 1. - 1.e-9:
                    raise ValueError(
                        f"MODAL_MVI_HARMONIC_PURITY_LOST: n={n}, fraction={fraction}")
                candidates.append((lam, n, m, q, p, sweep, source))
            frequency = float(np.sqrt(max(0., members[0][0]))/(2*np.pi))
            family_summary.append({"n":n,"axial_mode_index":m, "axial_mode_index_verified":False,
                                   "frequency_hz":frequency,
                                   "degeneracy":len(members)})
            if detail_callback is not None:
                detail_callback(stage="MVI_HARMONIC_DONE", harmonic=n,
                                axial_mode_index=m, frequency_hz=frequency,
                                sweeps=members[0][3])
    candidates.sort(key=lambda c:c[0])
    if len(candidates) < requested:
        raise ValueError("MODAL_MVI_INSUFFICIENT_HARMONIC_FAMILIES")
    best = candidates[:requested]
    # A requested prefix can stop between the two members of a degenerate
    # rotational pair. Keep the user-selected mode count, but disclose the
    # incomplete pair rather than presenting the prefix as a closed family.
    last = best[-1]
    cutoff_pair_incomplete = bool(
        last[1] != 0 and
        sum(c[1] == last[1] and c[2] == last[2] for c in best) == 1)
    for i, mode in enumerate(best):
        for prev in best[:i]:
            if abs(float(mode[3] @ operator.mass(prev[3]))) > 1.e-6:
                raise ValueError("MODAL_MVI_HARMONIC_MODES_NOT_ORTHOGONAL")
    eigs = np.asarray([m[0] for m in best])
    qout = np.stack([m[3].reshape(operator.shape) for m in best])
    pout = (None if not operator.ne else np.stack([
        m[4].reshape(operator.p0.shape) for m in best]))
    # The configured MVI rank is per component, not the rank of the
    # assembled vector mode across all CUF components and axial nodes.
    global_mode_ranks = []
    for field in qout:
        assembled = field.reshape(field.shape[0] * field.shape[1],
                                  field.shape[2])
        singular = np.linalg.svd(assembled, compute_uv=False)
        cutoff = 1.e-10 * singular[0] if singular.size else 0.
        global_mode_ranks.append(int(np.count_nonzero(singular > cutoff)))
    if progress_callback is not None:
        for i in range(requested):
            progress_callback(i+1, requested)
    if detail_callback is not None:
        for i, m in enumerate(best):
            detail_callback(stage="MVI_MODE_DONE", mode=i+1, n_modes=requested,
                            rank=operator.rank, sweep=m[5],
                            circumferential_wavenumber=m[1],
                            axial_mode_index=m[2], source=m[6],
                            frequency_hz=float(np.sqrt(eigs[i])/(2*np.pi)),
                            modal_amplitude_max=float(np.max(abs(m[3]))))
    return {
        "theory":operator.f.theory, "method":"MVI", "rank":operator.rank,
        "rank_scope":"per_component", "global_mode_ranks":global_mode_ranks,
        "angular_sector_dimension":2 if any(m[1] != 0 for m in best) else 1,
        "angular_sector_dimension_by_mode":[2 if m[1] else 1 for m in best],
        "angular_rank_saturated_by_mode":[operator.rank >= (2 if m[1] else 1) for m in best],
        "x_halfstep_can_span_full_fourier_sector_by_mode":[
            bool(m[1] == 0 or operator.rank >= 2) for m in best],
        "full_fourier_sector_note":(
            "When frozen angular partners span all phase directions, the X "
            "halfstep contains every admissible axial degree of freedom in "
            "that Fourier sector; FDM Fourier and MVI projected spectra "
            "then need not be independent comparisons. The flag describes "
            "capacity, not verified partner independence."),
        "rank_saturation_note":"A complete Fourier sector has at most two angular phases per component; ranks beyond this do not add angular directions.",
        "density_kg_m3":density,
        "frequencies_hz":np.sqrt(np.maximum(eigs,0.))/(2*np.pi),
        "omega_rad_s":np.sqrt(np.maximum(eigs,0.)),
        "omega_squared_rad2_s2":eigs,
        "mechanical_modes":qout,"electrical_modes":pout,
        "n_mechanical_free_dofs":operator.n_mechanical_free_dofs,
        "n_electric_free_dofs":operator.n_electric_free_dofs,
        "mvi_sweeps":[m[5] for m in best],
        "circumferential_wavenumbers":[m[1] for m in best],
        "axial_mode_indices":[m[2] for m in best],
        "axial_mode_index_interpretation":"frequency_order_within_circumferential_sector_not_verified_axial_half_wave_count",
        "harmonic_family_scan":family_summary,
        "mode_generation":[m[6] for m in best],
        "mode_independently_optimized":[m[6] == "MVI_YX" for m in best],
        "mvi_modal_algorithm":"fixed_rank_constrained_variational_YX_stationarity",
        "modal_spectrum_selection":"circumferential_harmonic_axial_families",
        "axial_modes_per_harmonic":axial_modes_per_harmonic,
        "cutoff_rotational_pair_incomplete":cutoff_pair_incomplete,
        "axial_overtone_search_complete":False,
        "global_spectrum_completeness_verified":False,
    }
