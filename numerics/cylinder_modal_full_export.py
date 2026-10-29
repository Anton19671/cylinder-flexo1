"""Saved-state modal FULL output: physical shapes, diagnostics and charts.

No eigenproblem is solved and no CASE/solver arrays are modified.
Postprocessing differentiates the original shell kinematics near the
undeformed state, so strain and stress are LINEAR modal shapes, not
large-amplitude nonlinear responses.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cylinder_excel_tables import LabeledWorkbook, write_labeled_table
from cylinder_modal_diagnostics import inspect_modal_shapes
from cylinder_modal_reference import _modal_spec
from cylinder_physical_fields import recover_thickness_fields


CHARACTERISTICS_XLSX = "01_МОДАЛЬНЫЕ_ХАРАКТЕРИСТИКИ.xlsx"
FIELDS_XLSX = "02_МОДАЛЬНЫЕ_ПОЛЯ_И_СЕЧЕНИЯ.xlsx"
PLOTS_DIR = "ГРАФИКИ_СОБСТВЕННЫХ_ФОРМ"


def _table(book, name, fields, rows):
    write_labeled_table(book, name, fields, rows)


def _linear_modal_physics(functional, mode, electric, dimensions):
    """Linearize the already implemented G3D shell field-recovery kernel."""
    length, radius, thickness = dimensions
    displacement_scale = max(float(np.max(np.abs(mode))), np.finfo(float).tiny)
    # A tiny physical deformation avoids evaluating nonlinear modal amplitudes.
    eps = min(1., 1.e-7 * min(length, radius, thickness) / displacement_scale)
    positive = recover_thickness_fields(
        functional, eps * mode, None if electric is None else eps * electric,
        n_thickness=3)
    negative = recover_thickness_fields(
        functional, -eps * mode, None if electric is None else -eps * electric,
        n_thickness=3)
    result = {}
    for name in ("displacement3_m", "strain6", "stress6_pa",
                 "gradient6x3", "hyperstress6x3_pa_m"):
        result[name] = (positive[name] - negative[name]) / (2. * eps)
    # The symmetric quadratic coefficient of the original enthalpy functional.
    result["quadratic_enthalpy_j_per_m3"] = (
        positive["enthalpy_j_per_m3"] + negative["enthalpy_j_per_m3"]
    ) / (2. * eps * eps)
    result["z_m"] = positive["z_m"]
    return result


def _crossings(sample):
    """Interior axial sign changes resolved by the nodal grid (not exact nodes)."""
    arr = np.asarray(sample, float)
    if not len(arr):
        return 0
    tol = float(np.max(np.abs(arr))) * 1.e-8
    signs = np.sign(arr[np.abs(arr) > tol])
    return int(np.sum(signs[:-1] * signs[1:] < 0.))


def _nodal_line_data(displacements):
    mid = displacements[1]                 # (nx,ns,3)
    energies = np.sum(mid**2, axis=(0, 1))
    dominant = int(np.argmax(energies))
    values = mid[:, :, dominant]
    j = int(np.argmax(np.sum(values * values, axis=0)))
    return dominant, _crossings(values[:, j])


def _plot_shape(destination, x, theta, radius, displacement, mode_no, freq):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    mid = displacement[1]                  # (nx,ns,3)
    maxmag = float(np.max(np.linalg.norm(mid, axis=-1)))
    scale = 0. if maxmag == 0. else .14 * min(float(radius), float(x[-1]-x[0])) / maxmag
    t = np.r_[theta, theta[0] + 2.*np.pi]
    values = np.concatenate((mid, mid[:, :1]), axis=1)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    undeformed = (xx, radius*np.cos(tt), radius*np.sin(tt))
    X = xx + scale * values[:, :, 0]
    Y = (radius + scale * values[:, :, 2])*np.cos(tt) - scale*values[:, :, 1]*np.sin(tt)
    Z = (radius + scale * values[:, :, 2])*np.sin(tt) + scale*values[:, :, 1]*np.cos(tt)
    figure = Figure(figsize=(10, 7))
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_wireframe(*undeformed, color="0.73", alpha=.48, linewidth=.55)
    ax.plot_surface(X, Y, Z, facecolors=__import__("matplotlib").cm.viridis(
        (values[:, :, 2] - values[:, :, 2].min()) /
        max(float(np.ptp(values[:, :, 2])), 1.e-30)),
        linewidth=0, alpha=.88, antialiased=True, shade=False)
    ax.set(xlabel="x [m]", ylabel="Y [m]", zlabel="Z [m]",
           title=f"Mode {mode_no}, f={freq:.7g} Hz | visual scale x{scale:.3g}")
    box = np.array([np.ptp(X), np.ptp(Y), np.ptp(Z)], float)
    ax.set_box_aspect(np.maximum(box, 1.e-9))
    figure.tight_layout()
    figure.savefig(destination, dpi=150)


def _plot_map(destination, x, theta, data, heading, *, nodal=False):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9, 5))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    t = np.r_[theta, theta[0] + 2.*np.pi]
    plane = np.concatenate((data, data[:, :1]), axis=1).T
    image = ax.pcolormesh(x, t, plane, shading="auto")
    figure.colorbar(image, ax=ax, label="Mass-normalized modal shape")
    if nodal and float(np.min(plane)) < 0. < float(np.max(plane)):
        ax.contour(x, t, plane, levels=(0.,), colors="k", linewidths=1.5)
    ax.set(xlabel="x [m]", ylabel="theta [rad]", title=heading,
           ylim=(0, 2.*np.pi))
    figure.tight_layout()
    figure.savefig(destination, dpi=150)


def _plot_sections(destination, x, theta, displacement, title, *, along):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9, 5))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    if along == "X":
        for angle in (0., np.pi/2, np.pi, 3.*np.pi/2):
            j = int(np.argmin(np.abs((theta-angle+np.pi)%(2*np.pi)-np.pi)))
            ax.plot(x, displacement[:, j], marker=".", label=f"theta={theta[j]:.3g}")
        ax.set_xlabel("x [m]")
    else:
        i = int(len(x)//2)
        ax.plot(np.r_[theta, theta[0]+2*np.pi],
                np.r_[displacement[i], displacement[i, 0]], marker=".",
                label=f"x={x[i]:.3g} m")
        ax.set_xlabel("theta [rad]")
    ax.set(ylabel="Normalized modal displacement", title=title)
    ax.grid(True, alpha=.25)
    ax.legend(loc="best")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)


def _physical_mode_data(prepared, density, fields, electric, frequency, case, diagnostics):
    func = prepared.problem.functional
    f = np.asarray(frequency, float)
    modes = np.asarray(fields, float)
    num, ncomp, nx, ns = modes.shape
    q_flat = modes.reshape(num, -1)
    zero = np.zeros_like(modes[0])
    mass_actions = np.stack([
        np.asarray(func.mass_action(modes[i], density_kg_m3=density,
                                   mechanical=zero, velocity=zero)).ravel()
        for i in range(num)])
    grams = q_flat @ mass_actions.T
    grams = .5*(grams + grams.T)
    diag = np.diag(grams)
    if np.any(~np.isfinite(diag)) or np.any(diag <= 0):
        raise ValueError("Invalid consistent modal masses in saved modes")
    cosine = grams / np.sqrt(diag[:, None]*diag[None, :])
    mass_mac = np.square(np.clip(cosine, -1., 1.))
    names = ("U", "V", "W")
    first = (0, (ncomp//3), 2*(ncomp//3)) if str(func.theory).startswith("CUF") else (0,1,2)
    participation = []
    for axis, component in zip(names, first):
        influence = np.zeros_like(zero)
        influence[component, :, :] = 1.
        ma = np.asarray(func.mass_action(influence, density_kg_m3=density,
                                        mechanical=zero, velocity=zero)).ravel()
        reference_mass = float(influence.ravel() @ ma)
        if reference_mass <= 0:
            raise ValueError("Invalid unit-direction reference mass")
        for i in range(num):
            numerator = float(q_flat[i] @ ma)
            participation.append((i+1, axis, numerator / diag[i],
                                  numerator*numerator / diag[i],
                                  100.*numerator*numerator / (diag[i]*reference_mass),
                                  reference_mass))
    times = case.get("_mode_times", [])
    rows = []
    physical = []
    for i in range(num):
        p = None if electric is None else electric[i]
        physical_mode = _linear_modal_physics(
            func, modes[i], p,
            (float(case["geometry"]["L"]), float(case["geometry"]["R"]),
             float(case["geometry"]["h"])))
        physical.append(physical_mode)
        direction, axial_nodes = _nodal_line_data(physical_mode["displacement3_m"])
        mode_meta = diagnostics["modes"][i]
        total_shapes = physical_mode["displacement3_m"]
        rows.append((i+1, float(f[i]), float(2*np.pi*f[i]),
                     float(diag[i]), float((2*np.pi*f[i])**2*diag[i]),
                     float(np.max(np.linalg.norm(total_shapes, axis=-1))),
                     "UVW"[direction], mode_meta["mode_type"],
                     int(mode_meta["harmonic"]), int(axial_nodes),
                     int(2*mode_meta["harmonic"]),
                     float(np.max(np.abs(cosine[i, np.arange(num)!=i])))
                     if num > 1 else 0.,
                     times[i] if i < len(times) else None))
    return rows, physical, grams, mass_mac, participation


def _write_mode_workbooks(root, case, summary, x, theta, rows, physical,
                          grams, mac, participation, doublets):
    with LabeledWorkbook(str(root / CHARACTERISTICS_XLSX)) as wb:
        _table(wb, "ФОРМЫ", (
            "mode", "frequency_hz", "omega_rad_s", "modal_mass", "modal_stiffness",
            "max_shape", "dominant_component", "mode_type", "circ_wave_n",
            "axial_nodal_count", "circ_nodal_estimate",
            "max_mass_overlap", "mode_time_s"), rows)
        _table(wb, "КРАТНЫЕ_ЧАСТОТЫ", (
            "mode_i", "mode_j", "frequency_gap_hz", "euclidean_mac",
            "phase_difference_deg", "classification"), (
            (p["mode_i"], p["mode_j"], p["frequency_gap_hz"],
             p["euclidean_mac"], p["phase_difference_deg"], p["classification"])
            for p in doublets))
        _table(wb, "ОРТОГОНАЛЬНОСТЬ", ("mode", *("M_"+str(i+1) for i in range(len(rows)))),
               ((i+1, *map(float, grams[i])) for i in range(len(rows))))
        _table(wb, "MAC_ПО_МАССЕ", ("mode", *("MAC_"+str(i+1) for i in range(len(rows)))),
               ((i+1, *map(float, mac[i])) for i in range(len(rows))))
        _table(wb, "УЧАСТИЕ", (
            "mode", "direction", "participation_coefficient",
            "effective_modal_mass", "effective_mass_percent", "reference_mass"),
            participation)
        _table(wb, "ПРИМЕЧАНИЯ", ("setting", "value"), (
            ("normalization", "Saved solver eigenvectors, physical mass metric"),
            ("modal_stiffness", "Derived from eigenvalue: omega_squared * modal_mass"),
            ("effective_mass", "Uniform LOCAL U0/V0/W0 coefficient directions; not global Cartesian"),
            ("mode_time_s", "Available for sequential MVI modes only, if recorded"),
            ("strain_stress", "Linearized near zero displacement, not physical response amplitude"),
            ("quadratic_enthalpy", "Second-order electromechanical enthalpy; not always positive strain energy"),
            ("axial_nodes", "Estimated sign changes at sampled axial nodes; coarse grids may miss lines"),
            ("circ_nodal_estimate", "2*n from circumferential harmonic; harmonic analysis of sampled grid"),
            ("mvi_reference", "Never infer an MVI/MKR accuracy comparison from one run"),
            ("solve_wall_s", summary.get("modal_solve_wall_s")),
        ))
    head = ("mode", "ix", "js", "x_m", "theta_rad", "z_m",
            "u_shape", "v_shape", "w_shape")
    sections = []
    through = []
    axial = []
    circum = []
    for i, item in enumerate(physical, start=1):
        z = np.asarray(item["z_m"])
        d = item["displacement3_m"]              # (z,nx,ns,3)
        strain = item["strain6"]
        stress = item["stress6_pa"]
        enthalpy = item["quadratic_enthalpy_j_per_m3"]
        for iz in range(len(z)):
            for ix in range(len(x)):
                for js in range(len(theta)):
                    through.append((i, ix, js, float(x[ix]), float(theta[js]),
                                    float(z[iz]), *map(float, d[iz,ix,js])))
                    sections.append((
                        i, ix, js, float(x[ix]), float(theta[js]), float(z[iz]),
                        *map(float, strain[iz,ix,js]),
                        *map(float, stress[iz,ix,js]), float(enthalpy[iz,ix,js])))
        # Actual sampled sections, not polynomial interpolation of nodal values.
        for component, axis in enumerate(("U","V","W")):
            for angle in (0.,np.pi/2,np.pi,3*np.pi/2):
                js = int(np.argmin(np.abs((theta-angle+np.pi)%(2*np.pi)-np.pi)))
                for ix in range(len(x)):
                    axial.append((i, axis, float(x[ix]),float(theta[js]),
                                  float(d[1,ix,js,component])))
            ix=len(x)//2
            for js in range(len(theta)):
                circum.append((i,axis,float(x[ix]),float(theta[js]),
                               float(d[1,ix,js,component])))
    with LabeledWorkbook(str(root / FIELDS_XLSX)) as wb:
        _table(wb, "UVW_ПО_ТОЛЩИНЕ", head, through)
        _table(wb, "ДЕФОРМАЦИИ_НАПРЯЖЕНИЯ",
               ("mode","ix","js","x_m","theta_rad","z_m",
                "E_xx","E_ss","E_zz","E_xs","E_xz","E_sz",
                "S_xx","S_ss","S_zz","S_xs","S_xz","S_sz",
                "quadratic_enthalpy_j_m3"), sections)
        _table(wb, "ПРОДОЛЬНЫЕ_СЕЧЕНИЯ",
               ("mode","direction","x_m","theta_rad","displacement_shape"), axial)
        _table(wb, "ОКРУЖНЫЕ_СЕЧЕНИЯ",
               ("mode","direction","x_m","theta_rad","displacement_shape"), circum)
    return [CHARACTERISTICS_XLSX, FIELDS_XLSX]


def _make_plots(dest, x, theta, radius, frequencies, diagnostics, physical):
    plots = dest / PLOTS_DIR
    plots.mkdir(parents=True, exist_ok=True)
    for i, mode in enumerate(physical, start=1):
        folder = plots / f"ФОРМА_{i:02d}"
        folder.mkdir(exist_ok=True)
        displ = mode["displacement3_m"]
        _plot_shape(folder/"01_3D_ФОРМА.png", x, theta, radius,
                    displ, i, frequencies[i-1])
        for iz, surface in enumerate(("ВНУТРЕННЯЯ", "СРЕДИННАЯ", "НАРУЖНАЯ")):
            for j, component in enumerate(("U","V","W")):
                _plot_map(folder/f"02_{surface}_{component}.png", x,theta,
                          displ[iz,:,:,j],
                          f"Mode {i}, f={frequencies[i-1]:.7g} Hz; {component}, {surface}")
        dominant = str(diagnostics["modes"][i-1]["dominant_component"])
        axis = ("U","V","W").index(dominant)
        _plot_map(folder/"03_УЗЛОВЫЕ_ЛИНИИ.png",x,theta,displ[1,:,:,axis],
                  f"Mode {i}, nodal lines in {dominant}",nodal=True)
        for j, component in enumerate(("U","V","W")):
            _plot_sections(folder/f"04_ПРОДОЛЬНЫЕ_{component}.png",
                           x,theta,displ[1,:,:,j],
                           f"Mode {i}; {component}(x) at theta sections",along="X")
            _plot_sections(folder/f"05_ОКРУЖНЫЕ_{component}.png",
                           x,theta,displ[1,:,:,j],
                           f"Mode {i}; {component}(theta) at x mid",along="S")
        for iz, surface in enumerate(("ВНУТРЕННЯЯ","СРЕДИННАЯ","НАРУЖНАЯ")):
            for name, source in (("ENERGY",mode["quadratic_enthalpy_j_per_m3"][iz]),
                                 ("EXX",mode["strain6"][iz,:,:,0]),
                                 ("SXX",mode["stress6_pa"][iz,:,:,0])):
                _plot_map(folder/f"06_{surface}_{name}.png",x,theta,source,
                          f"Mode {i}; {name}, {surface} (linearized)")
    return plots


def export_modal_full(root, destination, summary, frequencies, fields, electrical,
                      x, s, diagnostics):
    """Create physical FULL books in the RUN root, charts under EXPORT/FULL."""
    root, destination = Path(root), Path(destination)
    case = json.loads((root / "REFERENCE_CASE.json").read_text(encoding="utf-8"))
    prepared, density, requested = _modal_spec(case,base_dir=root)
    modes = np.asarray(fields, float)
    if modes.shape[0] != requested:
        raise ValueError("Saved modal mode count differs from CASE")
    electric = None if electrical is None else np.asarray(electrical, float)
    if electric is not None and electric.shape[0] != requested:
        raise ValueError("Saved electric modal mode count differs from CASE")
    times = summary.get("mode_wall_time_s", [])
    case["_mode_times"] = times if isinstance(times,list) else []
    rows, physical, grams, mac, participation = _physical_mode_data(
        prepared,density,modes,electric,frequencies,case,diagnostics)
    R = float(case["geometry"]["R"])
    theta = np.asarray(s,float) / R
    names = _write_mode_workbooks(root,case,summary,x,theta,rows,physical,
                                  grams,mac,participation,diagnostics["pairs"])
    plots = _make_plots(destination,x,theta,R,frequencies,diagnostics,physical)
    return {"workbooks": names, "plots": str(plots), "modes": len(rows)}


__all__ = ("export_modal_full","CHARACTERISTICS_XLSX","FIELDS_XLSX","PLOTS_DIR")
