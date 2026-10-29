"""Read accepted reference solver states for GUI, without legacy basis conversion.

The solver stores (component,x,s) mechanical coefficients and physical fields
(z,x,s,component). Display arrays are returned in (z,s,x) or (s,x) order.
This is *read-only*: never reconstruct physical data using a different basis.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from cylinder_reference_progress_adapter import accepted_reference_steps

_COMPONENTS = ("xx", "ss", "zz", "xs", "xz", "sz")


def _summary(root):
    root = Path(root)
    summary = json.loads((root / "FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("solver_backend") != "VARIATIONAL_REFERENCE":
        raise ValueError("Not a VARIATIONAL_REFERENCE result folder")
    return summary


def reference_catalog(root):
    summary = _summary(root)
    rows = accepted_reference_steps(root)
    return summary, rows


def _finite(data, name, shape=None):
    if name not in data:
        raise ValueError(f"Reference state is missing {name}")
    a = np.asarray(data[name], dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError(f"Invalid reference field {name}: {a.shape}")
    return a


def reference_step_data(root, step):
    """Read a *verified accepted* step, including exact through-thickness fields."""
    root = Path(root)
    summary, rows = reference_catalog(root)
    step = int(step)
    record = next((row for row in rows if row["index"] == step), None)
    if record is None:
        raise ValueError(f"Reference step {step} is not accepted")
    path = root / record["state_file"]
    with np.load(path, allow_pickle=False) as zf:
        data = {k: np.asarray(zf[k]) for k in zf.files}
    q = _finite(data, "mechanical")
    if q.ndim != 3 or min(q.shape) < 1:
        raise ValueError("Invalid reference mechanical component/x/s ordering")
    ncomp, nx, ns = q.shape
    theory = str(summary.get("theory", "")).upper()
    if theory.startswith("CUF_TE") and theory[-1] in "1234":
        expected = 3 * (int(theory[-1]) + 1)
        radial = 2 * (int(theory[-1]) + 1)
    elif theory in ("KL", "FSDT", "REDDY"):
        expected, radial = (3 if theory == "KL" else 5), 2
    else:
        raise ValueError("Unknown shell theory in reference result")
    if ncomp != expected:
        raise ValueError("Reference theory and mechanical ordering disagree")
    z = _finite(data, "z_m")
    if z.ndim != 1 or z.size < 3 or np.any(np.diff(z) <= 0):
        raise ValueError("Invalid reference thickness stations")
    nz = z.size
    disp = _finite(data, "displacement3_m", (nz, nx, ns, 3))
    volumes = {
        "u(z)": disp[:, :, :, 0].transpose(0, 2, 1),
        "v(z)": disp[:, :, :, 1].transpose(0, 2, 1),
        "w(z)": disp[:, :, :, 2].transpose(0, 2, 1),
    }
    for prefix, source in (("E", "strain6"), ("S", "stress6_pa")):
        if source in data:
            a = _finite(data, source, (nz, nx, ns, 6))
            for index, suffix in enumerate(_COMPONENTS):
                volumes[f"{prefix}_{suffix}"] = a[:, :, :, index].transpose(0, 2, 1)
    if "det_F" in data:
        volumes["detF"] = _finite(data, "det_F", (nz, nx, ns)).transpose(0, 2, 1)
    if "electric3_v_per_m" in data:
        e = _finite(data, "electric3_v_per_m", (nz, nx, ns, 3))
        for index, suffix in enumerate(("x", "s", "z")):
            volumes[f"Efield_{suffix}"] = e[:, :, :, index].transpose(0, 2, 1)
    if "induction3_c_per_m2" in data:
        d = _finite(data, "induction3_c_per_m2", (nz, nx, ns, 3))
        for index, suffix in enumerate(("x", "s", "z")):
            volumes[f"D_{suffix}"] = d[:, :, :, index].transpose(0, 2, 1)
    return {
        "z": z, "volumes": volumes,
        "u0": q[0].T, "v0": q[1].T, "w0": q[radial].T,
        "theory": theory, "index": step,
        "state_file": record["state_file"],
        "reference_saved_physical_fields": True,
    }


def reference_field_3d(data, field):
    if field in ("u0", "v0", "w0"):
        field = field[0] + "(z)"
    if field not in data["volumes"]:
        raise ValueError(f"Physical reference field {field} not saved")
    return data["z"], data["volumes"][field], field


def reference_field_2d(data, field, z_m=0.0):
    if field in ("u0", "v0", "w0"):
        return data[field], field + " (midsurface coefficient)"
    z, vol, name = reference_field_3d(data, field)
    iz = int(np.argmin(np.abs(z - float(z_m))))
    return vol[iz], f"{name}, z={z[iz]:.6g} m"


__all__ = ("reference_catalog", "reference_step_data", "reference_field_2d", "reference_field_3d")
