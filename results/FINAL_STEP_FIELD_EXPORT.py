"""
Lightweight step-by-step field exporter.
Does not modify solver equations.
Saves coordinates and displacements after accepted load steps.
"""

from pathlib import Path
import numpy as np


def save_step_field(folder, step, load, x, y, z, u, v, w):
    out = Path(folder) / f"step_{int(step):04d}"
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "field.npz",
        step=step,
        load=load,
        x=np.asarray(x),
        y=np.asarray(y),
        z=np.asarray(z),
        u=np.asarray(u),
        v=np.asarray(v),
        w=np.asarray(w),
    )

    with open(out / "info.txt", "w", encoding="utf-8") as f:
        f.write(f"step = {step}\n")
        f.write(f"load = {load}\n")
        f.write(f"points = {len(np.asarray(x).ravel())}\n")


def export_final_state(folder, load, x, y, z, u, v, w):
    save_step_field(folder, "final", load, x, y, z, u, v, w)
