"""Direct strong finite-difference interior equations for cylindrical-shell jets.

Coefficients here are continuum thickness-integrated conjugate densities, not
area-weighted discrete actions. Explicit derivatives of these coefficients
are independent of the weighted-transpose variational FDM. End actions are
derived from Green integration; physical end-row closure is NOT assembled.
KL may require third-order displacement jets/sixth-order axial equations,
which cannot be silently replaced by fourth-order boundary conditions.

The periodic second derivative is the compatible three-point Laplacian,
never a squared centered first derivative (which loses the Nyquist mode).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np


@dataclass(frozen=True)
class DirectStrongGrid:
    nx: int
    ns: int
    length_m: float
    radius_m: float

    def __post_init__(self):
        if (isinstance(self.nx, (bool, np.bool_)) or
                isinstance(self.ns, (bool, np.bool_)) or
                not isinstance(self.nx, (int, np.integer)) or
                not isinstance(self.ns, (int, np.integer)) or
                self.nx < 7 or self.ns < 4):
            raise ValueError("direct strong grid requires nx>=7, ns>=4")
        for value, key in ((self.length_m, "length"), (self.radius_m, "radius")):
            if (isinstance(value, (bool, np.bool_)) or
                    not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{key} must be positive and finite")

    @property
    def dx(self):
        return float(self.length_m) / (self.nx - 1)

    @property
    def ds(self):
        return 2 * math.pi * float(self.radius_m) / self.ns


@dataclass(frozen=True)
class DirectStrongBalance:
    mechanical_bulk: np.ndarray
    electrical_bulk: np.ndarray | None
    mechanical_end_displacement: np.ndarray
    mechanical_end_slope: np.ndarray | None
    mechanical_end_curvature: np.ndarray | None
    electrical_end_potential: np.ndarray | None
    safe_interior_x_nodes: tuple[int, ...]
    highest_axial_jet: int
    spatial_method_status: str = "DIRECT_STRONG_INTERIOR_NO_PHYSICAL_END_CLOSURE"


def _finite_field(field, shape, label):
    data = np.asarray(field, dtype=float)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"{label}: expected finite shape {shape}")
    return data


def _fornberg_weights(nodes, location, order):
    """Seven-node explicit interpolation weights, without matrix inversion."""
    n = len(nodes)
    table = np.zeros((n, order + 1), dtype=float)
    table[0, 0] = 1.0
    c1 = 1.0
    c4 = float(nodes[0]) - location
    for i in range(1, n):
        upto = min(i, order)
        c2 = 1.0
        c5 = c4
        c4 = float(nodes[i]) - location
        for j in range(i):
            c3 = float(nodes[i] - nodes[j])
            c2 *= c3
            if j == i - 1:
                for k in range(upto, 0, -1):
                    table[i, k] = c1 * (
                        k * table[i-1, k-1] - c5 * table[i-1, k]) / c2
                table[i, 0] = -c1 * c5 * table[i-1, 0] / c2
            for k in range(upto, 0, -1):
                table[j, k] = (c4 * table[j, k] -
                               k * table[j, k-1]) / c3
            table[j, 0] = c4 * table[j, 0] / c3
        c1 = c2
    return table[:, order]


@lru_cache(maxsize=96)
def _axial_weights(nx, order):
    if order not in (1, 2, 3) or nx < 7:
        raise ValueError("axial derivative orders 1..3 require nx>=7")
    result = np.zeros((nx, nx), dtype=float)
    for i in range(nx):
        left = min(max(0, i - 3), nx - 7)
        indices = np.arange(left, left + 7)
        result[i, indices] = _fornberg_weights(indices, float(i), order)
    result.setflags(write=False)
    return result


def direct_coefficient_derivative(field, axial_order, circum_order, grid):
    """Direct spatial differentiation of unweighted constitutive fields."""
    if not isinstance(grid, DirectStrongGrid):
        raise TypeError("direct strong grid object required")
    values = np.asarray(field, dtype=float)
    if (values.ndim < 2 or
            values.shape[-2:] != (grid.nx, grid.ns) or
            not np.isfinite(values).all()):
        raise ValueError("finite coefficients must end with (nx,ns)")
    if (type(axial_order) is not int or type(circum_order) is not int or
            axial_order < 0 or circum_order < 0 or
            axial_order > 3 or circum_order > 3 or
            axial_order + circum_order > 3):
        raise ValueError("surface derivative total order must be <=3")
    out = values.copy()
    ds = grid.ds
    if circum_order == 1:
        out = (out - np.roll(out, 1, axis=-1)) / ds
    elif circum_order in (2, 3):
        lap = (np.roll(out, -1, axis=-1) -
               2*out + np.roll(out, 1, axis=-1)) / ds**2
        out = (lap if circum_order == 2 else
               (lap - np.roll(lap, 1, axis=-1)) / ds)
    if axial_order:
        out = np.einsum("ij,...jk->...ik",
                        _axial_weights(grid.nx, axial_order),
                        out, optimize=True) / grid.dx**axial_order
    if not np.isfinite(out).all():
        raise ValueError("strong derivative produced nonfinite values")
    return out


def _outward_ends(physical_flux):
    return np.stack((-physical_flux[..., 0, :],
                      physical_flux[..., -1, :]))


def direct_strong_jet_balance(mechanical_coefficients, derivative_orders, grid,
                              electrical_coefficients=None, *,
                              mechanical_load=None, electrical_charge=None):
    """Physical strong balance and axial end work (not boundary conditions).

    Let B_a = sum_b (-1)^b d_s^b A_(a,b).
    Mechanical bulk: B_0-d_x B_1+d_xx B_2-d_xxx B_3-f.
    End actions conjugate to (q,q_x,q_xx):
       n_x*(B_1-d_x B_2+d_xx B_3, B_2-d_x B_3, B_3).
    Electric bulk: E_0-d_x E_1-d_s E_2-charge.
    Electric end potential action: n_x*E_1.

    No SVD, pseudoinverse, weighted-weak adjoint, or implicit BC closure.
    """
    if not isinstance(grid, DirectStrongGrid):
        raise TypeError("direct strong grid object required")
    mc = np.asarray(mechanical_coefficients, dtype=float)
    if (mc.ndim != 4 or mc.shape[0] < 1 or mc.shape[1] < 1 or
            mc.shape[2:] != (grid.nx, grid.ns) or
            len(derivative_orders) != mc.shape[0] or
            not np.isfinite(mc).all()):
        raise ValueError("finite mechanical coefficients (jet,field,nx,ns) required")
    seen = set()
    by_axial = [np.zeros_like(mc[0]) for _ in range(4)]
    highest = 0
    for k, entry in enumerate(derivative_orders):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError("each jet requires (name,axial,circum)")
        _, ax, az = entry
        if (type(ax) is not int or type(az) is not int or ax < 0 or
                az < 0 or ax + az > 3 or (ax, az) in seen):
            raise ValueError("duplicate or unsupported surface jet")
        seen.add((ax, az))
        highest = max(highest, ax)
        by_axial[ax] += (-1)**az * direct_coefficient_derivative(
            mc[k], 0, az, grid)
    if (0, 0) not in seen:
        raise ValueError("mechanical zero derivative jet is required")
    strong = sum(((-1)**ax * direct_coefficient_derivative(
        by_axial[ax], ax, 0, grid)
        for ax in range(highest+1)), np.zeros_like(mc[0]))
    if mechanical_load is not None:
        strong -= _finite_field(mechanical_load, mc[0].shape, "mechanical load")
    q_action = by_axial[1].copy()
    if highest >= 2:
        q_action -= direct_coefficient_derivative(by_axial[2], 1, 0, grid)
    if highest >= 3:
        q_action += direct_coefficient_derivative(by_axial[3], 2, 0, grid)
    qx_action = None
    if highest >= 2:
        qx_action = by_axial[2].copy()
        if highest >= 3:
            qx_action -= direct_coefficient_derivative(by_axial[3], 1, 0, grid)
    qxx_action = by_axial[3] if highest >= 3 else None
    elec_bulk = None
    elec_end = None
    if electrical_coefficients is not None:
        ec = np.asarray(electrical_coefficients, dtype=float)
        if (ec.ndim != 4 or ec.shape[0] != 3 or ec.shape[1] < 1 or
                ec.shape[2:] != (grid.nx, grid.ns) or
                not np.isfinite(ec).all()):
            raise ValueError("finite electrical coefficients (3,mode,nx,ns) required")
        elec_bulk = ec[0] - direct_coefficient_derivative(
            ec[1], 1, 0, grid)
        elec_bulk -= direct_coefficient_derivative(ec[2], 0, 1, grid)
        if electrical_charge is not None:
            elec_bulk -= _finite_field(electrical_charge,
                                       ec[0].shape, "electric charge")
        elec_end = _outward_ends(ec[1])
    elif electrical_charge is not None:
        raise ValueError("electric charge requires electrical coefficients")
    return DirectStrongBalance(
        strong, elec_bulk, _outward_ends(q_action),
        _outward_ends(qx_action) if qx_action is not None else None,
        _outward_ends(qxx_action) if qxx_action is not None else None,
        elec_end, tuple(range(highest, grid.nx-highest)), highest)


def direct_strong_from_functional(functional, mechanical, electrical=None, *,
                                  length_m, mechanical_load=None,
                                  electrical_charge=None,
                                  velocity=None, acceleration=None,
                                  density_kg_m3=None, eta_s=0.,
                                  mass_damping_rate_per_s=0.):
    """Evaluate the existing nonlinear 7-theory constitutive jets with a
    DIFFERENT direct-strong spatial differentiation.

    This diagnostic is not dispatched as a solver. It deliberately uses the
    same continuum constitutive coefficients but no reference weak-adjoint
    assembly. It requires a uniform physical x-grid and ignores physical
    boundary constraints until a validated theory-specific closure exists.
    """
    from cylinder_functional_assembly import MECHANICAL_DERIVATIVES
    if not hasattr(functional, "_dynamic_g3d_weak_jet_coefficients"):
        raise TypeError("CylinderFunctional constitutive jet producer required")
    nx,ns=functional.ops.shape
    grid=DirectStrongGrid(nx,ns,length_m,functional.R)
    weights=np.asarray(functional.ops.wx,float)
    if (weights.shape!=(nx,) or
            not np.allclose(weights[1:-1],grid.dx,rtol=1e-9,atol=1e-14) or
            not np.allclose(weights[[0,-1]],grid.dx/2,rtol=1e-9,atol=1e-14)):
        raise ValueError("direct strong diagnostic requires a uniform trapezoid x-grid")
    mc,ec=functional._dynamic_g3d_weak_jet_coefficients(
        mechanical,electrical,velocity=velocity,acceleration=acceleration,
        density_kg_m3=density_kg_m3,eta_s=eta_s,
        mass_damping_rate_per_s=mass_damping_rate_per_s)
    return direct_strong_jet_balance(
        mc,MECHANICAL_DERIVATIVES[:functional.n_derivatives],grid,ec,
        mechanical_load=mechanical_load,electrical_charge=electrical_charge)


__all__ = ("DirectStrongGrid", "DirectStrongBalance",
           "direct_coefficient_derivative", "direct_strong_jet_balance",
           "direct_strong_from_functional")
