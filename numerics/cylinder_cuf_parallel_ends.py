"""CUF TE1--TE4 direct-strong stationary problem with the source end program.

Krysko--Zhong corrected 2026-10-24, §§24,30--33. This connects existing
constitutive jets, physical end equations and a single volume gauge to the
experimental strong Newton driver. It is NOT a new GUI dispatch or a claim
that the direct-strong discretization has been physically certified.

The constitutive adapter is explicitly the normalized-basis CylinderFunctional.
It is shared continuum/reference infrastructure, not independent MVI assembly.
Differential half-orders and dimensional equation scales remain explicit.
"""
from dataclasses import dataclass
import numpy as np

from cylinder_direct_strong_boundary import (
    CUFParallelEndConditions, assemble_physical_end_rows,
)
from cylinder_direct_strong_fdm import DirectStrongGrid, direct_strong_from_functional
from cylinder_direct_strong_newton import (
    StrongNewtonSettings, cuf_mean_potential_gauge, solve_strong_nonlinear_equations,
)


def _field(value, shape, name):
    data = np.asarray(value, dtype=float)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"{name}: finite shape {shape} required")
    return data


@dataclass(frozen=True)
class CUFParallelEndEvaluation:
    mechanical: np.ndarray
    electrical: np.ndarray | None
    mean_potential_v: float | None
    mean_potential_error_v: float | None


class CUFParallelEndProblem:
    """Reusable physical equation callback for a stationary CUF end problem.

    Every CUF displacement coefficient is fixed at both ends. For a declared
    fourth-order field the second equation is zero double traction, NOT zero
    slope. All active electric modes have zero axial flux at both ends.
    Outer/inner face insulation is represented by the existing thickness
    variational projection; no additional face collocation rows are added.

    Loads are thickness-integrated generalized surface force/charge densities,
    as in direct_strong_from_functional; pressure in Pa is not interchangeable.
    """
    def __init__(self, functional, *, length_m, axial_orders,
                 potential_order=None, mechanical_load=None,
                 electrical_charge=None, mean_potential_v=0.,
                 potential_scale_v=1.):
        # Check the actual basis-bearing adapter, not a loosely named theory.
        from cylinder_functional_assembly import CylinderFunctional
        if not isinstance(functional, CylinderFunctional):
            raise TypeError("normalized-basis CylinderFunctional required")
        if functional.theory not in ("CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4"):
            raise ValueError("parallel CUF ends require CUF_TE1..CUF_TE4")
        ends = CUFParallelEndConditions(int(functional.theory[-1]),
                                       axial_orders, potential_order)
        if functional.electrical_on != (potential_order is not None):
            raise ValueError("potential order must agree with electrical ON/OFF")
        if potential_order is None and (electrical_charge is not None or
                                        mean_potential_v != 0.):
            raise ValueError("charge and mean potential require active electricity")
        nx, ns = functional.ops.shape
        grid = DirectStrongGrid(nx, ns, length_m, functional.R)
        # Enforce this before constructing the gauge; don't mix another
        # quadrature or circumference with the direct-strong grid.
        wx = np.full(nx, grid.dx)
        wx[[0, -1]] *= .5
        expected_area = np.broadcast_to(wx[:, None]*grid.ds, (nx, ns))
        if not np.allclose(functional.W, expected_area, rtol=1.e-10, atol=0.):
            raise ValueError("uniform trapezoid x and periodic circumference quadrature required")
        self.functional = functional
        self.grid = grid
        self.ends = ends
        self.mechanical_shape = (functional.n_mechanical, nx, ns)
        self.electrical_shape = (None if potential_order is None else
                                 (potential_order+1, nx, ns))
        self.mechanical_load = self._load(mechanical_load, self.mechanical_shape,
                                          "generalized mechanical load")
        self.electrical_charge = self._load(electrical_charge, self.electrical_shape,
                                            "generalized electric charge")
        if self.electrical_charge is not None:
            weighted_charge = expected_area*self.electrical_charge[0]
            net = float(np.sum(weighted_charge))
            total = float(np.sum(np.abs(weighted_charge)))
            if abs(net) > 1.e-10*max(1.e-30, total):
                raise ValueError("incompatible insulated free charge: nonzero net charge")
        self.electric_gauge = (None if potential_order is None else
            cuf_mean_potential_gauge(expected_area, radius_m=functional.R,
                thickness_m=functional.h, potential_order=potential_order,
                target_v=mean_potential_v, potential_scale_v=potential_scale_v))

    @staticmethod
    def _load(value, shape, name):
        if value is None:
            return None
        result = _field(value, shape, name).copy()
        result.setflags(write=False)
        return result

    def _state(self, mechanical, electrical):
        q = _field(mechanical, self.mechanical_shape, "CUF mechanical state")
        if self.electrical_shape is None:
            if electrical is not None:
                raise ValueError("electric state must be None when OFF")
            p = None
        else:
            p = _field(electrical, self.electrical_shape, "CUF potential state")
        return q, p

    def evaluate(self, mechanical, electrical=None):
        q, p = self._state(mechanical, electrical)
        balance = direct_strong_from_functional(
            self.functional, q, p, length_m=self.grid.length_m,
            mechanical_load=self.mechanical_load,
            electrical_charge=self.electrical_charge)
        rows = assemble_physical_end_rows(
            balance, q, self.grid, axial_orders=self.ends.axial_orders,
            mechanical_ends=self.ends.mechanical_ends, electrical=p,
            electrical_ends=self.ends.electrical_ends,
            gauge_closed_elsewhere=self.electric_gauge is not None)
        mean = error = None
        if self.electric_gauge is not None:
            mean = float(np.sum(self.electric_gauge.weights*p))
            error = mean-self.electric_gauge.target_v
        return CUFParallelEndEvaluation(rows.mechanical, rows.electrical, mean, error)

    def equations(self, mechanical, electrical=None):
        result = self.evaluate(mechanical, electrical)
        return result.mechanical, result.electrical

    def solve(self, mechanical, electrical=None, *,
              mechanical_state_scale, mechanical_equation_scale,
              electrical_state_scale=None, electrical_equation_scale=None,
              settings=StrongNewtonSettings()):
        """Explicit opt-in solve; construction/evaluation never starts Newton.

        Equation scales must cover the physical units of each bulk/end row;
        fourth-order natural rows and displacement rows generally differ.
        """
        q, p = self._state(mechanical, electrical)
        return solve_strong_nonlinear_equations(
            self.equations, q, p,
            mechanical_state_scale=mechanical_state_scale,
            mechanical_equation_scale=mechanical_equation_scale,
            electrical_state_scale=electrical_state_scale,
            electrical_equation_scale=electrical_equation_scale,
            electric_gauge=self.electric_gauge, settings=settings)


__all__ = ("CUFParallelEndEvaluation", "CUFParallelEndProblem")
