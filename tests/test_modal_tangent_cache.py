"""Modal local-operator and mass-orthogonality regression checks."""
import unittest
import numpy as np
from scipy.linalg import cholesky
from cylinder_modal_mvi import (_SeparatedModalProblem,
                                _significant_modal_constraints)
from cylinder_modal_reference import _modal_spec


class ModalCacheTests(unittest.TestCase):
    def test_constraint_filter_uses_mass_norm(self):
        mass = np.diag([4., 9., 16.])
        rows = np.array([[2e-8, 0., 0.], [0., 3e-4, 0.]])
        kept = _significant_modal_constraints(rows, cholesky(mass, lower=True))
        np.testing.assert_array_equal(kept, rows[1:])

    def test_cuf_local_action_matches_variational_tangent_with_spring(self):
        case = {
            'solver_backend': 'VARIATIONAL_REFERENCE', 'theory': 'CUF',
            'cuf_model': 'TE2', 'method': 'MVI', 'rank': 1,
            'reference_mvi': {'max_sweeps': 4}, 'calculation_kind': 'MODAL',
            'scale_model': 'MACRO', 'mode': 'LINEAR_ELASTIC',
            'geometry_model': 'GREEN_LAGRANGE', 'material_model': 'HOOKE',
            'units': {'system': 'SI_N_M_PA'},
            'geometry': {'L': 2., 'R': 1., 'h': .05},
            'material': {'E0': 210e9, 'nu0': .3},
            'quadrature': {'type': 'GAUSS', 'n_gauss': 5},
            'boundary_conditions': {'variational': {'mechanical': {
                'left': {'support': 'CLAMPED', 'higher': 'NATURAL'},
                'right': {'support': 'FREE', 'higher': 'NATURAL',
                          'spring': [10000.] * 9}}, 'electrical': {}}},
            'modal': {'density_kg_m3': 7800., 'n_modes': 1},
            'output_dir': 'modal_regression_unused',
            'reference_newton': {'tangent_step_scaled': 2e-6, 'max_free_dofs': 512},
            'mvi_grid': {'nx': 5, 'ns': 5, 'operator_scheme': 'SBP21_PERIODIC_ODD'},
        }
        prepared, density, _ = _modal_spec(case)
        operator = _SeparatedModalProblem(prepared, density, 1)
        # Nine components, 5x5 nodes, one clamped end (9x5 traces).
        self.assertEqual(operator.n_mechanical_free_dofs, 9 * 5 * 5 - 9 * 5)
        self.assertEqual(operator.n_electric_free_dofs, 0)
        direction = np.random.default_rng(829).normal(size=operator.nm) * operator.scales
        def force(sign):
            q = (sign * operator.step * direction).reshape(operator.shape)
            return prepared.problem.evaluate_effective_residual(q, None)[1]
        expected = (force(1.) - force(-1.)) / (2. * operator.step)
        actual = operator.stiffness(direction)
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 5e-7)


if __name__ == '__main__':
    unittest.main()

