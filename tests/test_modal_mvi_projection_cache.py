"""Identical frozen partner matrices may be reused across modal starts."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cylinder_modal_mvi import _SeparatedModalProblem


class ProjectionCacheTests(unittest.TestCase):
    def test_reuse_only_identical_basis_and_keep_cached_values_immutable(self):
        op = _SeparatedModalProblem.__new__(_SeparatedModalProblem)
        op.ne, op.rank, op.density = 0, 1, 7800.
        op.prepared = SimpleNamespace(problem=object())
        op._local_tangent = object()
        a = np.eye(6, 2)
        b = a.copy()
        b[0, 0] = 2.
        with patch('cylinder_modal_separated_projector.exact_separated_cuf_projection',
                   side_effect=lambda *args: (args[-1].T @ args[-1],
                                              2. * args[-1].T @ args[-1])) as project:
            first = op.cached_cuf_projection(a)
            first[0][0, 0] = -99.
            again = op.cached_cuf_projection(a.copy())
            np.testing.assert_array_equal(again[0], np.eye(2))
            np.testing.assert_array_equal(again[1], 2 * np.eye(2))
            self.assertEqual(project.call_count, 1)
            changed = op.cached_cuf_projection(b)
            self.assertEqual(project.call_count, 2)
            self.assertEqual(changed[0][0, 0], 4.)
            np.testing.assert_array_equal(op.cached_cuf_projection(a)[0], np.eye(2))


if __name__ == '__main__':
    unittest.main()
