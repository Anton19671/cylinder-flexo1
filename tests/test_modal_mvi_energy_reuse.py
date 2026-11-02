"""Verify that physical Rayleigh actions are reused without changing modes.

Only duplicated evaluations of an UNCHANGED physical field are avoided:
independent checks of EVERY newly computed Y/X field still run.
"""
from unittest.mock import patch

import numpy as np

from cylinder_modal_reference import solve_modal_case
from cylinder_modal_mvi import _SeparatedModalProblem
from modal_mvi_physical_smoke import reference_case


def test_modal_mvi_energy_reuse_keeps_exact_sweep_and_detailed_journal():
    case = reference_case()
    # A single fixed random start isolates Rayleigh action counts without
    # changing solver physics. This is NOT a spectrum-completeness test;
    # the unchanged multi-start benchmarks check that separately.
    seed = [{"kind": "random", "index": 0}]
    original = _SeparatedModalProblem.eigenvalue

    def run(detail):
        calls = [0]
        events = []

        def count(self, q, p):
            calls[0] += 1
            return original(self, q, p)

        with patch("cylinder_modal_mvi._modal_seed_plan", return_value=seed):
            with patch.object(_SeparatedModalProblem, "eigenvalue", count):
                output = solve_modal_case(
                    case,
                    detail_callback=(lambda **entry: events.append(entry))
                    if detail else None)
        sweeps = int(output["mvi_sweeps"][0])
        assert output["mvi_multistart_audit"][0]["tried"] == 1
        # One initial full physical energy plus one for each NEW Y/X field.
        # No original physical check is eliminated or weakened.
        assert calls[0] == 1 + 2 * sweeps
        if detail:
            assert sum(e["stage"] == "MVI_HALFSTEP_HARMONIC_AUDIT"
                       for e in events) == 2 * sweeps
            assert sum(e["stage"] == "MVI_VARIATIONAL_CHECK"
                       for e in events) == 2 * sweeps
            assert sum(e["stage"] == "MVI_CANDIDATE_INITIAL_STATE"
                       for e in events) == 1
        return output

    quiet = run(False)
    verbose = run(True)
    np.testing.assert_allclose(
        quiet["frequencies_hz"], verbose["frequencies_hz"],
        rtol=1e-13, atol=1e-10)
    np.testing.assert_allclose(
        quiet["mechanical_modes"], verbose["mechanical_modes"],
        rtol=1e-10, atol=1e-10)
