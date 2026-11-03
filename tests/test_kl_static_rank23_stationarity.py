"""Real static regression of KL C-C MVI-2 and MVI-3 vs independent FDM."""
import json
from pathlib import Path
import numpy as np
import pytest
from cylinder_reference_case import prepare_reference_case
from cylinder_variational_mvi import StaticMVISettings, solve_static_mvi
from cylinder_variational_fdm import solve_static_fdm

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "config/verification/kl_cc_uniform_static"

def load(label):
    path = CASES / (label + ".json")
    return prepare_reference_case(json.loads(path.read_text()), base_dir=path.parent)

@pytest.mark.parametrize("rank", [2, 3])
def test_kl_cc_low_rank_stationarity(rank):
    case = load(f"KL_CC_9x9_100KPA_MVI{rank}")
    result = solve_static_mvi(
        case.problem, case.mechanical, case.electrical,
        dead_load=case.dead_load, settings=case.settings,
        mvi_settings=StaticMVISettings(rank=rank, max_sweeps=5))
    print("MVI", rank, "status", result.status, "sweeps", result.sweeps)
    for record in result.history:
        if record.get("event") == "sweep":
            print("SWEEP", record["sweep"], "potential_j",
                  record["potential_j"], "change",
                  record.get("sweep_field_change_relative"))
    assert result.converged, result.status
    assert result.status == "CONVERGED_FULL"
    assert result.mvi_certificate["accepted"]
    assert result.sweeps <= 5
    assert np.max(np.abs(result.mechanical[:, [0, -1], :])) < 1e-11

    fdm = load("KL_CC_9x9_100KPA_MKR_FDM")
    reference = solve_static_fdm(fdm.problem, fdm.mechanical, fdm.electrical,
                                 dead_load=fdm.dead_load, settings=fdm.settings)
    assert reference.converged
    relative = np.linalg.norm(result.mechanical - reference.mechanical) / np.linalg.norm(reference.mechanical)
    print("MVI", rank, "relative field difference", relative)
    assert relative < 1e-6
