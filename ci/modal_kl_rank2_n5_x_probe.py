#!/usr/bin/env python3
"""Regression for production KL MVI-2 missing n=5 doublet, C-C 13x13."""
import runpy
import sys
import numpy as np

from cylinder_paths import activate
activate()
from modal_mvi_rawat_te1_cc_audit import CONFIG, read_case
from cylinder_modal_reference import _modal_spec
from modal_mvi_spectrum_audit import degenerate_subspace_mass_mac

sys.argv = ["ci/modal_mvi_all_ranks_theories_audit.py", "KL", "2"]
runpy.run_path(sys.argv[0], run_name="__main__")
with np.load("modal_KL_rank2_FDM.npz") as a, np.load("modal_KL_rank2_MVI_2.npz") as b:
    ref_freq, freq = a["frequencies_hz"], b["frequencies_hz"]
    reference, trial = a["mechanical_modes"], b["mechanical_modes"]
np.testing.assert_allclose(freq, ref_freq, rtol=1e-6, atol=1e-5)
case = read_case(CONFIG / "fdm_13x13" / "RAWAT2016_KL_13x13_MODAL_FDM.json")
case["modal"]["n_modes"] = 8
prepared, density, _ = _modal_spec(case)
for first in range(0, 8, 2):
    overlap = degenerate_subspace_mass_mac(
        reference[first:first+2], trial[first:first+2],
        prepared.problem.functional, density)
    print(f"[KL-RANK2] modes {first+1}-{first+2}: mass subspace overlaps {overlap}", flush=True)
    if np.min(overlap) < 1. - 1.e-6:
        raise AssertionError("Degenerate modal subspace differs from FDM")
print("[KL-RANK2] production spectrum and paired shapes verified", flush=True)
