#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from cylinder_final_stage11a_dispatcher import _build_auto_mvi_profiles


def _as_direction_columns(array: np.ndarray, rank_axis: int) -> np.ndarray:
    return np.moveaxis(np.asarray(array, dtype=float), rank_axis, 0).reshape(
        array.shape[rank_axis], -1
    ).T


def run_checks(verbose: bool = True) -> dict:
    results = {}
    for cuf_model in ("TE1","TE2","TE3","TE4"):
        case = {
            "theory": "CUF",
            "cuf_model": cuf_model,
            "nx": 41,
            "ns": 41,
            "stencil": 7,
            "L": 2000.0,
            "R": 1000.0,
            "h": 50.0,
        }
        reference_x, reference_y = _build_auto_mvi_profiles(case, 1)
        model_results = {}
        for rank in range(1, 9):
            X, Y = _build_auto_mvi_profiles(case, rank)
            if not np.array_equal(X[:, :, :1], reference_x):
                raise AssertionError(f"{cuf_model} MVI-{rank}: rank-1 X seed changed")
            if not np.array_equal(Y[:, :, :1], reference_y):
                raise AssertionError(f"{cuf_model} MVI-{rank}: rank-1 Y seed changed")

            x_columns = _as_direction_columns(X, rank_axis=2)
            numerical_rank = int(np.linalg.matrix_rank(x_columns))
            if numerical_rank != rank:
                raise AssertionError(
                    f"{cuf_model} MVI-{rank}: combined X bank rank "
                    f"{numerical_rank}, expected {rank}"
                )
            block_ranks = {}
            for component in range(3):
                for tau in range(X.shape[1]):
                    block = np.asarray(X[component,tau],float).T
                    block_rank = int(np.linalg.matrix_rank(block))
                    key=f"component={component},tau={tau}"
                    block_ranks[key] = block_rank
                    if block_rank != rank:
                        raise AssertionError(
                            f"{cuf_model} MVI-{rank}: {key} X bank rank "
                            f"{block_rank}, expected {rank}"
                        )

            max_abs_y = float(np.max(np.abs(Y)))
            if max_abs_y != 0.0:
                raise AssertionError(
                    f"{cuf_model} MVI-{rank}: plate-style initial Y is not zero: "
                    f"{max_abs_y:.6e}"
                )
            physical_field = np.einsum("cirk,cirs->cisk", X, Y)
            max_abs_field = float(np.max(np.abs(physical_field)))
            if max_abs_field != 0.0:
                raise AssertionError(
                    f"{cuf_model} MVI-{rank}: initial physical field is not zero: "
                    f"{max_abs_field:.6e}"
                )
            model_results[str(rank)] = {
                "X_combined_numerical_rank": numerical_rank,
                "X_block_numerical_ranks": block_ranks,
                "Y_max_abs": max_abs_y,
                "physical_field_max_abs": max_abs_field,
            }
        results[cuf_model] = model_results

    report = {
        "status": "PASS",
        "scope": "CUF TE1--TE4 automatic MVI profile bank, ranks 1..8",
        "rank_1_seed_unchanged": True,
        "initial_Y_zero": True,
        "initial_physical_field_zero": True,
        "results": results,
    }
    if verbose:
        print("MVI_1_8_PROFILE_BANK_SELF_CHECK_V14: PASS")
        for cuf_model, model_results in results.items():
            for rank, item in model_results.items():
                print(
                    f"  {cuf_model} MVI-{rank}: "
                    f"X-rank={item['X_combined_numerical_rank']}, "
                    f"max|Y|={item['Y_max_abs']:.1e}, "
                    f"max|field|={item['physical_field_max_abs']:.1e}"
                )
    return report

if __name__ == "__main__":
    run_checks(True)
