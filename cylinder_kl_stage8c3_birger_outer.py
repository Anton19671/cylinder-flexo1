#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8C-3: KL + Birger outer-cycle adapter.

The Birger outer algorithm is NOT rewritten.
This module injects KL functions into the already verified generic
Stage5A solve_birger_outer_cycle.

Pipeline:
state_guess, GB^k
 -> complete strict KL inner MVI at frozen GB^k
 -> KL plane-stress epsilon_i,sigma_i
 -> generic Stage5A GB update
 -> MANDATORY complete strict KL inner MVI at updated GB^(k+1)
 -> final material evaluation at the same updated GB
 -> final eta_G, eta_sigma, eta_u acceptance criteria.

No GB update inside inner MVI.
No new relaxation, line-search or solver.
"""
from __future__ import annotations
from dataclasses import dataclass

import cylinder_te1_stage5a_birger as b5
import cylinder_kl_stage8c2_frozen_gb_mvi as bi
import cylinder_kl_stage8b2_profiles as kp


@dataclass
class KLInnerSolveCounter:
    calls: int = 0


def make_kl_inner_solve(
    ops,maps,bc,*,K0,quadrature,geometry,generalized_load,mvi_settings,
    counter=None
):
    if counter is None:
        counter=KLInnerSolveCounter()

    def inner_solve(state_guess,GB):
        counter.calls += 1
        return bi.solve_frozen_birger_inner(
            state_guess,ops,maps,bc,
            K0=K0,GB=GB,quadrature=quadrature,geometry=geometry,
            generalized_load=generalized_load,mvi_settings=mvi_settings
        )

    inner_solve.counter=counter
    return inner_solve


def make_kl_material_evaluator(ops,*,K0,quadrature,geometry):
    def evaluate_material(state,GB):
        return bi.evaluate_frozen_material_state(
            state,ops,K0=K0,GB=GB,quadrature=quadrature,geometry=geometry
        )
    return evaluate_material


def solve_kl_birger_load_step(
    initial_state,initial_GB,
    *,
    G0,
    material_diagram,
    birger_settings,
    ops,maps,bc,
    K0,quadrature,geometry,generalized_load,mvi_settings,
    previous_accepted_epsilon_i=None,
    inner_counter=None,
):
    inner_solve=make_kl_inner_solve(
        ops,maps,bc,K0=K0,quadrature=quadrature,geometry=geometry,
        generalized_load=generalized_load,mvi_settings=mvi_settings,
        counter=inner_counter
    )
    evaluator=make_kl_material_evaluator(
        ops,K0=K0,quadrature=quadrature,geometry=geometry
    )

    result=b5.solve_birger_outer_cycle(
        initial_state,
        initial_GB,
        G0=G0,
        material_diagram=material_diagram,
        inner_solve=inner_solve,
        evaluate_material=evaluator,
        state_vector=lambda st: kp.reconstruct_vector(st,ops.grid),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
    )
    return result,inner_solve.counter
