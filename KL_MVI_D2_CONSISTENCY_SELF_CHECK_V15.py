#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small checks for the V15 KL-MVI D2 consistency correction.

No load series and no shell solve are started.  The check proves that every
MVI variation is differentiated by the same direct D2 operators as the
current KL kinematics, then compares the projected residual with an
independent finite difference of the discrete total potential.
"""
from __future__ import annotations

import numpy as np

import cylinder_final_stage11a_dispatcher as dispatch
import cylinder_kl_stage8b1_boundary as kb
import cylinder_kl_stage8b2_profiles as kp
import cylinder_kl_stage8b3_essential_bc as ke
import cylinder_kl_stage8b4_weak_projected as kw
import cylinder_te1_mvi_core as core
import cylinder_te1_stage3a_connector as c3


def _perturbed(state,maps,labels,direction,delta):
    out=state.copy()
    if direction=="Y":
        for value,(a,r,j) in zip(delta,labels):
            out.Y[a,r,j]+=value
        return out
    reduced=[a.copy() for a in ke.reduce_constrained_state_X(state,maps)]
    for value,(a,r,j) in zip(delta,labels):
        reduced[a][r,j]+=value
    return ke.constrain_state_X_from_reduced(reduced,state,maps)


def _energy_direction_error(state,ops,maps,bc,load,direction,rng):
    r,labels,_,_=kw.elastic_projected_residual(
        state,ops,maps,bc,E=210000.0,nu=0.3,h=50.0,
        geometry="VON_KARMAN",generalized_load=load,direction=direction
    )
    d=rng.standard_normal(len(labels)); d/=np.linalg.norm(d)
    predicted=float(np.dot(r,d))

    def energy(alpha):
        st=_perturbed(state,maps,labels,direction,alpha*d)
        return kw.total_potential_elastic(
            st,ops,bc,E=210000.0,nu=0.3,h=50.0,
            geometry="VON_KARMAN",generalized_load=load
        )

    # Use the better of two independent central differences to avoid making
    # this guard sensitive to one particular floating-point step.
    errors=[]
    for hfd in (1.0e-5,3.0e-6):
        measured=(energy(hfd)-energy(-hfd))/(2.0*hfd)
        errors.append(abs(measured-predicted)/max(abs(measured),abs(predicted),1.0))
    return min(errors)


def main():
    grid=core.CylinderGrid(2000.0,1000.0,9,9)
    ops=c3.CylinderSpatialOperators.build(grid,7)
    bc=kb.KLCylinderEndBC.clamped_clamped()
    maps=ke.build_x_constraint_maps(ops,bc)
    cfg={"theory":"KL","cuf_model":None,"nx":grid.nx,"ns":grid.ns,
         "stencil":7,"L":grid.length_x,"R":grid.radius,"h":50.0}
    X,Y=dispatch._build_auto_mvi_profiles(cfg,2,bc=bc)
    state=kp.ProfileStateKL(500.0*X,Y)

    worst=0.0
    for direction in ("X","Y"):
        for var in kw.direction_variations(state,ops,maps,direction):
            for a in range(3):
                worst=max(worst,float(np.max(np.abs(
                    var.dUxx[a]-ops.Dxx(var.dU[a])
                ))))
                worst=max(worst,float(np.max(np.abs(
                    var.dUss[a]-ops.Dss(var.dU[a])
                ))))
    if worst>5.0e-14:
        raise AssertionError(f"KL MVI variation D2 mismatch: {worst:.3e}")

    # Ensure the test grid would detect an accidental return to D1@D1.
    if np.allclose(ops.D2x,ops.D1x@ops.D1x) or np.allclose(ops.D2s,ops.D1s@ops.D1s):
        raise AssertionError("self-check grid does not distinguish D2 from D1@D1")

    load=np.zeros((3,grid.ns,grid.nx),float); load[2]=-500.0
    rng=np.random.default_rng(20260822)
    err_x=_energy_direction_error(state,ops,maps,bc,load,"X",rng)
    err_y=_energy_direction_error(state,ops,maps,bc,load,"Y",rng)
    if max(err_x,err_y)>2.0e-6:
        raise AssertionError(
            f"projected residual / potential derivative mismatch: X={err_x:.3e}, Y={err_y:.3e}"
        )

    print("KL_MVI_D2_CONSISTENCY_SELF_CHECK_V15: PASS")
    print(f"  variation D2 identity max error = {worst:.3e}")
    print(f"  energy-gradient relative X     = {err_x:.3e}")
    print(f"  energy-gradient relative Y     = {err_y:.3e}")
    print("  full KL solve                  = NOT RUN")


if __name__=="__main__":
    main()
