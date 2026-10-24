#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculation-free identities for exact classical KL Green--Lagrange kinematics."""
import numpy as np

import cylinder_kl_full_gl_kinematics as gl


class _Grid:
    ns=4; nx=5; radius=3.2; dx=0.3; ds=0.4


class _Ops:
    grid=_Grid()
    rng=np.random.default_rng(41)
    D1x=0.2*rng.standard_normal((grid.nx,grid.nx))
    D1s=0.2*rng.standard_normal((grid.ns,grid.ns))
    D2x=0.1*rng.standard_normal((grid.nx,grid.nx))
    D2s=0.1*rng.standard_normal((grid.ns,grid.ns))
    def Dx(self,a):return np.asarray(a)@self.D1x.T
    def Ds(self,a):return self.D1s@np.asarray(a)
    def Dxx(self,a):return np.asarray(a)@self.D2x.T
    def Dss(self,a):return self.D2s@np.asarray(a)


class _Resultants:
    pass


def _zeros(shape):
    return np.zeros((3,)+shape,float)


def _max_state_measure(state):
    return max(float(np.max(np.abs(a))) for a in (
        state.eps_x,state.eps_s,state.gamma_xs,
        state.kappa_x,state.kappa_s,state.kappa_xs,
    ))


def _rigid_motion_checks():
    R=2.7
    theta=np.linspace(0.0,2.0*np.pi,17,endpoint=False)[:,None]
    shape=theta.shape

    # Arbitrary axial translation.
    U=_zeros(shape); U[0]=4.3
    state=gl.state_from_derivatives(
        U,_zeros(shape),_zeros(shape),_zeros(shape),_zeros(shape),_zeros(shape),R=R
    )
    assert _max_state_measure(state)<1e-14

    # Finite rigid rotation about the cylinder axis.
    alpha=0.63
    U=_zeros(shape)
    U[1]=R*np.sin(alpha)
    U[2]=R*(np.cos(alpha)-1.0)
    state=gl.state_from_derivatives(
        U,_zeros(shape),_zeros(shape),_zeros(shape),_zeros(shape),_zeros(shape),R=R
    )
    assert _max_state_measure(state)<2e-14

    # Rigid transverse translation written in the rotating local basis.
    shift=0.41
    U=_zeros(shape); Us=_zeros(shape); Uss=_zeros(shape)
    U[1]=-shift*np.sin(theta); U[2]=shift*np.cos(theta)
    Us[1]=-shift*np.cos(theta)/R; Us[2]=-shift*np.sin(theta)/R
    Uss[1]=shift*np.sin(theta)/(R*R); Uss[2]=-shift*np.cos(theta)/(R*R)
    state=gl.state_from_derivatives(
        U,_zeros(shape),Us,_zeros(shape),Uss,_zeros(shape),R=R
    )
    assert _max_state_measure(state)<2e-14


def _directional_derivative_check():
    rng=np.random.default_rng(20260909)
    shape=(4,5); R=3.2
    fields=[2e-2*rng.standard_normal((3,)+shape) for _ in range(6)]
    directions=[rng.standard_normal((3,)+shape) for _ in range(6)]
    state=gl.state_from_derivatives(*fields,R=R)
    tangent=gl.directional_variation(state,*directions,R=R)
    step=2e-7
    plus=gl.state_from_derivatives(
        *(a+step*d for a,d in zip(fields,directions)),R=R
    )
    minus=gl.state_from_derivatives(
        *(a-step*d for a,d in zip(fields,directions)),R=R
    )
    exact=(
        tangent.d_eps_x,tangent.d_eps_s,tangent.d_gamma_xs,
        tangent.d_kappa_x,tangent.d_kappa_s,tangent.d_kappa_xs,
    )
    fd=tuple((a-b)/(2.0*step) for a,b in zip(
        (plus.eps_x,plus.eps_s,plus.gamma_xs,plus.kappa_x,plus.kappa_s,plus.kappa_xs),
        (minus.eps_x,minus.eps_s,minus.gamma_xs,minus.kappa_x,minus.kappa_s,minus.kappa_xs),
    ))
    err=max(float(np.max(np.abs(a-b))) for a,b in zip(exact,fd))
    scale=max(1.0,max(float(np.max(np.abs(a))) for a in exact))
    assert err/scale<2e-8, (err,scale)


def _variational_residual_check():
    rng=np.random.default_rng(87)
    ops=_Ops()
    U=1e-2*rng.standard_normal((3,ops.grid.ns,ops.grid.nx))
    state=gl.field_kinematics(U,ops)
    nm=_Resultants()
    for name in ("Nx","Ns","Nxs","Mx","Ms","Mxs"):
        setattr(nm,name,rng.standard_normal((ops.grid.ns,ops.grid.nx)))
    p=np.zeros_like(U)
    residual,_=gl.variational_strong_residual(state,nm,ops,p)
    dU=rng.standard_normal(U.shape)
    dUx=np.stack([ops.Dx(q) for q in dU])
    dUs=np.stack([ops.Ds(q) for q in dU])
    dUxx=np.stack([ops.Dxx(q) for q in dU])
    dUss=np.stack([ops.Dss(q) for q in dU])
    dUxs=np.stack([ops.Ds(ops.Dx(q)) for q in dU])
    dv=gl.directional_variation(
        state,dU,dUx,dUs,dUxx,dUss,dUxs,R=ops.grid.radius
    )
    density=(nm.Nx*dv.d_eps_x+nm.Ns*dv.d_eps_s+nm.Nxs*dv.d_gamma_xs
             +nm.Mx*dv.d_kappa_x+nm.Ms*dv.d_kappa_s
             +2.0*nm.Mxs*dv.d_kappa_xs)
    W=gl.surface_weights(ops.grid)
    lhs=float(np.sum(W*residual*dU))
    rhs=float(np.sum(W*density))
    assert abs(lhs-rhs)<2e-11*max(1.0,abs(rhs)),(lhs,rhs)


def main():
    _rigid_motion_checks()
    _directional_derivative_check()
    _variational_residual_check()
    print("KL_FULL_GL_KINEMATICS_SELF_CHECK: PASS")


if __name__=="__main__":
    main()
