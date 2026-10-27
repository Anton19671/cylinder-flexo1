"""Small-grid physical variational FDM adapter for two-stage Gauss IRK4.

Uses the same seven-shell G3D/Gz enthalpy, exact inertia, damping, charge
equations and affine boundary compiler as the Newmark reference. This is a
FULL NODAL FDM diagnostic, emphatically NOT rank-fixed MVI+IRK.

Physical essential mechanical traces must be stationary on the time step.
For moving supports, special IRK end constraints must first be derived.
"""
from __future__ import annotations

import numpy as np

from cylinder_boundary_spatial import affine_trace_space
from cylinder_gauss2_rk import (Gauss2Settings, Gauss2State, GAUSS2_C,
                                 gauss2_step)
from cylinder_variational_fdm import (StaticFDMSettings, state_scales,
                                      solve_checked_reduced_tangent)


def _stage_loads(loads_at, time_s):
    result=loads_at(float(time_s))
    if not isinstance(result, tuple) or len(result)!=2:
        raise TypeError("loads_at(t) must return (mechanical_load,free_charge)")
    return result


def _physical_assembled(problem, q, p, *, time_s, density_kg_m3,
                        acceleration=None, velocity=None,
                        damping_rate=0., eta_s=0., loads_at):
    dead, charge=_stage_loads(loads_at,time_s)
    kwargs=dict(dead_load=dead,free_charge=charge,time_s=time_s)
    if acceleration is not None:
        kwargs.update(acceleration=acceleration,density_kg_m3=density_kg_m3)
        if damping_rate or eta_s or problem.functional.theory=="KL":
            kwargs.update(velocity=velocity,
                          mass_damping_rate_per_s=damping_rate,eta_s=eta_s)
    return problem.assemble(q,p,**kwargs)


def _mechanical_trace(assembled, mechanical_size):
    matrix=np.asarray(assembled.space.matrix,float)
    target=np.asarray(assembled.space.target,float)
    if matrix.ndim!=2 or matrix.shape[1]<mechanical_size:
        raise ValueError("invalid physical trace matrix")
    # A combined electric gauge/Dirichlet row is not a mechanical support.
    mechanical_only=~np.any(matrix[:,mechanical_size:]!=0.,axis=1)
    return matrix[mechanical_only,:mechanical_size],target[mechanical_only]


def _require_fixed_supports(problem, state, settings, density, loads_at):
    q,p=state.mechanical,state.electrical
    t0=state.time_s
    times=(t0,*(t0+settings.dt_s*GAUSS2_C),t0+settings.dt_s)
    traces=[]
    for t in times:
        a=_physical_assembled(problem,q,p,time_s=float(t),
                              density_kg_m3=density,loads_at=loads_at)
        traces.append(_mechanical_trace(a,q.size))
    c0,d0=traces[0]
    for c,d in traces[1:]:
        if (c.shape!=c0.shape or d.shape!=d0.shape or
                not np.allclose(c,c0,rtol=0.,atol=1e-13) or
                not np.allclose(d,d0,rtol=0.,atol=1e-13)):
            raise ValueError("GAUSS2_MOVING_MECHANICAL_SUPPORT_REQUIRES_END_CLOSURE")
    q0=state.mechanical.ravel()
    v0=state.velocity.ravel()
    if (np.max(np.abs(c0@q0-d0),initial=0)>1e-10 or
            np.max(np.abs(c0@v0),initial=0)>1e-10):
        raise ValueError("GAUSS2_INITIAL_SUPPORT_INCONSISTENT")
    return c0,d0


def _electric_endpoint(problem, q, v, t, guess, *,
                       loads_at, scales, energy_scale, newton):
    """Solve true endpoint electric saddle equations at frozen q(t_end).

    Treat mechanical DOFs as fixed data; only the p-block of physical
    Dirichlet/gauge constraints and the electric first variation is solved.
    """
    assembled=_physical_assembled(problem,q,guess,time_s=t,density_kg_m3=None,
                                  loads_at=loads_at)
    qsize=q.size
    c=np.asarray(assembled.space.matrix,float)
    d=np.asarray(assembled.space.target,float)
    e=c[:,qsize:]
    qpart=c[:,:qsize]@q.ravel()
    has_e=np.any(e!=0.,axis=1)
    if np.max(np.abs(qpart[~has_e]-d[~has_e]),initial=0)>1e-10:
        raise ValueError("GAUSS2_ENDPOINT_MECHANICAL_SUPPORT_INCONSISTENT")
    ps=scales[qsize:]
    space=affine_trace_space(e[has_e]*ps[None,:],
                             d[has_e]-qpart[has_e],guess.ravel()/ps)
    b=space.basis
    dead,charge=_stage_loads(loads_at,t)

    def make_p(z):
        return (ps*(space.origin+b@z)).reshape(guess.shape)

    def evaluate(z):
        potential=make_p(z)
        _,full,_=problem.evaluate_effective_residual(
            q,potential,time_s=t,dead_load=dead,free_charge=charge)
        electric=np.asarray(full,float)[qsize:]
        if electric.shape!=(guess.size,) or not np.isfinite(electric).all():
            raise ValueError("invalid Gauss electric endpoint equations")
        return b.T@(ps*electric)/energy_scale

    z=np.zeros(b.shape[1])
    if not z.size:
        return make_p(z)
    g=evaluate(z)
    for iteration in range(newton.max_iterations+1):
        if float(np.max(np.abs(g),initial=0))<=newton.absolute_tolerance_scaled:
            return make_p(z)
        if iteration==newton.max_iterations:
            break
        jac=np.empty((len(z),len(z)),float)
        for j in range(len(z)):
            eps=newton.tangent_step_scaled*(1.+abs(z[j]))
            unit=np.zeros_like(z);unit[j]=eps
            jac[:,j]=(evaluate(z+unit)-evaluate(z-unit))/(2*eps)
        try:
            step,_=solve_checked_reduced_tangent(jac,g)
        except np.linalg.LinAlgError as exc:
            raise ValueError("GAUSS2_ENDPOINT_ELECTRIC_TANGENT_FAILED") from exc
        old=float(np.max(np.abs(g),initial=0))
        for halve in range(newton.max_halvings+1):
            ztry=z+2.**(-halve)*step
            new_g=evaluate(ztry)
            if (float(np.max(np.abs(new_g),initial=0))<old or
                    float(np.max(np.abs(new_g),initial=0))<=newton.absolute_tolerance_scaled):
                z,g=ztry,new_g
                break
        else:
            raise ValueError("GAUSS2_ENDPOINT_ELECTRIC_LINE_SEARCH_FAILED")
    raise ValueError("GAUSS2_ENDPOINT_ELECTRIC_NOT_SOLVED")


def solve_gauss2_variational_fdm_step(
        problem, initial, settings, *, density_kg_m3,
        loads_at, newton_settings=None,
        mass_damping_rate_per_s=0., eta_s=0.):
    """One *physical shell* Gauss-IRK4 FDM step, with no MVI substitution.

    Every interior stage evaluates the actual coupled shell functional,
    inertia and selected dissipation, and imposes the time-specific
    essential/electric/gauge constraints via affine elimination. Calls
    to this routine do not change the default Newmark CASE or GUI route.

    This small-grid diagnostic refuses moving essential mechanical supports
    until endpoint-compatible constrained Gauss collocation is implemented.
    """
    if not isinstance(initial,Gauss2State) or not isinstance(settings,Gauss2Settings):
        raise TypeError("Gauss2State and Gauss2Settings required")
    if not callable(loads_at) or getattr(problem,"kind",None)!="DYNAMIC":
        raise ValueError("DYNAMIC variational problem and time-dependent loads callback required")
    newton=StaticFDMSettings() if newton_settings is None else newton_settings
    if not isinstance(newton,StaticFDMSettings):
        raise TypeError("StaticFDMSettings required")
    try:
        density=float(density_kg_m3)
        damping=float(mass_damping_rate_per_s)
        eta=float(eta_s)
    except (TypeError,ValueError) as exc:
        raise ValueError("finite density, damping and viscosity required") from exc
    if (not np.isfinite([density,damping,eta]).all() or
            density<=0 or damping<0 or eta<0):
        raise ValueError("density>0 and nonnegative finite damping required")
    f=problem.functional
    if bool(f.electrical_on)!=(initial.electrical is not None):
        raise ValueError("initial electric modes disagree with physical switch")
    q0,p0=initial.mechanical,initial.electrical
    scales=state_scales(f,q0,p0,newton)
    qsize=q0.size
    energy_scale=float(f.material.young_pa*f.h*np.sum(f.W))
    if not np.isfinite(energy_scale) or energy_scale<=0:
        raise ValueError("invalid variational energy scale")
    c0,d0=_require_fixed_supports(problem,initial,settings,density,loads_at)

    def stage_equations(q,v,a,p,t):
        assembled=_physical_assembled(
            problem,q,p,time_s=t,acceleration=a,velocity=v,
            density_kg_m3=density,damping_rate=damping,eta_s=eta,
            loads_at=loads_at)
        u=q.ravel() if p is None else np.concatenate((q.ravel(),p.ravel()))
        scaled=u/scales
        space=affine_trace_space(assembled.space.matrix*scales[None,:],
                                 assembled.space.target,scaled)
        b=space.basis
        full=np.asarray(assembled.full_residual,float)
        if full.shape!=scales.shape or not np.isfinite(full).all():
            raise ValueError("invalid physical stage equations")
        # Orthogonal admissible balance + complementary essential equations.
        # Both are dimensionless; their subspaces cannot cancel each other.
        equations=b@(b.T@(scales*full)/energy_scale)+(scaled-space.project(scaled))
        return (equations[:qsize].reshape(q0.shape),
                None if p is None else equations[qsize:].reshape(p.shape))

    endpoint=None
    if p0 is not None:
        def endpoint(q,v,t,pguess):
            return _electric_endpoint(
                problem,q,v,t,pguess,loads_at=loads_at,scales=scales,
                energy_scale=energy_scale,newton=newton)
    result=gauss2_step(
        initial,settings,equations=stage_equations,
        mechanical_state_scale=scales[:qsize].reshape(q0.shape),
        mechanical_equation_scale=1.,
        electrical_state_scale=(None if p0 is None else
                                 scales[qsize:].reshape(p0.shape)),
        electrical_equation_scale=None if p0 is None else 1.,
        end_electrical_solver=endpoint)
    # Gauss stage constraints do not automatically guarantee a physical
    # endpoint under arbitrary prescribed support motion: never hide a
    # poststep projection that would change q, v or its temporal order.
    if (np.max(np.abs(c0@result.final.mechanical.ravel()-d0),initial=0)>1e-9 or
            np.max(np.abs(c0@result.final.velocity.ravel()),initial=0)>1e-9):
        raise ValueError("GAUSS2_ENDPOINT_MECHANICAL_SUPPORT_INCONSISTENT")
    return result


__all__=("solve_gauss2_variational_fdm_step",)
