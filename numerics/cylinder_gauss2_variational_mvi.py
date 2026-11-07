"""Opt-in seven-shell variational adapter for two-stage fixed-rank MVI+IRK.

The same physical coupled enthalpy, natural boundary contributions, inertia,
viscous laws and quasi-static electric equations as the Newmark path are
sampled at both true Gauss times. Spatial stages are solved by coupled rank-R
Y/Y then X/X iterations, never by a full-nodal FDM stage solve.

For now essential mechanical traces MUST remain stationary across the step.
The independent dynamic Newmark solver and GUI dispatch are not changed.
"""
from __future__ import annotations

import numpy as np

from cylinder_gauss2_rk import Gauss2Settings, Gauss2State
from cylinder_gauss2_mvi import (
    Gauss2MVISettings, solve_gauss2_mvi_step, _rank_factors, _profile_field,
    _normalize, _directions, _constrained_directions, _reduced_newton,
)
from cylinder_mvi_subspace_newton import (
    mvi_settings_from_options,mvi_state_scales,
)
from cylinder_gauss2_mvi_physics import (_physical_assembled,
                                       _require_fixed_supports,
                                       _stage_loads)


def _end_electric_mvi(problem, q, v, t, guess, rank, *, scales,
                      loads_at, settings, energy_scale):
    """Quasi-static electric saddle solve in endpoint rank-R X/Y space ONLY."""
    qs = q.size
    scale = np.asarray(scales, float).ravel()[qs:].reshape(guess.shape)
    x, y = _rank_factors(guess / scale, rank,
                         settings.endpoint_rank_tolerance,
                         'GAUSS2_MVI_ENDPOINT_ELECTRIC_INITIAL')

    def constraints(p):
        a = _physical_assembled(problem, q, p, time_s=t,
                                density_kg_m3=None, loads_at=loads_at)
        c = np.asarray(a.space.matrix, float)
        d = np.asarray(a.space.target, float)
        if c.shape[1] != qs + p.size:
            raise ValueError('GAUSS2_MVI_ENDPOINT_ELECTRIC_TRACE_SHAPE')
        electric = c[:, qs:]
        target = d - c[:, :qs] @ q.ravel()
        active = np.any(electric != 0., axis=1)
        if np.max(abs(target[~active]), initial=0) > 1e-10:
            raise ValueError('GAUSS2_MVI_ENDPOINT_MECHANICAL_SUPPORT_MISMATCH')
        return electric[active], target[active]

    def force(p):
        dead, charge = _stage_loads(loads_at, t)
        _, r, _ = problem.evaluate_effective_residual(
            q, p, dead_load=dead, free_charge=charge, time_s=t)
        r = np.asarray(r, float)
        if r.shape != (qs+p.size,) or not np.isfinite(r).all():
            raise ValueError('invalid physical electric endpoint equations')
        return (scale.ravel() * r[qs:]) / energy_scale

    # The electric constant mode can remain zero if its driving derivative
    # is zero; no artificial electric inertia is ever introduced.
    if not np.any(y):
        r0 = force(_profile_field(x, y)*scale)
        grad = r0.reshape(guess.shape)
        from scipy.linalg import qr
        for k in range(len(x)):
            if np.any(grad[k]):
                qbasis, _, _ = qr(grad[k], mode='economic', pivoting=True,
                                 check_finite=False)
                x[k] = qbasis[:, :rank].T

    tolerance = settings.absolute_tolerance
    for sweep in range(settings.max_sweeps):
        prior = _profile_field(x, y)
        for axis in ('Y', 'X'):
            x, y = _normalize(x, y, axis)
            current = _profile_field(x, y).ravel()
            directions = _directions(x, y, axis)
            mat, target = constraints((scale.ravel()*current).reshape(guess.shape))
            origin, basis = _constrained_directions(
                current, directions, mat*scale.ravel()[None, :], target)
            def reduced(values):
                trial = (scale.ravel()*values[0]).reshape(guess.shape)
                return [force(trial)]
            states, _ = _reduced_newton([origin], [basis], reduced, settings)
            difference = states[0]-current
            change = directions.T @ difference
            if np.max(abs(directions@change-difference), initial=0) > (
                    5e-10*max(1.,float(np.max(abs(difference),initial=0)))):
                raise ValueError('GAUSS2_MVI_ELECTRIC_FORBIDDEN_NODAL_CORRECTION')
            if axis == 'Y':
                y += change.reshape(y.shape)
            else:
                x += change.reshape(x.shape)
        new = _profile_field(x,y)
        p = new*scale
        mat,target=constraints(p)
        if np.max(abs(mat@p.ravel()-target),initial=0)>1e-9*max(
                1.,float(np.max(abs(target),initial=0))):
            raise ValueError('GAUSS2_MVI_ELECTRIC_ENDPOINT_TRACE_NOT_SATISFIED')
        all_projected = []
        for axis in ('Y', 'X'):
            xx, yy = _normalize(x, y, axis)
            tmat = _directions(xx,yy,axis)
            _, basis = _constrained_directions(new.ravel(), tmat,
                                               mat*scale.ravel()[None,:],target)
            all_projected.append(float(np.max(abs(basis.T@force(p)),initial=0)))
        difference = np.linalg.norm(new-prior)/max(
            np.linalg.norm(new),np.linalg.norm(prior),1e-30)
        if max(all_projected)<=tolerance and difference<=settings.field_change_tolerance:
            return p
    raise ValueError('GAUSS2_MVI_ELECTRIC_ENDPOINT_NOT_CONVERGED')


def solve_gauss2_variational_mvi_step(
        problem, initial, settings, *, mvi_settings, density_kg_m3,
        loads_at, newton_settings=None, mass_damping_rate_per_s=0.,
        eta_s=0., previous=None):
    """One physical Gauss--IRK4 step with BOTH stages solved by coupled X/Y.

    Fully variational shell stage operators are shared with Newmark. Supports
    cannot move within the step until collocation-compatible moving mechanical
    trace kinematics are implemented; electric electrode values may vary.
    """
    if (getattr(problem, 'kind', None) != 'DYNAMIC' or
            not isinstance(initial, Gauss2State) or
            not isinstance(settings, Gauss2Settings) or
            not isinstance(mvi_settings, Gauss2MVISettings)):
        raise TypeError('DYNAMIC problem, Gauss2State/Settings and MVI settings required')
    if not callable(loads_at):
        raise TypeError('physical loads_at(t) callback required')
    newton = mvi_settings_from_options(newton_settings)
    try:
        density,damping,viscosity=map(float,(
            density_kg_m3,mass_damping_rate_per_s,eta_s))
    except (TypeError,ValueError) as exc:
        raise ValueError('finite physical dynamic coefficients required') from exc
    if (not np.isfinite([density,damping,viscosity]).all() or
            density<=0 or damping<0 or viscosity<0):
        raise ValueError('density>0 and damping/viscosity nonnegative required')
    f = problem.functional
    if bool(f.electrical_on)!=(initial.electrical is not None):
        raise ValueError('Gauss2 MVI electric switch/state mismatch')
    q0,p0 = initial.mechanical, initial.electrical
    scale = mvi_state_scales(f,q0,p0,newton)
    combined_shape=(q0.shape[0]+(0 if p0 is None else p0.shape[0]),*q0.shape[1:])
    scale=scale.reshape(combined_shape)
    energy=float(f.material.young_pa*f.h*np.sum(f.W))
    if not np.isfinite(energy) or energy<=0:
        raise ValueError('invalid physical energy scale')
    # Check mechanical constraints at t0, Gauss stages, and endpoint before
    # any stage Newton evaluations. This is a strict refusal of moving BCs.
    c0,d0=_require_fixed_supports(problem,initial,settings,density,loads_at)
    qsize=q0.size

    def eq(q,v,a,p,t):
        dead,charge = _stage_loads(loads_at,t)
        _, r, _ = problem.evaluate_effective_residual(
            q,p,dead_load=dead,free_charge=charge,time_s=t,
            acceleration=a,density_kg_m3=density,
            velocity=v if (damping or viscosity or f.theory=='KL') else None,
            mass_damping_rate_per_s=damping,eta_s=viscosity)
        r=np.asarray(r,float)
        if r.size!=scale.size or not np.isfinite(r).all():
            raise ValueError('invalid physical Gauss MVI stage equations')
        return (r[:qsize].reshape(q0.shape),
                None if p0 is None else r[qsize:].reshape(p0.shape))

    def constraints(stage,t,field):
        q=field[:q0.shape[0]].copy()
        p=None if p0 is None else field[q0.shape[0]:].copy()
        assembled=_physical_assembled(
            problem,q,p,time_s=t,density_kg_m3=density,loads_at=loads_at)
        return np.asarray(assembled.space.matrix,float),np.asarray(assembled.space.target,float)

    end_solver=None
    if p0 is not None:
        def end_solver(q,v,t,guess,rank):
            return _end_electric_mvi(problem,q,v,t,guess,rank,
                scales=scale,loads_at=loads_at,settings=mvi_settings,
                energy_scale=energy)

    # Physical dimensionless stationarity coefficient = (state_scale/E)*F.
    out=solve_gauss2_mvi_step(
        initial,settings,mvi_settings=mvi_settings,
        equations=eq,constraints=constraints,
        state_scales=scale,equation_scales=energy/scale,
        end_electrical_solver=end_solver,previous=previous)
    # Stationary essential supports also require zero end velocity in the
    # prescribed directions. This cannot be repaired by rank truncation.
    if (np.max(abs(c0@out.final.mechanical.ravel()-d0),initial=0)>1e-9 or
            np.max(abs(c0@out.final.velocity.ravel()),initial=0)>1e-9):
        raise ValueError('GAUSS2_MVI_ENDPOINT_MECHANICAL_SUPPORT_INCONSISTENT')
    return out


__all__=('solve_gauss2_variational_mvi_step',)
