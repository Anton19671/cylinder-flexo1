"""Experimental two-stage Gauss collocation with genuine fixed-rank X/Y MVI.

Both physical Gauss stage displacement and electric potential fields are
represented ONLY by rank-R X/Y products. At each Y then X half-sweep, both
stages are solved together through the Gauss collocation coupling. Mechanical
velocity and acceleration remain FULL surface fields. No nodal fallback,
additive field, rank growth, truncation or SVD is permitted.

A rank-R stage pair does NOT in general produce a rank-R final displacement.
The endpoint is accepted only if it is exactly representable at the same rank
(to numerical roundoff), and physical final trace conditions are satisfied.
This is a constrained experimental collocation path; fourth order for generic
low-rank nonlinear states is NOT asserted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import qr, solve_triangular, LinAlgWarning, lu_factor, lu_solve, get_lapack_funcs
import warnings

from cylinder_gauss2_rk import (GAUSS2_A, GAUSS2_B, GAUSS2_C,
                                 Gauss2Settings, Gauss2State, Gauss2Stage)


@dataclass(frozen=True)
class Gauss2MVISettings:
    rank: int = 1
    max_sweeps: int = 20
    max_newton_iterations: int = 10
    max_free_coordinates: int = 256
    absolute_tolerance: float = 1.e-9
    relative_tolerance: float = 1.e-7
    field_change_tolerance: float = 1.e-8
    tangent_step: float = 1.e-6
    max_halvings: int = 14
    endpoint_rank_tolerance: float = 2.e-10

    def __post_init__(self):
        for key in ('rank', 'max_sweeps', 'max_newton_iterations',
                    'max_free_coordinates', 'max_halvings'):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == 'max_halvings' else 1):
                raise ValueError(key + ': invalid integer')
        if self.rank > 8:
            raise ValueError('MVI fixed rank must be 1..8')
        for key in ('absolute_tolerance', 'relative_tolerance',
                    'field_change_tolerance', 'tangent_step', 'endpoint_rank_tolerance'):
            v = getattr(self, key)
            if isinstance(v, (bool, np.bool_)) or not math.isfinite(float(v)) or float(v) <= 0:
                raise ValueError(key + ': finite positive value required')
        if self.tangent_step >= .1 or self.endpoint_rank_tolerance >= .001:
            raise ValueError('invalid tangent or endpoint rank threshold')


@dataclass(frozen=True)
class Gauss2MVIResult:
    final: Gauss2State
    stages: tuple[Gauss2Stage, Gauss2Stage]
    x_profiles: np.ndarray
    y_profiles: np.ndarray
    scales: np.ndarray
    sweeps: int
    iterations: int
    converged: bool = True
    status: str = 'GAUSS2_MVI_PROJECTED_STAGES_ENDPOINT_RANK_CERTIFIED'
    method: str = 'EXPERIMENTAL_GAUSS2_FIXED_RANK_MVI'


def _component_scales(scales, shape):
    s = np.asarray(scales, dtype=float)
    if s.ndim == 0:
        s = np.full(shape, float(s))
    elif s.shape == (shape[0],):
        s = np.broadcast_to(s[:, None, None], shape).copy()
    if s.shape != shape or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('MVI state scales require finite positive component fields')
    if not np.array_equal(s, np.broadcast_to(s[:, :1, :1], shape)):
        raise ValueError('MVI_XY_REQUIRES_COMPONENTWISE_CONSTANT_SCALES')
    return s


def _rank_factors(field, rank, tolerance, label):
    """Pivoted QR: accept exact fixed-rank fields, never truncate them."""
    z = np.asarray(field, float)
    if z.ndim != 3 or rank > min(z.shape[1:]) or not np.isfinite(z).all():
        raise ValueError(label + ': finite field and rank<=grid required')
    nc, nx, ns = z.shape
    x = np.empty((nc, rank, nx))
    y = np.empty((nc, rank, ns))
    for c in range(nc):
        # For zero fields a deterministic independent X partner exposes
        # the first Y half-sweep without adding any physical displacement.
        if np.max(abs(z[c]), initial=0) == 0:
            basis = np.cos(np.pi * (np.arange(rank)[:, None]) *
                           (np.arange(nx) + .5)[None, :] / nx)
            x[c] = np.linalg.qr(basis.T, mode='reduced')[0].T
            y[c] = 0.
        else:
            basis, _, _ = qr(z[c].T, mode='economic', pivoting=True,
                             check_finite=False)
            y[c] = basis[:, :rank].T
            x[c] = (z[c] @ basis[:, :rank]).T
        difference = x[c].T @ y[c] - z[c]
        # A unit-sized absolute floor silently deletes weak electric or
        # mechanical modes: compare relative to THIS component's amplitude.
        scale = float(np.max(np.abs(z[c])))
        if float(np.max(abs(difference), initial=0)) > tolerance * scale:
            raise ValueError(label + ': GAUSS2_MVI_FIELD_OUTSIDE_FIXED_RANK')
    return x, y


def _profile_field(x, y):
    return np.einsum('crx,crs->cxs', x, y)


def _normalize(x, y, axis):
    x, y = x.copy(), y.copy()
    for c in range(x.shape[0]):
        if axis == 'Y':
            q, r = np.linalg.qr(x[c].T, mode='reduced')
            x[c] = q.T
            y[c] = r @ y[c]
        else:
            q, r = np.linalg.qr(y[c].T, mode='reduced')
            y[c] = q.T
            x[c] = r @ x[c]
    return x, y


def _directions(x, y, axis):
    nc, rank, nx = x.shape
    ns = y.shape[2]
    count = ns if axis == 'Y' else nx
    t = np.zeros((nc * nx * ns, nc * rank * count))
    for c in range(nc):
        for r in range(rank):
            for k in range(count):
                field = np.zeros((nx, ns))
                if axis == 'Y':
                    field[:, k] = x[c, r]
                else:
                    field[k, :] = y[c, r]
                t[c * nx * ns:(c + 1) * nx * ns,
                  (c * rank + r) * count + k] = field.ravel()
    if not np.allclose(t.T @ t, np.eye(t.shape[1]), rtol=1.e-12, atol=1.e-12):
        raise ValueError('MVI frozen partner directions not orthonormal')
    return t


def _constrained_directions(base, t, c, d):
    """Affine trace elimination within current frozen-partner directions ONLY."""
    base = np.asarray(base, float)
    c = np.asarray(c, float)
    d = np.asarray(d, float)
    if (base.ndim != 1 or c.shape != (d.size, base.size) or
            t.shape[0] != base.size or not all(np.isfinite(a).all()
                                                  for a in (base, t, c, d))):
        raise ValueError('GAUSS2_MVI_INVALID_STAGE_TRACES')
    if not d.size:
        return base, t
    # Every row's norm is removed before testing attainable traces.
    row_norm = np.linalg.norm(c, axis=1)
    if np.any(row_norm == 0):
        raise ValueError('GAUSS2_MVI_ZERO_TRACE_ROW')
    c, d = c / row_norm[:, None], d / row_norm
    a = c @ t
    rhs = d - c @ base
    q, r, piv = qr(a.T, mode='full', pivoting=True, check_finite=False)
    diag = abs(np.diag(r))
    # A frozen X/Y partner can already satisfy the CLAMPED trace exactly;
    # the computed normalized C*T then contains only floating-point noise.
    # A solely relative QR rank cutoff would turn that noise into false
    # independent boundary constraints and move an accepted stage state on
    # the next sweep. Match the protected static/Newmark trace treatment.
    # The actual prescribed trace equations remain mandatory: the strict
    # physical defect check below still rejects unreachable displacements.
    direction_scaled_floor = 1.e-10 * float(np.max(
        np.linalg.norm(t, axis=0), initial=0.0))
    cutoff = max(1.e-12 * float(np.max(diag, initial=0.0)),
                 direction_scaled_floor)
    rank = int(np.count_nonzero(diag > cutoff))
    if rank:
        z = q[:, :rank] @ solve_triangular(r[:rank, :rank].T,
                                          rhs[np.asarray(piv[:rank], int)],
                                          lower=True, check_finite=False)
    else:
        z = np.zeros(t.shape[1])
    defect = a @ z - rhs
    tol = 1.e-10 * np.maximum(1.e-12,
                              abs(d) + abs(c) @ abs(base) + abs(a) @ abs(z))
    tol += 20 * np.finfo(float).eps * max(a.shape) * max(
        1., float(np.max(abs(base), initial=0)))
    if np.any(abs(defect) > tol):
        raise ValueError('GAUSS2_MVI_TRACE_NOT_REPRESENTABLE_AT_FIXED_RANK')
    origin = base + t @ z
    directions = t @ q[:, rank:]
    return origin, directions


def _checked_lu(jac, g):
    if not jac.size:
        return np.empty(0)
    norm1 = float(np.linalg.norm(jac, ord=1))
    if norm1 <= 0 or not np.isfinite(norm1):
        raise ValueError('GAUSS2_MVI_SINGULAR_HALFSTEP_TANGENT')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            lu, piv = lu_factor(jac, check_finite=False)
        rcond, info = get_lapack_funcs('gecon', (jac,))(lu, norm1)
        if info or not np.isfinite(rcond) or rcond <= np.finfo(float).eps:
            raise ValueError('GAUSS2_MVI_SINGULAR_HALFSTEP_TANGENT')
        delta = lu_solve((lu, piv), -g, check_finite=False)
    except (LinAlgWarning, ValueError) as exc:
        raise ValueError('GAUSS2_MVI_SINGULAR_HALFSTEP_TANGENT') from exc
    if not np.isfinite(delta).all():
        raise ValueError('GAUSS2_MVI_SINGULAR_HALFSTEP_TANGENT')
    return delta


def _reduced_newton(origins, bases, equation, settings):
    """Solve coupled stage Y/Y or X/X unknowns in their actual MVI spaces."""
    dim = sum(b.shape[1] for b in bases)
    if dim > settings.max_free_coordinates:
        raise ValueError('GAUSS2_MVI_ACTIVE_DOF_LIMIT')
    sizes = [b.shape[1] for b in bases]

    def states(z):
        result = []
        offset = 0
        for origin, b, n in zip(origins, bases, sizes):
            result.append(origin + b @ z[offset:offset + n])
            offset += n
        return result

    def evaluate(z):
        values = states(z)
        force = equation(values)
        if len(force) != len(bases):
            raise ValueError('invalid coupled stage equation count')
        parts = []
        for v, b, f in zip(values, bases, force):
            f = np.asarray(f, float)
            if f.shape != v.shape or not np.isfinite(f).all():
                raise ValueError('invalid coupled stage equations')
            parts.append(b.T @ f)
        return np.concatenate(parts)

    z = np.zeros(dim)
    g = evaluate(z)
    initial = float(np.max(abs(g), initial=0))
    tolerance = settings.absolute_tolerance + settings.relative_tolerance * initial
    iterations = 0
    for _ in range(settings.max_newton_iterations + 1):
        norm = float(np.max(abs(g), initial=0))
        if norm <= tolerance:
            return states(z), iterations
        if iterations >= settings.max_newton_iterations:
            raise ValueError('GAUSS2_MVI_HALFSTEP_NOT_CONVERGED')
        jac = np.empty((dim, dim), float)
        for j in range(dim):
            h = settings.tangent_step * (1. + abs(z[j]))
            unit = np.zeros(dim)
            unit[j] = h
            jac[:, j] = (evaluate(z + unit) - evaluate(z - unit)) / (2. * h)
        step = _checked_lu(jac, g)
        for halvings in range(settings.max_halvings + 1):
            trial = z + 2.**(-halvings) * step
            try:
                new_g = evaluate(trial)
            except ValueError as exc:
                if "INVALID_" not in str(exc):
                    raise
                continue
            except (FloatingPointError, OverflowError):
                continue
            next_norm = float(np.max(abs(new_g), initial=0))
            if next_norm < norm or next_norm <= tolerance:
                z, g = trial, new_g
                break
        else:
            raise ValueError('GAUSS2_MVI_HALFSTEP_LINE_SEARCH_FAILED')
        iterations += 1
    raise ValueError('GAUSS2_MVI_HALFSTEP_NOT_CONVERGED')


def _seed_inactive_factors(xs, ys, rhs, rank):
    """Seed only zero Y partners from current stage driving field, preserving q=0."""
    for stage in range(xs.shape[0]):
        for c in range(xs.shape[1]):
            if np.any(ys[stage, c]) or not np.any(rhs[stage, c]):
                continue
            basis, _, _ = qr(rhs[stage, c], mode='economic', pivoting=True,
                             check_finite=False)
            xs[stage, c] = basis[:, :rank].T
    return xs, ys


def _convergence_components(xs, ys, scales, stage_times, equation, constraints,
                            settings):
    result = []
    for axis in ('Y', 'X'):
        projected = []
        for i in range(2):
            xx, yy = _normalize(xs[i], ys[i], axis)
            state = _profile_field(xx, yy).ravel()
            physical = scales.ravel() * state
            t = _directions(xx, yy, axis)
            c, d = constraints(i, float(stage_times[i]), physical.reshape(scales.shape))
            origin, b = _constrained_directions(state, t,
                                                np.asarray(c, float)*scales.ravel()[None, :],
                                                np.asarray(d, float))
            if float(np.max(abs(origin-state), initial=0)) > 1e-9:
                # The represented current field itself must obey traces after
                # a completed sweep; do not permit an unseen affine lift.
                raise ValueError('GAUSS2_MVI_COMPLETED_SWEEP_TRACE_DRIFT')
            projected.append(b)
        force = equation([_profile_field(xs[i], ys[i]).ravel() for i in range(2)])
        g = [projected[i].T @ force[i] for i in range(2)]
        result.append(float(np.max(abs(np.concatenate(g)), initial=0)))
    return max(result)


def solve_gauss2_mvi_step(initial, settings, *, mvi_settings, equations,
                          constraints, state_scales, equation_scales,
                          end_electrical_solver=None, previous=None):
    """One two-stage rank-R X/Y Gauss collocation step for coupled equations.

    equations(q,v,a,p,t)->(mechanical equations, electrical equations or None)
    constraints(stage_index,t,combined_physical_field)->(C,d)
        where stage_index 0,1,2 denotes stage 1, stage 2, endpoint.
    These must be the *physical* stage equations and all essential/gauge
    traces. Stage accelerations and velocities are not approximated by MVI.
    end_electrical_solver(q_end,v_end,t_end,stage_2_p,rank)->physical p_end,
    solved in a separate fixed-rank electric MVI, never copied from stage 2.
    """
    if not isinstance(initial, Gauss2State) or not isinstance(settings, Gauss2Settings):
        raise TypeError('Gauss2State and Gauss2Settings required')
    if not isinstance(mvi_settings, Gauss2MVISettings):
        raise TypeError('Gauss2MVISettings required')
    if not callable(equations) or not callable(constraints):
        raise TypeError('physical equations and constraints callbacks required')
    q0, v0, p0 = initial.mechanical, initial.velocity, initial.electrical
    nc = q0.shape[0] + (0 if p0 is None else p0.shape[0])
    shape = (nc, *q0.shape[1:])
    scale = _component_scales(state_scales, shape)
    force_scale = _component_scales(equation_scales, shape)
    r = mvi_settings.rank
    if r > min(q0.shape[1:]):
        raise ValueError('MVI rank exceeds physical grid')
    u0 = q0 if p0 is None else np.concatenate((q0, p0))
    normalized = u0 / scale
    if previous is None:
        x0, y0 = _rank_factors(normalized, r,
                               mvi_settings.endpoint_rank_tolerance,
                               'GAUSS2_MVI_INITIAL_STATE')
    else:
        if not previous.converged or previous.final.time_s != initial.time_s:
            raise ValueError('GAUSS2_MVI_CONTINUATION_REQUIRES_ACCEPTED_PREVIOUS_STEP')
        x0 = np.asarray(previous.x_profiles, float).copy()
        y0 = np.asarray(previous.y_profiles, float).copy()
        old = np.asarray(previous.scales, float)
        if (x0.shape != (nc, r, q0.shape[1]) or
                y0.shape != (nc, r, q0.shape[2]) or old.shape != shape or
                not np.isfinite(old).all() or np.any(old <= 0)):
            raise ValueError('GAUSS2_MVI_CONTINUATION_FIXED_RANK_MISMATCH')
        if not np.array_equal(old, np.broadcast_to(old[:, :1, :1], shape)):
            raise ValueError('GAUSS2_MVI_CONTINUATION_INVALID_OLD_SCALES')
        # Preserve every physical component, not only the numerically
        # largest one. A small electric mode must not be swallowed by a
        # unit-sized displacement or an absolute error floor.
        previous_field = _profile_field(x0, y0)
        normalized_actual = u0 / old
        if not np.isfinite(normalized_actual).all():
            raise ValueError('GAUSS2_MVI_CONTINUATION_INVALID_OLD_SCALES')
        per_component_amplitude = np.max(
            np.abs(normalized_actual), axis=(1, 2), keepdims=True)
        limit = mvi_settings.endpoint_rank_tolerance * np.maximum(
            per_component_amplitude, np.finfo(float).tiny)
        if np.any(np.abs(previous_field-normalized_actual)>limit):
            raise ValueError('GAUSS2_MVI_CONTINUATION_FIELD_MISMATCH')
        ratio=old[:, 0, 0]/scale[:, 0, 0]
        x0*=ratio[:, None, None]
    # Each Gauss stage begins from the accepted previous physical X/Y field.
    xs = np.stack((x0.copy(), x0.copy()))
    ys = np.stack((y0.copy(), y0.copy()))
    dt = settings.dt_s
    stage_times = initial.time_s + dt * GAUSS2_C
    if stage_times[0] <= initial.time_s:
        raise ValueError('physical time cannot advance at supplied precision')
    inverse_a2 = np.linalg.inv(GAUSS2_A @ GAUSS2_A)
    energy_scale = 1.  # caller supplies independent physical equation scales
    qsize = q0.size
    psize = 0 if p0 is None else p0.size

    def evaluate(fields):
        raw = [np.asarray(field, float).reshape(shape)*scale for field in fields]
        qs = np.stack([u[:q0.shape[0]] for u in raw])
        ps = None if p0 is None else np.stack([u[q0.shape[0]:] for u in raw])
        targets = qs - q0[None, ...] - dt*GAUSS2_C[:,None,None,None]*v0[None,...]
        acc = np.einsum('ij,j...->i...',inverse_a2,targets)/(dt*dt)
        vel = v0[None,...] + dt*np.einsum('ij,j...->i...',GAUSS2_A,acc)
        force=[]
        for i in range(2):
            out=equations(qs[i].copy(),vel[i].copy(),acc[i].copy(),
                          None if ps is None else ps[i].copy(),float(stage_times[i]))
            if not isinstance(out, tuple) or len(out)!=2:
                raise TypeError('Gauss MVI equations must return two blocks')
            rm=np.asarray(out[0], float)
            ep=None if out[1] is None else np.asarray(out[1],float)
            if rm.shape!=q0.shape or not np.isfinite(rm).all():
                raise ValueError('invalid Gauss mechanical stage equations')
            if (p0 is None and ep is not None) or (p0 is not None and
                    (ep is None or ep.shape!=p0.shape or not np.isfinite(ep).all())):
                raise ValueError('invalid Gauss electric stage equations')
            both=rm if ep is None else np.concatenate((rm,ep))
            force.append((both/force_scale).ravel())
        return force,qs,vel,acc,ps

    # Initial zero fields need nontrivial partner directions for a nonzero
    # physical stage force. Seed only inactive Y, never alter the field.
    full0, *_ = evaluate([_profile_field(xs[i],ys[i]).ravel() for i in range(2)])
    gradient=np.stack([f.reshape(shape) for f in full0])
    xs,ys=_seed_inactive_factors(xs,ys,gradient,r)
    initial_force,_q,_v,_a,_p=evaluate([_profile_field(xs[i],ys[i]).ravel() for i in range(2)])
    initial_norm=max(float(np.max(abs(f),initial=0)) for f in initial_force)
    tolerance=mvi_settings.absolute_tolerance+mvi_settings.relative_tolerance*initial_norm
    iterations=0
    sweeps=0
    for sweep in range(1,mvi_settings.max_sweeps+1):
        before = np.stack([_profile_field(xs[i],ys[i]) for i in range(2)])
        for axis in ('Y','X'):
            origins=[];bases=[];directions=[];current=[]
            for stage in range(2):
                xx,yy=_normalize(xs[stage],ys[stage],axis)
                xs[stage],ys[stage]=xx,yy
                base=_profile_field(xx,yy).ravel()
                tmat=_directions(xx,yy,axis)
                physical=(base*scale.ravel()).reshape(shape)
                c,d=constraints(stage,float(stage_times[stage]),physical)
                c=np.asarray(c,float)
                d=np.asarray(d,float)
                if d.ndim!=1:
                    raise ValueError('GAUSS2_MVI_INVALID_STAGE_TRACES')
                origin,b=_constrained_directions(base,tmat,c*scale.ravel()[None,:],d)
                origins.append(origin);bases.append(b)
                directions.append(tmat);current.append(base)

            def eq(states):
                return evaluate(states)[0]
            solved, its=_reduced_newton(origins,bases,eq,mvi_settings)
            iterations+=its
            for stage in range(2):
                change=solved[stage]-current[stage]
                delta=directions[stage].T@change
                if (float(np.max(abs(directions[stage]@delta-change),initial=0)) >
                        5e-10*max(1.,float(np.max(abs(change),initial=0)))):
                    raise ValueError('GAUSS2_MVI_FORBIDDEN_NODAL_CORRECTION')
                if axis=='Y':
                    ys[stage]+=delta.reshape(ys[stage].shape)
                else:
                    xs[stage]+=delta.reshape(xs[stage].shape)
        sweeps=sweep
        after=np.stack([_profile_field(xs[i],ys[i]) for i in range(2)])
        change=float(np.linalg.norm(after-before)/max(
            np.linalg.norm(after),np.linalg.norm(before),1.e-30))
        projected=_convergence_components(xs,ys,scale,stage_times,
                                          lambda states:evaluate(states)[0],
                                          constraints,mvi_settings)
        if projected<=tolerance and change<=mvi_settings.field_change_tolerance:
            break
    else:
        raise ValueError('GAUSS2_MVI_STAGES_NOT_CONVERGED')

    final_stage_fields=[_profile_field(xs[i],ys[i]).ravel() for i in range(2)]
    stage_forces,qs,vel,acc,ps=evaluate(final_stage_fields)
    if not all(np.isfinite(x).all() for x in (qs,vel,acc)):
        raise ValueError('GAUSS2_MVI_INVALID_STAGE_FIELDS')
    qend=q0+dt*np.einsum('i,i...->...',GAUSS2_B,vel)
    vend=v0+dt*np.einsum('i,i...->...',GAUSS2_B,acc)
    if not np.isfinite(qend).all() or not np.isfinite(vend).all():
        raise ValueError('GAUSS2_MVI_INVALID_ENDPOINT')
    tend=initial.time_s+dt
    if p0 is None:
        pend=None
    else:
        if not callable(end_electrical_solver):
            raise ValueError('GAUSS2_MVI_ELECTRIC_ENDPOINT_SOLVER_REQUIRED')
        pend=np.asarray(end_electrical_solver(qend.copy(),vend.copy(),float(tend),
                            ps[1].copy(),r),float)
        if pend.shape!=p0.shape or not np.isfinite(pend).all():
            raise ValueError('GAUSS2_MVI_INVALID_ELECTRIC_ENDPOINT')
    end_state=qend if pend is None else np.concatenate((qend,pend))
    xend,yend=_rank_factors(end_state/scale,r,
                           mvi_settings.endpoint_rank_tolerance,
                           'GAUSS2_MVI_ENDPOINT')
    c,d=constraints(2,float(tend),end_state.copy())
    c=np.asarray(c,float);d=np.asarray(d,float)
    if c.shape!=(d.size,end_state.size) or not all(np.isfinite(v).all() for v in (c,d)):
        raise ValueError('GAUSS2_MVI_INVALID_ENDPOINT_TRACES')
    if np.max(abs(c@end_state.ravel()-d),initial=0) > 1e-9*max(
            1.,float(np.max(abs(d),initial=0))):
        raise ValueError('GAUSS2_MVI_ENDPOINT_TRACES_NOT_SATISFIED')
    stages=tuple(Gauss2Stage(float(stage_times[i]),qs[i].copy(),vel[i].copy(),
                             acc[i].copy(),None if ps is None else ps[i].copy())
                 for i in range(2))
    final=Gauss2State(float(tend),qend,vend,pend)
    return Gauss2MVIResult(final,stages,xend,yend,scale.copy(),sweeps,iterations)


__all__=('Gauss2MVISettings','Gauss2MVIResult', 'solve_gauss2_mvi_step')
