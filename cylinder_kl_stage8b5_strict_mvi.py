#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8B-5: strict internal MVI for the displacement-only KL cylinder.

Algorithm is the already accepted strict MVI mechanics:
    Y halfstep -> converge its own projected equations
    X halfstep -> converge its own projected equations
    then check both projected residuals + reconstructed field increment.

KL-specific change:
The residual supplied by Stage8B-4 is already the weak variational derivative
    r_j = dPi/dc_j.
Therefore it must NOT be multiplied again by T^T W.

Normalization:
For each raw profile parameter c_j, let
    T_j = dU/dc_j, n_j = ||T_j||_2.
The halfstep equations are written in normalized field directions:
    rbar_j = r_j / n_j.
A tangent column is evaluated by
    c_j -> c_j + h/n_j
and finite differencing rbar.

Convergence measure:
``rbar`` still has force units, so its absolute infinity norm is retained only
for diagnostics.  The stopping test uses the dimensionless ratio

    Rproj_rel = ||rbar||_inf / Rproj_scale,

where one fixed scale is formed once per load step from the applied resultant
load and the initial projected residual.  The reduced equations and their
tangent are not scaled or otherwise changed.

Only np.linalg.solve is used for the square reduced Newton system.
Correction-step control is enabled for the strict MVI Newton update:
trial full correction is checked and alpha is reduced if the projected
residual increases. No relaxation, KKT, SVD/pinv/lstsq/QR/Cholesky,
regularization, or full-grid correction. A deterministic algebraic factor gauge
normalization is applied between halfsteps; it preserves the reconstructed field
exactly and only removes the X/Y scale ambiguity of a separated rank.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import copy
import final_progress as fprog
import final_v3_fd as v3fd

import cylinder_kl_stage8b2_profiles as kp
import cylinder_kl_stage8b3_essential_bc as ke
import cylinder_kl_stage8b4_weak_projected as kw


class KLMVIError(RuntimeError):
    pass

class KLSingularReducedSystem(KLMVIError):
    pass

class KLHalfstepDidNotConverge(KLMVIError):
    pass


@dataclass(frozen=True)
class KLInternalMVISettings:
    projected_tolerance: float
    increment_tolerance: float
    tangent_fd_step: float
    max_halfstep_iterations: int
    max_xy_cycles: int
    max_correction_halvings: int = 5

    def __post_init__(self):
        if self.projected_tolerance<=0: raise ValueError("projected_tolerance must be positive")
        if self.increment_tolerance<=0: raise ValueError("increment_tolerance must be positive")
        if self.tangent_fd_step<=0: raise ValueError("tangent_fd_step must be positive")
        if self.max_halfstep_iterations<1: raise ValueError("max_halfstep_iterations must be >=1")
        if self.max_xy_cycles<1: raise ValueError("max_xy_cycles must be >=1")
        if self.max_correction_halvings<0: raise ValueError("max_correction_halvings must be >=0")


@dataclass(frozen=True)
class KLHalfstepRecord:
    direction: str
    iterations: int
    projected_inf: float
    correction_norm: float
    status: str
    projected_relative: float = float("nan")
    projected_scale: float = float("nan")


@dataclass
class KLInternalMVIResult:
    state: kp.ProfileStateKL
    converged: bool
    status: str
    cycles: int
    history: list
    final_projected_inf: float
    final_increment_rel: float
    final_projected_relative: float = float("nan")
    projected_scale: float = float("nan")


def _field_vector(state,grid):
    return kp.reconstruct_vector(state,grid)


def _relative_field_change(old,new,grid):
    a=_field_vector(old,grid)
    b=_field_vector(new,grid)
    return float(np.linalg.norm(b-a)/max(np.linalg.norm(b),1e-300))


def _x_reduced_arrays(state,maps):
    return [x.copy() for x in ke.reduce_constrained_state_X(state,maps)]


def _raw_parameters_and_labels(state,ops,maps,direction):
    d=str(direction).upper()
    if d=="Y":
        labels=[]
        vals=[]
        for a in range(3):
            for rr in range(state.rank):
                for j in range(ops.grid.ns):
                    labels.append((a,rr,j))
                    vals.append(state.Y[a,rr,j])
        return np.asarray(vals,float),tuple(labels)

    if d=="X":
        reduced=_x_reduced_arrays(state,maps)
        labels=[]
        vals=[]
        for a,m in enumerate(maps.maps):
            for rr in range(state.rank):
                for j in range(m.n_free):
                    labels.append((a,rr,j))
                    vals.append(reduced[a][rr,j])
        return np.asarray(vals,float),tuple(labels)
    raise ValueError("direction must be X or Y")


def _set_raw_parameters(state,ops,maps,direction,labels,values):
    d=str(direction).upper()
    values=np.asarray(values,float)
    if values.shape!=(len(labels),):
        raise ValueError("values shape mismatch")

    if d=="Y":
        for val,(a,rr,j) in zip(values,labels):
            state.Y[a,rr,j]=val
        return

    if d=="X":
        reduced=_x_reduced_arrays(state,maps)
        for val,(a,rr,j) in zip(values,labels):
            reduced[a][rr,j]=val
        rebuilt=ke.constrain_state_X_from_reduced(reduced,state,maps)
        state.X[:]=rebuilt.X
        return
    raise ValueError("direction must be X or Y")


def _direction_norms_labels(state,ops,maps,direction):
    variations=kw.direction_variations(state,ops,maps,direction)
    labels=tuple(v.label for v in variations)
    norms=np.asarray([
        np.linalg.norm(v.dU.reshape(-1))
        for v in variations
    ],float)
    return norms,labels


def _raw_projected(projected_fn,state,direction):
    r,labels=projected_fn(state,direction)
    r=np.asarray(r,float)
    if r.shape!=(len(labels),):
        raise KLMVIError("projected residual/label shape mismatch")
    if not np.all(np.isfinite(r)):
        raise KLMVIError("NONFINITE_PROJECTED_RESIDUAL")
    return r,tuple(labels)


def _prepare_direction_if_supported(projected_fn,state,direction):
    prepare=getattr(projected_fn,"mvi_prepare_direction",None)
    return None if prepare is None else prepare(state,direction)


def _raw_projected_prepared(projected_fn,state,direction,prepared):
    evaluate=getattr(projected_fn,"mvi_evaluate_prepared",None)
    if prepared is None or evaluate is None:
        return _raw_projected(projected_fn,state,direction)
    r,labels=evaluate(state,direction,prepared)
    r=np.asarray(r,float)
    if r.shape!=(len(labels),):
        raise KLMVIError("projected residual/label shape mismatch")
    if not np.all(np.isfinite(r)):
        raise KLMVIError("NONFINITE_PROJECTED_RESIDUAL")
    return r,tuple(labels)


def _direction_data_prepared(state,ops,maps,direction,projected_fn,prepared):
    raw,labels_r=_raw_projected_prepared(projected_fn,state,direction,prepared)
    params,labels_p=_raw_parameters_and_labels(state,ops,maps,direction)
    labels_n=tuple(prepared["labels"])
    norms=np.asarray(prepared["norms"],float)
    if labels_r!=labels_p or labels_r!=labels_n:
        raise KLMVIError("direction parameter ordering mismatch")
    active=norms>1e-14
    normalized=raw[active]/norms[active] if np.any(active) else np.zeros(0,float)
    return raw,params,labels_r,norms,active,normalized


def _direction_data(state,ops,maps,direction,projected_fn):
    raw,labels_r=_raw_projected(projected_fn,state,direction)
    params,labels_p=_raw_parameters_and_labels(state,ops,maps,direction)
    norms,labels_n=_direction_norms_labels(state,ops,maps,direction)
    if labels_r!=labels_p or labels_r!=labels_n:
        raise KLMVIError("direction parameter ordering mismatch")
    active=norms>1e-14
    if np.any(active):
        normalized=raw[active]/norms[active]
    else:
        normalized=np.zeros(0,float)
    return raw,params,labels_r,norms,active,normalized


def normalized_projected_inf(state,ops,maps,direction,projected_fn):
    raw,_params,_labels,_norms,active,rbar=_direction_data(
        state,ops,maps,direction,projected_fn
    )
    if not np.any(active):
        if ma(raw)==0.0:
            return 0.0
        raise KLMVIError(f"{str(direction).upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS")
    return float(np.max(np.abs(rbar))) if rbar.size else 0.0


def ma(a):
    a=np.asarray(a,float)
    return float(np.max(np.abs(a))) if a.size else 0.0


def _positive_projected_scale(value):
    scale=float(value)
    if not np.isfinite(scale) or scale<=0.0:
        raise ValueError("projected reference scale must be finite and positive")
    return scale


def _projected_relative(projected_inf,projected_scale):
    return float(projected_inf)/_positive_projected_scale(projected_scale)


def _projected_converged(projected_relative,settings):
    """Apply the user-requested projected tolerance without hidden relaxation."""
    tol=float(settings.projected_tolerance)
    value=float(projected_relative)
    if not np.isfinite(tol) or tol<=0.0:
        raise ValueError("projected_tolerance must be finite and positive")
    if not np.isfinite(value):
        return False
    return value<=tol


def _surface_load_resultant_scale(grid,generalized_load):
    p=np.asarray(generalized_load,float)
    expected=(3,grid.ns,grid.nx)
    if p.shape!=expected or not np.all(np.isfinite(p)):
        raise ValueError(f"generalized_load must be finite with shape {expected}")
    W=kw.surface_weights(grid)
    return float(np.sum(W*np.sqrt(np.sum(p*p,axis=0))))


def _fixed_projected_scale(state,ops,maps,projected_fn):
    """One force scale for the complete Y->X solve at the current load step."""
    supplied=float(getattr(projected_fn,"mvi_projected_reference_scale",0.0))
    if not np.isfinite(supplied) or supplied<0.0:
        raise KLMVIError("INVALID_MVI_PROJECTED_REFERENCE_SCALE")
    initial=[]
    for direction in ("Y","X"):
        try:
            initial.append(normalized_projected_inf(
                state,ops,maps,direction,projected_fn
            ))
        except KLMVIError:
            # The halfstep itself will report an inactive-direction error with
            # its established diagnostic if that direction is actually needed.
            pass
    return max([supplied,*initial,np.finfo(float).tiny])


def directional_tangent_fd(
    state,ops,maps,direction,projected_fn,*,tangent_fd_step,quadrature,
    cycle=None,halfstep_iteration=None,max_halfstep_iterations=None,
    max_xy_cycles=None,direction_data=None,prepared=None,
    force_finite_difference=False
):
    """J-12 KL tangent by V3-scaled projected differences.

    ``direction_data`` may contain the already evaluated base-state tuple from
    ``_direction_data``.  Reusing it removes one duplicate projected residual
    evaluation per half-step iteration without changing the central +h/-h
    tangent or any reduced equation.
    """
    if direction_data is None:
        _raw,p0,labels,norms,active,rbar0=_direction_data(
            state,ops,maps,direction,projected_fn
        )
    else:
        _raw,p0,labels,norms,active,rbar0=direction_data
    positions=np.flatnonzero(active)
    if positions.size==0:
        raise KLMVIError(f"{str(direction).upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS")

    assemble=getattr(projected_fn,"mvi_assemble_tangent",None)
    if prepared is not None and assemble is not None and not force_finite_difference:
        A=assemble(state,direction,prepared,norms,active)
        if A is not None:
            A=np.asarray(A,float)
            expected=(positions.size,positions.size)
            if A.shape!=expected or not np.all(np.isfinite(A)):
                raise KLMVIError("invalid algorithmic projected tangent")
            fprog.emit(
                "mvi_tangent_progress",stage="MVI_TANGENT",
                halfstep=str(direction).upper(),cycle=cycle,max_xy_cycles=max_xy_cycles,
                halfstep_iteration=halfstep_iteration,
                max_halfstep_iterations=max_halfstep_iterations,
                tangent_column=int(positions.size),tangent_columns=int(positions.size),
                tangent_percent=100.0,
            )
            return A,rbar0,p0,labels,norms,active

    h_shell=v3fd.thickness_from_quadrature(quadrature)
    amplitude=p0*norms
    char=np.full_like(amplitude,h_shell,dtype=float)
    hj=v3fd.relative_steps(amplitude,char,tangent_fd_step)
    A=np.empty((positions.size,positions.size),float)
    tangent_columns=int(positions.size)
    progress_stride=max(1,(tangent_columns+9)//10)
    scheme=str(getattr(projected_fn,"mvi_tangent_fd_scheme","CENTRAL")).upper()
    if scheme not in ("CENTRAL","FORWARD"):
        raise KLMVIError(f"unsupported MVI tangent difference scheme: {scheme}")

    for jcol,pos in enumerate(positions):
        plus=state.copy()
        pp=p0.copy()
        pp[pos]+=hj[pos]/norms[pos]
        _set_raw_parameters(plus,ops,maps,direction,labels,pp)

        rawp,labelsp=_raw_projected_prepared(projected_fn,plus,direction,prepared)
        if labelsp!=labels:
            raise KLMVIError("projected residual ordering changed inside halfstep")

        # Partner is frozen during a halfstep.  Prepared directions retain
        # their norms.  The fallback verifies the same invariant.
        if prepared is None:
            normsp,labelspn=_direction_norms_labels(plus,ops,maps,direction)
            if (
                labelspn!=labels
                or np.max(np.abs(normsp-norms))>1e-11*max(ma(norms),1.0)
            ):
                raise KLMVIError("normalized tangent directions changed inside halfstep")

        rbarp=rawp[active]/norms[active]
        if scheme=="FORWARD":
            # The J2 trial response is evaluated from one committed history.
            # A one-sided derivative is its consistent directional tangent and
            # avoids a second complete material solve for the -h trial.
            A[:,jcol]=(rbarp-rbar0)/hj[pos]
        else:
            minus=state.copy()
            pm=p0.copy()
            pm[pos]-=hj[pos]/norms[pos]
            _set_raw_parameters(minus,ops,maps,direction,labels,pm)
            rawm,labelsm=_raw_projected_prepared(projected_fn,minus,direction,prepared)
            if labelsm!=labels:
                raise KLMVIError("projected residual ordering changed inside halfstep")
            if prepared is None:
                normsm,labelsmn=_direction_norms_labels(minus,ops,maps,direction)
                if (
                    labelsmn!=labels
                    or np.max(np.abs(normsm-norms))>1e-11*max(ma(norms),1.0)
                ):
                    raise KLMVIError("normalized tangent directions changed inside halfstep")
            rbarm=rawm[active]/norms[active]
            A[:,jcol]=(rbarp-rbarm)/(2.0*hj[pos])
        completed=int(jcol+1)
        if completed==1 or completed==tangent_columns or completed%progress_stride==0:
            fprog.emit(
                "mvi_tangent_progress",stage="MVI_TANGENT",
                halfstep=str(direction).upper(),cycle=cycle,max_xy_cycles=max_xy_cycles,
                halfstep_iteration=halfstep_iteration,
                max_halfstep_iterations=max_halfstep_iterations,
                tangent_column=completed,tangent_columns=tangent_columns,
                tangent_percent=100.0*completed/max(tangent_columns,1),
            )

    return A,rbar0,p0,labels,norms,active


def _direct_solve(A,rbar,direction):
    try:
        delta=np.linalg.solve(np.asarray(A,float),-np.asarray(rbar,float))
    except np.linalg.LinAlgError as exc:
        raise KLSingularReducedSystem(
            f"SINGULAR_REDUCED_MVI_SYSTEM / {str(direction).upper()}_HALFSTEP"
        ) from exc
    if not np.all(np.isfinite(delta)):
        raise KLSingularReducedSystem(
            f"NONFINITE_REDUCED_MVI_CORRECTION / {str(direction).upper()}_HALFSTEP"
        )
    return delta


def solve_strict_halfstep(
    state,ops,maps,direction,projected_fn,settings,*,quadrature,cycle=None,
    projected_scale=1.0
):
    d=str(direction).upper()
    pscale=_positive_projected_scale(projected_scale)
    last_corr=0.0

    for it in range(1,settings.max_halfstep_iterations+1):
        prepared=_prepare_direction_if_supported(projected_fn,state,d)
        if prepared is None:
            direction_data=_direction_data(state,ops,maps,d,projected_fn)
        else:
            direction_data=_direction_data_prepared(
                state,ops,maps,d,projected_fn,prepared
            )
        raw,_p0,labels,norms,active,rbar=direction_data
        if not np.any(active):
            if ma(raw)==0.0:
                return KLHalfstepRecord(d,it-1,0.0,0.0,"CONVERGED_REDUCED_MVI")
            raise KLMVIError(f"{d}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS")

        pinf=float(np.max(np.abs(rbar))) if rbar.size else 0.0
        prel=_projected_relative(pinf,pscale)
        fprog.emit(
            "mvi_halfstep_iteration",stage="MVI_HALFSTEP",
            halfstep=d,cycle=cycle,max_xy_cycles=int(settings.max_xy_cycles),
            halfstep_iteration=int(it),
            max_halfstep_iterations=int(settings.max_halfstep_iterations),
            projected_inf=float(pinf),projected_relative=float(prel),
            projected_scale=float(pscale),
            active_directions=int(np.count_nonzero(active)),
        )
        if _projected_converged(prel,settings):
            return KLHalfstepRecord(
                d,it-1,pinf,last_corr,"CONVERGED_REDUCED_MVI",prel,pscale
            )

        # Prefer the supplied reduced tangent.  If it cannot produce any
        # decreasing correction (notably at a J2 active-set switch), rebuild
        # the same reduced MVI Jacobian from actual projected-residual trials.
        # This remains an X/Y MVI correction; no full-grid Newton step is used.
        tangent_modes=["PREFERRED"]
        if prepared is not None and callable(
            getattr(projected_fn,"mvi_assemble_tangent",None)
        ):
            tangent_modes.append("PROJECTED_FD_RETRY")

        current_prel=prel
        accepted=False
        for tangent_mode in tangent_modes:
            A,rbar0,pbase,labels2,_norms2,active2=directional_tangent_fd(
                state,ops,maps,d,projected_fn,
                tangent_fd_step=settings.tangent_fd_step,quadrature=quadrature,
                cycle=cycle,halfstep_iteration=int(it),
                max_halfstep_iterations=int(settings.max_halfstep_iterations),
                max_xy_cycles=int(settings.max_xy_cycles),
                direction_data=direction_data,prepared=prepared,
                force_finite_difference=(tangent_mode=="PROJECTED_FD_RETRY"),
            )
            if labels2!=labels or not np.array_equal(active2,active):
                raise KLMVIError("direction data changed inside halfstep")
            if np.max(np.abs(rbar0-rbar))>1e-12*max(ma(rbar),1.0):
                raise KLMVIError("projected residual changed inside halfstep assembly")

            delta_norm=_direct_solve(A,rbar,d)
            pp=pbase.copy()
            pos=np.flatnonzero(active)
            raw_delta=np.zeros_like(pp)
            raw_delta[pos]=delta_norm/norms[active]
            undamped_corr=float(np.linalg.norm(raw_delta))

            # Backtrack only on trial states.  The accepted update is written
            # to the live state explicitly below.
            alpha=1.0
            max_halvings=int(getattr(settings,"max_correction_halvings",5))
            min_alpha=1.0/(2.0**max_halvings)
            for _ih in range(max_halvings+1):
                trial_state=copy.deepcopy(state)
                _set_raw_parameters(
                    trial_state,ops,maps,d,labels,pp+alpha*raw_delta
                )
                trial_pinf=normalized_projected_inf(
                    trial_state,ops,maps,d,projected_fn
                )
                trial_prel=_projected_relative(trial_pinf,pscale)

                if trial_prel <= current_prel:
                    _set_raw_parameters(
                        state,ops,maps,d,labels,pp+alpha*raw_delta
                    )
                    last_corr=float(alpha*undamped_corr)
                    accepted=True
                    if tangent_mode=="PROJECTED_FD_RETRY":
                        fprog.emit(
                            "mvi_tangent_retry_accepted",stage="MVI_HALFSTEP",
                            halfstep=d,cycle=cycle,halfstep_iteration=int(it),
                            tangent_mode=tangent_mode,damping_factor=float(alpha),
                            projected_relative_before=float(current_prel),
                            projected_relative_after=float(trial_prel),
                        )
                    break

                fprog.emit(
                    "mvi_correction_rejected",stage="MVI_HALFSTEP",
                    halfstep=d,cycle=cycle,halfstep_iteration=int(it),
                    tangent_mode=tangent_mode,damping_factor=float(alpha),
                    projected_relative_before=float(current_prel),
                    projected_relative_after=float(trial_prel),
                )
                alpha*=0.5
                if alpha < min_alpha:
                    break

            if accepted:
                break
            if tangent_mode=="PREFERRED" and len(tangent_modes)>1:
                fprog.emit(
                    "mvi_tangent_retry",stage="MVI_HALFSTEP",
                    halfstep=d,cycle=cycle,halfstep_iteration=int(it),
                    reason="PREFERRED_TANGENT_HAS_NO_DECREASING_CORRECTION",
                    action="REBUILD_REDUCED_PROJECTED_FD_TANGENT",
                )

        if not accepted:
            raise KLMVIError(
                f"CORRECTION_STEP_FAILED / {d}_HALFSTEP / "
                f"projected_relative={current_prel:.17g}"
            )

    pinf=normalized_projected_inf(state,ops,maps,d,projected_fn)
    prel=_projected_relative(pinf,pscale)
    if _projected_converged(prel,settings):
        return KLHalfstepRecord(
            d,settings.max_halfstep_iterations,pinf,last_corr,
            "CONVERGED_REDUCED_MVI",prel,pscale
        )
    raise KLHalfstepDidNotConverge(
        f"MAX_HALFSTEP_ITERATIONS / {d}_HALFSTEP / "
        f"projected_inf={pinf:.17g} / projected_relative={prel:.17g} / "
        f"projected_scale={pscale:.17g}"
    )


def _rebalance_separated_factors(state):
    """Remove factor scale ambiguity while preserving every physical field.

    For every component/rank pair, Y^T X is invariant under
        X <- s X,  Y <- Y/s.
    Only pairs with a meaningful physical amplitude are normalized.  A
    near-zero pair is left untouched: balancing numerical noise in an
    identically zero component would activate spurious reduced directions and
    make the Newton system ill-conditioned.
    """
    amplitudes=[]
    norms=[]
    for a in range(state.X.shape[0]):
        for rr in range(state.rank):
            nx=float(np.linalg.norm(state.X[a,rr]))
            ny=float(np.linalg.norm(state.Y[a,rr]))
            norms.append((a,rr,nx,ny))
            if np.isfinite(nx) and np.isfinite(ny):
                amplitudes.append(nx*ny)
    amplitude_scale=max(amplitudes,default=0.0)
    floor=max(1.0e-12*amplitude_scale,1.0e-14)
    for a,rr,nx,ny in norms:
        if (
            not np.isfinite(nx) or not np.isfinite(ny)
            or nx<=0.0 or ny<=0.0
        ):
            continue
        if nx*ny<=floor:
            # The pair carries below-threshold physical amplitude.  Remove
            # numerical noise explicitly so it cannot re-enter the active
            # tangent solely because a previous gauge normalization enlarged
            # both factors.
            state.X[a,rr]=0.0
            state.Y[a,rr]=0.0
            continue
        scale=float(np.sqrt(ny/nx))
        if not np.isfinite(scale) or scale<=0.0:
            continue
        state.X[a,rr]*=scale
        state.Y[a,rr]/=scale



def solve_internal_mvi(
    initial_state,ops,maps,bc,projected_fn,settings,*,quadrature
):
    bc.check_rigid_mode_requirement()
    state=initial_state.copy()
    state.validate_grid(ops.grid)
    _rebalance_separated_factors(state)

    # Refuse incompatible X rather than silently projecting it.
    ke.reduce_constrained_state_X(state,maps)

    history=[]
    final_p=float("inf")
    final_p_rel=float("inf")
    final_inc=float("inf")
    projected_scale=_fixed_projected_scale(state,ops,maps,projected_fn)

    for cyc in range(1,settings.max_xy_cycles+1):
        before=state.copy()

        hy=solve_strict_halfstep(
            state,ops,maps,"Y",projected_fn,settings,quadrature=quadrature,
            cycle=cyc,projected_scale=projected_scale
        )
        # Keep the separated representation well-scaled before switching the
        # active block.  The reconstructed U is unchanged identically.
        _rebalance_separated_factors(state)
        fprog.emit("mvi_halfstep", halfstep="Y", half_iterations=int(hy.iterations), projected_inf=float(hy.projected_inf), projected_relative=float(hy.projected_relative), projected_scale=float(projected_scale), correction_norm=float(hy.correction_norm), stage="MVI")
        hx=solve_strict_halfstep(
            state,ops,maps,"X",projected_fn,settings,quadrature=quadrature,
            cycle=cyc,projected_scale=projected_scale
        )
        _rebalance_separated_factors(state)
        fprog.emit("mvi_halfstep", halfstep="X", half_iterations=int(hx.iterations), projected_inf=float(hx.projected_inf), projected_relative=float(hx.projected_relative), projected_scale=float(projected_scale), correction_norm=float(hx.correction_norm), stage="MVI")

        py=normalized_projected_inf(state,ops,maps,"Y",projected_fn)
        px=normalized_projected_inf(state,ops,maps,"X",projected_fn)
        final_p=max(px,py)
        final_p_rel=_projected_relative(final_p,projected_scale)
        final_inc=_relative_field_change(before,state,ops.grid)

        history.append({
            "cycle":cyc,
            "Y_half_iterations":hy.iterations,
            "X_half_iterations":hx.iterations,
            "Y_projected_inf_after_cycle":py,
            "X_projected_inf_after_cycle":px,
            "projected_inf":final_p,
            "projected_relative":final_p_rel,
            "projected_scale":projected_scale,
            "field_increment_rel":final_inc,
        })
        try:
            _U=kp.reconstruct_fields(state); _w=_U[2]; _js,_ix=np.unravel_index(int(np.argmax(np.abs(_w))),_w.shape)
            _wmax=float(np.max(np.abs(_w))); _wval=float(_w[_js,_ix])
        except Exception:
            _wmax=None; _wval=None; _js=None; _ix=None
        fprog.emit("mvi_cycle", cycle=int(cyc), projected_inf=float(final_p), projected_relative=float(final_p_rel), projected_scale=float(projected_scale), field_increment_rel=float(final_inc),
                   Y_half_iterations=int(hy.iterations), X_half_iterations=int(hx.iterations), stage="MVI",
                   w_abs_max=_wmax,w_at_abs_max=_wval,wmax_is=_js,wmax_ix=_ix)

        if (
            _projected_converged(final_p_rel,settings)
            and final_inc<=settings.increment_tolerance
        ):
            return KLInternalMVIResult(
                state,True,"CONVERGED_REDUCED_MVI",cyc,history,
                final_p,final_inc,final_p_rel,projected_scale
            )

    return KLInternalMVIResult(
        state,False,"MAX_XY_CYCLES",settings.max_xy_cycles,history,
        final_p,final_inc,final_p_rel,projected_scale
    )


def make_elastic_projected_function(
    ops,maps,bc,*,E,nu,h,geometry,generalized_load,
    resultants_method="ANALYTIC",quadrature=None
):
    p=np.asarray(generalized_load,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")

    def projected(state,direction):
        r,labels,_,_=kw.elastic_projected_residual(
            state,ops,maps,bc,E=E,nu=nu,h=h,geometry=geometry,
            generalized_load=p,direction=direction,
            resultants_method=resultants_method,quadrature=quadrature
        )
        return r,labels
    def _prepare(state,direction):
        variations=kw.direction_variations(state,ops,maps,direction)
        labels=tuple(v.label for v in variations)
        norms=np.asarray([np.linalg.norm(v.dU.reshape(-1)) for v in variations],float)
        return {"variations":variations,"labels":labels,"norms":norms}

    def _evaluate_prepared(state,direction,prepared):
        kin,nm=kw.current_elastic_kinematics_resultants(
            state,ops,E=E,nu=nu,h=h,geometry=geometry,
            resultants_method=resultants_method,quadrature=quadrature
        )
        r,labels=kw.projected_residual_from_variations(
            kin,nm,prepared["variations"],ops,bc,geometry=geometry,
            generalized_load=p
        )
        return r,labels

    projected.mvi_prepare_direction=_prepare
    projected.mvi_evaluate_prepared=_evaluate_prepared

    # Resultant magnitude of the applied surface load.  It has the same force
    # dimension as the normalized weak projected residual.
    projected.mvi_projected_reference_scale=_surface_load_resultant_scale(
        ops.grid,p
    )
    return projected
