#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7D: GENERAL CUF plate-style MVI solver for 3m fields.

This is the Stage 3C algorithm generalized only in storage dimension:
TE1 six fields -> general CUF 3m fields.

Plate-reference algorithm used by ``plate_vim_j2.py``:
- strict Y half-cycle, then X half-cycle;
- partner family frozen during each half-cycle;
- exactly one projected correction per half-cycle and per XY cycle;
- r = T^T W R;
- A = T^T W (K_T T);
- K_T*T by explicit residual perturbation in normalized profile directions;
- A dc = -r using only np.linalg.solve;
- no full-grid Newton correction;
- exact QR gauge transform of the frozen partner and pair balancing preserve
  every reconstructed physical field without rank truncation;
- no SVD/pinv/lstsq/Cholesky/regularization or fallback solver;
- no inner half-step loop and no half-step line search/damping.

Convergence measure:
The physical projected residual is retained unchanged for the reduced
equations and tangent.  A dimensionally homogenized copy is used only for the
stopping measure and diagnostics.  Each Y/X projected system keeps its own
fixed reference from its first active half-step.  This makes
``projected_tolerance`` dimensionless without changing the MVI equations.

For m=2 with F0=1,F1=z the solver is regression-tested against verified TE1
Stage 3C.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import final_progress as fprog
import final_v3_fd as v3fd

import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7b_general_mvi_profiles as g8
import cylinder_cuf_stage7c_general_boundary as g9


class GeneralMVIInternalError(RuntimeError):
    pass

class GeneralSingularReducedSystem(GeneralMVIInternalError):
    pass

class GeneralHalfstepDidNotConverge(GeneralMVIInternalError):
    pass


@dataclass(frozen=True)
class GeneralInternalMVISettings:
    projected_tolerance: float
    increment_tolerance: float
    tangent_fd_step: float
    max_halfstep_iterations: int
    max_xy_cycles: int
    detf_guarded_correction: bool = True
    max_correction_halvings: int = 24
    # Keep the Python-API default backward compatible.  CASE/launcher selects
    # STRICT_HALFSTEP by default for normal TE1--TE4 runs.
    algorithm: str = "PLATE_STYLE_ONE_CORRECTION"
    # Residual formulation. STRONG is the audited production default;
    # VARIATIONAL is an opt-in verification path for full GL CUF elasticity.
    formulation: str = "STRONG"
    # Full residual is a quality/classification diagnostic for fixed-rank MVI.
    # It never invalidates a state whose projected equations and field
    # increment have converged.
    full_residual_tolerance: float | None = None

    def __post_init__(self):
        if self.projected_tolerance<=0: raise ValueError("projected_tolerance must be positive")
        if self.increment_tolerance<=0: raise ValueError("increment_tolerance must be positive")
        if self.tangent_fd_step<=0: raise ValueError("tangent_fd_step must be positive")
        if self.max_halfstep_iterations<1: raise ValueError("max_halfstep_iterations must be >=1")
        if self.max_xy_cycles<1: raise ValueError("max_xy_cycles must be >=1")
        if self.max_correction_halvings<0: raise ValueError("max_correction_halvings must be >=0")
        if self.full_residual_tolerance is not None and self.full_residual_tolerance<=0:
            raise ValueError("full_residual_tolerance must be positive when supplied")
        algorithm=str(self.algorithm).upper()
        if algorithm not in ("PLATE_STYLE_ONE_CORRECTION","STRICT_HALFSTEP"):
            raise ValueError("algorithm must be PLATE_STYLE_ONE_CORRECTION or STRICT_HALFSTEP")
        formulation=str(self.formulation).upper()
        if formulation not in ("STRONG","VARIATIONAL"):
            raise ValueError("formulation must be STRONG or VARIATIONAL")


@dataclass(frozen=True)
class GeneralHalfstepRecord:
    direction: str
    iterations: int
    projected_inf: float
    correction_norm: float
    status: str
    projected_relative: float = float("nan")
    projected_scale: float = float("nan")


@dataclass
class GeneralInternalMVIResult:
    state: g8.ProfileStateCUF
    converged: bool
    status: str
    cycles: int
    history: list
    final_projected_inf: float
    final_increment_rel: float
    final_projected_relative: float = float("nan")
    projected_scale: float = float("nan")
    final_x_projected_relative: float = float("nan")
    final_y_projected_relative: float = float("nan")
    full_residual_relative_to_load: float | None = None
    full_equilibrium_satisfied: bool = False
    finite_order_equilibrium_only: bool = False
    full_residual_tolerance: float = float("nan")


def _field_vector(state,grid):
    return g8.reconstruct_vector(state,grid)


def _relative_field_change(old,new,grid):
    a=_field_vector(old,grid)
    b=_field_vector(new,grid)
    return float(np.linalg.norm(b-a)/max(np.linalg.norm(b),1e-300))


def _full_residual_diagnostics(state,grid,residual_fn,measure_multipliers=None):
    """Return full/interior/boundary strong-residual diagnostics only.

    The physical residual is never modified.  When V-4 measure multipliers
    are available, the reported norms use their dimensionally homogenized
    copy.  Relative values are normalized by the correspondingly scaled
    applied-load reference attached to the residual function.
    """
    R=np.asarray(_residual_vector(residual_fn,state,grid),float)
    shape=(state.n_fields,grid.ns,grid.nx)
    if R.size!=int(np.prod(shape)):
        raise GeneralMVIInternalError("FULL_RESIDUAL_DIAGNOSTIC_SHAPE_MISMATCH")
    rr=R.reshape(shape)
    if measure_multipliers is None:
        mm=np.ones_like(rr)
    else:
        mm=np.asarray(measure_multipliers,float).reshape(shape)
        if not np.all(np.isfinite(mm)) or np.any(mm<=0.0):
            raise GeneralMVIInternalError("INVALID_FULL_RESIDUAL_MEASURE_MULTIPLIERS")
    rm=rr*mm

    full_inf=float(np.max(np.abs(rm))) if rm.size else 0.0
    if grid.nx>2:
        interior_inf=float(np.max(np.abs(rm[:,:,1:-1])))
    else:
        interior_inf=0.0
    boundary_inf=max(
        float(np.max(np.abs(rm[:,:,0]))) if grid.nx else 0.0,
        float(np.max(np.abs(rm[:,:,-1]))) if grid.nx else 0.0,
    )

    reference=getattr(residual_fn,"mvi_projected_reference_residual",None)
    load_scale=None
    if reference is not None:
        ref=np.asarray(reference,float).reshape(shape)
        if np.all(np.isfinite(ref)):
            refm=ref*mm
            # End rows are replaced by BC equations.  Normalize the full
            # strong residual by the physical applied-load scale carried by
            # interior equations so the ratio remains meaningful for q.
            if grid.nx>2:
                load_scale=float(np.max(np.abs(refm[:,:,1:-1])))
            else:
                load_scale=float(np.max(np.abs(refm)))
            if not np.isfinite(load_scale) or load_scale<=1.0e-300:
                load_scale=None

    def rel(v):
        return None if load_scale is None else float(v)/float(load_scale)

    return {
        "full_residual_inf_scaled":full_inf,
        "interior_residual_inf_scaled":interior_inf,
        "boundary_residual_inf_scaled":boundary_inf,
        "full_residual_relative_to_load":rel(full_inf),
        "interior_residual_relative_to_load":rel(interior_inf),
        "boundary_residual_relative_to_load":rel(boundary_inf),
        "full_residual_load_scale":load_scale,
    }

def _active_mask_for_direction(state,grid,bc,direction):
    d=str(direction).upper()
    if d=="X":
        return g9.homogeneous_geometric_x_active_mask(state,grid,bc)
    if d=="Y":
        return np.ones_like(state.Y,dtype=bool)
    raise ValueError("direction must be X or Y")


def _direction_data(state,grid,bc,direction):
    mask=_active_mask_for_direction(state,grid,bc,direction)
    T,ids=g8.build_direction_tangent(state,grid,direction,mask)
    norms=np.linalg.norm(T,axis=0)
    # Same relative activity rule as the audited sandwich MVI: a profile
    # family is inactive only when its tangent is negligible relative to the
    # largest partner family in this half-step.  This removes zero-sensitivity
    # coordinates before the reduced solve; it neither changes state nor rank.
    scale=max(float(np.max(norms)) if norms.size else 0.0,1.0)
    active=norms>1.0e-14*scale
    if not np.any(active):
        return mask,T,ids,norms,active,np.zeros((T.shape[0],0))
    Tn=T[:,active]/norms[active][None,:]
    return mask,T,ids,norms,active,Tn


def _balance_all_profile_pairs(state):
    """Plate-MVI pair balancing; every separated product is unchanged."""
    for component in range(3):
        for tau in range(state.m):
            for term in range(state.rank):
                xp=state.X[component,tau,term]
                yp=state.Y[component,tau,term]
                nx=float(np.linalg.norm(xp))
                ny=float(np.linalg.norm(yp))
                if nx<=1.0e-30 or ny<=1.0e-30:
                    continue
                scale=float(np.sqrt(ny/nx))
                xp*=scale
                yp/=scale


def _orthonormalise_frozen_partner(state,grid,bc,partner_direction):
    """Exact plate-MVI QR gauge transform with no truncation."""
    d=str(partner_direction).upper()
    if d=="X":
        partner=state.X
        opposite=state.Y
    elif d=="Y":
        partner=state.Y
        opposite=state.X
    else:
        raise ValueError("partner_direction must be X or Y")

    mask=_active_mask_for_direction(state,grid,bc,d)
    for component in range(3):
        for tau in range(state.m):
            free_mask=np.asarray(mask[component,tau,0],bool)
            if not np.all(mask[component,tau]==free_mask[None,:]):
                raise GeneralMVIInternalError("rank-dependent partner mask")
            free=np.flatnonzero(free_mask)
            if free.size<state.rank:
                raise GeneralMVIInternalError(
                    "MVI rank exceeds independent partner profile nodes"
                )
            matrix=np.asarray(partner[component,tau][:,free],float)
            if float(np.linalg.norm(matrix))<=1.0e-30:
                continue
            q,r=np.linalg.qr(matrix.T,mode="reduced")
            # Same fixed-rank QR gauge as the audited sandwich MVI.
            #
            # QR is an exact change of separated coordinates:
            #       partner.T = Q R,   opposite_new = R opposite,
            # hence opposite_new.T @ Q.T == opposite.T @ partner.
            # This identity holds also when R has a very small diagonal entry.
            # In that case Q supplies a completed numerical basis; no MVI term
            # is removed, no pseudo-inverse is used and the reconstructed CUF
            # field is preserved.  Skipping the transform left a degenerate
            # partner in the next half-step and was the direct source of the
            # rank-3+ singular reduced systems.
            rdiag=np.abs(np.diag(r))
            rmax=float(np.max(rdiag)) if rdiag.size else 0.0
            rmin=float(np.min(rdiag)) if rdiag.size else 0.0
            rank_tolerance=(
                np.finfo(float).eps*max(matrix.shape)*max(rmax,1e-300)*100.0
            )
            transformed=np.zeros_like(partner[component,tau])
            transformed[:,free]=q.T
            partner[component,tau]=transformed
            opposite[component,tau]=r@opposite[component,tau]
            if rdiag.size!=state.rank or rmin<=rank_tolerance:
                fprog.emit(
                    "mvi_partner_qr_completed_rank_deficient",
                    stage="MVI_GAUGE",
                    direction=d,
                    component=int(component),
                    tau=int(tau),
                    rank=int(state.rank),
                    min_R_diag=float(rmin),
                    max_R_diag=float(rmax),
                    rank_tolerance=float(rank_tolerance),
                    action="COMPLETE_GAUGE_WITHOUT_RANK_TRUNCATION",
                )

    if d=="X":
        g9.enforce_homogeneous_geometric_x_profiles(state,grid,bc)


def _plate_style_single_halfcycle(
    state,grid,bc,direction,residual_fn,settings,*,weights,
    measure_multipliers,projected_scale,quadrature,cycle
):
    """Apply exactly one plate-reference projected correction."""
    d=str(direction).upper()
    partner="X" if d=="Y" else "Y"
    before_field=_field_vector(state,grid)
    _orthonormalise_frozen_partner(state,grid,bc,partner)

    R=_residual_vector(residual_fn,state,grid)
    mask,_T,ids,norms,active,Tn=_direction_data(state,grid,bc,d)
    if Tn.shape[1]==0:
        raise GeneralMVIInternalError(
            f"{d}_HALFCYCLE_HAS_NO_ACTIVE_PROFILE_DIRECTIONS"
        )
    projected=_weighted_projected(Tn,R,weights)
    raw_pinf=float(np.max(np.abs(projected))) if projected.size else 0.0
    if measure_multipliers is None:
        Rmeasure=R
    else:
        Rmeasure=np.asarray(measure_multipliers,float).reshape(-1)*R
    projected_measure=_weighted_projected(Tn,Rmeasure,weights)
    pinf=float(np.max(np.abs(projected_measure))) if projected_measure.size else 0.0
    pscale=_positive_projected_scale(projected_scale)
    fprog.emit(
        "mvi_halfstep_iteration",stage="MVI_HALFSTEP",halfstep=d,
        cycle=cycle,max_xy_cycles=int(settings.max_xy_cycles),
        halfstep_iteration=1,max_halfstep_iterations=1,
        projected_inf=pinf,projected_relative=pinf/pscale,
        projected_scale=pscale,raw_projected_inf=raw_pinf,
        active_directions=int(Tn.shape[1]),
        algorithm="PLATE_STYLE_ONE_CORRECTION",
    )

    KT_Tn,Tn2,_mask2,ids2,_norms2,active2=directional_tangent_action_fd(
        state,grid,bc,d,residual_fn,
        tangent_fd_step=settings.tangent_fd_step,
        quadrature=quadrature,base_residual=R,
        cycle=cycle,halfstep_iteration=1,
        max_halfstep_iterations=1,max_xy_cycles=int(settings.max_xy_cycles),
        detf_guarded_correction=bool(settings.detf_guarded_correction),
        max_correction_halvings=int(settings.max_correction_halvings),
    )
    if not np.array_equal(ids,ids2) or not np.array_equal(active,active2):
        raise GeneralMVIInternalError("direction data changed inside halfcycle")
    if np.max(np.abs(Tn-Tn2))>1e-12:
        raise GeneralMVIInternalError("normalized tangent changed inside halfcycle")

    reduced=Tn.T@(weights[:,None]*KT_Tn)
    delta_norm=_direct_solve(reduced,projected,d)
    current,ids3=g8.get_active_parameters(state,d,mask)
    if not np.array_equal(ids,ids3):
        raise GeneralMVIInternalError("direction data changed before correction")
    correction=np.zeros_like(current)
    positions=np.flatnonzero(active)
    correction[positions]=delta_norm/norms[active]

    candidate=state.copy()
    g8.set_active_parameters(candidate,d,ids,current+correction)
    _balance_all_profile_pairs(candidate)
    g9.enforce_homogeneous_geometric_x_profiles(candidate,grid,bc)

    for check_direction in ("Y","X"):
        _mask,_T,_ids,_norms,_active,Tcheck=_direction_data(
            candidate,grid,bc,check_direction
        )
        if Tcheck.shape[1]==0:
            fprog.emit(
                "mvi_halfstep_candidate_rejected",stage="MVI_HALFSTEP",
                halfstep=d,cycle=cycle,
                reason="NO_ACTIVE_PARTNER_DIRECTIONS",
            )
            return state,GeneralHalfstepRecord(
                d,1,pinf,0.0,"SKIPPED_INACTIVE_CANDIDATE",pinf/pscale,pscale
            )

    correction_field=float(np.linalg.norm(
        _field_vector(candidate,grid)-before_field
    ))
    return candidate,GeneralHalfstepRecord(
        d,1,pinf,correction_field,"ONE_PLATE_STYLE_CORRECTION",pinf/pscale,pscale
    )


def _residual_vector(residual_fn,state,grid):
    r=np.asarray(residual_fn(state),float)
    field_shape=(3,state.m,grid.ns,grid.nx)
    nfull=g8.n_full_dofs(state,grid)
    if r.shape==field_shape:
        r=r.reshape(-1)
    if r.shape!=(nfull,):
        raise GeneralMVIInternalError(
            f"residual must have shape {(nfull,)} or {field_shape}, got {r.shape}"
        )
    if not np.all(np.isfinite(r)):
        raise GeneralMVIInternalError("NONFINITE_FULL_RESIDUAL")
    return r


def _weighted_projected(Tn,residual,weights):
    return Tn.T@(weights*residual)


def _mvi_characteristic_scale(state,grid,quadrature,direction,ids):
    """J-12 V3 scale for active Taylor-CUF profile parameters.

    The final general-CUF branch exposes Taylor TE1..TE4, so q_tau has the
    same physical coefficient scale used by V3: h/L**tau.  For a normalized
    MVI direction the physical generalized-field amplitude is c_j*||T_j||.
    """
    h_shell=v3fd.thickness_from_quadrature(quadrature)
    L=float(grid.length_x)
    n_dir=grid.nx if str(direction).upper()=="X" else grid.ns
    out=np.empty(len(ids),dtype=float)
    for k,flat_idx in enumerate(ids):
        _component,tau,_rank,_node=g8._decode(
            flat_idx,state.m,state.rank,n_dir
        )
        out[k]=h_shell/(L**tau)
    return out



def _surface_first_derivatives_full_direction(dq,ops,basis):
    """First surface derivatives of a full CUF direction field."""
    dq=np.asarray(dq,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if dq.shape!=expected:
        raise ValueError(f"full CUF tangent direction must have shape {expected}")
    dqx=np.empty_like(dq)
    dqs=np.empty_like(dq)
    for a in range(3):
        for tau in range(basis.m):
            dqx[a,tau]=ops.Dx(dq[a,tau])
            dqs[a,tau]=ops.Ds(dq[a,tau])
    return dqx,dqs


def _assemble_exact_tangent_with_end_bcs(draw,dq,dresultants,bc):
    """Derivative of the Stage7C strong end-BC row replacement."""
    draw=np.asarray(draw,float)
    dq=np.asarray(dq,float)
    if draw.shape!=dq.shape:
        raise ValueError("draw/dq shape mismatch")
    m,_ns,nx=dq.shape[1],dq.shape[2],dq.shape[3]
    if m!=bc.m:
        raise ValueError("BC m mismatch")
    out=draw.copy()
    natural=(dresultants.Au_x,dresultants.Av_x,dresultants.Aw_x)
    for _side,end,ix,normal in (
        ("LEFT",bc.left,0,-1.0),("RIGHT",bc.right,nx-1,+1.0)
    ):
        for a in range(3):
            for tau in range(m):
                cond=end.conditions[a][tau]
                if cond.kind=="GEOMETRIC":
                    out[a,tau,:,ix]=dq[a,tau,:,ix]
                elif cond.kind=="NATURAL":
                    out[a,tau,:,ix]=normal*natural[a][tau,:,ix]
                elif cond.kind=="ELASTIC":
                    out[a,tau,:,ix]=(normal*natural[a][tau,:,ix]
                                     +cond.stiffness_array(dq.shape[2])*dq[a,tau,:,ix])
                else:
                    raise ValueError(f"unsupported CUF boundary kind: {cond.kind}")
    return out



def _axis_two_layer_supports(D,layers=2):
    """Conservative axis support of q -> first derivatives -> strong residual."""
    D=np.asarray(D,float)
    if D.ndim!=2 or D.shape[0]!=D.shape[1]:
        raise ValueError("derivative matrix must be square")
    rows_by_input=tuple(
        np.flatnonzero(D[:,j] != 0.0).astype(int) for j in range(D.shape[1])
    )
    supports=[]
    for j0 in range(D.shape[1]):
        reached={int(j0)}
        frontier={int(j0)}
        for _ in range(int(layers)):
            nxt=set()
            for j in frontier:
                nxt.update(int(i) for i in rows_by_input[j])
            nxt-=reached
            reached|=nxt
            frontier=nxt
            if not frontier:
                break
        supports.append(np.asarray(sorted(reached),dtype=int))
    return tuple(supports)


def _profile_coloring_groups(state,ops,direction,active_ids):
    """Safe exact-action groups for separated MVI profile directions.

    An X profile parameter changes a complete s-line at one x node; therefore
    only x supports matter for disjoint extraction.  A Y profile parameter
    changes a complete x-line at one s node; analogously only s supports
    matter.  Full residual-field coupling is treated conservatively by never
    sharing any axis support inside one group.
    """
    d=str(direction).upper()
    if d=="X":
        n_dir=int(ops.grid.nx)
        supports=_axis_two_layer_supports(ops.D1x,layers=2)
    elif d=="Y":
        n_dir=int(ops.grid.ns)
        supports=_axis_two_layer_supports(ops.D1s,layers=2)
    else:
        raise ValueError("direction must be X or Y")

    ids=np.asarray(active_ids,dtype=int).reshape(-1)
    nodes=np.asarray([int(i)%n_dir for i in ids],dtype=int)
    groups=[]
    occupied=[]
    # Large end-stencil supports first, then stable original column order.
    order=sorted(range(ids.size),key=lambda j:(-len(supports[nodes[j]]),j))
    for jcol in order:
        supp=supports[nodes[jcol]]
        placed=False
        for g,occ in zip(groups,occupied):
            if not np.any(occ[supp]):
                g.append(jcol)
                occ[supp]=True
                placed=True
                break
        if not placed:
            occ=np.zeros(n_dir,dtype=bool)
            occ[supp]=True
            groups.append([jcol])
            occupied.append(occ)

    # Hard invariant: no two columns in a group may share an output axis row.
    for group in groups:
        seen=np.zeros(n_dir,dtype=bool)
        for jcol in group:
            supp=supports[nodes[jcol]]
            if np.any(seen[supp]):
                raise GeneralMVIInternalError("MVI_EXACT_PROFILE_COLOR_OVERLAP")
            seen[supp]=True
    return tuple(tuple(g) for g in groups),supports,nodes


def _profile_output_rows(state,grid,direction,axis_support):
    """Full flattened residual rows for one conservative profile support."""
    nf=int(state.n_fields); ns=int(grid.ns); nx=int(grid.nx)
    supp=np.asarray(axis_support,dtype=int)
    mask=np.zeros((nf,ns,nx),dtype=bool)
    if str(direction).upper()=="X":
        mask[:,:,supp]=True
    else:
        mask[:,supp,:]=True
    return np.flatnonzero(mask.reshape(-1)).astype(int)


def directional_tangent_action_exact(
    state,grid,bc,direction,residual_fn,*,
    cycle=None,halfstep_iteration=None,max_halfstep_iterations=None,
    max_xy_cycles=None
):
    """Return K_T*Tn from an exact tangent factory attached to residual_fn.

    The factory is prepared exactly once at the current base state, so the
    Green--Lagrange/von-Karman base constitutive data are cached across every
    normalized MVI profile direction of this half-step iteration.
    """
    factory=getattr(residual_fn,"mvi_prepare_exact_tangent_action",None)
    if factory is None:
        raise GeneralMVIInternalError("EXACT_MVI_TANGENT_FACTORY_NOT_AVAILABLE")

    mask,_T,ids,norms,active,Tn=_direction_data(state,grid,bc,direction)
    if Tn.shape[1]==0:
        raise GeneralMVIInternalError(
            f"{str(direction).upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS"
        )

    action=factory(state)
    if not callable(action):
        raise GeneralMVIInternalError("INVALID_EXACT_MVI_TANGENT_ACTION")

    tangent_columns=int(Tn.shape[1])
    out=np.zeros((g8.n_full_dofs(state,grid),tangent_columns),float)
    active_ids=np.asarray(ids,dtype=int)[np.asarray(active,dtype=bool)]
    ops=getattr(residual_fn,"mvi_exact_tangent_ops",None)
    use_coloring=bool(
        ops is not None and
        getattr(residual_fn,"mvi_exact_tangent_profile_coloring",True)
    )
    if use_coloring:
        groups,supports,nodes=_profile_coloring_groups(
            state,ops,direction,active_ids
        )
    else:
        groups=tuple((j,) for j in range(tangent_columns))
        supports=None; nodes=None

    tangent_actions=len(groups)
    row_cache={}
    if use_coloring:
        for node in np.unique(nodes):
            row_cache[int(node)]=_profile_output_rows(
                state,grid,direction,supports[int(node)]
            )
    progress_stride=max(1,(tangent_columns+9)//10)
    fprog.emit(
        "mvi_tangent_mode",stage="MVI_TANGENT",
        halfstep=str(direction).upper(),cycle=cycle,
        halfstep_iteration=halfstep_iteration,
        tangent_mode=("EXACT_FROZEN_GB_PROFILE_COLORED" if use_coloring
                      else "EXACT_FROZEN_GB"),
        tangent_columns=tangent_columns,tangent_actions=tangent_actions,
    )
    completed=0
    next_progress=1
    for iaction,group in enumerate(groups,1):
        combined=np.sum(Tn[:,np.asarray(group,dtype=int)],axis=1)
        summed=np.asarray(action(combined),float).reshape(-1)
        if summed.shape!=(out.shape[0],):
            raise GeneralMVIInternalError("exact tangent action size mismatch")
        if not np.all(np.isfinite(summed)):
            raise GeneralMVIInternalError("non-finite exact tangent action")
        if use_coloring:
            for jcol in group:
                rows=row_cache[int(nodes[jcol])]
                out[rows,jcol]=summed[rows]
        else:
            out[:,group[0]]=summed
        completed+=len(group)
        if (completed>=next_progress or iaction==tangent_actions):
            fprog.emit(
                "mvi_tangent_progress",stage="MVI_TANGENT",
                halfstep=str(direction).upper(),cycle=cycle,max_xy_cycles=max_xy_cycles,
                halfstep_iteration=halfstep_iteration,
                max_halfstep_iterations=max_halfstep_iterations,
                tangent_column=completed,tangent_columns=tangent_columns,
                tangent_percent=100.0*completed/max(tangent_columns,1),
                tangent_action=iaction,tangent_actions=tangent_actions,
                tangent_mode=("EXACT_FROZEN_GB_PROFILE_COLORED" if use_coloring
                              else "EXACT_FROZEN_GB"),
            )
            next_progress=completed+progress_stride
    return out,Tn,mask,ids,norms,active


def directional_tangent_action_prefer_exact(
    state,grid,bc,direction,residual_fn,*,
    tangent_fd_step,quadrature,base_residual=None,
    cycle=None,halfstep_iteration=None,max_halfstep_iterations=None,
    max_xy_cycles=None,detf_guarded_correction=True,max_correction_halvings=12
):
    """Use exact frozen-GB tangent when the active mechanics branch supplies it.

    Legacy/unsupported residual functions keep the verified central-FD path.
    """
    if callable(getattr(residual_fn,"mvi_prepare_exact_tangent_action",None)):
        return directional_tangent_action_exact(
            state,grid,bc,direction,residual_fn,
            cycle=cycle,halfstep_iteration=halfstep_iteration,
            max_halfstep_iterations=max_halfstep_iterations,
            max_xy_cycles=max_xy_cycles,
        )
    return directional_tangent_action_fd(
        state,grid,bc,direction,residual_fn,
        tangent_fd_step=tangent_fd_step,quadrature=quadrature,
        base_residual=base_residual,cycle=cycle,
        halfstep_iteration=halfstep_iteration,
        max_halfstep_iterations=max_halfstep_iterations,
        max_xy_cycles=max_xy_cycles,
        detf_guarded_correction=detf_guarded_correction,
        max_correction_halvings=max_correction_halvings,
    )


def directional_tangent_action_fd(
    state,grid,bc,direction,residual_fn,*,
    tangent_fd_step,quadrature,base_residual=None,
    cycle=None,halfstep_iteration=None,max_halfstep_iterations=None,
    max_xy_cycles=None,detf_guarded_correction=True,max_correction_halvings=12
):
    """Return K_T*Tn by J-12 V3-scaled central residual differences."""
    mask,_T,ids,norms,active,Tn=_direction_data(
        state,grid,bc,direction
    )
    if Tn.shape[1]==0:
        raise GeneralMVIInternalError(
            f"{str(direction).upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS"
        )

    positions=np.flatnonzero(active)
    out=np.empty((g8.n_full_dofs(state,grid),positions.size),float)

    current,ids2=g8.get_active_parameters(state,direction,mask)
    if not np.array_equal(ids,ids2):
        raise GeneralMVIInternalError("active parameter ordering changed")
    char=_mvi_characteristic_scale(
        state,grid,quadrature,direction,ids
    )
    amplitude=current*norms
    hj=v3fd.relative_steps(amplitude,char,tangent_fd_step)
    tangent_columns=int(positions.size)
    progress_stride=max(1,(tangent_columns+9)//10)

    max_halvings=int(max_correction_halvings)
    guard_enabled=bool(detf_guarded_correction)

    for jcol,pos in enumerate(positions):
        hcol=float(hj[pos])
        last_exc=None
        success=False
        for halvings in range(max_halvings+1):
            plus=state.copy()
            cp,ids3=g8.get_active_parameters(plus,direction,mask)
            if not np.array_equal(ids,ids3):
                raise GeneralMVIInternalError("active parameter ordering changed")
            cp[pos]+=hcol/norms[pos]
            g8.set_active_parameters(plus,direction,ids,cp)

            minus=state.copy()
            cm,ids4=g8.get_active_parameters(minus,direction,mask)
            if not np.array_equal(ids,ids4):
                raise GeneralMVIInternalError("active parameter ordering changed")
            cm[pos]-=hcol/norms[pos]
            g8.set_active_parameters(minus,direction,ids,cm)

            try:
                Rp=_residual_vector(residual_fn,plus,grid)
                Rm=_residual_vector(residual_fn,minus,grid)
            except Exception as exc:
                if (not guard_enabled) or (not _is_invalid_deformation_error(exc)):
                    raise
                last_exc=exc
                if halvings>=max_halvings:
                    break
                hcol*=0.5
                continue

            out[:,jcol]=(Rp-Rm)/(2.0*hcol)
            success=True
            if halvings:
                fprog.emit(
                    "mvi_tangent_damped",stage="MVI_TANGENT",
                    halfstep=str(direction).upper(),cycle=cycle,
                    halfstep_iteration=halfstep_iteration,
                    tangent_column=int(jcol+1),
                    damping_factor=float(hcol/float(hj[pos])),
                    tangent_halvings=int(halvings),
                    reason="INVALID_DEFORMATION_MAPPING_GUARD",
                )
            break
        if not success:
            raise GeneralMVIInternalError(
                f"DET_F_GUARDED_TANGENT_FAILED / {str(direction).upper()}_HALFSTEP / "
                f"column={int(jcol+1)} / max_halvings={max_halvings} / "
                f"last_error={type(last_exc).__name__}: {last_exc}"
            )
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
    return out,Tn,mask,ids,norms,active


def _direct_solve(A,projected,direction):
    try:
        delta=np.linalg.solve(np.asarray(A,float),-np.asarray(projected,float))
    except np.linalg.LinAlgError as exc:
        raise GeneralSingularReducedSystem(
            f"SINGULAR_REDUCED_MVI_SYSTEM / {str(direction).upper()}_HALFSTEP"
        ) from exc
    if not np.all(np.isfinite(delta)):
        raise GeneralSingularReducedSystem(
            f"NONFINITE_REDUCED_MVI_CORRECTION / {str(direction).upper()}_HALFSTEP"
        )
    return delta


def _is_invalid_deformation_error(exc):
    text=f"{type(exc).__name__}: {exc}"
    return ("INVALID_DEFORMATION_MAPPING" in text) or ("InvalidDeformationMapping" in text)


def _commit_detf_guarded_update(
    state,grid,d,residual_fn,mask,ids,current,dp,settings,*,
    cycle=None,halfstep_iteration=None,Tn=None,weights=None,
    measure_multipliers=None,current_projected_relative=None,
    projected_scale=1.0
):
    if not settings.detf_guarded_correction:
        g8.set_active_parameters(state,d,ids,current+dp)
        return 1.0,0

    alpha=1.0
    last_exc=None
    for halvings in range(int(settings.max_correction_halvings)+1):
        trial=state.copy()
        g8.set_active_parameters(trial,d,ids,current+alpha*dp)
        try:
            Rtrial=_residual_vector(residual_fn,trial,grid)
        except Exception as exc:
            if not _is_invalid_deformation_error(exc):
                raise
            last_exc=exc
            if halvings>=int(settings.max_correction_halvings):
                break
            alpha*=0.5
            continue

        trial_rel=None
        if Tn is not None and weights is not None and current_projected_relative is not None:
            if measure_multipliers is None:
                Rmeasure=Rtrial
            else:
                mm=np.asarray(measure_multipliers,float).reshape(-1)
                Rmeasure=mm*Rtrial
            projected_trial=_weighted_projected(Tn,Rmeasure,np.asarray(weights,float))
            pinf_trial=float(np.max(np.abs(projected_trial))) if projected_trial.size else 0.0
            trial_rel=_projected_relative(pinf_trial,projected_scale)
            if trial_rel>float(current_projected_relative):
                last_exc=GeneralMVIInternalError(
                    f"PROJECTED_RESIDUAL_WOULD_INCREASE "
                    f"{float(current_projected_relative):.17g}->{trial_rel:.17g}"
                )
                if halvings>=int(settings.max_correction_halvings):
                    break
                alpha*=0.5
                continue

        g8.set_active_parameters(state,d,ids,current+alpha*dp)
        if halvings:
            fprog.emit(
                "mvi_correction_damped",stage="MVI_HALFSTEP",
                halfstep=str(d).upper(),cycle=cycle,
                halfstep_iteration=halfstep_iteration,
                damping_factor=float(alpha),
                correction_halvings=int(halvings),
                projected_relative_after=None if trial_rel is None else float(trial_rel),
                reason="DETF_OR_PROJECTED_RESIDUAL_GUARD",
            )
        return float(alpha),int(halvings)

    raise GeneralMVIInternalError(
        f"DET_F_GUARDED_CORRECTION_FAILED / {str(d).upper()}_HALFSTEP / "
        f"max_correction_halvings={int(settings.max_correction_halvings)} / "
        f"last_error={type(last_exc).__name__}: {last_exc}"
    )


def _projected_inf_from_residual(
    state,grid,bc,direction,R,weights,*,measure_multipliers=None
):
    """Projected stopping measure for a residual already evaluated at state.

    This helper is algebraically identical to ``projected_inf_for_direction``
    and exists only so the same physical residual can be reused for X and Y
    checks after a completed XY cycle.
    """
    R=np.asarray(R,float).reshape(-1)
    expected=(g8.n_full_dofs(state,grid),)
    if R.shape!=expected or not np.all(np.isfinite(R)):
        raise ValueError("invalid precomputed residual")
    _,_,_,_,_active,Tn=_direction_data(state,grid,bc,direction)
    if Tn.shape[1]==0:
        if float(np.max(np.abs(R)))==0.0:
            return 0.0
        raise GeneralMVIInternalError(
            f"{str(direction).upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS"
        )
    if measure_multipliers is None:
        Rmeasure=R
    else:
        mm=np.asarray(measure_multipliers,float).reshape(-1)
        if mm.shape!=R.shape or not np.all(np.isfinite(mm)) or np.any(mm<=0.0):
            raise ValueError("invalid V-4 measure_multipliers")
        Rmeasure=mm*R
    projected=_weighted_projected(Tn,Rmeasure,weights)
    return float(np.max(np.abs(projected))) if projected.size else 0.0


def projected_inf_for_direction(
    state,grid,bc,direction,residual_fn,weights,*,measure_multipliers=None
):
    R=_residual_vector(residual_fn,state,grid)
    return _projected_inf_from_residual(
        state,grid,bc,direction,R,weights,
        measure_multipliers=measure_multipliers
    )

def _positive_projected_scale(value):
    scale=float(value)
    if not np.isfinite(scale) or scale<=0.0:
        raise ValueError("projected reference scale must be finite and positive")
    return scale


def _projected_relative(projected_inf,projected_scale):
    return float(projected_inf)/_positive_projected_scale(projected_scale)


def _fixed_projected_scale(
    state,grid,bc,residual_fn,weights,*,measure_multipliers=None,
    directions=("Y","X")
):
    """Fixed measure scale for selected CUF half-step directions.

    Besides the initial residual, retain the projected applied-load scale.
    The latter is independent of Birger's current GB update and prevents an
    already equilibrated outer iterate from defining a near-zero reference.
    """
    supplied=float(getattr(residual_fn,"mvi_projected_reference_scale",0.0))
    if not np.isfinite(supplied) or supplied<0.0:
        raise GeneralMVIInternalError("INVALID_MVI_PROJECTED_REFERENCE_SCALE")
    reference=getattr(residual_fn,"mvi_projected_reference_residual",None)
    if reference is not None:
        reference=np.asarray(reference,float).reshape(-1)
        expected=(g8.n_full_dofs(state,grid),)
        if reference.shape!=expected or not np.all(np.isfinite(reference)):
            raise GeneralMVIInternalError("INVALID_MVI_PROJECTED_REFERENCE_RESIDUAL")
        if measure_multipliers is not None:
            mm=np.asarray(measure_multipliers,float).reshape(-1)
            if mm.shape!=reference.shape or not np.all(np.isfinite(mm)) or np.any(mm<=0.0):
                raise ValueError("invalid V-4 measure_multipliers")
            reference=mm*reference
    initial=[]
    applied=[]
    for direction in tuple(str(d).upper() for d in directions):
        try:
            initial.append(projected_inf_for_direction(
                state,grid,bc,direction,residual_fn,weights,
                measure_multipliers=measure_multipliers
            ))
            if reference is not None:
                _,_,_,_,_active,Tn=_direction_data(
                    state,grid,bc,direction
                )
                if Tn.shape[1]:
                    rload=_weighted_projected(Tn,reference,weights)
                    applied.append(
                        float(np.max(np.abs(rload))) if rload.size else 0.0
                    )
        except GeneralMVIInternalError:
            # Preserve the established inactive-direction diagnostic in the
            # halfstep if that direction is actually required.
            pass
    # Unit-scale floor follows the established V2 residual-scale definition.
    return max([supplied,*applied,*initial,1.0])


def solve_strict_halfstep(
    state,grid,bc,direction,residual_fn,settings,*,weights=None,
    measure_multipliers=None,quadrature,cycle=None,projected_scale=1.0,
    equation_scale=1.0
):
    d=str(direction).upper()
    pscale=_positive_projected_scale(projected_scale)
    # IMPORTANT: equation_scale belongs to the original reduced equation
    # T^T W R = 0 and is fixed from this direction's first active halfstep.
    # Re-initialising it from each current residual makes rank>1 MVI demand an
    # artificial absolute ~1e-8...1e-10 residual after the first cycle.
    escale=_positive_projected_scale(equation_scale)
    w=g8.residual_projection_weights(state,grid) if weights is None else np.asarray(weights,float)
    if w.shape!=(g8.n_full_dofs(state,grid),):
        raise ValueError("weights shape mismatch")

    # Gauge only: QR transfers the reciprocal factor exactly, hence it leaves
    # every physical generalized displacement unchanged.  Strict halfsteps
    # must use the same conditioning transform as the plate-style branch.
    partner="X" if d=="Y" else "Y"
    _orthonormalise_frozen_partner(state,grid,bc,partner)

    last_corr=0.0
    # A single detF line-search reduction can be a legitimate Newton
    # correction near the admissible boundary.  A persistent sequence of deep
    # reductions means that the outer load increment is too large; allowing
    # those tiny updates to count as a successful halfstep creates the long
    # stagnation visible in MVI-5 logs.  The counter is local to this halfstep
    # and is reset by any correction that does not need deep detF damping.
    deep_detf_damping_streak=0
    deep_detf_halvings=4
    for it in range(1,settings.max_halfstep_iterations+1):
        R=_residual_vector(residual_fn,state,grid)
        mask,_T,ids,norms,active,Tn=_direction_data(state,grid,bc,d)

        if Tn.shape[1]==0:
            if float(np.max(np.abs(R)))==0.0:
                return GeneralHalfstepRecord(d,it-1,0.0,0.0,"CONVERGED_REDUCED_MVI")
            raise GeneralMVIInternalError(f"{d}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS")

        # V-4: preserve the original reduced equations, but use the
        # dimensionally scaled residual only for the projected stopping MEASURE.
        projected=_weighted_projected(Tn,R,w)
        # The frozen-partner Newton system below solves the ORIGINAL reduced
        # equation Tn.T W R = 0.  Its half-step stopping test and correction
        # guard must use that same equation.  V-4 row scaling is a measure of
        # the completed physical residual; projecting M*R defines a different
        # reduced equation and can remain nonzero after the original Newton
        # equation has already been solved.  That mismatch made linear MVI-5+
        # repeat and damp an already converged X correction.
        equation_inf=float(np.max(np.abs(projected))) if projected.size else 0.0
        equation_relative=_projected_relative(equation_inf,escale)
        if measure_multipliers is None:
            Rmeasure=R
        else:
            mm=np.asarray(measure_multipliers,float).reshape(-1)
            if mm.shape!=R.shape or not np.all(np.isfinite(mm)) or np.any(mm<=0.0):
                raise ValueError("invalid V-4 measure_multipliers")
            Rmeasure=mm*R
        projected_measure=_weighted_projected(Tn,Rmeasure,w)
        pinf=float(np.max(np.abs(projected_measure))) if projected_measure.size else 0.0
        prel=_projected_relative(pinf,pscale)
        fprog.emit(
            "mvi_halfstep_iteration",stage="MVI_HALFSTEP",
            halfstep=d,cycle=cycle,max_xy_cycles=int(settings.max_xy_cycles),
            halfstep_iteration=int(it),
            max_halfstep_iterations=int(settings.max_halfstep_iterations),
            projected_inf=float(equation_inf),
            projected_relative=float(equation_relative),
            projected_scale=float(escale),
            v4_measure_projected_inf=float(pinf),
            v4_measure_projected_relative=float(prel),
            v4_measure_projected_scale=float(pscale),
            active_directions=int(Tn.shape[1]),
        )
        if equation_relative<=settings.projected_tolerance:
            return GeneralHalfstepRecord(
                d,it-1,equation_inf,last_corr,"CONVERGED_REDUCED_MVI",
                equation_relative,escale
            )

        KT_Tn,Tn2,_mask2,ids2,_norms2,active2=directional_tangent_action_prefer_exact(
            state,grid,bc,d,residual_fn,
            tangent_fd_step=settings.tangent_fd_step,
            quadrature=quadrature,
            base_residual=R,
            cycle=cycle,halfstep_iteration=int(it),
            max_halfstep_iterations=int(settings.max_halfstep_iterations),
            max_xy_cycles=int(settings.max_xy_cycles),
            detf_guarded_correction=bool(settings.detf_guarded_correction),
            max_correction_halvings=int(settings.max_correction_halvings),
        )
        if not np.array_equal(ids,ids2):
            raise GeneralMVIInternalError("direction data changed inside halfstep")
        if not np.array_equal(active,active2):
            raise GeneralMVIInternalError("active directions changed inside halfstep")
        if np.max(np.abs(Tn-Tn2))>1e-12:
            raise GeneralMVIInternalError("normalized tangent changed inside halfstep")

        A=Tn.T@(w[:,None]*KT_Tn)
        delta_norm=_direct_solve(A,projected,d)

        current,ids3=g8.get_active_parameters(state,d,mask)
        if not np.array_equal(ids,ids3):
            raise GeneralMVIInternalError("active parameter ordering changed")

        dp=np.zeros_like(current)
        positions=np.flatnonzero(active)
        dp[positions]=delta_norm/norms[active]
        alpha,_halvings=_commit_detf_guarded_update(
            state,grid,d,residual_fn,mask,ids,current,dp,settings,
            cycle=cycle,halfstep_iteration=int(it),Tn=Tn,weights=w,
            # Guard the same original equation solved by A*delta=-projected.
            # This corrects the residual mismatch; damping limits and the
            # accepted half-step algorithm are unchanged.
            measure_multipliers=None,
            current_projected_relative=equation_relative,
            projected_scale=escale,
        )
        if int(_halvings)>=deep_detf_halvings:
            deep_detf_damping_streak+=1
        else:
            deep_detf_damping_streak=0
        if deep_detf_damping_streak>=3:
            fprog.emit(
                "mvi_step_subdivision_requested",
                stage="MVI_HALFSTEP",
                halfstep=d,cycle=cycle,
                halfstep_iteration=int(it),
                correction_halvings=int(_halvings),
                consecutive_deep_detf_damping=int(deep_detf_damping_streak),
                reason="DET_F_STEP_TOO_LARGE",
                action="REJECT_LOAD_INCREMENT_AND_SUBDIVIDE",
            )
            raise GeneralMVIInternalError(
                f"DET_F_STEP_TOO_LARGE / {d}_HALFSTEP / "
                f"deep_damping_halvings={int(_halvings)} / "
                f"consecutive={int(deep_detf_damping_streak)}"
            )
        last_corr=float(np.linalg.norm(alpha*dp))

    equation_inf=projected_inf_for_direction(
        state,grid,bc,d,residual_fn,w,measure_multipliers=None
    )
    equation_relative=_projected_relative(equation_inf,escale)
    if equation_relative<=settings.projected_tolerance:
        return GeneralHalfstepRecord(
            d,settings.max_halfstep_iterations,equation_inf,last_corr,
            "CONVERGED_REDUCED_MVI",equation_relative,escale
        )
    v4_inf=projected_inf_for_direction(
        state,grid,bc,d,residual_fn,w,
        measure_multipliers=measure_multipliers
    )
    v4_relative=_projected_relative(v4_inf,pscale)
    raise GeneralHalfstepDidNotConverge(
        f"MAX_HALFSTEP_ITERATIONS / {d}_HALFSTEP / "
        f"projected_inf={equation_inf:.17g} / "
        f"projected_relative={equation_relative:.17g} / "
        f"projected_scale={escale:.17g} / "
        f"v4_measure_projected_inf={v4_inf:.17g} / "
        f"v4_measure_projected_relative={v4_relative:.17g}"
    )


def solve_internal_mvi(
    initial_state,grid,bc,residual_fn,settings,*,weights=None,
    measure_multipliers=None,quadrature
):
    bc.check_rigid_mode_requirement()
    state=initial_state.copy()
    state.validate_grid(grid)
    if state.m!=bc.m:
        raise ValueError("state/BC m mismatch")
    g9.enforce_homogeneous_geometric_x_profiles(state,grid,bc)

    w=g8.residual_projection_weights(state,grid) if weights is None else np.asarray(weights,float)
    history=[]
    final_p=float("inf")
    final_p_rel=float("inf")
    final_inc=float("inf")
    final_px_rel=float("inf")
    final_py_rel=float("inf")
    final_full_rel=None
    full_tol=float(
        settings.projected_tolerance
        if settings.full_residual_tolerance is None
        else settings.full_residual_tolerance
    )
    # X is inactive at the physical zero state because Y=0.  A common
    # Y->X scale therefore makes the X halfstep inherit an unrelated Y
    # magnitude.  Freeze one reference per direction when that direction first
    # becomes active; the equations and their tangent are unchanged.
    projected_scales={
        "Y":_fixed_projected_scale(
            state,grid,bc,residual_fn,w,measure_multipliers=measure_multipliers,
            directions=("Y",)
        ),
        "X":None,
    }
    equation_scales={
        "Y":_fixed_projected_scale(
            state,grid,bc,residual_fn,w,measure_multipliers=None,
            directions=("Y",)
        ),
        "X":None,
    }

    def _direction_scales(direction):
        d=str(direction).upper()
        if projected_scales[d] is None:
            projected_scales[d]=_fixed_projected_scale(
                state,grid,bc,residual_fn,w,
                measure_multipliers=measure_multipliers,directions=(d,)
            )
        if equation_scales[d] is None:
            equation_scales[d]=_fixed_projected_scale(
                state,grid,bc,residual_fn,w,
                measure_multipliers=None,directions=(d,)
            )
        return float(projected_scales[d]),float(equation_scales[d])

    algorithm=str(getattr(settings,"algorithm","PLATE_STYLE_ONE_CORRECTION")).upper()

    try:
        _full0=_full_residual_diagnostics(
            state,grid,residual_fn,measure_multipliers=measure_multipliers
        )
        fprog.emit("mvi_full_residual",stage="MVI",phase="INITIAL",cycle=0,**_full0)
    except Exception as exc:
        fprog.emit("mvi_full_residual",stage="MVI",phase="INITIAL",cycle=0,diagnostic_error=f"{type(exc).__name__}: {exc}")
    for cyc in range(1,settings.max_xy_cycles+1):
        before=state.copy()

        if algorithm=="STRICT_HALFSTEP":
            # Establish each fixed scale in the same exact QR gauge that the
            # corresponding strict halfstep will use.
            _orthonormalise_frozen_partner(state,grid,bc,"X")
            pscale_y,escale_y=_direction_scales("Y")
            hy=solve_strict_halfstep(
                state,grid,bc,"Y",residual_fn,settings,weights=w,
                measure_multipliers=measure_multipliers,quadrature=quadrature,
                cycle=cyc,projected_scale=pscale_y,equation_scale=escale_y,
            )
            fprog.emit("mvi_halfstep", halfstep="Y", half_iterations=int(hy.iterations), projected_inf=float(hy.projected_inf), projected_relative=float(hy.projected_relative), projected_scale=float(hy.projected_scale), correction_norm=float(hy.correction_norm), stage="MVI", algorithm=algorithm)
            _orthonormalise_frozen_partner(state,grid,bc,"Y")
            pscale_x,escale_x=_direction_scales("X")
            hx=solve_strict_halfstep(
                state,grid,bc,"X",residual_fn,settings,weights=w,
                measure_multipliers=measure_multipliers,quadrature=quadrature,
                cycle=cyc,projected_scale=pscale_x,equation_scale=escale_x,
            )
            fprog.emit("mvi_halfstep", halfstep="X", half_iterations=int(hx.iterations), projected_inf=float(hx.projected_inf), projected_relative=float(hx.projected_relative), projected_scale=float(hx.projected_scale), correction_norm=float(hx.correction_norm), stage="MVI", algorithm=algorithm)
        else:
            pscale_y,_escale_y=_direction_scales("Y")
            state,hy=_plate_style_single_halfcycle(
                state,grid,bc,"Y",residual_fn,settings,weights=w,
                measure_multipliers=measure_multipliers,
                projected_scale=pscale_y,quadrature=quadrature,cycle=cyc,
            )
            fprog.emit("mvi_halfstep", halfstep="Y", half_iterations=1, projected_inf=float(hy.projected_inf), projected_relative=float(hy.projected_relative), projected_scale=float(hy.projected_scale), correction_norm=float(hy.correction_norm), stage="MVI", algorithm=algorithm)
            pscale_x,_escale_x=_direction_scales("X")
            state,hx=_plate_style_single_halfcycle(
                state,grid,bc,"X",residual_fn,settings,weights=w,
                measure_multipliers=measure_multipliers,
                projected_scale=pscale_x,quadrature=quadrature,cycle=cyc,
            )
            fprog.emit("mvi_halfstep", halfstep="X", half_iterations=1, projected_inf=float(hx.projected_inf), projected_relative=float(hx.projected_relative), projected_scale=float(hx.projected_scale), correction_norm=float(hx.correction_norm), stage="MVI", algorithm=algorithm)

        g9.enforce_homogeneous_geometric_x_profiles(state,grid,bc)

        # One physical residual at the final state of the XY cycle is enough
        # for both orthogonal projected checks.  The old code evaluated the
        # same residual twice (once for Y and once for X).
        R_cycle=_residual_vector(residual_fn,state,grid)
        full_diag=_full_residual_diagnostics(
            state,grid,residual_fn,measure_multipliers=measure_multipliers
        )
        fprog.emit("mvi_full_residual",stage="MVI",phase="AFTER_XY",cycle=int(cyc),**full_diag)
        # The strict halfsteps solve the original reduced equations
        # T^T W R=0.  Use those same equations for the cycle acceptance gate.
        # V-4 row-scaled projections are diagnostics only; they are not an
        # additional convergence equation and may remain nonzero by design.
        py=_projected_inf_from_residual(
            state,grid,bc,"Y",R_cycle,w,measure_multipliers=None
        )
        px=_projected_inf_from_residual(
            state,grid,bc,"X",R_cycle,w,measure_multipliers=None
        )
        py_rel=_projected_relative(py,equation_scales["Y"])
        px_rel=_projected_relative(px,equation_scales["X"])
        final_py_rel=float(py_rel)
        final_px_rel=float(px_rel)
        py_v4=_projected_inf_from_residual(
            state,grid,bc,"Y",R_cycle,w,measure_multipliers=measure_multipliers
        )
        px_v4=_projected_inf_from_residual(
            state,grid,bc,"X",R_cycle,w,measure_multipliers=measure_multipliers
        )
        py_v4_rel=_projected_relative(py_v4,projected_scales["Y"])
        px_v4_rel=_projected_relative(px_v4,projected_scales["X"])
        final_p=max(px,py)
        final_p_rel=max(py_rel,px_rel)
        report_scale=max(float(equation_scales["Y"]),float(equation_scales["X"]))
        final_inc=_relative_field_change(before,state,grid)
        full_rel=full_diag["full_residual_relative_to_load"]
        final_full_rel=(None if full_rel is None else float(full_rel))
        full_ok=bool(
            final_full_rel is not None
            and np.isfinite(final_full_rel)
            and final_full_rel<=full_tol
        )

        history.append({
            "cycle":cyc,
            "Y_half_iterations":hy.iterations,
            "X_half_iterations":hx.iterations,
            "Y_projected_inf_after_cycle":py,
            "X_projected_inf_after_cycle":px,
            "Y_projected_relative_after_cycle":py_rel,
            "X_projected_relative_after_cycle":px_rel,
            "Y_v4_projected_inf_after_cycle":py_v4,
            "X_v4_projected_inf_after_cycle":px_v4,
            "Y_v4_projected_relative_after_cycle":py_v4_rel,
            "X_v4_projected_relative_after_cycle":px_v4_rel,
            "projected_inf":final_p,
            "projected_relative":final_p_rel,
            "projected_scale":report_scale,
            "Y_projected_scale":equation_scales["Y"],
            "X_projected_scale":equation_scales["X"],
            "Y_v4_projected_scale":projected_scales["Y"],
            "X_v4_projected_scale":projected_scales["X"],
            "field_increment_rel":final_inc,
            "full_residual_inf_scaled":full_diag["full_residual_inf_scaled"],
            "interior_residual_inf_scaled":full_diag["interior_residual_inf_scaled"],
            "boundary_residual_inf_scaled":full_diag["boundary_residual_inf_scaled"],
            "full_residual_relative_to_load":full_diag["full_residual_relative_to_load"],
            "interior_residual_relative_to_load":full_diag["interior_residual_relative_to_load"],
            "boundary_residual_relative_to_load":full_diag["boundary_residual_relative_to_load"],
            "full_residual_load_scale":full_diag["full_residual_load_scale"],
            "full_residual_tolerance":full_tol,
            "full_equilibrium_satisfied":full_ok,
            "solution_status":("CONVERGED_FULL" if full_ok else "CONVERGED_PROJECTED"),
        })
        try:
            _q=g8.reconstruct_fields(state); _w=_q[2,0]; _js,_ix=np.unravel_index(int(np.argmax(np.abs(_w))),_w.shape)
            _wmax=float(np.max(np.abs(_w))); _wval=float(_w[_js,_ix])
        except Exception:
            _wmax=None; _wval=None; _js=None; _ix=None
        fprog.emit("mvi_cycle", cycle=int(cyc), projected_inf=float(final_p), projected_relative=float(final_p_rel), projected_scale=float(report_scale), field_increment_rel=float(final_inc),
                   Y_projected_relative=float(py_rel), X_projected_relative=float(px_rel),
                   Y_projected_scale=float(equation_scales["Y"]), X_projected_scale=float(equation_scales["X"]),
                   Y_v4_projected_relative=float(py_v4_rel), X_v4_projected_relative=float(px_v4_rel),
                   Y_v4_projected_scale=float(projected_scales["Y"]), X_v4_projected_scale=float(projected_scales["X"]),
                   Y_half_iterations=int(hy.iterations), X_half_iterations=int(hx.iterations), stage="MVI",
                   full_residual_inf_scaled=float(full_diag["full_residual_inf_scaled"]), interior_residual_inf_scaled=float(full_diag["interior_residual_inf_scaled"]), boundary_residual_inf_scaled=float(full_diag["boundary_residual_inf_scaled"]), full_residual_relative_to_load=full_diag["full_residual_relative_to_load"], interior_residual_relative_to_load=full_diag["interior_residual_relative_to_load"], boundary_residual_relative_to_load=full_diag["boundary_residual_relative_to_load"], full_residual_load_scale=full_diag["full_residual_load_scale"],
                   full_residual_tolerance=float(full_tol),full_equilibrium_satisfied=bool(full_ok),
                   solution_status=("CONVERGED_FULL" if full_ok else "CONVERGED_PROJECTED"),
                   w_abs_max=_wmax,w_at_abs_max=_wval,wmax_is=_js,wmax_ix=_ix)

        if final_p_rel<=settings.projected_tolerance and final_inc<=settings.increment_tolerance:
            status="CONVERGED_FULL" if full_ok else "CONVERGED_PROJECTED"
            return GeneralInternalMVIResult(
                state,True,status,cyc,history,
                final_p,final_inc,final_p_rel,report_scale,
                final_px_rel,final_py_rel,final_full_rel,full_ok,
                not full_ok,full_tol
            )

    return GeneralInternalMVIResult(
        state,False,"MAX_XY_CYCLES",settings.max_xy_cycles,history,
        final_p,final_inc,final_p_rel,report_scale,
        final_px_rel,final_py_rel,final_full_rel,False,False,full_tol
    )


def make_physical_residual_function(
    ops,*,K0,GB,quadrature,basis,generalized_load,bc
):
    p=np.asarray(generalized_load,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if basis.m!=bc.m:
        raise ValueError("basis/BC m mismatch")

    def residual(state):
        raw,resultants,(q,_qx,_qs)=g8.raw_strong_residual_from_profiles(
            state,ops,K0=K0,GB=GB,quadrature=quadrature,
            basis=basis,generalized_load=p
        )
        return g9.assemble_strong_residual_with_end_bcs(
            raw,q,resultants,bc
        ).reshape(-1)
    def prepare_exact_tangent_action(state):
        # Exact partial derivative dR/dq at the current state with K0/GB frozen.
        # This is exactly the derivative required by one Birger outer iteration.
        q,qx,qs=g8.profile_fields_and_first_derivatives(state,ops)
        ctx=g7.prepare_general_elastic_tangent_context(
            q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
            quadrature=quadrature,basis=basis
        )
        zload=np.zeros_like(q)
        expected=q.shape

        def action(dqvec):
            dq=np.asarray(dqvec,float).reshape(expected)
            dqx,dqs=_surface_first_derivatives_full_direction(dq,ops,basis)
            dres=g7.general_thickness_resultants_tangent_from_context(
                ctx,dq,dqx,dqs
            )
            draw=g7.general_residuals(
                dres,Dx=ops.Dx,Ds=ops.Ds,generalized_load=zload
            )
            return _assemble_exact_tangent_with_end_bcs(
                draw,dq,dres,bc
            ).reshape(-1)
        return action

    residual.mvi_prepare_exact_tangent_action=prepare_exact_tangent_action
    residual.mvi_exact_tangent_ops=ops
    residual.mvi_exact_tangent_profile_coloring=True
    residual.mvi_tangent_mode="EXACT_GREEN_LAGRANGE_FROZEN_GB"
    # Reference for the stopping measure only.  The physical reduced equations
    # and their tangent continue to use the unmodified residual above.
    residual.mvi_projected_reference_residual=p.reshape(-1).copy()
    return residual
