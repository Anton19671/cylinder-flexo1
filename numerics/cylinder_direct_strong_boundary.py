"""Explicit physical axial end rows for independent strong cylinder FDM.

Diagnostic only. Differential order per component must be certified separately
from the principal symbol: 2, 4, or 6. This does NOT dispatch a solver.
"""
from dataclasses import dataclass
import numpy as np
from cylinder_direct_strong_fdm import DirectStrongGrid, DirectStrongBalance
from cylinder_mkr_field_jets import MKRFieldJets

@dataclass(frozen=True)
class StrongEndSystem:
    mechanical: np.ndarray
    electrical: np.ndarray | None
    axial_orders: tuple
    status: str = "STRONG_END_ROWS_NOT_SOLVER_CERTIFIED"


@dataclass(frozen=True)
class CUFParallelEndConditions:
    """Source §§30--33: immobile sections, microfree higher traces.

    Axial half-orders are explicit input, not inferred from the number of
    jets. This distinction matters when a G3D model degenerates to Gz or
    classical elasticity. Electrical insulation leaves ONE global gauge.
    """
    te_order: int
    axial_orders: tuple
    potential_order: int | None = None

    def __post_init__(self):
        if type(self.te_order) is not int or not 1 <= self.te_order <= 4:
            raise ValueError("CUF TE order must be 1..4")
        orders = tuple(self.axial_orders)
        if (len(orders) != 3*(self.te_order+1) or
                any(type(m) is not int or m not in (1, 2) for m in orders)):
            raise ValueError("explicit CUF axial half-order 1 or 2 per field required")
        if self.potential_order is not None and (
                type(self.potential_order) is not int or
                not 0 <= self.potential_order <= 4):
            raise ValueError("potential order must be None (OFF) or 0..4")
        object.__setattr__(self, "axial_orders", orders)

    @property
    def mechanical_ends(self):
        # Fresh dictionaries: changing one end/field cannot change another.
        return {side: [[{"kind": "ESSENTIAL", "value": 0.}] +
                       [{"kind": "NATURAL", "value": 0.} for _ in range(m-1)]
                       for m in self.axial_orders]
                for side in ("left", "right")}

    @property
    def electrical_ends(self):
        if self.potential_order is None:
            return None
        return {side: [{"kind": "NATURAL", "value": 0.}
                       for _ in range(self.potential_order+1)]
                for side in ("left", "right")}

def _condition(value, ns, label):
    if not isinstance(value, dict) or set(value) != {"kind", "value"} or value["kind"] not in ("ESSENTIAL", "NATURAL"):
        raise ValueError(label + ": explicit ESSENTIAL/NATURAL kind and value required")
    raw=value["value"]
    if isinstance(raw, (bool,np.bool_)):
        raise ValueError(label + ": boolean value")
    v=np.asarray(raw,float)
    if v.ndim==0:v=np.full(ns,float(v))
    if v.shape!=(ns,) or not np.isfinite(v).all():
        raise ValueError(label + ": nonfinite or incorrect shape")
    return value["kind"],v

def stationary_essential_scales(grid, shape, ends, base_scale, *,
                                scaled_absolute_tolerance, electric=False,
                                absolute_trace_tolerance=1.e-9):
    """Cap ONLY essential end-row scales, in their own physical units.

    A force/charge scale applied to a prescribed displacement, normal
    derivative or voltage row could otherwise falsely accept a support
    violation. Natural rows and bulk equations retain the original scale.
    This uses the same row positions as assemble_physical_end_rows.
    """
    from cylinder_direct_strong_newton import _positive_scale
    scale=_positive_scale(base_scale,shape,"stationary strong equation").copy()
    if (len(shape)!=3 or tuple(shape[1:])!=(grid.nx,grid.ns)
            or not isinstance(ends,dict) or set(ends)!={"left","right"}):
        raise ValueError("STATIONARY_STRONG_END_SCALE_INVALID")
    tol=float(absolute_trace_tolerance)
    relative=float(scaled_absolute_tolerance)
    if not (np.isfinite(tol) and tol>0. and np.isfinite(relative) and relative>0.):
        raise ValueError("STATIONARY_STRONG_TRACE_TOLERANCE_INVALID")
    cap=tol/relative
    if not np.isfinite(cap) or cap<=0.:
        raise ValueError("STATIONARY_STRONG_TRACE_SCALE_UNREPRESENTABLE")
    for side in ("left","right"):
        data=ends[side]
        if not isinstance(data,(tuple,list)) or len(data)!=shape[0]:
            raise ValueError("STATIONARY_STRONG_END_FIELD_COUNT_INVALID")
        for c,items in enumerate(data):
            entries=(items,) if electric else items
            if not isinstance(entries,(tuple,list)) or not 1<=len(entries)<=3:
                raise ValueError("STATIONARY_STRONG_END_TRACE_COUNT_INVALID")
            for order,item in enumerate(entries):
                kind,_=_condition(item,grid.ns,"stationary strong "+side+" trace")
                if kind=="ESSENTIAL":
                    if grid.uses_sat_end_closure and not electric and order:
                        raise ValueError("MATCHED_SAT_HIGHER_ESSENTIAL_TRACE_UNSUPPORTED")
                    node=order if side=="left" else grid.nx-1-order
                    scale[c,node,:]=np.minimum(scale[c,node,:],cap)
    return scale


def native_kinematic_normal_trace(field, grid, component, normal_order, side,
                                  *, prepared_jet=None):
    """Outward-normal essential trace from the SAME MKR kinematic stencil.

    The strong coefficient divergence is deliberately a different
    approximation. Reusing that divergence operator for an essential
    slope/curvature would constrain a different field than the one in
    the actual constitutive strains (and can pass a false clamp).
    """
    if side not in ("left","right"):
        raise ValueError("invalid physical axial side")
    if (type(normal_order) is not int or normal_order not in (0,1,2)):
        raise ValueError("essential normal trace order must be 0, 1 or 2")
    if not isinstance(grid,DirectStrongGrid):
        raise TypeError("direct strong grid required")
    values=np.asarray(field,float)
    if (values.ndim!=3 or values.shape[1:]!=(grid.nx,grid.ns)
            or not np.isfinite(values).all()
            or type(component) is not int
            or not 0<=component<values.shape[0]):
        raise ValueError("invalid physical mechanical trace field/component")
    if prepared_jet is None:
        jet=MKRFieldJets(grid).derivative(values,normal_order,0)
    else:
        jet=np.asarray(prepared_jet,float)
        if jet.shape!=values.shape or not np.isfinite(jet).all():
            raise ValueError("invalid prepared physical trace jet")
    physical_end=0 if side=="left" else -1
    outward_sign=-1. if side=="left" and normal_order%2 else 1.
    return outward_sign*jet[component,physical_end,:]


def _matched_immobile_sat_mechanical(balance,q,grid,axial_orders,mechanical_ends):
    """Keep all interior strong equations; lift higher natural end work.

    Supported physical program: fixed q at both sections and natural
    conditions for ALL higher normal traces. Explicit entries prescribe
    their values; omitted higher trace actions have zero target. In
    particular, coupled nonlinear KL is not artificially assigned a
    pointwise fieldwise PDE order by this SAT construction.
    """
    from cylinder_matched_strong_sat import add_natural_trace_sat
    highest=balance.highest_axial_jet
    shape=(2,len(q),grid.ns)
    targets=[np.zeros(shape) for k in range(1,highest)]
    essential=np.empty(shape)
    for side_index,side in enumerate(("left","right")):
        conditions=mechanical_ends[side]
        if not isinstance(conditions,(tuple,list)) or len(conditions)!=len(q):
            raise ValueError("MATCHED_SAT_END_FIELD_COUNT_INVALID")
        for c,m in enumerate(axial_orders):
            if not isinstance(conditions[c],(tuple,list)) or len(conditions[c])!=m:
                raise ValueError("MATCHED_SAT_END_TRACE_COUNT_INVALID")
            for k,item in enumerate(conditions[c]):
                kind,value=_condition(item,grid.ns,"matched SAT "+side+" trace "+str(k))
                if k==0:
                    if kind!="ESSENTIAL":
                        raise ValueError("MATCHED_SAT_REQUIRES_FIXED_DISPLACEMENT_ENDS")
                    essential[side_index,c]=value
                else:
                    if kind!="NATURAL":
                        raise ValueError("MATCHED_SAT_HIGHER_ESSENTIAL_TRACE_UNSUPPORTED")
                    targets[k-1][side_index,c]=value
    traces=[(np.array([-1.,1.])[:,None]**k)*
            grid.matched_axial.kinematic[k][[0,-1]]/grid.dx**k
            for k in range(1,highest)]
    actions=(balance.mechanical_end_slope,
             balance.mechanical_end_curvature)[:highest-1]
    result=add_natural_trace_sat(balance.mechanical_bulk,grid.axial_weights,
                                 traces,actions,targets)
    result[:,0,:]=q[:,0,:]-essential[0]
    result[:,-1,:]=q[:,-1,:]-essential[1]
    return result


def assemble_physical_end_rows(balance, mechanical, grid, *, axial_orders,
                               mechanical_ends, electrical=None,
                               electrical_ends=None, gauge_closed_elsewhere=False):
    """Make one physical equation per node; no automatic edge or gauge assumptions.

    Each end supplies one list per component; list length m for axial
    equation order 2*m. Entries correspond to q, d_n q, d_nn q and their
    conjugate natural forces. Electric modes each need a potential or flux.
    """
    if not isinstance(grid,DirectStrongGrid) or not isinstance(balance,DirectStrongBalance):
        raise TypeError("strong balance/grid required")
    bulk=np.asarray(balance.mechanical_bulk,float)
    q=np.asarray(mechanical,float)
    if (q.shape!=bulk.shape or q.ndim!=3 or q.shape[1:]!=(grid.nx,grid.ns)
            or not np.isfinite(q).all()):
        raise ValueError("mechanical field shape/values invalid")
    if (not isinstance(axial_orders,(list,tuple)) or len(axial_orders)!=len(q)
            or any(type(v)!=int or not 1<=v<=3 or v>balance.highest_axial_jet for v in axial_orders)):
        raise ValueError("certified axial orders 1,2,3 per component required")
    if not isinstance(mechanical_ends,dict) or set(mechanical_ends)!={"left","right"}:
        raise ValueError("left and right mechanical end data required")
    if grid.uses_sat_end_closure:
        if gauge_closed_elsewhere:
            raise ValueError("MATCHED_SAT_FLOATING_GAUGE_UNSUPPORTED")
        result=_matched_immobile_sat_mechanical(
            balance,q,grid,axial_orders,mechanical_ends)
    else:
        result=bulk.copy()
    # The legacy pointwise closure alone replaces additional interior rows.
    kinematic=MKRFieldJets(grid)
    essential_jets={0:q}
    for side_index, side in enumerate(() if grid.uses_sat_end_closure else ("left","right")):
        conditions=mechanical_ends[side]
        if not isinstance(conditions,(tuple,list)) or len(conditions)!=len(q):
            raise ValueError(side+": condition list per component required")
        for i,m in enumerate(axial_orders):
            if not isinstance(conditions[i],(tuple,list)) or len(conditions[i])!=m:
                raise ValueError(side+": exactly m physical end rows required")
            for k,item in enumerate(conditions[i]):
                kind,value=_condition(item,grid.ns,f"{side} component {i} trace {k}")
                node=k if side_index==0 else grid.nx-1-k
                if kind=="ESSENTIAL":
                    if k not in essential_jets:
                        essential_jets[k]=kinematic.derivative(q,k,0)
                    trace=native_kinematic_normal_trace(
                        q,grid,i,k,side,prepared_jet=essential_jets[k])
                else:
                    actions=(balance.mechanical_end_displacement,
                             balance.mechanical_end_slope,
                             balance.mechanical_end_curvature)
                    if actions[k] is None:
                        raise ValueError("missing full constitutive natural action")
                    trace=actions[k][side_index,i,:]
                result[i,node,:]=trace-value
    electric=None
    if balance.electrical_bulk is None:
        if electrical is not None or electrical_ends is not None:
            raise ValueError("electrical field disabled")
    else:
        pde=np.asarray(balance.electrical_bulk,float)
        p=np.asarray(electrical,float)
        if p.shape!=pde.shape or not np.isfinite(p).all():
            raise ValueError("electric potential shape invalid")
        if not isinstance(electrical_ends,dict) or set(electrical_ends)!={"left","right"}:
            raise ValueError("explicit electric end data required")
        electric=pde.copy()
        constant_kinds=[]
        for side_index,side in enumerate(("left","right")):
            items=electrical_ends[side]
            if not isinstance(items,(tuple,list)) or len(items)!=len(p):
                raise ValueError("each electric mode requires an end condition")
            node=0 if side_index==0 else -1
            for i,item in enumerate(items):
                kind,value=_condition(item,grid.ns,side+" electric mode")
                if grid.uses_sat_end_closure and kind!="ESSENTIAL":
                    raise ValueError("MATCHED_SAT_REQUIRES_BOTH_ELECTRIC_ENDS_PRESCRIBED")
                if i==0:constant_kinds.append(kind)
                trace=(p[i,node] if kind=="ESSENTIAL" else
                       balance.electrical_end_potential[side_index,i])
                electric[i,node,:]=trace-value
        if gauge_closed_elsewhere and "ESSENTIAL" in constant_kinds:
            raise ValueError("constant electric mode already has a prescribed potential")
        if constant_kinds==["NATURAL","NATURAL"] and not gauge_closed_elsewhere:
            raise ValueError("constant electric mode requires electrode or gauge closure")
    if grid.uses_sat_end_closure:
        return StrongEndSystem(result,electric,tuple(axial_orders),
                               "MATCHED_SAT_NATURAL_WORK_RESEARCH_NOT_BOUNDARY_CERTIFIED")
    return StrongEndSystem(result,electric,tuple(axial_orders))

__all__=("StrongEndSystem","CUFParallelEndConditions","assemble_physical_end_rows",
         "native_kinematic_normal_trace")
