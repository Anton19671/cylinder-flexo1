"""Explicit physical axial end rows for independent strong cylinder FDM.

Diagnostic only. Differential order per component must be certified separately
from the principal symbol: 2, 4, or 6. This does NOT dispatch a solver.
"""
from dataclasses import dataclass
import numpy as np
from cylinder_direct_strong_fdm import DirectStrongGrid, DirectStrongBalance, direct_coefficient_derivative

@dataclass(frozen=True)
class StrongEndSystem:
    mechanical: np.ndarray
    electrical: np.ndarray | None
    axial_orders: tuple
    status: str = "STRONG_END_ROWS_NOT_SOLVER_CERTIFIED"

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
    result=bulk.copy()
    for side_index, side in enumerate(("left","right")):
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
                    field=q[i] if k==0 else direct_coefficient_derivative(q[i],k,0,grid)
                    # Evaluate every essential trace at the ACTUAL physical end.\n                    # The adjacent rows hold additional equations; they are\n                    # not sampling locations for d_n q or d_nn q.\n                    physical_end=0 if side_index==0 else -1\n                    trace=field[physical_end,:] * (-1 if k==1 and side_index==0 else 1)
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
                if i==0:constant_kinds.append(kind)
                trace=(p[i,node] if kind=="ESSENTIAL" else
                       balance.electrical_end_potential[side_index,i])
                electric[i,node,:]=trace-value
        if constant_kinds==["NATURAL","NATURAL"] and not gauge_closed_elsewhere:
            raise ValueError("constant electric mode requires electrode or gauge closure")
    return StrongEndSystem(result,electric,tuple(axial_orders))

__all__=("StrongEndSystem","assemble_physical_end_rows")
