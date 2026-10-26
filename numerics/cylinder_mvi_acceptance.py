"""Final acceptance certificate for fixed-rank coupled static/Newmark MVI.

No rank change, physical-field correction, SVD or regularization.
"""
from __future__ import annotations
import numpy as np
from cylinder_residual_norms import admissible_nodal_metrics


def certify_mvi_solution(*, x_profiles, y_profiles, lift_scaled, scales,
                         mechanical, electrical, full_basis, reduced_residual,
                         projected_residual_scaled, full_tolerance_scaled,
                         constraint_defect_scaled, solver_converged,
                         max_constraint_defect_scaled=1.e-9):
    x=np.asarray(x_profiles,float)
    y=np.asarray(y_profiles,float)
    lift=np.asarray(lift_scaled,float)
    s=np.asarray(scales,float)
    q=np.asarray(mechanical,float)
    p=None if electrical is None else np.asarray(electrical,float)
    if (x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=y.shape[:2]
            or not 1<=x.shape[1]<=8 or x.shape[1]>min(x.shape[2],y.shape[2])):
        raise ValueError("invalid fixed-rank MVI X/Y profiles")
    shape=(x.shape[0],x.shape[2],y.shape[2])
    if (lift.shape!=shape or s.shape!=shape or q.ndim!=3
            or q.shape[1:]!=shape[1:]
            or q.shape[0]+(0 if p is None else p.shape[0])!=shape[0]
            or (p is not None and (p.ndim!=3 or p.shape[1:]!=shape[1:]))
            or not all(np.isfinite(a).all() for a in (x,y,lift,s,q))
            or (p is not None and not np.isfinite(p).all())
            or np.any(s<=0)):
        raise ValueError("MVI profile/scaling/physical field mismatch or nonfinite value")
    for name,value in (("full_tolerance_scaled",full_tolerance_scaled),
                       ("constraint_defect_scaled",constraint_defect_scaled),
                       ("projected_residual_scaled",projected_residual_scaled),
                       ("max_constraint_defect_scaled",max_constraint_defect_scaled)):
        if isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or value<0:
            raise ValueError(name+" must be finite nonnegative")
    if not isinstance(solver_converged,(bool,np.bool_)):
        raise ValueError("solver_converged must be Boolean")
    b=np.asarray(full_basis,float)
    g=np.asarray(reduced_residual,float)
    if b.ndim!=2 or b.shape[0]!=int(np.prod(shape)) or b.shape[1]!=g.size:
        raise ValueError("full admissible basis/physical field size mismatch")
    metrics=admissible_nodal_metrics(b,g,q.size)
    represented=lift+np.einsum("crx,crs->cxs",x,y)
    actual=q if p is None else np.concatenate((q,p))
    scaled_actual=actual/s
    error=float(np.max(np.abs(represented-scaled_actual),initial=0))
    bound=256*np.finfo(float).eps*max(1.,
        float(np.max(np.abs(represented),initial=0)),
        float(np.max(np.abs(scaled_actual),initial=0)))
    shape_ok=error<=bound
    full_ok=metrics["full"]<=full_tolerance_scaled
    bc_ok=constraint_defect_scaled<=max_constraint_defect_scaled
    return {
        "accepted":bool(solver_converged and full_ok and bc_ok and shape_ok),
        "solver_converged":bool(solver_converged),"fixed_rank":int(x.shape[1]),
        "full_equilibrium_satisfied":bool(full_ok),
        "boundary_constraint_satisfied":bool(bc_ok),
        "profile_reconstruction_satisfied":bool(shape_ok),
        "projected_equilibrium_satisfied":bool(projected_residual_scaled<=full_tolerance_scaled),
        "full_admissible_residual_scaled":metrics["full"],
        "mechanical_residual_scaled":metrics["mechanical"],
        "electrical_residual_scaled":metrics["electrical"],
        "basis_coordinate_residual_scaled":metrics["basis_coordinates"],
        "projected_residual_scaled":float(projected_residual_scaled),
        "full_tolerance_scaled":float(full_tolerance_scaled),
        "constraint_defect_scaled":float(constraint_defect_scaled),
        "profile_reconstruction_error_scaled":error,
        "profile_reconstruction_tolerance_scaled":float(bound),
    }


__all__=("certify_mvi_solution",)
