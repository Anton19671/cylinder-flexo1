"""Explicit preflight until an inhomogeneous additive lift is implemented.

The fixed-rank MVI state is ONLY sum_r X_r(x)Y_r(s); arbitrary
inhomogeneous essential traces are not guaranteed representable. Reject
before the solver, without a rank change or a fallback to FDM.
"""
from __future__ import annotations
import numpy as np
from cylinder_variational_boundary import _fixed_components, _vector


def require_homogeneous_mvi_traces(spec, *, theory, n_mechanical):
    if not isinstance(spec,dict):
        raise ValueError('explicit variational BC object required')
    mechanical=spec.get('mechanical') or {}
    electrical=spec.get('electrical') or {}
    for side in ('left','right'):
        data=mechanical.get(side) or {}
        support=str(data.get('support','FREE')).strip().upper()
        fixed=_fixed_components(support,theory,n_mechanical,data)
        if not fixed:continue
        prescribed=_vector(data.get('prescribed'),n_mechanical,side+' prescribed')
        if np.any(prescribed[list(fixed)]!=0):
            raise ValueError('MVI_NONZERO_ESSENTIAL_TRACE_UNSUPPORTED: '
                             +side+' mechanical displacement; arbitrary prescribed '
                             'values require an additive lift, not implemented')
    for side in ('left','right','inner','outer'):
        data=electrical.get(side) or {}
        if str(data.get('mode','INSULATED')).strip().upper()!='POTENTIAL':continue
        value=float(data['value'])
        if not np.isfinite(value):
            raise ValueError(side+': finite electric potential required')
        if value!=0.:
            raise ValueError('MVI_NONZERO_ESSENTIAL_TRACE_UNSUPPORTED: '
                             +side+' electric potential; arbitrary prescribed '
                             'values require an additive lift, not implemented')
    return True


__all__=('require_homogeneous_mvi_traces',)
