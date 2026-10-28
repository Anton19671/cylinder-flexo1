"""Report-only applicability monitor for FSDT/Reddy linear directors.

At the midsurface, the stored director kinematics is F_z = e_r + theta_x e_x
+ theta_s e_theta, hence a non-objective E_zz = |theta|**2/2 for any nonzero
slope. The 0.1-rad recommendation is advisory, NOT a validated error bound.
Do not alter the G3D/Gz functional, mass, fixed-rank MVI or load history.
"""
from __future__ import annotations
import numpy as np


def director_rotation_diagnostic(theory, mechanical=None, *, advisory_limit_rad=0.1):
    theory=str(theory).upper()
    if theory not in ('KL','FSDT','REDDY','CUF_TE1','CUF_TE2','CUF_TE3','CUF_TE4'):
        raise ValueError('unknown director theory')
    bad=theory in ('FSDT','REDDY')
    info={
        'director_type':('LINEARIZED_NONUNIT' if bad else 'NOT_APPLICABLE'),
        'finite_rotation_objectivity_verified':False,
        'moderate_rotation_advisory_limit_rad':float(advisory_limit_rad) if bad else None,
        'theta_parameter_max':None,
        'spurious_Ezz_midplane_max':None,
        'advisory_limit_exceeded':None,
        'quantitative_finite_rotation_validated':False,
        'warning':('FSDT/Reddy linear directors are nonunit; finite rotations create '
                   'spurious Ezz=0.5*(theta_x^2+theta_s^2). Use only for '
                   'moderate rotations with independent validation. 0.1 rad '
                   'is a warning threshold, not a proved accuracy bound.' if bad else
                   'This diagnostic does not certify finite-rotation accuracy.'),
    }
    if not bad or mechanical is None:
        return info
    q=np.asarray(mechanical,float)
    if q.ndim!=3 or q.shape[0]!=5 or not np.isfinite(q).all():
        raise ValueError('FSDT/Reddy rotation diagnostic requires finite (5,nx,ns) field')
    theta2=q[3]*q[3]+q[4]*q[4]
    theta_max=float(np.sqrt(np.max(theta2)))
    info['theta_parameter_max']=theta_max
    info['spurious_Ezz_midplane_max']=float(.5*np.max(theta2))
    info['advisory_limit_exceeded']=bool(theta_max>advisory_limit_rad)
    return info


__all__=('director_rotation_diagnostic',)
