"""Fail-closed fixed-rank endpoint field reconstruction for coupled MVI.

An implicit Gauss collocation RK endpoint is a linear combination of
stage velocities and is not, in general, on the same rank-R X(x)Y(s)
manifold as its stages. Never silently truncate it via SVD or fabricate
a rank-R endpoint. This fitter uses alternating QR/Gram solves in the
independent MVI factors, preserves declared homogeneous essential traces,
and accepts ONLY if the reconstructed field matches the requested endpoint
to the explicitly supplied tolerance.

An unmet tolerance reports RANK_R_ENDPOINT_NOT_REPRESENTABLE. A singular
factor Gram matrix reports RANK_R_ENDPOINT_FACTOR_SINGULAR.
No physical field solve or numerical method fallback is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from independent_mvi.coupled_surface import (
    CoupledMVIFactors, NativeCoupledMVISurface,
)
from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram, _check_end_factors,
)


@dataclass(frozen=True)
class MVIRankEndpointFit:
    factors: CoupledMVIFactors | None
    accepted: bool
    status: str
    relative_error: float
    iterations: int


def fit_mvi_endpoint_field(surface: NativeCoupledMVISurface,
                           seed: CoupledMVIFactors,
                           field, program: CoupledMVIEndProgram, *,
                           relative_tolerance=1.e-9,
                           max_iterations=30):
    if not isinstance(surface,NativeCoupledMVISurface):
        raise TypeError("NativeCoupledMVISurface required")
    free=_check_end_factors(surface,seed,program)
    target=np.asarray(field,dtype=float)
    if target.shape!=surface.shape or not np.isfinite(target).all():
        raise ValueError("MVI_ENDPOINT_FIELD_SHAPE_INVALID")
    if (isinstance(relative_tolerance,(bool,np.bool_)) or
            not np.isfinite(relative_tolerance) or
            not 0<relative_tolerance<1):
        raise ValueError("MVI_ENDPOINT_FIT_TOLERANCE_INVALID")
    if type(max_iterations) is not int or max_iterations<1:
        raise ValueError("MVI_ENDPOINT_FIT_ITERATIONS_INVALID")
    for c,ids in enumerate(free):
        fixed=np.ones(surface.grid.nx,dtype=bool)
        fixed[ids]=False
        if np.max(np.abs(target[c,fixed]),initial=0.)>(
                relative_tolerance*max(1.,float(np.max(np.abs(target[c]))))):
            return MVIRankEndpointFit(None,False,
                                      "MVI_ENDPOINT_ESSENTIAL_TRACE_CONFLICT",
                                      float("inf"),0)

    def error(factors):
        actual=surface.reconstruct(factors)
        scale=max(1.e-30,float(np.linalg.norm(target)))
        return float(np.linalg.norm(actual-target)/scale)

    first_error=error(seed)
    if first_error<=relative_tolerance:
        return MVIRankEndpointFit(seed,True,"MVI_ENDPOINT_EXACT_RANK",
                                  first_error,0)

    x=seed.x.copy()
    y=seed.y.copy()
    r=seed.rank
    for iteration in range(1,max_iterations+1):
        for c,ids in enumerate(free):
            A=x[c][:,ids]
            B=y[c]
            T=target[c,ids,:]
            # Orthogonal transformations of the currently available
            # rank-R spans. No SVD, pseudoinverse or rank expansion.
            try:
                gram=B@B.T
                if np.linalg.cond(gram,1)>1.e12:
                    return MVIRankEndpointFit(
                        None,False,"MVI_ENDPOINT_FACTOR_SINGULAR",
                        float("inf"),iteration)
                Anew=np.linalg.solve(gram,(T@B.T).T)
                x[c]=0.
                x[c][:,ids]=Anew
                gram=Anew@Anew.T
                if np.linalg.cond(gram,1)>1.e12:
                    return MVIRankEndpointFit(
                        None,False,"MVI_ENDPOINT_FACTOR_SINGULAR",
                        float("inf"),iteration)
                y[c]=np.linalg.solve(gram,Anew@T)
            except np.linalg.LinAlgError:
                return MVIRankEndpointFit(
                    None,False,"MVI_ENDPOINT_FACTOR_SINGULAR",
                    float("inf"),iteration)
        candidate=CoupledMVIFactors(x,y)
        err=error(candidate)
        if err<=relative_tolerance:
            return MVIRankEndpointFit(candidate,True,
                                     "MVI_ENDPOINT_EXACT_RANK",err,iteration)
    return MVIRankEndpointFit(None,False,
                              "MVI_RANK_R_ENDPOINT_NOT_REPRESENTABLE",
                              err,max_iterations)


__all__=("MVIRankEndpointFit","fit_mvi_endpoint_field")
