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
            not isinstance(relative_tolerance,(int,float,np.integer,np.floating)) or
            not np.isfinite(relative_tolerance) or
            not 0<relative_tolerance<1):
        raise ValueError("MVI_ENDPOINT_FIT_TOLERANCE_INVALID")
    if type(max_iterations) is not int or max_iterations<1:
        raise ValueError("MVI_ENDPOINT_FIT_ITERATIONS_INVALID")
    for c,ids in enumerate(free):
        fixed=np.ones(surface.grid.nx,dtype=bool)
        fixed[ids]=False
        # Essential traces are homogeneous physical conditions, not a
        # relative fitting target. max(1, amplitude) silently admitted
        # O(1e-9 m) nonzero support displacement for micrometre fields.
        # Permit only a component-scaled floating-point roundoff floor;
        # never relax the boundary using the endpoint fitting tolerance.
        amplitude=float(np.max(np.abs(target[c]),initial=0.))
        trace_roundoff=64.*np.finfo(float).eps*max(
            amplitude,np.finfo(float).tiny)
        if np.max(np.abs(target[c,fixed]),initial=0.)>trace_roundoff:
            return MVIRankEndpointFit(None,False,
                                      "MVI_ENDPOINT_ESSENTIAL_TRACE_CONFLICT",
                                      float("inf"),0)

    # The mechanical and electric modes have different physical units.
    # A global norm across all of them could conceal a poorly reconstructed
    # small-amplitude component behind a large displacement or voltage.
    # Use one dimensionless reference per component from the requested
    # field or the seed; any zero target/seed mode is already exact.
    seed_field=surface.reconstruct(seed)
    scales=np.maximum(
        np.linalg.norm(target.reshape(surface.n_fields,-1),axis=1),
        np.linalg.norm(seed_field.reshape(surface.n_fields,-1),axis=1))
    scales=np.maximum(scales,1.e-30)

    def component_errors(factors):
        difference=surface.reconstruct(factors)-target
        return (np.linalg.norm(difference.reshape(surface.n_fields,-1),
                               axis=1)/scales)

    first_error=float(np.max(component_errors(seed)))
    if first_error<=relative_tolerance:
        return MVIRankEndpointFit(seed,True,"MVI_ENDPOINT_EXACT_RANK",
                                  first_error,0)

    x=seed.x.copy()
    y=seed.y.copy()
    for iteration in range(1,max_iterations+1):
        for c,ids in enumerate(free):
            # A rank-R representation does not require ALL R factors to
            # be nonzero. A component that already matches (especially
            # identically zero electric potential) must not be forced
            # through a singular Gram factorization.
            component=np.einsum("ri,rj->ij",x[c],y[c],
                                optimize=True)
            difference=float(np.linalg.norm(component-target[c])/scales[c])
            if difference<=relative_tolerance:
                continue
            B=y[c]
            T=target[c,ids,:]
            if not np.any(T):
                # The zero field has a valid representation at every R;
                # preserve the angular partner and zero its axial factor.
                x[c]=0.
                continue
            # Alternating linear Gram equations only; no SVD, no rank
            # truncation, no pseudo-inverse or method substitution.
            try:
                gram=B@B.T
                if not np.isfinite(gram).all() or np.linalg.cond(gram,1)>1.e12:
                    return MVIRankEndpointFit(
                        None,False,"MVI_ENDPOINT_FACTOR_SINGULAR",
                        float("inf"),iteration)
                Anew=np.linalg.solve(gram,(T@B.T).T)
                x[c]=0.
                x[c][:,ids]=Anew
                gram=Anew@Anew.T
                if not np.isfinite(gram).all() or np.linalg.cond(gram,1)>1.e12:
                    return MVIRankEndpointFit(
                        None,False,"MVI_ENDPOINT_FACTOR_SINGULAR",
                        float("inf"),iteration)
                y[c]=np.linalg.solve(gram,Anew@T)
            except np.linalg.LinAlgError:
                return MVIRankEndpointFit(
                    None,False,"MVI_ENDPOINT_FACTOR_SINGULAR",
                    float("inf"),iteration)
        candidate=CoupledMVIFactors(x,y)
        errors=component_errors(candidate)
        err=float(np.max(errors))
        if err<=relative_tolerance:
            return MVIRankEndpointFit(candidate,True,
                                     "MVI_ENDPOINT_EXACT_RANK",err,iteration)
    return MVIRankEndpointFit(None,False,
                              "MVI_RANK_R_ENDPOINT_NOT_REPRESENTABLE",
                              err,max_iterations)


__all__=("MVIRankEndpointFit","fit_mvi_endpoint_field")
