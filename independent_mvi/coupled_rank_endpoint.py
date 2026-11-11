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
from scipy.linalg import qr as pivoted_qr

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


def _trace_roundoff_limit(component):
    """Only numerical roundoff is permissible on homogeneous fixed traces.

    Scale by this component's actual maximum magnitude, not an artificial
    unit displacement. This is independent of rank-fit acceptance tolerance.
    """
    amplitude=float(np.max(np.abs(component),initial=0.))
    return 64.*np.finfo(float).eps*max(amplitude,np.finfo(float).tiny)


def _target_qr_angular_partner(target_free, rank):
    """Restart a singular angular partner without SVD or reducing rank.

    Only a rank-R angular *seed* is constructed. Its reconstructed
    field is still checked against the full target before acceptance.
    Inactive modes remain allocated: no zero rank slot is discarded.
    """
    # Zero/dependent leading axial rows must not occupy the retained
    # angular span. Pivot columns before selecting the R partner vectors.
    q,_,_=pivoted_qr(target_free.T,mode="economic",pivoting=True,
                     check_finite=True)
    if q.shape[1]<rank or not np.isfinite(q).all():
        raise np.linalg.LinAlgError("MVI_ENDPOINT_QR_PARTNER_UNAVAILABLE")
    return q[:,:rank].T.copy()


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
        trace_roundoff=_trace_roundoff_limit(target[c])
        if np.max(np.abs(target[c,fixed]),initial=0.)>trace_roundoff:
            return MVIRankEndpointFit(None,False,
                                      "MVI_ENDPOINT_ESSENTIAL_TRACE_CONFLICT",
                                      float("inf"),0)

    # The mechanical and electric modes have different physical units.
    # A global norm across all of them could conceal a poorly reconstructed
    # small-amplitude component behind a large displacement or voltage.
    # Measure fidelity against the REQUESTED endpoint, never against a
    # much larger seed. Otherwise a large but admissible Gauss-stage seed
    # can hide a completely inaccurate small endpoint response.
    # The subnormal floor exists only for exactly zero target components:
    # those must reconstruct to zero within numerical roundoff.
    scales=np.linalg.norm(target.reshape(surface.n_fields,-1),axis=1)
    scales=np.maximum(scales,np.finfo(float).tiny)

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
                if (not np.isfinite(gram).all() or
                        np.linalg.cond(gram,p=1)>1.e12):
                    # A dormant angular slot makes its Gram singular,
                    # even when the requested physical endpoint has a
                    # perfectly valid representation of rank <= R.
                    # Reinitialize the full R slots by QR of the target.
                    B=_target_qr_angular_partner(T,seed.rank)
                    y[c]=B
                    gram=B@B.T
                Anew=np.linalg.solve(gram,(T@B.T).T)
                x[c]=0.
                x[c][:,ids]=Anew
                # An exact rank-1 field is also valid at declared rank
                # 2..8. Do not invert its necessarily singular X Gram.
                if (float(np.linalg.norm(x[c].T@y[c]-target[c]))/
                        scales[c]<=relative_tolerance):
                    continue
                gram=Anew@Anew.T
                if (not np.isfinite(gram).all() or
                        np.linalg.cond(gram,p=1)>1.e12):
                    # The current X span cannot determine all Y slots.
                    # QR gives another full-size seed; the outer field
                    # acceptance test remains mandatory.
                    y[c]=_target_qr_angular_partner(T,seed.rank)
                    x[c]=0.
                    x[c][:,ids]=(T@y[c].T).T
                    continue
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
