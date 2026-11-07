"""Consistent physical acceptance labels for coupled MICRO solver campaigns.

Shared by real 112-case and refinement studies. This module does not
import the solver or advance a physical load/time step. Fixed-rank
MVI acceptance is *not* external physical accuracy certification.
"""
from __future__ import annotations

def classify_accepted_steps(summary,entry):
    """Interpret accepted fixed-rank MVI as rank-limited, never full-space.

    For STATIC MVI, CONVERGED_PROJECTED is an accepted *rank-constrained*
    solution. Its ability to represent the true full-space problem remains
    unverified and must not be reported as CONVERGED_FULL.
    """
    steps=summary.get("steps",[])
    n=entry["steps"]
    if (summary.get("converged") is not True or
        summary.get("accepted_steps")!=n or len(steps)!=n or
        any(step.get("converged") is not True for step in steps)):
        raise ValueError("SCIENTIFIC_CONVERGENCE_STEPS_NOT_ACCEPTED")
    actual_status=summary.get("status")
    if entry["method"]=="MVI" and entry["kind"]=="STATIC":
        if actual_status not in ("CONVERGED_FULL","CONVERGED_PROJECTED"):
            raise ValueError("SCIENTIFIC_MVI_UNRECOGNIZED_ACCEPTED_STATUS")
        cert=steps[-1].get("mvi_certificate",{})
        is_full=actual_status=="CONVERGED_FULL"
        if (steps[-1].get("status")!=actual_status or
            cert.get("solution_status")!=actual_status or
            cert.get("accepted") is not True or
            cert.get("projected_accepted") is not True or
            cert.get("boundary_constraint_satisfied") is not True or
            cert.get("profile_reconstruction_satisfied") is not True or
            cert.get("projected_equilibrium_satisfied") is not True or
            cert.get("full_equilibrium_satisfied") is not is_full or
            steps[-1].get("full_equilibrium_satisfied") is not is_full):
            raise ValueError("SCIENTIFIC_MVI_CERTIFICATE_INCONSISTENT")
        return {
            "solution_class":("FULL_SPACE_STATIONARY" if is_full
                              else "RANK_CONSTRAINED_ONLY"),
            "full_space_stationarity_confirmed":is_full,
            "fixed_rank_stationarity_confirmed":True,
            "spatial_rank_limit_detected":not is_full,
        }
    if actual_status!="CONVERGED":
        raise ValueError("SCIENTIFIC_FULL_SPACE_SOLVE_NOT_CONVERGED")
    return {
        "solution_class":("DYNAMIC_STEP_ACCEPTED" if entry["kind"]=="DYNAMIC"
                          else "FULL_SPACE_STATIONARY"),
        # A dynamic MVI step is a time-advance acceptance, not an external
        # full-space rank certification; do not infer such a claim.
        "full_space_stationarity_confirmed":(
            True if entry["kind"]=="STATIC" else None),
        "fixed_rank_stationarity_confirmed":None,
        "spatial_rank_limit_detected":None,
    }

